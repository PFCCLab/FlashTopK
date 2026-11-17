# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The indexer's top-k as one call: Fleet's `cudnn_indexer_topk_fwd` on [B, S, H, 128] query batches.

    indices, length, scores = flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=scale,
                                                    valid_range=valid_range, return_scores=True)

Row (b, t) selects the top_k keys of its window -- the document mask's [valid_start, valid_end) clipped to the
ratio-causal limit, or the causal prefix -- by

    score(b, t, k) = softmax_scale * sum_h weights[b, t, h] * relu(index_q[b, t, h] . index_k[b, k])

The scale multiplies the head sum, as the cuDNN path's does, so the two agree to within the head sum's association.
Ids are global per batch ([0, S_k)), the padding is -1 (indices) and -inf (scores), and topk_length is the number of
keys the row's window held -- min(top_k, window), which is Fleet's `(topk_indices >= 0).sum(-1)`.

Written with the torch API only, so it runs on PyTorch and on PaddlePaddle tensors under
`paddle.enable_compat(scope={"flashtopk"})` (see kernels/launch.py): every allocation goes through the active framework's
allocator, and the kernels are pointed at its current stream.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch

from .kernels.launch import device_index as _device_index
from .topk import MAX_TOPK, Selector

__all__ = ["indexer_topk"]

HEAD_DIM = 128
HEADS = (32, 64)
# One selector per (device, dtype, heads, top_k, pruning), whose workspace grows to the largest call seen. It is
# shared by calls that want the scores and calls that do not: return_scores only decides whether the entry allocates a
# [B, S, top_k] scores buffer of its own and hands it to the selector as an output.
_SELECTORS: dict = {}


def _check_inputs(index_q, index_k, weights, valid_range):
    """The shapes and dtypes Fleet's validator requires, plus the valid_range contract."""
    for name, tensor in (("index_q", index_q), ("index_k", index_k), ("weights", weights)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be on a GPU, got device {tensor.device}; the kernels are CUDA ones")
    if index_q.dim() != 4:
        raise ValueError(f"index_q must be [B, S, H, D], got {tuple(index_q.shape)}")
    if index_k.dim() != 3:
        raise ValueError(f"index_k must be [B, S_k, D], got {tuple(index_k.shape)}")
    if weights.dim() != 3:
        raise ValueError(f"weights must be [B, S, H], got {tuple(weights.shape)}")
    batch, seq_len, heads, dim = (int(v) for v in index_q.shape)
    batch_k, seq_kv, dim_k = (int(v) for v in index_k.shape)
    batch_w, seq_len_w, heads_w = (int(v) for v in weights.shape)
    if batch_k != batch or batch_w != batch:
        raise ValueError(f"batch mismatch: index_q={tuple(index_q.shape)}, index_k={tuple(index_k.shape)}, "
                         f"weights={tuple(weights.shape)}")
    if dim_k != dim or seq_kv < 1:
        raise ValueError(f"index_k must be [B, S_k >= 1, {dim}], got {tuple(index_k.shape)}")
    if (seq_len_w, heads_w) != (seq_len, heads):
        raise ValueError(f"weights must be [{batch}, {seq_len}, {heads}], got {tuple(weights.shape)}")
    if heads not in HEADS:
        raise ValueError(f"H must be one of {HEADS}, got {heads}")
    if dim != HEAD_DIM:
        raise ValueError(f"D must be {HEAD_DIM}, got {dim}")
    if index_q.dtype != torch.bfloat16 or index_k.dtype != torch.bfloat16:
        raise ValueError(f"index_q and index_k must be bfloat16, got {index_q.dtype} and {index_k.dtype}")
    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"weights must be bfloat16 or float32, got {weights.dtype}")
    if index_k.device != index_q.device or weights.device != index_q.device:
        raise ValueError(f"index_q, index_k and weights must share a device, got {index_q.device}, "
                         f"{index_k.device}, {weights.device}")
    if valid_range is not None:
        if not isinstance(valid_range, torch.Tensor):
            raise TypeError(f"valid_range must be a tensor or None, got {type(valid_range)!r}")
        if tuple(int(v) for v in valid_range.shape) != (batch, seq_len, 2):
            raise ValueError(f"valid_range must be [{batch}, {seq_len}, 2], got {tuple(valid_range.shape)}")
        if valid_range.device != index_q.device:
            raise ValueError(f"valid_range is on {valid_range.device}, the queries on {index_q.device}")
    return batch, seq_len, heads, dim, seq_kv


@functools.lru_cache(maxsize=8)
def _causal_windows(batch: int, seq_len: int, seq_kv: int, ratio: int, seq_offset: int, device_index: int):
    """Causal prefix windows over the batches' concatenated key buffers: (batch * seq_len,) int32, or (None, ends) for
    B == 1, where every start is 0. Row t's end is min(S_k, (seq_offset + t + 1) // ratio)."""
    device = torch.device("cuda", device_index)
    query = torch.arange(seq_len, dtype=torch.int32, device=device)
    causal = ((seq_offset + 1 + query) // ratio).clamp(max=seq_kv)
    if batch == 1:
        return None, causal
    offset = torch.arange(batch, dtype=torch.int32, device=device) * seq_kv
    ends = (causal.unsqueeze(0) + offset.unsqueeze(1)).reshape(batch * seq_len)
    starts = offset.unsqueeze(1).expand(batch, seq_len).reshape(batch * seq_len)
    return starts, ends


@functools.lru_cache(maxsize=8)
def _tail_offsets(batch: int, seq_kv: int, device_index: int) -> torch.Tensor:
    """b * S_k for the batches 1 .. B-1, [B-1, 1, 1] int32, to fold the concatenated buffer's ids back per batch."""
    device = torch.device("cuda", device_index)
    return (torch.arange(1, batch, dtype=torch.int32, device=device) * seq_kv).reshape(batch - 1, 1, 1)


@functools.lru_cache(maxsize=8)
def _causal_length(batch: int, seq_len: int, seq_kv: int, ratio: int, seq_offset: int, top_k: int,
                   device_index: int) -> torch.Tensor:
    """The causal path's length [batch * seq_len] int32: min(top_k, causal), the same for every batch."""
    _, ends = _causal_windows(1, seq_len, seq_kv, ratio, seq_offset, device_index)
    return ends.clamp(max=top_k).reshape(1, seq_len).expand(batch, seq_len).reshape(batch * seq_len)


@functools.lru_cache(maxsize=8)
def _zeros(rows: int, device_index: int) -> torch.Tensor:
    return torch.zeros(rows, dtype=torch.int32, device=torch.device("cuda", device_index))


def _windows(device_index, valid_range, batch, seq_len, seq_kv, ratio, seq_offset):
    """(ks, ke) int32 [batch * seq_len] over the concatenated key buffers, in FlashTopK's convention: row r's keys are
    [ks[r], ke[r]) of the flattened KV.

    The causal windows depend on nothing per call, so they are kept -- a row's start is 0 with B == 1 and its batch's
    first key id otherwise. A document mask's starts and ends do depend on the call, and cost a handful of elementwise
    kernels over [batch * seq_len] (nothing next to a selection)."""
    starts, ends = _causal_windows(batch, seq_len, seq_kv, ratio, seq_offset, device_index)
    if valid_range is None:
        return (_zeros(batch * seq_len, device_index) if starts is None else starts), ends
    if valid_range.dtype != torch.int32:
        valid_range = valid_range.to(torch.int32)
    window = valid_range.reshape(batch * seq_len, 2)
    start, end = window[:, 0], window[:, 1]
    if starts is not None:
        start, end = start + starts, end + starts
    # A start past the last key leaves the window empty (ks >= ke), as Fleet's radix kernel sees a count of 0.
    return start.clamp(min=0), torch.minimum(ends, end)


def _selector(device_index: int, heads: int, top_k: int, pruning: str, rows: int, seq_kv: int) -> Selector:
    """The cached selector of this configuration, planned (workspace allocated, kernels compiled) for this call."""
    key = (device_index, torch.bfloat16, heads, top_k, pruning)
    selector = _SELECTORS.get(key)
    if selector is None:
        selector = _SELECTORS[key] = Selector(top_k, num_heads=heads, dtype=torch.bfloat16,
                                              device=torch.device("cuda", device_index), pruning=pruning)
    if selector.max_rows < min(rows, selector.chunk_size) or selector.max_kv < seq_kv:
        selector.plan(rows, seq_kv)
    return selector


def indexer_topk(index_q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor, top_k: int, *,
                 softmax_scale: float, ratio: int = 1, seq_offset: int = 0,
                 valid_range: Optional[torch.Tensor] = None, return_scores: bool = False,
                 pruning: str = "off"
                 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fleet's `cudnn_indexer_forward` + `_select_indexer_top_k` as one FlashTopK call.

    The annotations say `torch.Tensor`: under `paddle.enable_compat(scope={"flashtopk"})` (how paddlefleet_ops loads
    this package) the arguments and results are `paddle.Tensor`, and `torch.Tensor` *is* `paddle.Tensor` there.

    Args:
        index_q: [B, S, H, 128] bfloat16 indexer queries (H 32 or 64, 64 on SM100).
        index_k: [B, S_k, 128] bfloat16 compressed indexer keys.
        weights: [B, S, H] bfloat16 (or float32) per-head weights.
        top_k: entries selected per query row, 1 .. 4096.
        softmax_scale: score scale, D ** -0.5 * indexer_softmax_scale in Fleet's terms; applied to the head sum.
        ratio: compression ratio of the causal limit, `(seq_offset + t + 1) // ratio`.
        seq_offset: global query position of the first row (context-parallel causal mode).
        valid_range: optional [B, S, 2] int32 document-mask window [start, end) per row; None for a causal prefix.
        return_scores: also return the selected scores (fp32), -inf in the padding slots.
        pruning: FlashTopK pruning mode ("off", "always", "adaptive"). "off" scans every key; it is the right mode
            for the indexer at 32 or 64 heads, where pruning costs more than it saves.

    Returns:
        (topk_indices [B, S, top_k] int32 global ids with -1 padding, topk_length [B, S] int32, topk_scores [B, S,
        top_k] fp32 with -inf padding, or None without return_scores).
    """
    batch, seq_len, heads, dim, seq_kv = _check_inputs(index_q, index_k, weights, valid_range)
    top_k, ratio, seq_offset = int(top_k), int(ratio), int(seq_offset)
    if not 1 <= top_k <= MAX_TOPK:
        raise ValueError(f"top_k must be in [1, {MAX_TOPK}], got {top_k}")
    if ratio < 1:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if seq_offset < 0:
        raise ValueError(f"seq_offset must be >= 0, got {seq_offset}")
    scale = float(softmax_scale)
    rows = batch * seq_len
    device = index_q.device
    device_index = _device_index(index_q)

    # The active framework allocates the results (Paddle's allocator under the proxy) and the kernels write them in
    # place: no copy of an output either.
    indices = torch.empty([batch, seq_len, top_k], dtype=torch.int32, device=device)
    length = torch.empty([batch, seq_len], dtype=torch.int32, device=device)
    scores = torch.empty([batch, seq_len, top_k], dtype=torch.float32, device=device) if return_scores else None

    with torch.no_grad():
        selector = _selector(device_index, heads, top_k, pruning, rows, seq_kv)
        query = index_q.reshape(rows, heads, dim)
        key = index_k.reshape(batch * seq_kv, dim)
        weight = weights.reshape(rows, heads)
        if weight.dtype != torch.float32:
            weight = weight.to(torch.float32)  # the score contract's one input copy
        starts, ends = _windows(device_index, valid_range, batch, seq_len, seq_kv, ratio, seq_offset)
        index_out = indices.reshape(rows, top_k)
        score_out = scores.reshape(rows, top_k) if return_scores else None
        selector.run(query, key, weight, starts, ends, None,
                     (score_out, index_out) if return_scores else index_out)
        if batch > 1:
            # The kernels see the batches' key buffers as one, so batch b's ids come back b * S_k too high. Fold the
            # offset back on the rows of the batches past the first; a -1 there goes to -1 - b * S_k < -1 and clamps
            # back to -1, while a real id (>= b * S_k) lands in [0, S_k).
            shifted = index_out.view(batch, seq_len, top_k)[1:]
            shifted.sub_(_tail_offsets(batch, seq_kv, device_index)).clamp_(min=-1, max=seq_kv)
        if score_out is not None and scale != 1.0:
            score_out.copy_(score_out * scale)
        # A row's slots are non-negative exactly for its min(top_k, window) best keys (every key in a window scores
        # finite), which is the value of Fleet's (topk_indices >= 0).sum(-1). The causal path's lengths are the same on
        # every call, so they are copied rather than recomputed.
        length_out = length.reshape(rows)
        if valid_range is None:
            length_out.copy_(_causal_length(batch, seq_len, seq_kv, ratio, seq_offset, top_k, device_index))
        else:
            torch.sub(ends, starts, out=length_out)
            length_out.clamp_(min=0, max=top_k)
    return indices, length, scores
