# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""The vendored FlashTopK against Fleet's own indexer top-k, in the environment paddlefleet_ops installs it into.

This is not the package's own suite (tests/test_exact.py and friends run on PyTorch): it installs FlashTopK the way
`paddlefleet_ops/build_utils.py` installs every ecosystem library -- `pip install . --target <dir> --no-deps
--no-build-isolation --no-compile` -- imports the copy from there under `paddle.enable_compat(scope={"flashtopk"})`,
and runs it with **no PyTorch in the process at all** (`_NoRealTorch`, a meta_path finder behind Paddle's proxy that
answers ImportError for the real `torch` package), which is the environment the wheel lands in.

Reference A is `cudnn_indexer_topk_fwd(..., dsa_index_topk_backend="paddle")` of the local PaddleFleet -- the dense
cuDNN score matrix [B, S, S_k] fp32 plus the radix top-k that `flashtopk.indexer_topk` replaces.

Rows are compared with the tie-aware rule of tests/test_reference.py: a row whose index set differs is wrong only if a
key it swapped in scores more than a relative 1e-5 below the row's k-th fp64 reference score (or swapped out above it),
because cuDNN sums the heads in its own association and FlashTopK in four FMA chains, so near-ties at the k-th score may
swap. Every case also pins the answer's shape: canonical order, the [-1, -inf] padding, the lengths, the scores, and
the rule itself (candidates [valid_start, min(valid_end, causal)) computed in fp64 from the bf16 operands).

Run it on its own -- a real-torch process cannot share it, `torch` is Paddle's proxy here, and tests/conftest.py
imports PyTorch itself, hence --noconftest:

    cd FlashTopK && PYTHONPATH=<PaddleFleet>/src ../topk_env/bin/python -m pytest tests/test_paddle_indexer.py \
        -q --noconftest -W ignore -p no:cacheprovider
"""

from __future__ import annotations

import importlib
import importlib.machinery
import os
import subprocess
import sys
import tempfile
import types

import pytest

# The package's own suite (`pytest tests` on PyTorch) collects this file too, and it must not run there: its whole
# point is a process whose `torch` is Paddle's proxy and where PyTorch is not importable.
if "torch" in sys.modules and not hasattr(sys.modules["torch"], "_original_module"):
    pytest.skip("tests/test_paddle_indexer.py runs in its own process: see the module docstring", allow_module_level=True)

import numpy as np  # noqa: E402
import paddle  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL_TIE = 1e-5
HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5 * 1.3  # what Fleet passes as dim ** -0.5 * indexer_softmax_scale


# --- the vendored install, and the environment without PyTorch ----------------------------------------------------


class _NoRealTorch:
    """Real PyTorch is not importable in this process: paddlefleet_ops installs its ecosystem libraries with --no-deps
    into a Paddle environment, and `flashtopk` must run there. Paddle's compat proxy sits in front of this finder, so
    `import torch` inside `flashtopk.*` still resolves to Paddle; anything else asking for torch gets ImportError,
    which is exactly what a torch-less environment gives it (tvm_ffi and nvidia-cutlass-dsl both import torch inside
    `try: ... except ImportError`)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"this process has no PyTorch: {fullname}")
        return None


def _installed(prefix: str) -> str:
    """`pip install <repo> --target <dir> --no-deps --no-build-isolation --no-compile`, as build_utils.py does."""
    target = tempfile.TemporaryDirectory(prefix=prefix)  # noqa: SIM115 -- kept for the process's lifetime
    subprocess.run(
        [sys.executable, "-m", "pip", "install", REPO, "--target", target.name, "--no-deps", "--no-build-isolation",
         "--no-compile", "-q"],
        check=True, capture_output=True)
    return target


class _Guard:
    """How this process is set up: PyTorch blocked, the proxy enabled for `flashtopk`, the vendored copy first on the
    path. `flashtopk`'s assertion runs after everything else, so the package is imported by the tests below."""

    def __init__(self):
        self.vendored = _installed("flashtopk-vendored-")
        sys.path.insert(0, self.vendored.name)


paddle.enable_compat(scope={"flashtopk"}, silent=True)
# Behind the proxy (so in-scope `import torch` gets Paddle) but ahead of the PathFinder (so nothing else can read a
# PyTorch install off the disk).
sys.meta_path.insert(sys.meta_path.index(importlib.machinery.PathFinder), _NoRealTorch())
_VENDORED = _Guard()

import flashtopk  # noqa: E402 -- after the vendoring and the proxy, and only from the installed copy
import flashtopk.indexer as entry  # noqa: E402
from flashtopk.kernels import launch as kernellaunch  # noqa: E402


# --- plumbing -----------------------------------------------------------------------------------------------------


def _fleet():
    """`paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn`, importing the module alone if the package cannot.

    The checkout's `paddlefleet/__init__.py` pulls in the training stack, which needs a paddlefleet_ops build with the
    FlashMask cute kernels this environment does not have; the indexer module itself only needs paddlefleet_ops'
    cuDNN frontend, so a stub package holding the checkout's `__path__` gets it without the rest."""
    name = "paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn"
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    for entry_path in sys.path:
        candidate = os.path.join(entry_path, "paddlefleet")
        if os.path.isfile(os.path.join(candidate, "__init__.py")):
            stub = types.ModuleType("paddlefleet")
            stub.__path__ = [candidate]
            sys.modules.setdefault("paddlefleet", stub)
            break
    module = importlib.import_module(name)
    assert module.__file__, "paddlefleet is not importable; put its src directory on PYTHONPATH"
    return module


def _operands(batch, seq_len, heads, seq_kv, seed):
    paddle.seed(seed)
    index_q = paddle.randn([batch, seq_len, heads, HEAD_DIM], dtype="bfloat16")
    index_k = paddle.randn([batch, seq_kv, HEAD_DIM], dtype="bfloat16")
    weights = paddle.randn([batch, seq_len, heads], dtype="bfloat16")
    return index_q, index_k, weights


def _docmask(ranges, batch, seq_len):
    """[B, S, 2] int32 window tensor of the rows' (start, end) pairs."""
    return paddle.to_tensor(np.array(ranges, dtype="int32").reshape(batch, seq_len, 2))


def _rule_windows(batch, seq_len, seq_kv, ratio, seq_offset, ranges):
    """(ks, ke) int64 [batch * seq_len] of the documented rule: [valid_start, min(valid_end, causal)), with
    causal = min((seq_offset + t + 1) // ratio, S_k)."""
    starts, ends = [], []
    for b in range(batch):
        for t in range(seq_len):
            causal = min((seq_offset + t + 1) // ratio, seq_kv)
            start = 0 if ranges is None else int(ranges[b, t, 0])
            end = causal if ranges is None else min(int(ranges[b, t, 1]), causal)
            starts.append(start)
            ends.append(end)
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def _fp64(index_q, index_k, weights, ranges, ratio, seq_offset):
    """[batch * seq_len, S_k] float64 scores from the bf16 operands, -inf outside each row's window."""
    batch, seq_len, heads, dim = (int(v) for v in index_q.shape)
    seq_kv = int(index_k.shape[1])
    query, key, weight = index_q.astype("float64").numpy(), index_k.astype("float64").numpy(), \
        weights.astype("float64").numpy()
    scores = np.empty((batch * seq_len, seq_kv), dtype=np.float64)
    for b in range(batch):
        dots = np.einsum("shd,td->sht", query[b], key[b])
        np.maximum(dots, 0.0, out=dots)
        scores[b * seq_len:(b + 1) * seq_len] = np.einsum("sh,sht->st", weight[b], dots) * SCALE
    starts, ends = _rule_windows(batch, seq_len, seq_kv, ratio, seq_offset, ranges)
    columns = np.arange(seq_kv)
    inside = (columns[None] >= starts[:, None]) & (columns[None] < ends[:, None])
    scores[~inside] = -np.inf
    return scores, starts, ends


def _canonical(indices, scores, top_k):
    """The order contract: -1/-inf padding only at the end, scores descending, ids ascending on equal scores."""
    live = indices >= 0
    assert (live == (np.arange(top_k)[None] < live.sum(1, keepdims=True))).all(), "padding is not trailing"
    for row in range(indices.shape[0]):
        n = int(live[row].sum())
        ids, values = indices[row, :n], scores[row, :n]
        assert np.all(np.isfinite(values)), f"row {row} has a -inf in a live slot"
        assert (np.diff(values) <= 0).all(), f"row {row} is not sorted by score"
        equal = np.diff(values) == 0
        assert (np.diff(ids)[equal] > 0).all(), f"row {row} does not break score ties by id"
        assert (np.isneginf(scores[row, n:])).all(), f"row {row} padding is not -inf"


def _check(label, mine, theirs, scores_ref, starts, ends, top_k):
    """The context's acceptance: tie-aware set differences, lengths, scores on the shared ids, the order."""
    mine_idx, mine_len, mine_scores = mine
    ref_idx, ref_len, ref_scores = theirs
    ref_scores = np.where(ref_idx < 0, -np.inf, ref_scores)
    mine_scores = np.where(mine_idx < 0, -np.inf, mine_scores)
    live_mine, live_ref = mine_idx >= 0, ref_idx >= 0
    assert bool((mine_len == ref_len).all()), \
        f"{label}: topk_length differs (mine {mine_len.tolist()}, Fleet {ref_len.tolist()})"
    assert ref_idx.shape == mine_idx.shape == (len(starts), top_k), "the rows do not line up with the windows"
    rows_differ, differing, max_swaps, violations = 0, 0, 0, []
    for row in range(ref_idx.shape[0]):
        got = set(int(v) for v in mine_idx[row, live_mine[row]])
        want = set(int(v) for v in ref_idx[row, live_ref[row]])
        if got == want:
            continue
        rows_differ += 1
        differing += len(got ^ want)
        max_swaps = max(max_swaps, len(got - want))
        selected = min(top_k, max(int(ends[row] - starts[row]), 0))
        if selected == 0:
            violations.append((row, "a non-empty selection out of an empty window"))
            continue
        theta = float(np.sort(scores_ref[row])[::-1][selected - 1])
        tol = REL_TIE * abs(theta)
        for key in got - want:
            if scores_ref[row, key] < theta - tol:
                violations.append((row, f"swapped in {key}: {scores_ref[row, key]:.7f} < {theta:.7f} - {tol:.2e}"))
        for key in want - got:
            if scores_ref[row, key] > theta + tol:
                violations.append((row, f"swapped out {key}: {scores_ref[row, key]:.7f} > {theta:.7f} + {tol:.2e}"))
    assert not violations, f"{label}: {len(violations)} tie-rule violations, first: {violations[:4]}"
    shared = live_mine & live_ref & (mine_idx == ref_idx)
    if shared.any():
        scale = np.maximum(np.abs(ref_scores[shared]), 1e-30)
        score_error = float(np.max(np.abs(mine_scores[shared] - ref_scores[shared]) / scale))
        assert score_error < 1e-4, f"{label}: scores differ by {score_error:.3e} on the shared ids"
    else:
        score_error = 0.0
    _canonical(mine_idx, mine_scores, top_k)
    live = max(int(live_mine.sum()), int(live_ref.sum()), 1)
    stats = dict(rows=int(ref_idx.shape[0]), rows_differ=rows_differ, selections_differ=differing / live,
                 max_swaps=max_swaps, score_error=score_error)
    print(f"{label}: {stats}")
    return stats


def _compare(index_q, index_k, weights, top_k, *, ratio=1, seq_offset=0, valid_range=None, prune="off", label="case",
             both=True):
    """Fleet and FlashTopK on the same operands: the same ids and lengths, near-identical scores, the rule respected."""
    ranges = None if valid_range is None else valid_range.numpy()
    ref_idx, ref_len, ref_scores = _fleet().cudnn_indexer_topk_fwd(
        index_q, index_k, weights, ratio=ratio, topk_effective=top_k,
        indexer_softmax_scale=SCALE / HEAD_DIM**-0.5, valid_range=valid_range, seq_offset=seq_offset,
        return_topk_scores=True)
    mine_idx, mine_len, mine_scores = flashtopk.indexer_topk(
        index_q, index_k, weights, top_k, softmax_scale=SCALE, ratio=ratio, seq_offset=seq_offset,
        valid_range=valid_range, return_scores=True, pruning=prune)
    assert (ref_idx.numpy()[ref_idx.numpy() >= 0] < int(index_k.shape[1])).all(), \
        f"{label}: Fleet returned an id outside [0, S_k) -- out-of-contract window"
    theirs = (ref_idx.numpy().reshape(-1, top_k), ref_len.numpy().reshape(-1),
              ref_scores.numpy().reshape(-1, top_k))
    mine = (mine_idx.numpy().reshape(-1, top_k), mine_len.numpy().reshape(-1),
            mine_scores.numpy().reshape(-1, top_k))
    paddle.device.synchronize()
    scores_ref, starts, ends = _fp64(index_q, index_k, weights, ranges, ratio, seq_offset)
    stats = _check(label, mine, theirs, scores_ref, starts, ends, top_k)
    if both:  # the same inputs again, without return_scores
        again = flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, ratio=ratio,
                                       seq_offset=seq_offset, valid_range=valid_range, pruning=prune)
        assert again[2] is None, "return_scores=False must give None"
        assert np.array_equal(again[0].numpy().reshape(-1, top_k), mine[0])
        assert np.array_equal(again[1].numpy().reshape(-1), mine[1])
    return stats


# --- the vendoring, the proxy, the blocked PyTorch ----------------------------------------------------------------


def test_vendored_copy_is_the_one_imported():
    """The package under test is the one pip installed into a target directory, not the checkout's."""
    assert os.path.realpath(flashtopk.__file__).startswith(os.path.realpath(_VENDORED.vendored.name)), \
        f"imported {flashtopk.__file__}, not the vendored copy"
    assert os.path.isfile(os.path.join(_VENDORED.vendored.name, "flashtopk", "indexer.py"))


def test_no_real_torch_in_the_process():
    """`torch` is Paddle's proxy in every flashtopk module, and PyTorch itself is not importable: the wheel runs in the
    torch-less environment paddlefleet_ops installs into."""
    assert kernellaunch.PROXY, "the package did not see Paddle's compat proxy"
    try:
        imported = importlib.import_module("torch")
    except ImportError:
        imported = None
    assert imported is None or hasattr(imported, "_original_module"), \
        f"torch resolved to {getattr(imported, '__file__', imported)!r}, not to Paddle's proxy"
    for module in (flashtopk.indexer, flashtopk.topk, kernellaunch):
        assert hasattr(module.torch, "_original_module"), f"{module.__name__} holds a real torch"
    assert getattr(flashtopk.indexer.torch, "Tensor", None) is paddle.Tensor


def test_exports():
    """The entry point under both names, and no torch.library op in a process that has no torch.library."""
    assert flashtopk.indexer_topk is flashtopk.flashtopk_indexer_topk is entry.indexer_topk
    assert "ops" not in flashtopk.__all__, "the custom op was registered under Paddle's proxy"


def test_workspace_comes_from_paddles_allocator():
    """The selector's workspace is allocated through the active framework: Paddle's allocator holds it."""
    index_q, index_k, weights = _operands(1, 16, 32, 64, seed=101)
    paddle.device.synchronize()
    before = paddle.device.cuda.memory_allocated()
    flashtopk.indexer_topk(index_q, index_k, weights, 32, softmax_scale=SCALE, pruning="off")
    paddle.device.synchronize()
    grown = paddle.device.cuda.memory_allocated() - before
    arena = min(flashtopk.workspace_bytes(16, 64, 32, num_heads=32), 1 << 30)
    assert grown >= 0.5 * arena, f"the workspace ({grown} bytes) is not in Paddle's allocator ({arena} expected)"


def test_selector_is_shared_by_both_return_scores_calls():
    """One selector -- one workspace -- per (device, dtype, heads, top_k, pruning): return_scores only decides whether
    the entry allocates a scores buffer of its own."""
    index_q, index_k, weights = _operands(1, 8, 32, 16, seed=103)
    top_k = 4
    entry._SELECTORS.clear()
    first = flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, pruning="off")
    selector = entry._SELECTORS[(0, entry.torch.bfloat16, 32, top_k, "off")]
    with_scores = flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, pruning="off",
                                         return_scores=True)
    assert len(entry._SELECTORS) == 1, f"return_scores created a second selector: {list(entry._SELECTORS)}"
    assert entry._SELECTORS[(0, entry.torch.bfloat16, 32, top_k, "off")] is selector
    assert np.array_equal(first[0].numpy(), with_scores[0].numpy()), "the scores buffer changed the selection"
    assert with_scores[2] is not None and with_scores[2].shape == (1, 8, top_k)


def test_pruning_off_compiles_no_pruning_kernel():
    """pruning="off" never runs the mask, the refine or the Triton planner, so plan(warmup=True) compiles none of them
    (each pruning kernel is functools.cache'd, so an empty cache is a kernel that was never compiled)."""
    heads, top_k = 64, 8
    index_q, index_k, weights = _operands(1, 4, heads, 32, seed=107)
    flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, pruning="off")
    assert kernellaunch._compile_prune_sm100.cache_info().currsize == 0, "the mask kernel was compiled"
    assert kernellaunch._compile_refine_sm100.cache_info().currsize == 0, "the refine kernel was compiled"
    assert "flashtopk.plan" not in sys.modules, "the Triton planner was imported by a pruning=off call"
    assert flashtopk.topk._planner.cache_info().currsize == 0


# --- the same answer as Fleet ------------------------------------------------------------------------------------


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("ratio,seq_offset", [(1, 0), (4, 0), (4, 5), (1, 3)])
def test_causal_matches_fleet(heads, ratio, seq_offset):
    """Causal windows (valid_range None) at 32 and 64 heads, both compression ratios, with and without a CP query
    offset, against the dense cuDNN score matrix + radix top-k."""
    operands = _operands(1, 12, heads, 16, seed=7 + heads + ratio)
    _compare(*operands, 6, ratio=ratio, seq_offset=seq_offset, label=f"causal h{heads} r{ratio} so{seq_offset}")


def test_batch_flattens():
    """B > 1: one selection over the batches' concatenated key buffers, ids folded back into [0, S_k) per batch."""
    operands = _operands(2, 5, 32, 7, seed=11)
    assert _compare(*operands, 4, ratio=2, seq_offset=3, label="batch2")["rows_differ"] == 0


def test_batch_flattens_with_document_mask():
    """B > 1 and a document mask: the windows are shifted by the batch's key offset and the ids folded back.

    Two ratio-aligned documents per sequence (queries 0-3 in columns 0-1, queries 4-5 in columns 2-...), the same
    document structure in both batches, so the ids of one batch's keys are never the other's."""
    operands = _operands(2, 6, 32, 8, seed=12)
    ranges = [(0, 0), (0, 1), (0, 1), (0, 2), (2, 2), (2, 3)] * 2
    stats = _compare(*operands, 3, ratio=2, valid_range=_docmask(ranges, 2, 6), label="batch2 docmask")
    assert stats["rows_differ"] == 0


def test_document_mask_matches_fleet():
    """Document mask (valid_range): two packed documents, ratio 4, with empty rows, against Fleet's dense path."""
    operands = _operands(1, 8, 32, 8, seed=13)
    ranges = [(0, 0), (0, 0), (0, 0), (0, 1), (2, 1), (2, 1), (2, 1), (2, 2)]
    stats = _compare(*operands, 3, ratio=4, valid_range=_docmask(ranges, 1, 8), label="docmask")
    assert stats["rows_differ"] == 0


def test_document_mask_h64():
    """The same document mask at 64 index heads (SM100's 4-row query tile)."""
    operands = _operands(1, 8, 64, 8, seed=23)
    ranges = [(0, 0), (0, 0), (0, 0), (0, 1), (2, 1), (2, 1), (2, 1), (2, 2)]
    stats = _compare(*operands, 6, ratio=4, valid_range=_docmask(ranges, 1, 8), label="docmask h64")
    assert stats["rows_differ"] == 0


def test_document_mask_ratio_one_with_offset():
    """Document mask at ratio 1 with a CP query offset and a document whose first row is empty."""
    operands = _operands(1, 6, 32, 10, seed=17)
    ranges = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)]
    stats = _compare(*operands, 4, ratio=1, seq_offset=2, valid_range=_docmask(ranges, 1, 6), label="docmask r1")
    assert stats["rows_differ"] == 0


def test_top_k_above_candidates():
    """top_k above S_k: every row has fewer candidates than slots, so both pad the tail with -1/-inf."""
    operands = _operands(1, 4, 32, 4, seed=19)
    assert _compare(*operands, 8, ratio=1, label="top_k>candidates")["rows_differ"] == 0


def test_window_past_causal_is_clipped():
    """A valid_end past the ratio-causal limit: the documented rule clips the window at causal, so the answer is the
    causal-only one.

    Fleet's dense path is out of contract there -- its left-aligned row keeps the -inf columns inside its counted
    prefix and its slot mask then trusts them (it can return an id outside [0, S_k), or crash the gather of
    `return_topk_scores`) -- so this case is checked against the fp64 rule, not against Fleet."""
    index_q, index_k, weights = _operands(1, 6, 32, 8, seed=37)
    clipped = flashtopk.indexer_topk(index_q, index_k, weights, 4, softmax_scale=SCALE, ratio=4, return_scores=True,
                                     valid_range=_docmask([(0, 8)] * 6, 1, 6), pruning="off")
    causal = flashtopk.indexer_topk(index_q, index_k, weights, 4, softmax_scale=SCALE, ratio=4, return_scores=True,
                                    pruning="off")
    for a, b in zip(clipped, causal):
        assert np.array_equal(a.numpy(), b.numpy()), "a window end past causal must be the causal window"
    paddle.device.synchronize()
    scores_ref, starts, ends = _fp64(index_q, index_k, weights, np.tile(np.array([[[0, 8]]], dtype=np.int32), (1, 6, 1)),
                                     4, 0)
    live = clipped[0].numpy().reshape(-1, 4)
    _canonical(live, clipped[2].numpy().reshape(-1, 4), 4)
    for row in range(6):
        selected = min(4, int(ends[row] - starts[row]))
        got = scores_ref[row][live[row][live[row] >= 0]]
        want = np.sort(scores_ref[row])[::-1][:selected]
        assert np.allclose(np.sort(got)[::-1], want, rtol=1e-9), f"row {row} is not the window's exact top-k"
    lengths = clipped[1].numpy().reshape(-1)
    assert lengths.tolist() == np.clip(ends - starts, 0, 4).tolist() == [0, 0, 0, 1, 1, 1], \
        "the clipped windows are the causal prefixes, and ratio 4 leaves the first three rows empty"


def test_repeatable():
    """The cached selector, its windows and its arena are reused: a second call returns the same bits."""
    index_q, index_k, weights = _operands(1, 9, 32, 10, seed=29)
    ranges = _docmask([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (0, 0), (1, 2), (2, 3)], 1, 9)
    calls = [flashtopk.indexer_topk(index_q, index_k, weights, 5, softmax_scale=SCALE, ratio=2, valid_range=ranges,
                                    return_scores=True, pruning="off") for _ in range(3)]
    for other in calls[1:]:
        for a, b in zip(calls[0], other):
            paddle.device.synchronize()
            assert np.array_equal(a.numpy(), b.numpy())


# --- the stream, the copies, the rules ---------------------------------------------------------------------------


def test_no_input_copy_and_outputs_written_in_place(monkeypatch):
    """q and the keys reach the kernels as the caller's own memory, the weights as one float32 copy, and the results
    are written into the tensors Paddle allocated: no copy of an input or of an output."""
    batch, seq_len, heads, seq_kv, top_k = 2, 4, 32, 6, 4
    index_q, index_k, weights = _operands(batch, seq_len, heads, seq_kv, seed=41)
    seen = {}
    real = flashtopk.Selector.run

    def spy(self, q, kv, weight, ks, ke, kv_scales=None, out=None):
        seen.update(q=q, kv=kv, weight=weight, ks=ks, ke=ke, out=out)
        return real(self, q, kv, weight, ks, ke, kv_scales, out)

    monkeypatch.setattr(flashtopk.Selector, "run", spy)
    indices, length, scores = flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, ratio=1,
                                                     return_scores=True, pruning="off")
    assert seen["q"].data_ptr() == index_q.data_ptr(), "q was copied on the way to the kernels"
    assert seen["kv"].data_ptr() == index_k.data_ptr(), "the keys were copied on the way to the kernels"
    assert tuple(seen["q"].shape) == (batch * seq_len, heads, HEAD_DIM)
    assert tuple(seen["kv"].shape) == (batch * seq_kv, HEAD_DIM)
    assert seen["weight"].dtype == entry.torch.float32 and tuple(seen["weight"].shape) == (batch * seq_len, heads)
    assert seen["ks"].dtype == entry.torch.int32 and tuple(seen["ks"].shape) == (batch * seq_len,)
    assert seen["ke"].dtype == entry.torch.int32 and tuple(seen["ke"].shape) == (batch * seq_len,)
    out_scores, out_indices = seen["out"]
    assert out_indices.data_ptr() == indices.data_ptr(), "the indices were not written in place"
    assert out_scores.data_ptr() == scores.data_ptr(), "the scores were not written in place"
    ids = indices.numpy()
    assert ((ids == -1) | ((ids >= 0) & (ids < seq_kv))).all(), "ids are not in [0, S_k) per batch"


def test_stream_raw(monkeypatch):
    """Everything the call launches runs on Paddle's current stream: the TVM-FFI environment stream the kernels read
    is Paddle's raw handle while the selection runs, and it is not stream 0."""
    import tvm_ffi

    batch, seq_len, heads, seq_kv, top_k = 1, 8, 32, 8, 4
    index_q, index_k, weights = _operands(batch, seq_len, heads, seq_kv, seed=43)
    device = index_q.place.gpu_device_id()
    flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, ratio=1, pruning="off")
    seen = {}
    real = flashtopk.Selector._run_rows  # inside the run's stream scope, and inside the kernels' launch path

    def spy(self, q, kv, kv_scales, weight, ks, ke, scores, indices):
        seen["ffi"] = tvm_ffi.get_raw_stream(tvm_ffi.device(f"cuda:{device}"))
        seen["paddle"] = paddle.device.current_stream().__cuda_stream__()[1]
        return real(self, q, kv, kv_scales, weight, ks, ke, scores, indices)

    monkeypatch.setattr(flashtopk.Selector, "_run_rows", spy)
    stream = paddle.device.Stream()
    with paddle.device.stream_guard(stream):
        flashtopk.indexer_topk(index_q, index_k, weights, top_k, softmax_scale=SCALE, ratio=1, return_scores=True,
                               pruning="off")
    raw = stream.__cuda_stream__()[1]
    assert seen["paddle"] == raw, "the guard did not put Paddle on the new stream"
    assert seen["ffi"] == raw, f"the kernels' environment stream was {seen['ffi']:#x}, Paddle on {raw:#x}"
    assert raw != 0 and seen["ffi"] != 0, "the call fell back to the legacy default stream"


def test_producer_and_consumer_order_on_a_paddle_stream():
    """A producer and a consumer on Paddle's stream, with no synchronization anywhere in between.

    The queries are poisoned, a long burn runs on the stream, and only then the real queries are copied in -- as
    asynchronously as the copy before it. A call that launched its kernels anywhere else would read the poison and
    select different keys; a consumer that ran before the call would read a stale answer."""
    index_q, index_k, weights = _operands(1, 8, 32, 8, seed=47)
    real_q = index_q.clone()
    reference = flashtopk.indexer_topk(real_q, index_k, weights, 4, softmax_scale=SCALE, ratio=1, return_scores=True,
                                       pruning="off")[0].numpy()
    poison = paddle.randn(index_q.shape, dtype="bfloat16") * 64.0
    paddle.device.synchronize()
    stream = paddle.device.Stream()
    with paddle.device.stream_guard(stream):
        index_q.copy_(poison, blocking=False)
        burn = paddle.randn([1024, 1024], dtype="float32")
        for _ in range(12):
            burn = paddle.matmul(burn, burn)
        index_q.copy_(real_q, blocking=False)
        indices, length, scores = flashtopk.indexer_topk(index_q, index_k, weights, 4, softmax_scale=SCALE, ratio=1,
                                                        return_scores=True, pruning="off")
        counts = (indices >= 0).astype("int32").sum(-1)  # a consumer, also unsynchronized
    paddle.device.synchronize()
    assert np.array_equal(indices.numpy(), reference), "the selection read the queries before the producer wrote them"
    assert np.array_equal(counts.numpy().reshape(-1), length.numpy().reshape(-1)), \
        "the consumer read a stale result"


def test_grad_semantics():
    """Indices only: the results carry no gradient, whatever the inputs' flags are."""
    index_q, index_k, weights = _operands(1, 6, 32, 6, seed=59)
    index_q.stop_gradient = False
    weights.stop_gradient = False
    indices, length, scores = flashtopk.indexer_topk(index_q, index_k, weights, 3, softmax_scale=SCALE, ratio=1,
                                                     return_scores=True, pruning="off")
    assert indices.stop_gradient and length.stop_gradient and scores.stop_gradient
    assert (indices.dtype, length.dtype, scores.dtype) == (paddle.int32, paddle.int32, paddle.float32)
    assert not index_q.stop_gradient, "the call must not have touched the inputs' grad flags"


def test_rejects_bad_inputs():
    """The contracts Fleet's validator enforces, and the ones FlashTopK adds."""
    index_q, index_k, weights = _operands(1, 4, 32, 4, seed=61)
    with pytest.raises(TypeError):
        flashtopk.indexer_topk(index_q.numpy(), index_k, weights, 2, softmax_scale=SCALE)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q, index_k, weights[:, :, :16], 2, softmax_scale=SCALE)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q.astype("float32"), index_k, weights, 2, softmax_scale=SCALE)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q, index_k, weights, 0, softmax_scale=SCALE)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q, index_k, weights, 4097, softmax_scale=SCALE)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q, index_k, weights, 2, softmax_scale=SCALE, ratio=0)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q, index_k, weights, 2, softmax_scale=SCALE, seq_offset=-1)
    with pytest.raises(ValueError):
        flashtopk.indexer_topk(index_q, index_k, weights, 2, softmax_scale=SCALE,
                               valid_range=paddle.zeros([1, 5, 2], dtype="int32"))


def test_pruning_under_the_proxy():
    """Pruning needs the Triton planning kernel, which imports triton -- a third-party import that is *not* in the
    proxy's scope, so this checks whether that works at all rather than requiring it: pruning="off" is what the
    indexer uses, and the pruning paths keep their PyTorch behaviour."""
    index_q, index_k, weights = _operands(1, 8, 32, 8, seed=71)
    try:
        pruned = flashtopk.indexer_topk(index_q, index_k, weights, 4, softmax_scale=SCALE, ratio=1, pruning="always")
    except Exception as error:  # noqa: BLE001 -- reported, not required
        pytest.skip(f"pruning under the proxy is unsupported: {type(error).__name__}: {error}")
    plain = flashtopk.indexer_topk(index_q, index_k, weights, 4, softmax_scale=SCALE, ratio=1, pruning="off")
    assert np.array_equal(pruned[0].numpy(), plain[0].numpy()), "pruning changed the selection"
