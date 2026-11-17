# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Exact recovery of the rows the selection could not finish, enqueued on every call and decided on the device.

A row is bad when its candidates overflowed the buffer or it selected fewer than min(topk, window) keys. The
selection counts bad rows and records the first RECOVERY_ROWS of them; triage (kernels/triage.py) then picks:

  * per-row recovery, at most RECOVERY_ROWS bad rows: their q rows are gathered into slots, scored densely over their
    windows, and each slot's exact top-k replaces its row. Windows of at least SPLIT_MIN_KEYS keys are cut into
    slices whose top-ks are selected first, so a long row keeps the whole GPU busy;
  * whole-call recovery, more bad rows: every row is rescored densely, `chunk` rows at a time, and reselected;
  * nothing: every stage sees empty windows and unmapped slots, and returns at once.

A recovered row is the exact top-k of the dense scan's scores, which are the scan's own bits, in canonical order:
the output does not depend on which path produced it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .kernels import launch
from .kernels.common import HEAD_DIM, NEXT_N, NUM_HEADS
from .kernels.selection import RECOVERY_ROWS

SPLIT_MIN_KEYS = 32768
SPLIT_KEYS = 16384
MAX_SPLITS = 64


def recovery_splits(seq_len_kv: int) -> int:
    """Slices a per-row recovery slot is cut into: one per SPLIT_KEYS keys from SPLIT_MIN_KEYS keys."""
    if seq_len_kv < SPLIT_MIN_KEYS:
        return 1
    return min(MAX_SPLITS, math.ceil(seq_len_kv / SPLIT_KEYS))


def dense_rows(arena_elements: int, seq_len_kv: int, num_q: int) -> int:
    """Rows whole-call recovery scores at a time in an arena of `arena_elements` floats: whole 8-row tiles."""
    rows = arena_elements // max(seq_len_kv, 1) // NEXT_N * NEXT_N
    return max(NEXT_N, min(rows, math.ceil(num_q / NEXT_N) * NEXT_N))


class Recovery:
    """One plan's recovery buffers, for up to `max_rows` rows and `max_kv` keys. `arena` is the call's float
    workspace (free once the selection has run), at least RECOVERY_ROWS * max_kv floats."""

    def __init__(self, max_rows: int, max_kv: int, topk: int, dtype: torch.dtype, arena: torch.Tensor,
                 num_heads: int = NUM_HEADS, arena_elements: Optional[int] = None):
        dev = arena.device
        self.num_heads = num_heads
        # The arena's length as a Python int, from the caller that allocated it. Reading it back off the tensor
        # (`int(arena.numel())`) is a device-to-host read of a 0-d CUDA tensor under Paddle's compat proxy, and that
        # read waits for the stream -- i.e. for every kernel the call has already enqueued.
        if arena_elements is None:
            arena_elements = int(arena.numel())
        self.arena_elements = int(arena_elements)
        if self.arena_elements < RECOVERY_ROWS * max_kv:
            raise ValueError("the recovery arena must hold RECOVERY_ROWS rows of dense scores")
        i32 = dict(dtype=torch.int32, device=dev)
        words_per_head = HEAD_DIM * torch.empty((), dtype=dtype).element_size() // 4
        self.topk = topk
        self.arena = arena
        self.row_map = torch.empty(RECOVERY_ROWS, **i32)
        self.slot_ks = torch.empty(RECOVERY_ROWS, **i32)
        self.slot_ke = torch.empty(RECOVERY_ROWS, **i32)
        self.q_gather = torch.empty(RECOVERY_ROWS * num_heads * words_per_head, **i32)
        self.w_gather = torch.empty(RECOVERY_ROWS, num_heads, dtype=torch.float32, device=dev)
        self.whole_map = torch.empty(max_rows, **i32)
        self.whole_ks = torch.empty(max_rows, **i32)
        self.whole_ke = torch.empty(max_rows, **i32)
        self.scratch = torch.empty(max(max_rows, RECOVERY_ROWS), **i32)
        self.max_splits = recovery_splits(max_kv)
        stage_rows = RECOVERY_ROWS * self.max_splits if self.max_splits > 1 else 0
        self.stage_scores = torch.empty(stage_rows, topk, dtype=torch.float32, device=dev)
        self.stage_indices = torch.empty(stage_rows, topk, **i32)
        self._views: dict = {}

    def _plan_views(self, num_q: int, seq_len_kv: int, sm90: bool, dtype: torch.dtype):
        """The aliases of this plan's buffers that a call needs, built once per call shape: a call makes about twenty
        of them, and under Paddle's compat proxy each slice or view is a framework call. The buffers are allocated
        once, in this plan, so the views stay valid for the plan's lifetime."""
        key = (num_q, seq_len_kv, sm90, dtype)
        views = self._views.get(key)
        if views is None:
            # per-row: the gathered slots
            q_slots = self.q_gather.view(dtype).view(RECOVERY_ROWS, self.num_heads, HEAD_DIM)
            splits = min(self.max_splits, recovery_splits(seq_len_kv))
            stage = RECOVERY_ROWS * splits
            # whole call: `chunk` rows at a time
            chunk = dense_rows(self.arena_elements, seq_len_kv, num_q)
            seen: dict = {}
            chunks = []
            for off in range(0, num_q, chunk):
                rows = min(chunk, num_q - off)
                part = slice(off, off + rows)
                if rows not in seen:
                    dense = self.arena[: rows * seq_len_kv].view(rows, seq_len_kv)
                    seen[rows] = (dense, dense.view(torch.int32), self.scratch[:rows])
                dense, dense_i32, scratch = seen[rows]
                chunks.append((part, dense, dense_i32, scratch, self.whole_ks[part], self.whole_ke[part],
                               self.whole_map[part]))
            slot_dense = self.arena[: RECOVERY_ROWS * seq_len_kv].view(RECOVERY_ROWS, seq_len_kv)
            views = self._views[key] = dict(
                slots=(slot_dense, slot_dense.view(torch.int32), q_slots,
                       launch.as_operand(q_slots).view(-1, HEAD_DIM, 1) if sm90 else None,
                       self.scratch[:RECOVERY_ROWS], splits, stage,
                       self.stage_scores[:stage], self.stage_indices[:stage]),
                whole=(self.whole_map[:num_q], self.whole_ks[:num_q], self.whole_ke[:num_q]),
                chunks=chunks, chunk=chunk)
        return views

    def run(self, q, kv, kv_scales, weights, ks, ke, health, bad_rows, scores, indices) -> None:
        """Enqueue the recovery of this call's selection (scores / indices, its health and bad-row table)."""
        num_q, seq_len_kv = q.shape[0], kv.shape[0]
        sm90 = launch.arch(q) == 9
        views = self._plan_views(num_q, seq_len_kv, sm90, q.dtype)
        whole_map, whole_ks, whole_ke = views["whole"]
        launch.triage(health, bad_rows, ks, ke, q, weights, self.row_map, self.slot_ks, self.slot_ke, self.q_gather,
                      self.w_gather, whole_map, whole_ks, whole_ke)

        # per-row: the gathered slots
        dense, dense_i32, q_slots, operand, scratch, splits, stage, stage_scores, stage_indices = views["slots"]
        launch.scan_dense(q_slots, kv, kv_scales, self.w_gather, self.slot_ks, self.slot_ke, dense, scratch,
                          q_operand=operand, out_i32=dense_i32)
        if splits > 1:
            launch.select_dense(dense, self.slot_ks, self.slot_ke, self.row_map, stage_scores, stage_indices, splits,
                                dense_i32=dense_i32)
            launch.select_staged(stage_scores, stage_indices, self.row_map, scores, indices)
        else:
            launch.select_dense(dense, self.slot_ks, self.slot_ke, self.row_map, scores, indices, dense_i32=dense_i32)

        # whole call, `chunk` rows at a time
        full_operand = launch.q_operand_sm90(q) if sm90 else None
        for part, dense, dense_i32, scratch, ks_part, ke_part, map_part in views["chunks"]:
            operand = None
            if sm90:  # the operand is laid out per call, so the slices are taken here rather than cached
                rows = part.stop - part.start
                operand = full_operand[part.start * NUM_HEADS:
                                       (part.start + math.ceil(rows / NEXT_N) * NEXT_N) * NUM_HEADS]
            launch.scan_dense(q[part], kv, kv_scales, weights[part], ks_part, ke_part, dense, scratch,
                              q_operand=operand, out_i32=dense_i32)
            launch.select_dense(dense, ks_part, ke_part, map_part, scores, indices, dense_i32=dense_i32)
