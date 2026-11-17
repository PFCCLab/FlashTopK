# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""FlashTopK: exact, deterministic top-k selection for learned sparse attention.

    selector = flashtopk.Selector(topk=2048, dtype=torch.float8_e4m3fn).plan(max_rows, max_kv)
    indices = selector.run(q, kv, weights, ks, ke, kv_scales)

or flashtopk.select(...) for the same with a cached selector, and flashtopk.indexer_topk(...) for the indexer's own
entry point (a batch of [B, S, H, 128] queries, windows from a document mask or the causal limit). See topk.py for the
score and the output order.

The package runs on PyTorch, and on PaddlePaddle tensors under `paddle.enable_compat(scope={"flashtopk"})`, where its
`import torch` resolves to Paddle's compatibility proxy; see kernels/launch.py. `torch.ops.flashtopk.select` is a PyTorch
custom op (`flashtopk.ops`, for torch.compile), so it is only registered when the process has PyTorch itself.
"""

from .indexer import indexer_topk, indexer_topk as flashtopk_indexer_topk
from .kernels import launch
from .topk import DEFAULT_CHUNK_SIZE, Info, Selector, select, supported, windows_from_cu_seqlens, workspace_bytes

__all__ = ["DEFAULT_CHUNK_SIZE", "Info", "Selector", "flashtopk_indexer_topk", "indexer_topk", "select",
           "supported", "windows_from_cu_seqlens", "workspace_bytes"]
__version__ = "0.1.0"

if not launch.PROXY:
    from . import ops  # registers torch.ops.flashtopk.select; Paddle has no torch.library.custom_op to register with

    __all__.append("ops")
