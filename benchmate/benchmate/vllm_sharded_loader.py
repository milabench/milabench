"""vLLM load format ``milabench_sharded``: per-rank raw weight shards for fast reloads.

With tensor parallelism, every rank of vLLM's default loader walks the whole checkpoint and
slices its share out of each tensor on the CPU (an hour for a 1.5 TB MXFP4 MoE checkpoint).
vLLM's own ``sharded_state`` saves the model *after* ``process_weights_after_loading``, which
renames and repacks quantized weights: a fresh model cannot load that state back.

This loader saves each rank's weights right after loading, *before* that processing:

* first run (no shards for this rank yet): normal load from the checkpoint, then the rank's
  state dict is written to ``<shard_dir>/vllm-<version>/pp<P>-tp<T>/rank-pp<p>-tp<t>-part<n>``;
* later runs: each rank reads only its own files and copies them into its parameters.

``process_weights_after_loading`` runs once in both cases (vLLM's base loader calls it).
Each rank writes and reads on its own node, so a node-local ``shard_dir`` works.

    --load-format milabench_sharded
    --model-loader-extra-config '{"shard_dir": "/path/to/shards"}'   # or VLLM_MILABENCH_SHARD_DIR

Registered through the ``vllm.general_plugins`` entry point, so every vLLM process (Ray
workers included) has it.
"""

from __future__ import annotations

import glob
import os
import time

LOAD_FORMAT = "milabench_sharded"
PART_BYTES = 4 << 30


def register():
    try:
        from vllm.model_executor.model_loader import register_model_loader
    except ImportError:
        return
    register_model_loader(LOAD_FORMAT)(_loader_class())


def _loader_class():
    import torch
    from copy import copy

    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

    class MilabenchShardedLoader(DefaultModelLoader):
        def __init__(self, load_config):
            extra = copy(load_config.model_loader_extra_config or {})
            # VLLM_* variables reach the Ray workers (vLLM copies them), argv JSON needs no escaping
            self.shard_dir = extra.pop("shard_dir", None) or os.environ.get("VLLM_MILABENCH_SHARD_DIR")
            if not self.shard_dir:
                raise ValueError(
                    f"load format {LOAD_FORMAT} needs model_loader_extra_config.shard_dir or VLLM_MILABENCH_SHARD_DIR"
                )
            load_config = copy(load_config)
            load_config.model_loader_extra_config = extra
            # the default loader picks the checkpoint files by load format: ours is safetensors
            load_config.load_format = "safetensors"
            super().__init__(load_config)

        def _rank_dir_and_prefix(self):
            import vllm
            from vllm.distributed import (
                get_pipeline_model_parallel_rank,
                get_pp_group,
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            pp, tp = get_pp_group().world_size, get_tensor_model_parallel_world_size()
            layout = os.path.join(self.shard_dir, f"vllm-{vllm.__version__}", f"pp{pp}-tp{tp}")
            rank = f"rank-pp{get_pipeline_model_parallel_rank()}-tp{get_tensor_model_parallel_rank()}"
            return layout, rank

        def load_weights(self, model, model_config):
            layout, rank = self._rank_dir_and_prefix()
            done = os.path.join(layout, f"{rank}.done")
            if os.path.exists(done):
                self._load_shards(model, sorted(glob.glob(os.path.join(layout, f"{rank}-part*.safetensors"))))
                return
            super().load_weights(model, model_config)
            self._save_shards(model, layout, rank, done)

        def _load_shards(self, model, files):
            from safetensors.torch import safe_open

            start = time.perf_counter()
            state = ShardedStateLoader._filter_subtensors(model.state_dict())
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        param = state.pop(key)
                        param.data.copy_(tensor)
            if state:
                raise ValueError(f"{LOAD_FORMAT}: missing keys in the shards: {sorted(state)[:5]}")
            print(f"[{LOAD_FORMAT}] loaded {len(files)} shard files in {time.perf_counter() - start:.1f}s", flush=True)

        def _save_shards(self, model, layout, rank, done):
            from safetensors.torch import save_file

            start = time.perf_counter()
            os.makedirs(layout, exist_ok=True)
            part, size, chunk = 0, 0, {}

            def flush():
                nonlocal part, size, chunk
                if chunk:
                    tmp = os.path.join(layout, f".{rank}-part{part:05d}.tmp")
                    save_file(chunk, tmp)
                    os.replace(tmp, os.path.join(layout, f"{rank}-part{part:05d}.safetensors"))
                    part, size, chunk = part + 1, 0, {}

            for key, tensor in ShardedStateLoader._filter_subtensors(model.state_dict()).items():
                chunk[key] = tensor.detach().to("cpu", copy=True).contiguous()
                size += tensor.numel() * tensor.element_size()
                if size >= PART_BYTES:
                    flush()
            flush()
            # the marker last: an interrupted save is redone, never half-loaded
            with open(done, "w") as f:
                f.write(f"{part}\n")
            print(f"[{LOAD_FORMAT}] saved {part} shard files to {layout} in {time.perf_counter() - start:.1f}s", flush=True)

    return MilabenchShardedLoader
