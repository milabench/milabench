"""Runtime compatibility shims for upstream open-instruct on milabench/ROCm."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from benchmate.metrics import ManualTimedIterator, default_event, sumggle_push
from benchmate.monitor import get_rank

_deepspeed_args: dict[str, Any] = {}


def parse_milabench_argv() -> bool:
    """Remove milabench-only CLI flags before upstream HfArgumentParser runs."""
    skip_model_save = False
    argv: list[str] = [sys.argv[0]]
    it = iter(sys.argv[1:])
    for arg in it:
        if arg == "--skip_model_save":
            value = next(it, "true")
            skip_model_save = value.lower() in ("1", "true", "yes", "on")
            continue
        if arg.startswith("--skip_model_save="):
            skip_model_save = arg.split("=", 1)[1].lower() in ("1", "true", "yes", "on")
            continue
        argv.append(arg)
    sys.argv[:] = argv
    return skip_model_save


def apply_runtime_compat() -> None:
    """Apply all shims before importing open_instruct training code."""
    _torchvision_videoreader_shim()
    _patch_accelerator_for_milabench()


def bind_deepspeed_args(args) -> None:
    """Pass FlatArguments into the Accelerator.prepare wrapper."""
    _deepspeed_args["args"] = args


def enable_skip_model_save() -> None:
    import open_instruct.finetune as finetune
    import open_instruct.model_utils as model_utils

    def _noop(*_args, **_kwargs):
        return None

    model_utils.save_with_accelerate = _noop
    finetune.save_with_accelerate = _noop


def _count_batch_tokens(batch) -> int:
    """Match open-instruct/finetune.py batch token accounting."""
    if "attention_mask" in batch:
        return int(batch["attention_mask"].sum().item())
    if "position_ids" in batch:
        return int(batch["position_ids"].numel())
    if "cu_seq_lens_q" in batch:
        return int(batch["cu_seq_lens_q"][-1].item())
    raise ValueError(f"Cannot count tokens in batch keys: {sorted(batch)}")


def _milabench_rank() -> int:
    rank = get_rank()
    return 0 if rank < 0 else rank


def _wrap_train_dataloader(loader, accelerator):
    """CUDA-event timed loader; step() on each optimizer step."""
    return ManualTimedIterator(
        loader,
        event_fn=default_event(),
        rank=_milabench_rank(),
        push=sumggle_push(),
        device=accelerator.device,
        earlystop=10**9,
        raise_stop_program=False,
        batch_size_fn=_count_batch_tokens,
    )


def patch_upstream_model_utils(path: Path) -> None:
    """One-time file patch: ai2-olmo-core may lack AttentionBackendName.flash_4."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if "getattr(AttentionBackendName, _name, None)" in text:
        return
    needle = (
        "_OLMO_CORE_TO_HF_ATTN: dict[AttentionBackendName, str] = {\n"
        '    AttentionBackendName.flash_4: "flash_attention_4",\n'
        '    AttentionBackendName.flash_3: "flash_attention_3",\n'
        '    AttentionBackendName.flash_2: "flash_attention_2",\n'
        '    AttentionBackendName.torch: "sdpa",\n'
        '    AttentionBackendName.te: "sdpa",\n'
        "}\n"
    )
    if needle not in text:
        return
    replacement = '''_OLMO_CORE_TO_HF_ATTN: dict[AttentionBackendName, str] = {}
for _name, _hf in (
    ("flash_4", "flash_attention_4"),
    ("flash_3", "flash_attention_3"),
    ("flash_2", "flash_attention_2"),
    ("torch", "sdpa"),
    ("te", "sdpa"),
):
    _backend = getattr(AttentionBackendName, _name, None)
    if _backend is not None:
        _OLMO_CORE_TO_HF_ATTN[_backend] = _hf
'''
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def _torchvision_videoreader_shim() -> None:
    try:
        from torchvision.io import VideoReader  # noqa: F401
    except ImportError:
        import torchvision.io as tv_io

        class VideoReader:  # dummy for datasets 4.x on ROCm
            pass

        tv_io.VideoReader = VideoReader


def _patch_accelerator_for_milabench() -> None:
    from accelerate import Accelerator

    if getattr(Accelerator.prepare, "_milabench_openinstruct", False):
        return

    orig_prepare = Accelerator.prepare
    orig_accumulate = Accelerator.accumulate

    def prepare(self, *args, **kwargs):
        result = orig_prepare(self, *args, **kwargs)
        flat_args = _deepspeed_args.get("args")

        if flat_args is not None and len(result) >= 4:
            import open_instruct.finetune as finetune

            model, optimizer, train_dataloader, lr_scheduler = result[:4]

            if self.state.deepspeed_plugin is not None:
                num_steps = flat_args.max_train_steps
                lr_scheduler = finetune._create_scheduler(flat_args, optimizer, num_steps)

            timed = _wrap_train_dataloader(train_dataloader, self)
            self._milabench_timed_loader = timed
            self._milabench_logging_steps = flat_args.logging_steps or 1
            self._milabench_opt_step = 0
            train_dataloader = timed
            result = (model, optimizer, train_dataloader, lr_scheduler) + tuple(result[4:])

        return result

    @contextmanager
    def accumulate(self, *args, **kwargs):
        with orig_accumulate(self, *args, **kwargs):
            yield
        timed = getattr(self, "_milabench_timed_loader", None)
        if timed is None or not self.sync_gradients:
            return
        self._milabench_opt_step += 1
        if self._milabench_opt_step % self._milabench_logging_steps != 0:
            return
        if timed.acc_batch_size <= 0:
            return
        # Record CUDA event span + token count, sync, then push (benchmate pattern).
        timed.step()
        timed._push()

    prepare._milabench_openinstruct = True
    accumulate._milabench_openinstruct = True
    Accelerator.prepare = prepare
    Accelerator.accumulate = accumulate
