"""Runtime compatibility shims for upstream open-instruct on milabench/ROCm."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

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
    _patch_accelerator_prepare_for_deepspeed()


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


def install_rate_hook() -> None:
    import open_instruct.finetune as finetune

    tps_re = re.compile(r"TPS:\s*([0-9.eE+-]+)")
    orig_info = finetune.logger.info

    def info(msg, *args, **kwargs):
        orig_info(msg, *args, **kwargs)
        if get_rank() not in (-1, 0):
            return
        text = msg % args if args else str(msg)
        match = tps_re.search(text)
        if not match:
            return
        print(
            json.dumps(
                {
                    "task": "train",
                    "rate": float(match.group(1)),
                    "units": "items/s",
                    "time": time.time(),
                }
            ),
            flush=True,
        )

    finetune.logger.info = info


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


def _patch_accelerator_prepare_for_deepspeed() -> None:
    from accelerate import Accelerator

    if getattr(Accelerator.prepare, "_milabench_openinstruct", False):
        return

    orig_prepare = Accelerator.prepare

    def prepare(self, *args, **kwargs):
        result = orig_prepare(self, *args, **kwargs)
        if self.state.deepspeed_plugin is None:
            return result
        flat_args = _deepspeed_args.get("args")
        if flat_args is None or len(result) < 4:
            return result

        import open_instruct.finetune as finetune

        model, optimizer, train_dataloader, lr_scheduler = result[:4]
        num_steps = flat_args.max_train_steps
        lr_scheduler = finetune._create_scheduler(flat_args, optimizer, num_steps)
        return (model, optimizer, train_dataloader, lr_scheduler) + tuple(result[4:])

    prepare._milabench_openinstruct = True
    Accelerator.prepare = prepare
