"""Persist missing aiter shuffle helpers required by ATOM main worker processes."""

from __future__ import annotations

import site
import sys
from pathlib import Path


_PATCH = '''

def interleave_gate_up_rows(w: torch.Tensor) -> torch.Tensor:
    """(E, 2*I, ...) GGUU [g..,u..] -> GUGU [g0,u0,g1,u1,...] (rows)."""
    inter = w.shape[1] // 2
    return (
        torch.stack([w[:, :inter], w[:, inter:]], dim=2)
        .flatten(1, 2)
        .contiguous()
    )


def moe_shuffle_weight(
    src: torch.Tensor,
    experts_cnt: int | None = None,
    is_guinterleave: bool = False,
    gate_up: bool = False,
    layout=(16, 16),
) -> torch.Tensor:
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() == "gfx1250":
        if is_guinterleave and gate_up:
            src = interleave_gate_up_rows(src)
        return shuffle_weight(src, layout=layout)
    return shuffle_weight(
        src,
        layout=layout,
        is_guinterleave=is_guinterleave,
        gate_up=gate_up,
    )
'''


def _shuffle_path() -> Path:
    return Path(site.getsitepackages()[0]) / "aiter" / "ops" / "shuffle.py"


def _invalidate_shuffle_cache() -> None:
    cache_dir = _shuffle_path().parent / "__pycache__"
    if not cache_dir.is_dir():
        return
    for path in cache_dir.glob("shuffle*.pyc"):
        path.unlink(missing_ok=True)
    sys.modules.pop("aiter.ops.shuffle", None)


def patch_aiter_shuffle() -> None:
    shuffle_path = _shuffle_path()
    text = shuffle_path.read_text()
    if "def interleave_gate_up_rows" in text and "def moe_shuffle_weight" in text:
        return
    if _PATCH.strip() not in text:
        shuffle_path.write_text(text.rstrip() + _PATCH)
    _invalidate_shuffle_cache()


if __name__ == "__main__":
    patch_aiter_shuffle()
