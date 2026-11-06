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
    if not shuffle_path.is_file():
        return
    text = shuffle_path.read_text()
    if "def interleave_gate_up_rows" in text and "def moe_shuffle_weight" in text:
        return
    if _PATCH.strip() not in text:
        shuffle_path.write_text(text.rstrip() + _PATCH)
    _invalidate_shuffle_cache()


# ATOM 0.1.8 hashes the source of every file Dynamo traced into its compile-cache key, skipping only "<string>" and
# "<frozen os>": a traced os.path call (Kimi-K3 with MXFP4 lm_head / latent projections) lists "<frozen genericpath>"
# and open() raises FileNotFoundError in every worker. Skip every "<...>" pseudo-file.
_TRACED_FILES_OLD = 'if filepath == "<string>" or filepath == "<frozen os>":'
_TRACED_FILES_NEW = 'if filepath.startswith("<"):'


def patch_atom_traced_files() -> None:
    backends = Path(site.getsitepackages()[0]) / "atom" / "utils" / "backends.py"
    if not backends.is_file():
        return
    text = backends.read_text()
    if _TRACED_FILES_OLD not in text:
        return
    backends.write_text(text.replace(_TRACED_FILES_OLD, _TRACED_FILES_NEW))
    for path in (backends.parent / "__pycache__").glob("backends*.pyc"):
        path.unlink(missing_ok=True)
    sys.modules.pop("atom.utils.backends", None)


# ATOM 0.1.8 picks the MXFP4 GEMM inside the compiled forward: use_mxfp4_preshuffle_triton_gemm() calls aiter's
# get_gfx_runtime(), which Dynamo traces down to os.stat + a rocminfo subprocess (it ignores the lru_cache). The
# graph break specializes the token count (ConstraintViolationError: hidden_states size()[0] specialized to 16384)
# for any online MXFP4 linear. Detect the arch once at import, outside the traced code.
_GFX_OLD = """    if arch is None:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        arch = get_gfx_runtime()
"""
_GFX_NEW = """    if arch is None:
        arch = _GFX_RUNTIME or _gfx_runtime()
"""
_GFX_MODULE = """

def _gfx_runtime():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime()


# milabench: resolved at import so the compiled forward reads a constant (no os.stat / rocminfo in the graph)
try:
    _GFX_RUNTIME = _gfx_runtime()
except Exception:
    _GFX_RUNTIME = None


def use_mxfp4_preshuffle_triton_gemm("""


def patch_atom_mxfp4_arch() -> None:
    linear = Path(site.getsitepackages()[0]) / "atom" / "model_ops" / "linear.py"
    if not linear.is_file():
        return
    text = linear.read_text()
    if _GFX_OLD not in text or "_GFX_RUNTIME" in text:
        return
    text = text.replace(_GFX_OLD, _GFX_NEW, 1)
    text = text.replace("\n\ndef use_mxfp4_preshuffle_triton_gemm(", _GFX_MODULE, 1)
    linear.write_text(text)
    for path in (linear.parent / "__pycache__").glob("linear*.pyc"):
        path.unlink(missing_ok=True)
    sys.modules.pop("atom.model_ops.linear", None)


if __name__ == "__main__":
    patch_aiter_shuffle()
    patch_atom_traced_files()
    patch_atom_mxfp4_arch()
