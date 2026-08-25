"""Helpers used only by ``milabench cherrybin`` commands."""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

from milabench.common import run_with_loggers
from milabench.fs import XPath
from milabench.log import DataReporter, TerminalFormatter, TextReporter
from milabench.multi import MultiPackage
from milabench.utils import validation_layers

if TYPE_CHECKING:
    from milabench.pack import BasePackage


_GENERATED_MARKERS = ("FakeImageNet", "FakeVideo")


def uses_generated_dataset(pack: BasePackage) -> bool:
    """True when the bench builds its data locally (do not store in cherrybin).

    Opt out with ``generated: true``, tag ``generated``, ``cherrybin: false``,
    or argv that points at FakeImageNet / FakeVideo.
    """
    cfg = pack.config or {}
    if cfg.get("cherrybin") is False or cfg.get("generated") is True:
        return True
    tags = cfg.get("tags") or []
    if "generated" in tags:
        return True
    argv = cfg.get("argv", {})
    blob = _argv_blob(argv)
    return any(marker in blob for marker in _GENERATED_MARKERS)


def _argv_blob(argv) -> str:
    if argv is None:
        return ""
    if isinstance(argv, str):
        return argv
    if isinstance(argv, dict):
        return " ".join(f"{k} {v}" for k, v in argv.items())
    return " ".join(str(x) for x in argv)


def prepare_locally(pack: BasePackage, *, shortrace: bool = False) -> int:
    """Run the pack's normal prepare (generate data) into its current dirs."""
    name = pack.config["name"]
    single = MultiPackage({name: pack})
    return run_with_loggers(
        single.do_prepare(),
        loggers=[
            TerminalFormatter(),
            TextReporter("stdout"),
            TextReporter("stderr"),
            DataReporter(),
            *validation_layers("error", short=shortrace),
        ],
        mp=single,
        short=shortrace,
    )


def require_cherrybin():
    try:
        import cherrybin.core as core
    except ImportError as exc:
        raise SystemExit(
            "cherrybin is required for `milabench cherrybin`. "
            "Install it with: pip install cherrybin"
        ) from exc
    return core


def isolate_pack_dirs(pack: BasePackage) -> tuple[XPath, XPath]:
    """Point this pack at per-benchmark data/cache trees. Returns the originals."""
    name = pack.config["name"]
    orig_data = XPath(pack.dirs.data)
    orig_cache = XPath(pack.dirs.cache)
    pack.dirs.data = orig_data / name
    pack.dirs.cache = orig_cache / name
    pack.config.setdefault("dirs", {})
    pack.config["dirs"]["data"] = str(pack.dirs.data)
    pack.config["dirs"]["cache"] = str(pack.dirs.cache)
    return orig_data, orig_cache


def pack_roots(pack: BasePackage) -> list[tuple[str, str]]:
    return [
        (str(pack.dirs.data), "data"),
        (str(pack.dirs.cache), "cache"),
    ]


def blob_cache_dir(base: str | os.PathLike) -> str:
    return str(XPath(base) / "cache" / ".cherrybin-blobs")


def hardlink_tree(src: str | os.PathLike, dest: str | os.PathLike) -> int:
    """Hardlink (or copy) every file under ``src`` into ``dest``."""
    core = require_cherrybin()
    src = os.fspath(src)
    dest = os.fspath(dest)
    if not os.path.isdir(src):
        return 0
    n = 0
    for root, _, files in os.walk(src):
        for fname in files:
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, src)
            core.link_or_copy(full, os.path.join(dest, rel))
            n += 1
    return n


def materialize_checkout(
    db_path: str,
    name: str,
    standard_data: str | os.PathLike,
    standard_cache: str | os.PathLike,
    isolated_data: str | os.PathLike,
    isolated_cache: str | os.PathLike,
    blob_cache: str,
    dest_staging: str,
    *,
    io_chunk: int | None = None,
):
    """Checkout one bench into staging, then hardlink to standard and isolated trees."""
    core = require_cherrybin()
    result = core.checkout(db_path, name, dest_staging, blob_cache, io_chunk=io_chunk)
    data_src = os.path.join(dest_staging, "data")
    cache_src = os.path.join(dest_staging, "cache")
    if os.path.isdir(data_src):
        hardlink_tree(data_src, standard_data)
        hardlink_tree(data_src, isolated_data)
    if os.path.isdir(cache_src):
        hardlink_tree(cache_src, standard_cache)
        hardlink_tree(cache_src, isolated_cache)
    return result


def mirror_isolated_to_standard(
    pack: BasePackage,
    orig_data: str | os.PathLike,
    orig_cache: str | os.PathLike,
) -> None:
    hardlink_tree(pack.dirs.data, orig_data)
    hardlink_tree(pack.dirs.cache, orig_cache)


def remove_isolated_trees(pack: BasePackage) -> None:
    """Delete this pack's isolated download trees (data/<name>, cache/<name>)."""
    for path in (pack.dirs.data, pack.dirs.cache):
        path = os.fspath(path)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"cleaned {path}")
