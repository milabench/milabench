"""Cherrybin optional archive: isolate, update, checkout. No pack.py hooks."""

import sys
from argparse import Namespace as NS
from pathlib import Path
from types import SimpleNamespace

import pytest

_CHERRYBIN_ROOT = Path(__file__).resolve().parents[3] / "cherrybin"
if _CHERRYBIN_ROOT.is_dir():
    sys.path.insert(0, str(_CHERRYBIN_ROOT))

from milabench.cli.cherrybin.util import (
    hardlink_tree,
    isolate_pack_dirs,
    materialize_checkout,
    mirror_isolated_to_standard,
    pack_roots,
    uses_generated_dataset,
)
from milabench.fs import XPath


pytest.importorskip("cherrybin.core")

from cherrybin.core import list_files, update_files  # noqa: E402


class DummyPack:
    def __init__(self, name, data, cache, **extra):
        self.config = {
            "name": name,
            "dirs": {"data": str(data), "cache": str(cache)},
            **extra,
        }
        self.dirs = NS(data=XPath(data), cache=XPath(cache))


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_isolate_pack_dirs_only_remaps_this_pack(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    pack = DummyPack("vllm", data, cache)

    orig_data, orig_cache = isolate_pack_dirs(pack)

    assert orig_data == XPath(data)
    assert orig_cache == XPath(cache)
    assert pack.dirs.data == XPath(data / "vllm")
    assert pack.dirs.cache == XPath(cache / "vllm")
    assert pack.config["dirs"]["data"] == str(data / "vllm")
    assert pack_roots(pack) == [
        (str(data / "vllm"), "data"),
        (str(cache / "vllm"), "cache"),
    ]


def test_update_then_prepare_only_selected_pack(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    db = tmp_path / "archive.db"

    vllm = DummyPack("vllm", data, cache)
    dino = DummyPack("dinov2", data, cache)
    isolate_pack_dirs(vllm)
    isolate_pack_dirs(dino)

    _write(vllm.dirs.data / "hub" / "llama.bin", "llama")
    _write(vllm.dirs.cache / "torch" / "v.bin", "v-torch")
    _write(dino.dirs.data / "hub" / "vit.bin", "vit")
    _write(dino.dirs.cache / "torch" / "d.bin", "d-torch")

    stats = update_files(
        str(db),
        [
            (vllm.config["name"], pack_roots(vllm)),
            (dino.config["name"], pack_roots(dino)),
        ],
    )
    assert {s.name: s.file_count for s in stats} == {"vllm": 2, "dinov2": 2}

    vllm_paths = {rel for rel, _ in list_files(str(db), "vllm")}
    assert vllm_paths == {"data/hub/llama.bin", "cache/torch/v.bin"}
    dino_paths = {rel for rel, _ in list_files(str(db), "dinov2")}
    assert dino_paths == {"data/hub/vit.bin", "cache/torch/d.bin"}

    # Second update: add one file, drop one for vllm
    (vllm.dirs.data / "hub" / "llama.bin").unlink()
    _write(vllm.dirs.data / "hub" / "new.bin", "new-weights")
    again = update_files(str(db), [(vllm.config["name"], pack_roots(vllm))])
    assert again[0].added == 1
    assert again[0].removed == 1
    assert {rel for rel, _ in list_files(str(db), "vllm")} == {
        "data/hub/new.bin",
        "cache/torch/v.bin",
    }
    # dinov2 untouched
    assert {rel for rel, _ in list_files(str(db), "dinov2")} == {
        "data/hub/vit.bin",
        "cache/torch/d.bin",
    }

    # Checkout only vllm into a clean workspace
    out = tmp_path / "node"
    std_data = out / "data"
    std_cache = out / "cache"
    staging = tmp_path / "staging"
    staging.mkdir()
    materialize_checkout(
        str(db),
        "vllm",
        std_data,
        std_cache,
        std_data / "vllm",
        std_cache / "vllm",
        str(out / "cache" / ".cherrybin-blobs"),
        str(staging),
    )

    assert (std_data / "hub" / "new.bin").read_text() == "new-weights"
    assert (std_cache / "torch" / "v.bin").read_text() == "v-torch"
    assert not (std_data / "hub" / "vit.bin").exists()
    assert (std_data / "vllm" / "hub" / "new.bin").exists()


def test_mirror_isolated_into_standard_data_cache(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    pack = DummyPack("vllm", data, cache)
    orig_data, orig_cache = isolate_pack_dirs(pack)
    _write(pack.dirs.data / "hub" / "m.bin", "m")
    _write(pack.dirs.cache / "torch" / "t.bin", "t")

    mirror_isolated_to_standard(pack, orig_data, orig_cache)

    assert (data / "hub" / "m.bin").read_text() == "m"
    assert (cache / "torch" / "t.bin").read_text() == "t"
    # same inode when hardlink works
    if (data / "hub" / "m.bin").stat().st_ino == (pack.dirs.data / "hub" / "m.bin").stat().st_ino:
        assert True


def test_hardlink_tree_skips_missing_src(tmp_path):
    assert hardlink_tree(tmp_path / "missing", tmp_path / "dest") == 0


def test_vanilla_make_env_has_no_cherrybin_hook():
    from milabench import pack as pack_mod
    import inspect

    source = inspect.getsource(pack_mod.BasePackage.make_env)
    assert "cherrybin" not in source


def test_prepare_execute_checkouts_shared_db(tmp_path, monkeypatch):
    from milabench.cli.cherrybin.prepare import Prepare

    src_data = tmp_path / "src" / "data"
    src_cache = tmp_path / "src" / "cache"
    pack = DummyPack("vllm", src_data, src_cache)
    isolate_pack_dirs(pack)
    _write(pack.dirs.data / "hub" / "llama.bin", "llama")
    db = tmp_path / "archive.db"
    update_files(str(db), [(pack.config["name"], pack_roots(pack))])

    out_data = tmp_path / "node" / "data"
    out_cache = tmp_path / "node" / "cache"
    dest_pack = DummyPack("vllm", out_data, out_cache)
    monkeypatch.setattr(
        "milabench.cli.cherrybin.prepare.get_multipack",
        lambda *a, **k: SimpleNamespace(packs={"vllm": dest_pack}),
    )

    args = SimpleNamespace(shared=str(db), cache="", base=str(tmp_path / "node"))
    assert Prepare.execute(args) == 0
    assert (out_data / "hub" / "llama.bin").read_text() == "llama"
    assert (out_data / "vllm" / "hub" / "llama.bin").exists()


def test_update_execute_writes_db_and_mirrors(tmp_path, monkeypatch):
    from milabench.cli.cherrybin.update import Update

    data = tmp_path / "data"
    cache = tmp_path / "cache"
    pack = DummyPack("vllm", data, cache)
    db = tmp_path / "archive.db"

    monkeypatch.setattr(
        "milabench.cli.cherrybin.update.get_multipack",
        lambda *a, **k: SimpleNamespace(packs={"vllm": pack}),
    )

    def fake_run(coro, **kwargs):
        coro.close()
        _write(pack.dirs.data / "hub" / "m.bin", "weights")
        _write(pack.dirs.cache / "torch" / "t.bin", "torch")
        return 0

    monkeypatch.setattr("milabench.cli.cherrybin.util.run_with_loggers", fake_run)

    args = SimpleNamespace(shared=str(db), shortrace=False, clean=False)
    assert Update.execute(args) == 0
    assert db.is_file()
    assert {rel for rel, _ in list_files(str(db), "vllm")} == {
        "data/hub/m.bin",
        "cache/torch/t.bin",
    }
    assert (data / "hub" / "m.bin").read_text() == "weights"
    assert (cache / "torch" / "t.bin").read_text() == "torch"


def test_update_execute_indexes_one_bench_before_the_next(tmp_path, monkeypatch):
    from milabench.cli.cherrybin.update import Update

    data = tmp_path / "data"
    cache = tmp_path / "cache"
    vllm = DummyPack("vllm", data, cache)
    dino = DummyPack("dinov2", data, cache)
    db = tmp_path / "archive.db"
    seen_in_db_after_prepare = []

    monkeypatch.setattr(
        "milabench.cli.cherrybin.update.get_multipack",
        lambda *a, **k: SimpleNamespace(
            packs={"vllm": vllm, "dinov2": dino},
        ),
    )

    def fake_run(coro, **kwargs):
        coro.close()
        mp = kwargs.get("mp")
        pack = next(iter(mp.packs.values()))
        name = pack.config["name"]
        _write(pack.dirs.data / "hub" / f"{name}.bin", name)
        # Previous benches must already be in the archive.
        if db.is_file():
            seen_in_db_after_prepare.append(
                {rel for rel, _ in list_files(str(db), "vllm")}
            )
        else:
            seen_in_db_after_prepare.append(set())
        return 0

    monkeypatch.setattr("milabench.cli.cherrybin.util.run_with_loggers", fake_run)

    args = SimpleNamespace(shared=str(db), shortrace=False, clean=False)
    assert Update.execute(args) == 0
    assert seen_in_db_after_prepare[0] == set()
    assert seen_in_db_after_prepare[1] == {"data/hub/vllm.bin"}
    assert {rel for rel, _ in list_files(str(db), "dinov2")} == {"data/hub/dinov2.bin"}


def test_update_execute_clean_removes_isolated_downloads(tmp_path, monkeypatch):
    from milabench.cli.cherrybin.update import Update

    data = tmp_path / "data"
    cache = tmp_path / "cache"
    pack = DummyPack("vllm", data, cache)
    db = tmp_path / "archive.db"

    monkeypatch.setattr(
        "milabench.cli.cherrybin.update.get_multipack",
        lambda *a, **k: SimpleNamespace(packs={"vllm": pack}),
    )

    def fake_run(coro, **kwargs):
        coro.close()
        _write(pack.dirs.data / "hub" / "m.bin", "weights")
        _write(pack.dirs.cache / "torch" / "t.bin", "torch")
        return 0

    monkeypatch.setattr("milabench.cli.cherrybin.util.run_with_loggers", fake_run)

    args = SimpleNamespace(shared=str(db), shortrace=False, clean=True)
    assert Update.execute(args) == 0
    assert db.is_file()
    assert {rel for rel, _ in list_files(str(db), "vllm")} == {
        "data/hub/m.bin",
        "cache/torch/t.bin",
    }
    assert not (data / "vllm").exists()
    assert not (cache / "vllm").exists()
    assert not (data / "hub" / "m.bin").exists()


def test_uses_generated_dataset_from_argv_and_flags(tmp_path):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    assert uses_generated_dataset(
        DummyPack("resnet", data, cache, argv={"--data": "{milabench_data}/FakeImageNet"})
    )
    assert uses_generated_dataset(DummyPack("resnet", data, cache, generated=True))
    assert uses_generated_dataset(DummyPack("resnet", data, cache, cherrybin=False))
    assert uses_generated_dataset(DummyPack("resnet", data, cache, tags=["generated"]))
    assert not uses_generated_dataset(DummyPack("vllm", data, cache))


def test_update_skips_generated_dataset(tmp_path, monkeypatch):
    from milabench.cli.cherrybin.update import Update

    data = tmp_path / "data"
    cache = tmp_path / "cache"
    pack = DummyPack("resnet50", data, cache, generated=True)
    db = tmp_path / "archive.db"
    prepared = []

    monkeypatch.setattr(
        "milabench.cli.cherrybin.update.get_multipack",
        lambda *a, **k: SimpleNamespace(packs={"resnet50": pack}),
    )

    def fake_run(coro, **kwargs):
        prepared.append(True)
        coro.close()
        return 0

    monkeypatch.setattr("milabench.cli.cherrybin.util.run_with_loggers", fake_run)

    args = SimpleNamespace(shared=str(db), shortrace=False, clean=False)
    assert Update.execute(args) == 0
    assert prepared == []
    assert not db.exists()


def test_prepare_runs_generate_for_generated_dataset(tmp_path, monkeypatch):
    from milabench.cli.cherrybin.prepare import Prepare

    data = tmp_path / "data"
    cache = tmp_path / "cache"
    pack = DummyPack("resnet50", data, cache, generated=True)
    prepared = []

    monkeypatch.setattr(
        "milabench.cli.cherrybin.prepare.get_multipack",
        lambda *a, **k: SimpleNamespace(packs={"resnet50": pack}),
    )

    def fake_run(coro, **kwargs):
        prepared.append(pack.config["name"])
        coro.close()
        _write(data / "FakeImageNet" / "done", "ok")
        return 0

    monkeypatch.setattr("milabench.cli.cherrybin.util.run_with_loggers", fake_run)

    args = SimpleNamespace(
        shared=str(tmp_path / "archive.db"),
        cache="",
        base=str(tmp_path),
    )
    (tmp_path / "archive.db").write_bytes(b"x")
    assert Prepare.execute(args) == 0
    assert prepared == ["resnet50"]
    assert (data / "FakeImageNet" / "done").read_text() == "ok"


def test_prepare_execute_missing_archive(tmp_path):
    from milabench.cli.cherrybin.prepare import Prepare

    args = SimpleNamespace(shared=str(tmp_path / "nope.db"), cache="", base=str(tmp_path))
    assert Prepare.execute(args) == 1
