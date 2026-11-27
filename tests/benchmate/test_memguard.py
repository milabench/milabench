"""memguard: page cache dropped when free memory runs low (the GB10's driver can't reclaim it)."""

import json

from benchmate import memguard


def test_evict_page_cache_walks_the_folders(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "w.safetensors").write_bytes(b"x" * 4096)
    (tmp_path / "b.bin").write_bytes(b"y" * 100)
    (tmp_path / "link").symlink_to(tmp_path / "b.bin")
    assert memguard.evict_page_cache(f"{tmp_path}:/does/not/exist") == 4096 + 100


def test_low_free_memory_evicts_without_killing(tmp_path, monkeypatch):
    G = 2**30
    infos = iter([{"MemTotal": 120 * G}] + [{"MemTotal": 120 * G, "MemFree": 3 * G, "Cached": 30 * G,
                                              "MemAvailable": 33 * G}]
                 + [{"MemTotal": 120 * G, "MemFree": 28 * G, "Cached": 5 * G, "MemAvailable": 33 * G}] * 10)
    monkeypatch.setattr(memguard, "meminfo", lambda: next(infos))
    monkeypatch.setattr(memguard, "psi_full_avg10", lambda: 0.0)
    evicted = []
    monkeypatch.setattr(memguard, "evict_page_cache", lambda paths: evicted.append(paths) or 0)
    killed = []
    monkeypatch.setattr(memguard, "kill_benchmark", lambda *a: killed.append(a) or [])
    alive = iter([True, False])
    real_exists = memguard.os.path.exists
    monkeypatch.setattr(memguard.os.path, "exists", lambda p: next(alive) if p == "/proc/4242" else real_exists(p))
    log = tmp_path / "log"
    memguard.main(["--pid", "4242", "--no-kernel-log", "--evict-paths", "/hub", "--interval", "0", "--log", str(log)])
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert evicted == ["/hub"] and not killed
    evict = next(e for e in events if e["event"] == "evict")
    assert evict["free_gb"] == 3 and evict["free_after_gb"] == 28 and evict["cached_after_gb"] == 5
