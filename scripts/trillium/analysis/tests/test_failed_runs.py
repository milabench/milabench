"""A failed measured run leaves no synthetic point standing. Measured runs are made from rewritten copies of synthetic
runs (crashed before their summary, as a full-load out-of-memory does). No pytest needed:

    ../../analysis/.venv/bin/python tests/test_failed_runs.py       # from scripts/trillium/analysis
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tri_analysis import build, figures, style, synthetic, tables  # noqa: E402
from tri_analysis.runs import failed_operation  # noqa: E402
from tri_analysis.select import failed_runs  # noqa: E402


def crashed_copy(syn: Path, raw: Path, name: str) -> None:
    """Run ``name`` as a measured run killed before it wrote anything but run_start."""
    src = next((syn / "trillium" / "runs" / name).glob("*.data"))
    out = raw / "trillium" / "runs" / f"real-{name}"
    out.mkdir(parents=True)
    keep = []
    for line in src.read_text().splitlines():
        e = json.loads(line)
        d = e.get("data") or {}
        if e["event"] == "end":
            d["return_code"] = -9
        elif e["event"] == "data" and "run_start" not in d:
            continue
        keep.append(json.dumps(e))
    (out / src.name).write_text("\n".join(keep) + "\n")


def test_failed_runs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic.generate(tmp / "syn")
        for name in ("syn-f1-glm-full-load-sync-n13-r1",  # F1: from the entry's name
                     "syn-bg-weak-n26",  # two experiments: A-G and B-G-weak lose their 26-node point
                     "syn-aq-fsdp-12k-n13-r1"):  # Qwen: from run_start
            crashed_copy(tmp / "syn", tmp / "raw", name)
        base = build.build(tmp / "none", tmp / "syn", tmp / "t0")
        t = build.build(tmp / "raw", tmp / "syn", tmp / "t1")
        names = set(t["runs"].run_name)

        # F1: every synthetic full load at 13 nodes is gone, the other operations stay.
        gone = set(base["runs"].run_name) - names
        assert {n for n in gone if "load" in n and "n13" in n} == {
            f"syn-f1-glm-full-load-{m}-n13-r{r}" for m in ("sync", "async") for r in (1, 2, 3)}, gone
        assert "syn-f1-glm-full-sync-save-n13-r1" in names and "syn-f1-glm-weights-load-sync-n13-r1" in names
        f = t["failed_ops"]
        assert len(f) == 1 and tuple(f.iloc[0][["content", "op", "mode", "nodes"]]) == ("full", "load", "sync", 13), f

        # Throughput runs: the synthetic point goes, repetitions included, and the failure shows.
        assert "syn-bg-weak-n26" in gone and {"syn-aq-fsdp-12k-n13-r1", "syn-aq-fsdp-12k-n13-r2"} <= gone
        assert set(failed_runs(t["runs"], "A-G").nodes) == {26} and set(failed_runs(t["runs"], "B-G-weak").nodes) == {26}
        assert set(failed_runs(t["runs"], "A-Q").layout) == {"fsdp"}
        note = tables.failed_note(t)
        assert "3 configurations" in note and "full load, 13 nodes" in note, note

        # A success of the same configuration: no failure left to show.
        ok = tmp / "raw" / "trillium" / "runs" / "real-ok"
        shutil.copytree(tmp / "syn" / "trillium" / "runs" / "syn-bg-weak-n26", ok)
        t2 = build.build(tmp / "raw", tmp / "syn", tmp / "t2", other_clusters=True)
        assert not (failed_runs(t2["runs"], "B-G-weak").nodes == 26).any()

        style.apply()
        for fn in (figures.checkpoint, figures.glm_scaling, figures.glm_layouts):
            assert fn(t) is not None
        assert tables.over_memory_note(base)  # the synthetic 192 GiB says it would not fit


def test_measured_checkpoint_without_failure():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic.generate(tmp / "syn")
        ok = tmp / "raw" / "trillium" / "runs" / "real-ok"
        shutil.copytree(tmp / "syn" / "trillium" / "runs" / "syn-f1-glm-full-sync-save-n13-r1", ok)
        t = build.build(tmp / "raw", tmp / "syn", tmp / "t")
        names = set(t["runs"].run_name)
        assert "syn-f1-glm-full-sync-save-n13-r1" not in names and "syn-f1-glm-full-load-sync-n13-r1" in names
        assert t["failed_ops"].empty and t["checkpoints"].synthetic.dtype == bool


if __name__ == "__main__":
    test_measured_checkpoint_without_failure()
    print("ok test_measured_checkpoint_without_failure")
    test_failed_runs()
    print("ok test_failed_runs")
