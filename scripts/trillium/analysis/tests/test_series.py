"""Runs that serve two experiments: every GLM-5 series holds all its node counts. No pytest needed:

    ../../analysis/.venv/bin/python tests/test_series.py       # from scripts/trillium/analysis
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tri_analysis import build, synthetic  # noqa: E402
from tri_analysis.machine import GLM_TOP_NODES  # noqa: E402
from tri_analysis.select import completed, glm_series, series_gaps  # noqa: E402
from tri_analysis.style import GLM  # noqa: E402


def run(experiment, nodes, seqs, ac="full", precision="bf16", layout="pp13-ep8"):
    return {"run_id": f"{experiment}-{nodes}-{seqs}-{ac}-{precision}-{layout}", "experiment": experiment, "nodes": nodes,
            "layout": layout, "model": GLM, "status": "completed", "global_batch_tokens": seqs * 8192, "seq_len": 8192,
            "ac": ac, "precision": precision, "mb_seqs": 1, "synthetic": False}


def launcher_tags() -> pd.DataFrame:
    """The tags the launchers give (glm/10_ag.sh, glm/20_bg.sh), at 52 nodes."""
    return pd.DataFrame([run("A-G", 13, 256), run("A-G", 52, 1024), run("B-G-weak", 26, 512), run("B-G-weak", 39, 768),
                         run("B-G-strong", 13, 768), run("B-G-strong", 26, 768), run("B-G-strong", 52, 768)])


def test_second_experiments():
    r = launcher_tags()
    if GLM_TOP_NODES != 52:
        r = r[r.nodes <= GLM_TOP_NODES]
    assert series_gaps(r) == {}, series_gaps(r)
    weak = completed(r, "B-G-weak")
    assert sorted(weak.nodes) == [n for n in (13, 26, 39, 52) if n <= GLM_TOP_NODES]
    strong = completed(r, "B-G-strong")
    assert sorted(strong.nodes) == [n for n in (13, 26, 39, 52) if n <= GLM_TOP_NODES]
    assert 26 in set(completed(r, "A-G").nodes)
    both = completed(r, "B-G-weak", "B-G-strong")  # the 39-node run under each
    assert (both.nodes == 39).sum() == 2 and set(both[both.nodes == 39].experiment) == {"B-G-weak", "B-G-strong"}
    assert len(completed(r)) == len(r)  # no experiment asked: each run once


def test_gap_and_other_configurations():
    r = launcher_tags()
    assert series_gaps(r[r.nodes != 26])["B-G-weak"] == [26]  # a missing run shows
    # Same size, other configuration: serves nothing else.
    other = pd.DataFrame([run("A-G", 13, 256, ac="selective"), run("C-speed", 13, 256),
                          run("A-G", 13, 256, precision="mxfp8")])
    assert completed(other, "B-G-weak").empty


def test_synthetic_series():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        synthetic.generate(tmp / "syn")
        t = build.build(tmp / "raw", tmp / "syn", tmp / "tables")
    assert series_gaps(t["runs"]) == {}, series_gaps(t["runs"])
    for name, want in glm_series().items():
        d = completed(t["runs"], name)
        d = d[(d.layout == "pp13-ep8") & (d.ac == "full") & (d.precision == "bf16")]
        assert set(want) <= set(d.nodes), (name, sorted(d.nodes))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
