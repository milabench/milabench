#!/usr/bin/env python
"""Regenerate the Trillium tables, figures and report from raw results.

    analyze.py all                    # collect + synth + build + plot + report
    analyze.py all --no-collect       # from the local copy in data/raw
    analyze.py all --no-synthetic     # measured data only
    analyze.py collect | synth | build | plot | report | progress

Raw data is copied read-only from trillium-gpu:$SCRATCH/trillium/ (rsync over ssh) into
data/raw/trillium/; synthetic data (../FORMAT.md formats) goes to data/synthetic/trillium/.
Run with scripts/analysis/.venv/bin/python (same requirements as the Rorqual pipeline).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tri_analysis import build as build_mod  # noqa: E402
from tri_analysis import figures, progress, synthetic, tables  # noqa: E402
from tri_analysis.machine import CLUSTER  # noqa: E402
from tri_analysis.select import failed_runs, series_gaps  # noqa: E402

RAW = HERE / "data" / "raw"
SYN = HERE / "data" / "synthetic"
TABLES = HERE / "data" / "tables"
REPORT = HERE.parent / "report"
GENERATED = REPORT / "generated"
HOST = "trillium-gpu"
# Directories under $SCRATCH/trillium copied into data/raw/trillium/. Checkpoints: only their
# JSON metadata, never the checkpoint files themselves (up to ~9 TB).
REMOTE = ["runs", "nccl", "nodemon"]
GLM_FP4_RUNS = [f"glm-fp4.1271210.megatron-glm5-tri-pp13-ep8-n13-gbs256{s}" for s in ("", "-mxfp8", "-nvfp4")]
# Qwen3 32B dense with BF16, MXFP8 and NVFP4: C-speed at 13 nodes (GLM-FP4-03, one job) and C-loss on 1 node, one job per run
# (GLM-FP4-02: 1272019-22; GLM-FP4-04: 1272416 BF16, 1272419 and 1272423 MXFP8, 1272429 and 1272431 NVFP4; 3 runs per recipe)
QW_FP4_RUNS = [*[f"glm-fp4.1272011.megatron-qwen3-32b-n13-{r}" for r in ("bf16", "mxfp8", "nvfp4")],
               *[f"glm-fp4.{job}.megatron-qwen3-32b-loss-{r}-n1" for job, r in (("1272019", "bf16"), ("1272020", "bf16-rep2"),
                                                                             ("1272021", "mxfp8"), ("1272022", "nvfp4"),
                                                                             ("1272416", "bf16"), ("1272419", "mxfp8"), ("1272423", "mxfp8"),
                                                                             ("1272429", "nvfp4"), ("1272431", "nvfp4"))]]
# Qwen3 32B at 32k and 128k tokens with context parallelism (QW-CP-01, job 1273505: three entries, 13 nodes; the CP8 one on 12 of them),
# 128k CP4 with selective recompute (QW-CP-02, job 1273588), and 128k CP4 x DP4 in BF16 and MXFP8 on 4 nodes (job 1278205, `nw-qw-cp-n4-mxfp8`)
QW_CP_RUNS = [*[f"qw-cp.1273505.megatron-qwen3-32b-cp-{e}" for e in ("n13-32k", "n13-128k", "n12-128k-cp8")],
              "qw-cp.1273588.megatron-qwen3-32b-cp-n13-128k-sel",
              *[f"qw-cp.1278205.megatron-qwen3-32b-cp-n4-128k{s}" for s in ("", "-mxfp8")]]
# Kimi K3 EP16 over 2 nodes (K3-XN-01): job 1271952 all-gather, 1271953 DeepEP V2; on 4 nodes, prefill and decode on separate nodes
# (K3-XN-02: 1272790 P/D, 1272791 its colocated reference) and the experts over all 4 (K3-XN-03: 1272792, EP32, 128 sequences per engine)
# and the stall job (1278355: K3-INF-02's server on 4 other nodes, each concurrency twice in a row; 19 MB; its nodemon is not read)
# OpenFold3 training on the uniform subset (OF3 step 2: 1273887 on 1 node, 1273885 on 2, 1273884 on 4; OF3-01: 1274059 on 13), keyed on job and
# variant: published, DeepSpeed attention (1 and 13 nodes), cuEquivariance + TF32. Not the profile runs, nor the step-1 subset's (1273594-1273810).
OF3_RUNS = [f"of3.{job}.{v}" for job, variants in (("1273887", ("uniform", "uniform,cueq+tf32", "uniform,ds")),
                                                   ("1273885", ("uniform", "uniform,cueq+tf32")), ("1273884", ("uniform", "uniform,cueq+tf32")),
                                                   ("1274059", ("uniform", "uniform,cueq+tf32", "uniform,ds"))) for v in variants]
K3_XN_RUNS = ["kimi-vllm-2n-dp2tp8-ep16.1271952","kimi-vllm-2n-dp2tp8-ep16-deepep.1271953",
              "kimi-vllm-4n-pd-ep16.1272790", "kimi-vllm-4n-pd-ep16-colocated.1272791", "kimi-vllm-4n-dp4tp8-ep32-s128.1272792",
              "kimi-vllm-4n-tp8pp4-stall.1278355"]
# Other sources, not under the directories above: (path under $SCRATCH, directory under data/raw/trillium/).
# The trailing slash matters: health checks are one directory per job, directly under health/.
REMOTE_OTHER = [("slurm-healthcheck/results/", "health"),
                ("trillium/glm/ag/recovered/", "recovered"),
                ("trillium/hw/post-pstate/README.md", "post-pstate"),
                ("trillium/hw/post-pstate/before_after.tsv", "post-pstate"),
                ("trillium/hw/post-pstate/noise.tsv", "post-pstate"),
                ("trillium/hw/post-pstate/power_limits.tsv", "post-pstate"),
                ("trillium/hw/post-pstate/healthcheck_cpu_clocks.tsv", "post-pstate"),
                ("trillium/hw/post-pstate/boot_checks.tsv", "post-pstate"),
                ("trillium/hw/post-pstate/load_table.tsv", "post-pstate"),
                ("trillium/hw/caps-cleared/load_table.tsv", "caps-cleared"),
                ("trillium/hw/caps-cleared/healthcheck.tsv", "caps-cleared"),
                ("trillium/hw/caps-cleared/hc2.tsv", "caps-cleared"),
                ("trillium/hw/caps-cleared/load2/load_table.tsv", "caps-cleared/load2"),
                ("trillium/hw/slowfix/table.tsv", "slowfix"),
                ("trillium/hw/slowfix/healthcheck.tsv", "slowfix"),
                ("trillium/hw/allnodes/by_node.tsv", "allnodes"),
                ("trillium/hw/allnodes/parts_by_node.tsv", "allnodes"),
                ("trillium/hw/nodemon-gaps/probe_1node.tsv", "nodemon-gaps"),
                ("trillium/hw/cpu-drops/runs_1node.tsv", "cpu-drops"),
                ("trillium/hw/srygaard-hc/by_node.tsv", "srygaard-hc"),
                ("trillium/hw/srygaard-hc/metric_offsets.tsv", "srygaard-hc"),
                ("trillium/hw/fp8-glm/summary.txt", "fp8-glm"),
                # New workloads: Kimi K3 inference (vLLM), one directory per job; 91 MB on Nov 8. Outside runs/,
                # so the training build does not read it (tt_analysis.runs.data_files globs runs/*/*.data).
                ("trillium/new/k3-inf/runs/", "k3-inf/runs"),
                # Kimi K3 inference across nodes (K3-XN-01: EP16, all-gather and DeepEP all-to-all; K3-XN-02: P/D and colocated; K3-XN-03:
                # EP32; the stall job), keyed on job: those six run directories only (19-99 MB each); the bring-ups in k3-xn/runs are not read.
                *[(f"trillium/new/k3-xn/runs/{run}/", f"k3-xn/runs/{run}") for run in K3_XN_RUNS],
                # Kimi K3 pretraining (TorchTitan): run directories (172 MB on Nov 8), and the GPU telemetry of the two
                # completed 48-node runs only (gpu.tsv, 119 MB; the other nodemon files and jobs are not read).
                ("trillium/new/k3-trn/runs/", "k3-trn/runs"),
                *[(f"trillium/new/k3-trn/nodemon/{run}/nodes/*.gpu.tsv", f"k3-trn/nodemon/{run}")
                  for run in ("k3-pp48-full-rbf16.1263956", "k3-pp48-l49.1263476")],
                # GLM-5.3 GRPO with slime, 32 nodes: colocated without and with rollout routing replay (GLM-RL-03, GLM-RL-04),
                # disaggregated (GLM-RL-05), and colocated on B200's default rollout kernels (GLM-RL-06): their run directories
                # (21 MB each) and job logs (19-21 MB each) only; the other GLM-RL jobs are bring-up and tests, not read.
                *[(f"trillium/new/glm-rl/runs/{run}/", f"glm-rl/runs/{run}") for run in ("nw-glm-rl-glm5.1266065.slime-glm5-grpo",
                                                                                          "nw-glm-rl-glm5-r3.1267147.slime-glm5-grpo-r3",
                                                                                          "nw-glm-rl-glm5-disagg.1267151.slime-glm5-grpo-disagg",
                                                                                          "nw-glm-rl-glm5-trtllm.1275601.slime-glm5-grpo-trtllm-mrr2048")],
                *[(f"trillium/new/glm-rl/slurm/{log}", "glm-rl/slurm") for log in ("nw-glm-rl-glm5-1266065.out",
                                                                                   "nw-glm-rl-glm5-r3-1267147.out",
                                                                                   "nw-glm-rl-glm5-disagg-1267151.out",
                                                                                   "nw-glm-rl-glm5-trtllm-1275601.out")],
                # GLM-5 C-speed with BF16, MXFP8 and NVFP4 at 13 nodes (GLM-FP4-01, job 1271210): its three run directories (49-71 MB
                # each) and the GPU telemetry of each entry (gpu.tsv only, 23 MB in all). Outside runs/: not campaign runs.
                *[(f"trillium/new/glm-fp4/runs/{run}/", f"glm-fp4/runs/{run}") for run in GLM_FP4_RUNS],
                *[(f"trillium/new/glm-fp4/nodemon/{run}/nodes/*.gpu.tsv", f"glm-fp4/nodemon/{run}") for run in GLM_FP4_RUNS],
                # Qwen3 32B dense with BF16, MXFP8 and NVFP4 (GLM-FP4-03, job 1272011, 13 nodes; GLM-FP4-02 and -04, nine jobs, 1 node each):
                # the twelve run directories (9-16 MB each) and their gpu.tsv. Outside runs/: not campaign runs.
                *[(f"trillium/new/glm-fp4/runs/{run}/", f"glm-fp4/runs/{run}") for run in QW_FP4_RUNS],
                *[(f"trillium/new/glm-fp4/nodemon/{run}/nodes/*.gpu.tsv", f"glm-fp4/nodemon/{run}") for run in QW_FP4_RUNS],
                # Qwen3 32B with context parallelism (QW-CP-01, job 1273505; QW-CP-02, 1273588; MXFP8 on 4 nodes, 1278205): the six run directories (6-23 MB each), their gpu.tsv and ib.tsv
                # (InfiniBand counters), and the 8k reference's ib.tsv (GLM-FP4-03's BF16 entry, 1272011). Outside runs/: not campaign runs.
                *[(f"trillium/new/qw-cp/runs/{run}/", f"qw-cp/runs/{run}") for run in QW_CP_RUNS],
                *[(f"trillium/new/qw-cp/nodemon/{run}/nodes/*.{kind}.tsv", f"qw-cp/nodemon/{run}") for run in QW_CP_RUNS for kind in ("gpu", "ib")],
                (f"trillium/new/glm-fp4/nodemon/{QW_FP4_RUNS[0]}/nodes/*.ib.tsv", f"glm-fp4/nodemon/{QW_FP4_RUNS[0]}"),
                # OpenFold3 training (OF3 step 2 and OF3-01): each run's per-rank step files and runner.yml (1.5-3 MB a run), and its node monitor's
                # gpu, ib and cpu tables (up to 36 MB a run at 13 nodes; the monitor sits inside the run directory). Outside runs/: not campaign runs.
                *[(f"trillium/new/of3/runs/{run}/{name}", f"of3/runs/{run}") for run in OF3_RUNS for name in ("steps-r*.jsonl", "runner.yml")],
                *[(f"trillium/new/of3/runs/{run}/nodemon/nodes/*.{kind}.tsv", f"of3/nodemon/{run}") for run in OF3_RUNS for kind in ("gpu", "ib", "cpu")],
                # FLUX.1 training, MLPerf (FLUX-01, job 1276503, 8 nodes; the 4-node real-data bring-up 1276381): MLPerf log, compliance output and
                # nvidia-smi telemetry (4.1 MB and 0.4 MB). Florida's published logs are not on Trillium: data/raw/trillium/flux/florida/README.
                *[(f"trillium/new/flux/runs/{run}/", f"flux/runs/{run}") for run in ("nw-flux-run8.1276503", "nw-flux-real4.1276381")],
                # FLUX.1 step-time gap against Florida, 2 nodes, 200 steps a run: job 1278802's five runs (7.4 MB), and job 1277931's MLPerf log
                # and 1 s telemetry only (44 MB; its Nsight Systems profile is not read).
                ("trillium/new/flux/dev/runs/gap.1278802/", "flux/dev/gap.1278802"),
                *[(f"trillium/new/flux/dev/runs/gap1/{name}", "flux/dev/gap1") for name in ("*_1.log", "gpu-*.csv")],
                # The running agents' own tables: cross-checks only, never a source for the report.
                ("trillium/qwen/aq/results.tsv", "agent-tables/qwen-aq"),
                ("trillium/qwen/p3/results.tsv", "agent-tables/qwen-p3"),
                ("trillium/qwen/p3/results13.tsv", "agent-tables/qwen-p3"),
                ("trillium/qwen/large/results.tsv", "agent-tables/qwen-large"),
                ("trillium/glm/ag-results.tsv", "agent-tables/glm"),
                ("trillium/glm/p3-results.tsv", "agent-tables/glm"),
                ("trillium/glm/closs-results.tsv", "agent-tables/glm")]
SACCT = "sacct -u $USER -X -S 2026-11-04 -P -o JobID,JobName,State,Start,End,Elapsed,NNodes,NodeList"
CHECKPOINT_FILTER = ["--include", "*/", "--include", "*.json", "--include", "*.jsonl", "--exclude", "*",
                     "--prune-empty-dirs"]


def ssh(command: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, command], capture_output=True, text=True,
                          timeout=60, check=True).stdout


def collect() -> None:
    dest = RAW / CLUSTER
    dest.mkdir(parents=True, exist_ok=True)
    try:
        scratch = ssh("echo $SCRATCH").strip()
        root = scratch + "/trillium"
        existing = ssh(" ".join(f"test -e {root}/{d} && echo {d};" for d in [*REMOTE, "checkpoints"])).split()
        # Before the rsync: a job not in this snapshot, or still running in it, may have partial files (build.py).
        (dest / "sacct.txt").write_text(ssh(SACCT))
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[collect] {HOST}: ssh failed ({exc}); keeping the local copy", file=sys.stderr)
        return
    rsync = ["rsync", "-a", "-e", "ssh -o BatchMode=yes"]
    try:
        sources = [f"{HOST}:{root}/{d}" for d in existing if d != "checkpoints"]
        if sources:
            subprocess.run([*rsync, *sources, f"{dest}/"], check=True, timeout=3600)
        if "checkpoints" in existing:
            subprocess.run([*rsync, *CHECKPOINT_FILTER, f"{HOST}:{root}/checkpoints", f"{dest}/"], check=True,
                           timeout=600)
        for src, sub in REMOTE_OTHER:
            (dest / sub).mkdir(parents=True, exist_ok=True)
            subprocess.run([*rsync, f"{HOST}:{scratch}/{src}", f"{dest / sub}/"], check=True, timeout=600)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[collect] {HOST}: rsync failed ({exc}); the local copy may be partial", file=sys.stderr)
        return
    print(f"[collect] {HOST}: {', '.join([*existing, *sorted({d for _, d in REMOTE_OTHER})])}")


def build(use_synthetic: bool):
    if use_synthetic and not SYN.exists():
        synthetic.generate(SYN)
    t = build_mod.build(RAW, SYN if use_synthetic else None, TABLES)
    runs = t["runs"]
    n_syn = int(runs.synthetic.sum()) if len(runs) else 0
    print(f"[build] {len(runs)} runs ({n_syn} synthetic) -> {TABLES}")
    return t


def check_series(t) -> None:
    """Each GLM-5 series holds all its node counts: always true with the synthetic data filling the gaps."""
    if t["runs"].empty:
        return
    gaps = series_gaps(t["runs"])
    failed = failed_runs(t["runs"], *gaps)  # a failure leaves its node count empty on purpose: the figure marks it
    known = {(e, n) for e, n in zip(failed.experiment, failed.nodes)} if len(failed) else set()
    unexplained = {e: [n for n in nodes if (e, n) not in known] for e, nodes in gaps.items()}
    if any(unexplained.values()) and t["runs"].synthetic.any():
        raise SystemExit(f"[plot] GLM-5 series without some of their node counts: {unexplained}")
    if gaps:
        print(f"[plot] not run yet, GLM-5 series without these node counts: {gaps}")


def plot(t) -> None:
    check_series(t)
    done = figures.render_all(t, GENERATED / "figures")
    missing = [k for k, ok in done.items() if not ok]
    print(f"[plot] {sum(done.values())} figures -> {GENERATED / 'figures'}"
          + (f" (no data: {', '.join(missing)})" if missing else ""))
    tables.write_all(t, GENERATED / "tables")


def show_progress(t) -> None:
    points = progress.table(t)
    points.to_csv(TABLES / "progress.csv", index=False)
    print(progress.markdown(progress.summary(points, t, RAW / CLUSTER / "sacct.txt")))


def report() -> None:
    if not (REPORT / "main.tex").exists():
        print(f"[report] no {REPORT / 'main.tex'}; skipped")
        return
    for _ in range(2):  # second pass resolves references
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                              cwd=REPORT, capture_output=True, text=True)
        if proc.returncode:
            print(proc.stdout[-3000:], file=sys.stderr)
            raise SystemExit("[report] pdflatex failed")
    print(f"[report] {REPORT / 'main.pdf'}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["all", "collect", "synth", "build", "plot", "report", "progress"])
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args(argv)
    use_syn = not args.no_synthetic

    if args.command == "collect" or (args.command == "all" and not args.no_collect):
        collect()
    if args.command in ("synth", "all") and use_syn:
        synthetic.generate(SYN)
        print(f"[synth] -> {SYN}")
    if args.command in ("build", "all"):
        t = build(use_syn)
    elif args.command in ("plot", "report", "progress"):
        t = build_mod.load(TABLES)
    if args.command == "progress":
        show_progress(t)
    if args.command in ("plot", "all", "report"):
        plot(t)
    if args.command in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
