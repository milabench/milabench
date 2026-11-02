#!/usr/bin/env python
"""Tables and plots for the Kimi-K3 serving study (config/kimi_study.yaml).

Reads the ``pass_summary`` and ``eval_summary`` events that benchmarks/vllm/main.py pushes for every client
pass, from one or more run folders:

    python scripts/analysis/kimi_study.py $MILABENCH_BASE/runs/kimi-study-* --out kimi_report

Writes, under ``--out``:

* ``passes.csv``: one row per (config, pass): throughput, TTFT / TPOT / ITL / E2E percentiles
* ``itl_by_context.csv``: mean ITL per context-position bin, per (config, pass)
* ``accuracy.csv`` and ``accuracy.md``: one row per config, one column per eval task (+ delta vs the reference)
* Altair charts (``.html`` interactive + ``.png``): ``throughput_vs_latency``, ``efficiency`` (output tokens per
  joule and GPU power vs concurrency, power averaged over each pass's request window only), ``context_scaling``,
  ``itl_by_context``, ``accuracy``; ``report.html`` has them all
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import pandas as pd

# accuracy reference: no online quantization, bf16 KV cache
REFERENCE = "kimi-atom-tp8-bf16"


def bench_name(path: str) -> str:
    # <runs>/<run>/<bench>.<tag...>.data, the bench name has no dots
    return os.path.basename(path).split(".")[0]


def pass_energy(summary: dict, event_time: float, power: list, nodes: int) -> dict:
    """GPU power and energy over the window in which the pass's requests ran.

    The window is [start_time, end_time] (older runs: the pass_summary is pushed when the pass ends, so
    [event time - duration_s, event time]): server start, warmup and the gaps between passes are left out.
    ``power`` is [(time, watts summed over the node's GPUs)]; voir samples the main node only, so for a
    multi-node run the power is extrapolated to every node (``power_nodes_measured`` < ``nodes``).
    """
    end = summary.get("end_time") or event_time
    start = summary.get("start_time") or end - summary["duration_s"]
    samples = [w for t, w in power if start <= t <= end]
    if not samples or not summary.get("duration_s"):
        return {}
    watts = sum(samples) / len(samples) * nodes
    joules = watts * summary["duration_s"]
    output_tokens = summary["output_throughput"] * summary["duration_s"]
    total_tokens = summary["total_throughput"] * summary["duration_s"]
    return {
        "power_w": watts,
        "power_samples": len(samples),
        "power_nodes_measured": 1,
        "energy_kj": joules / 1000,
        "output_tok_per_j": output_tokens / joules,
        "total_tok_per_j": total_tokens / joules,
        "j_per_output_tok": joules / output_tokens if output_tokens else float("nan"),
    }


def read_events(run_dirs):
    passes, evals, itl = [], [], []
    for run in run_dirs:
        for path in sorted(glob.glob(os.path.join(run, "*.data"))):
            bench = bench_name(path)
            power, nodes, pass_events = [], 1, []
            with open(path) as f:
                for line in f:
                    if '"gpudata"' in line:
                        data = json.loads(line).get("data") or {}
                        gpus = data.get("gpudata") or {}
                        if data.get("time") and gpus:
                            power.append((data["time"], sum(g.get("power", 0) for g in gpus.values())))
                        continue
                    if line.startswith('{"event": "config"'):
                        nodes = int((json.loads(line).get("data") or {}).get("num_machines") or 1)
                        continue
                    if '"pass_summary"' not in line and '"eval_summary"' not in line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data = entry.get("data") or {}
                    common = {"run": os.path.basename(run.rstrip("/")), "config": bench}
                    if summary := data.get("pass_summary"):
                        summary = dict(summary)
                        for bin_name, value in (summary.pop("itl_by_context", None) or {}).items():
                            low, high = (int(x) for x in bin_name.split("-"))
                            itl.append({**common, "pass": summary["name"], "ctx_low": low, "ctx_high": high, **value})
                        summary.pop("time", None)
                        pass_events.append((common, summary, data.get("time")))
                    if summary := data.get("eval_summary"):
                        for metric, value in summary["scores"].items():
                            evals.append({
                                **common,
                                "task": summary["task"],
                                "metric": metric,
                                "value": value,
                                "stderr": summary["stderr"].get(metric.replace(",", "_stderr,", 1)),
                                "samples": summary.get("samples"),
                            })
            # the GPU power samples of the whole run are known only once the file is read
            for common, summary, event_time in pass_events:
                energy = pass_energy(summary, event_time, power, nodes)
                if energy and nodes > 1:
                    energy["power_nodes_measured"] = 1
                passes.append({**common, "pass": summary.pop("name"), "nodes": nodes, **summary, **energy})
    return pd.DataFrame(passes), pd.DataFrame(evals), pd.DataFrame(itl)


def accuracy_table(evals: pd.DataFrame) -> pd.DataFrame:
    if evals.empty:
        return evals
    # make-up entries (<config>-gpqa: one eval task re-run after a timeout) count for their server config
    evals = evals.assign(config=evals["config"].str.replace(r"-gpqa$", "", regex=True))
    # the last run of a config wins (re-runs after a fix)
    evals = evals.drop_duplicates(["config", "task", "metric"], keep="last")
    table = evals.pivot_table(index="config", columns="task", values="value", aggfunc="last") * 100
    if REFERENCE in table.index:
        for task in list(table.columns):
            table[f"{task} Δ"] = table[task] - table.loc[REFERENCE, task]
    return table.round(1)


def _save(chart, out: str, name: str):
    """Interactive HTML (tooltips, zoom, legend selection) and a PNG for documents."""
    chart.save(os.path.join(out, f"{name}.html"))
    try:
        chart.save(os.path.join(out, f"{name}.png"), scale_factor=2)
    except Exception as exc:  # PNG export needs vl-convert-python
        print(f"[plots] {name}.png not written: {exc}")


def plots(passes: pd.DataFrame, itl: pd.DataFrame, accuracy: pd.DataFrame, out: str):
    import warnings

    import altair as alt

    # every chart shares one legend selection on purpose (click a config: highlighted everywhere)
    warnings.filterwarnings("ignore", message="Automatically deduplicated selection parameter")
    alt.data_transformers.disable_max_rows()
    # click a config in any legend to highlight it everywhere
    pick = alt.selection_point(fields=["config"], bind="legend")
    color = alt.Color("config:N", legend=alt.Legend(orient="bottom", columns=3, labelLimit=320))
    opacity = alt.condition(pick, alt.value(1.0), alt.value(0.15))
    charts = []

    def line(data, x, y, title, tooltip, x_scale=None, y_scale=None):
        return (
            alt.Chart(data, title=title)
            .mark_line(point=True)
            .encode(
                x=alt.X(x, scale=x_scale or alt.Undefined) if isinstance(x, str) else x,
                y=alt.Y(y, scale=y_scale or alt.Undefined) if isinstance(y, str) else (y.scale(y_scale) if y_scale else y),
                color=color,
                opacity=opacity,
                tooltip=tooltip,
            )
            .add_params(pick)
            .properties(width=420, height=300)
        )

    log = alt.Scale(type="log")
    if not passes.empty:
        conc = passes[passes["pass"].str.match(r"conc\d+$")].copy()
        if not conc.empty:
            conc["concurrency"] = conc["pass"].str[4:].astype(int)
            tip = ["config", "concurrency", alt.Tooltip("output_throughput:Q", format=".1f"),
                   alt.Tooltip("tpot_ms_p50:Q", format=".1f"), alt.Tooltip("ttft_ms_p50:Q", format=".0f")]
            tradeoff = (
                line(conc, alt.X("tpot_ms_p50:Q", title="TPOT p50 (ms)", scale=alt.Scale(zero=False)),
                     alt.Y("output_throughput:Q", title="output tokens/s"),
                     "Throughput vs per-token latency (labels: concurrency)", tip)
                .encode(order="concurrency:Q")
            )
            labels = (
                alt.Chart(conc).mark_text(dx=8, dy=-6, fontSize=10)
                .encode(x="tpot_ms_p50:Q", y="output_throughput:Q", text="concurrency:Q", color=color, opacity=opacity)
            )
            row = [
                (tradeoff + labels).properties(width=420, height=300),
                line(conc, alt.X("concurrency:Q", scale=log), alt.Y("ttft_ms_p50:Q", title="TTFT p50 (ms)"),
                     "Time to first token (p50)", tip, y_scale=log),
            ]
            charts.append(alt.hconcat(*row))
            _save(alt.hconcat(*row), out, "throughput_vs_latency")

            if "output_tok_per_j" in conc:
                eff = conc.dropna(subset=["output_tok_per_j"])
                tip = ["config", "concurrency", alt.Tooltip("output_tok_per_j:Q", format=".3f"),
                       alt.Tooltip("power_w:Q", format=".0f"), alt.Tooltip("output_throughput:Q", format=".1f"),
                       "power_nodes_measured:Q", "nodes:Q"]
                row = [
                    line(eff, alt.X("concurrency:Q", scale=log), alt.Y("output_tok_per_j:Q", title="output tokens / J"),
                         "Efficiency: output tokens per joule (= tok/s per W)", tip),
                    line(eff, alt.X("concurrency:Q", scale=log), alt.Y("power_w:Q", title="GPU power (W, all GPUs)"),
                         "GPU power while the requests run", tip),
                ]
                charts.append(alt.hconcat(*row))
                _save(alt.hconcat(*row), out, "efficiency")

        ctx = passes[passes["pass"].str.match(r"ctx\d+k-c\d+$")].copy()
        if not ctx.empty:
            ctx["input_k"] = ctx["pass"].str.extract(r"ctx(\d+)k")[0].astype(int)
            ctx["concurrency"] = ctx["pass"].str.extract(r"-c(\d+)")[0].astype(int)
            ctx["series"] = ctx["config"] + " c" + ctx["concurrency"].astype(str)
            tip = ["config", "concurrency", "input_k", alt.Tooltip("ttft_ms_p50:Q", format=".0f"),
                   alt.Tooltip("tpot_ms_p50:Q", format=".1f")]
            def ctx_line(y, title, y_scale=None):
                return (
                    line(ctx, alt.X("input_k:Q", scale=log, title="input length (k tokens)"), y, title, tip,
                         y_scale=y_scale)
                    .encode(strokeDash=alt.StrokeDash("concurrency:N"), detail="series:N")
                )
            row = [
                ctx_line(alt.Y("ttft_ms_p50:Q", title="TTFT p50 (ms)"), "Prefill cost vs input length", log),
                ctx_line(alt.Y("tpot_ms_p50:Q", title="TPOT p50 (ms)"), "Decode cost vs input length"),
            ]
            charts.append(alt.hconcat(*row))
            _save(alt.hconcat(*row), out, "context_scaling")

    if not itl.empty:
        long = itl[itl["pass"].str.startswith("decodelong")].copy()
        if not long.empty:
            long["position"] = long["ctx_low"]
            long["series"] = long["config"] + " " + long["pass"]
            chart = (
                line(long, alt.X("position:Q", scale=alt.Scale(type="symlog"), title="position in the sequence (tokens, bin start)"),
                     alt.Y("itl_mean_ms:Q", title="mean ITL (ms)"), "Decode slowdown as the context fills",
                     ["config", "pass", "ctx_low", "ctx_high", alt.Tooltip("itl_mean_ms:Q", format=".1f"), "tokens"])
                .encode(strokeDash="pass:N", detail="series:N")
            )
            charts.append(chart)
            _save(chart, out, "itl_by_context")

    if not accuracy.empty:
        acc = accuracy[[c for c in accuracy.columns if not c.endswith(" Δ")]].reset_index()
        acc = acc.melt(id_vars="config", var_name="task", value_name="accuracy")
        chart = (
            alt.Chart(acc, title="Accuracy (%) per benchmark")
            .mark_bar()
            .encode(
                x=alt.X("config:N", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=260)),
                y=alt.Y("accuracy:Q", title="accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                color=color,
                opacity=opacity,
                column=alt.Column("task:N", title=None),
                tooltip=["config", "task", alt.Tooltip("accuracy:Q", format=".1f")],
            )
            .add_params(pick)
            .properties(width=220, height=260)
        )
        charts.append(chart)
        _save(chart, out, "accuracy")

    if charts:
        alt.vconcat(*charts).properties(title="Kimi-K3 serving study").save(os.path.join(out, "report.html"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run folders ($MILABENCH_BASE/runs/<run>)")
    parser.add_argument("--out", default="kimi_report")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    passes, evals, itl = read_events(args.runs)

    if not passes.empty:
        passes = passes.drop_duplicates(["config", "pass"], keep="last")
        passes.to_csv(os.path.join(args.out, "passes.csv"), index=False)
        cols = ["config", "pass", "output_throughput", "request_throughput", "ttft_ms_p50", "ttft_ms_p99",
                "tpot_ms_p50", "tpot_ms_p99", "e2el_ms_p50", "power_w", "j_per_output_tok", "failed"]
        print(passes[[c for c in cols if c in passes]].to_string(index=False, float_format="%.1f"))
    if not itl.empty:
        itl.to_csv(os.path.join(args.out, "itl_by_context.csv"), index=False)

    table = accuracy_table(evals)
    if not table.empty:
        evals.to_csv(os.path.join(args.out, "accuracy_raw.csv"), index=False)
        table.to_csv(os.path.join(args.out, "accuracy.csv"))
        with open(os.path.join(args.out, "accuracy.md"), "w") as f:
            try:
                f.write(table.to_markdown())
            except ImportError:  # to_markdown needs tabulate
                header = "| config | " + " | ".join(table.columns) + " |"
                rows = ["| " + " | ".join([str(i), *(f"{v:.1f}" for v in row)]) + " |" for i, row in table.iterrows()]
                f.write("\n".join([header, "|---" * (len(table.columns) + 1) + "|", *rows]) + "\n")
        print()
        print(table.to_string())

    plots(passes, itl, table, args.out)
    print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
