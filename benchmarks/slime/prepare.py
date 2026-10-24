#!/usr/bin/env python
"""Stage HF weights, the torch_dist checkpoint and data (sources are cloned by benchfile.install)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

import common


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def prepare_model(cfg):
    argv = cfg["argv"]
    hf_dir = common.resolve(argv["--hf-checkpoint"])
    # no --ref-load (e.g. GLM-5.3-Flash, recipe without KL loss): nothing to convert to
    torch_dist = common.resolve(argv["--ref-load"]) if "--ref-load" in argv else None
    snapshot_download(
        cfg["model"]["hf"],
        local_dir=hf_dir,
        token=os.environ.get("MILABENCH_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py"],
    )

    if torch_dist is None or (torch_dist / "latest_checkpointed_iteration.txt").exists():
        return
    if os.environ.get("SLIME_DOWNLOAD_ONLY"):
        # the conversion runs Megatron/TE on GPUs: do it later on a GPU node
        print(f"[slime] SLIME_DOWNLOAD_ONLY: skipping conversion to {torch_dist}", file=sys.stderr)
        return
    if cfg["model"].get("convert") is False and not os.environ.get("SLIME_CONVERT"):
        # e.g. 744B: the conversion needs a multi-node job; provide a converted
        # checkpoint at --ref-load or set SLIME_CONVERT=1.
        print(f"[slime] skipping conversion, expecting {torch_dist}", file=sys.stderr)
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(common.code_dir()), str(common.slime_src()), str(common.megatron_src()), env.get("PYTHONPATH", "")]
    )
    # >1 spreads the layers over the GPUs (the converter makes pipeline size = world size), needed for big models
    nproc = os.environ.get("SLIME_CONVERT_NPROC") or str(cfg["model"].get("convert_nproc", 1))
    run(
        [
            "torchrun", f"--nproc-per-node={nproc}",
            str(common.slime_src() / "tools" / "convert_hf_to_torch_dist.py"),
            *common.model_argv(),
            "--hf-checkpoint", str(hf_dir),
            "--save", str(torch_dist),
        ],
        env=env,
        cwd=str(common.slime_src()),
    )


def prepare_fineweb_data(out, rows=4096, max_chars=16000):
    """FineWeb is raw text: wrap each document as one assistant turn so slime's
    SFT rollout (chat template + assistant loss mask) trains on the whole text."""
    if out.exists():
        return
    from datasets import Dataset, load_dataset

    stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    messages = [
        [
            {"role": "user", "content": "Write a web document."},
            {"role": "assistant", "content": row["text"][:max_chars]},
        ]
        for row in stream.take(rows)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict({"messages": messages}).to_parquet(str(out))


def prepare_openhermes_data(out, rows=4096):
    if out.exists():
        return
    from datasets import load_dataset

    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    ds = load_dataset("teknium/OpenHermes-2.5", split=f"train[:{rows}]")
    ds = ds.map(
        lambda s: {"messages": [{"role": roles[t["from"]], "content": t["value"]} for t in s["conversations"]]},
        remove_columns=ds.column_names,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out))


def prepare_rl_data(out):
    if out.exists():
        return
    snapshot_download("zhuzilin/dapo-math-17k", repo_type="dataset", local_dir=out.parent)


DATASETS = {
    "openhermes": prepare_openhermes_data,
    "fineweb": prepare_fineweb_data,
    "dapo-math-17k": prepare_rl_data,
}


def main():
    cfg = common.config()
    prepare_model(cfg)
    DATASETS[cfg["dataset"]](common.resolve(cfg["argv"]["--prompt-data"]))


if __name__ == "__main__":
    main()
    # The streamed FineWeb download leaves a background HTTP thread behind; at interpreter shutdown it
    # aborts the process ("PyGILState_Release", exit -6) although everything was written. Everything is
    # done and flushed here, so skip the finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
