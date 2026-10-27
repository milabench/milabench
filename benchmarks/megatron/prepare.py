#!/usr/bin/env python
"""Stage the HF config and tokenizer of the model (the GLM-5 spec reads the DSA indexer sizes from the config) and a
FineWeb subset to train on. No weights are needed (the model is randomly initialised)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from benchmate.fineweb import prepare_fineweb


def main():
    cfg = json.loads(os.environ["MILABENCH_CONFIG"])
    target = Path(str(cfg["model"]["hf_path"]).replace("{milabench_data}", os.environ["MILABENCH_DIR_DATA"]))
    token = os.environ.get("MILABENCH_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not (target / "tokenizer.json").exists():
        snapshot_download(
            cfg["model"]["hf"],
            local_dir=target,
            token=token,
            allow_patterns=[
                "config.json",
                "*.py",
                "generation_config.json",
                "tokenizer*",
                "special_tokens_map.json",
                "chat_template*",
            ],
        )

    data = cfg.get("data") or {}
    prepare_fineweb(
        Path(os.environ["MILABENCH_DIR_DATA"]) / "fineweb" / data.get("subset", "sample-10BT"),
        int(data.get("samples", 20000)),
        subset=data.get("subset", "sample-10BT"),
        token=token,
    )


if __name__ == "__main__":
    main()
