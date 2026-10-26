#!/usr/bin/env python
"""Stage the HF config of the model: the GLM-5 spec reads the DSA indexer sizes from it. No weights are needed
(the model is randomly initialised, the data is random)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    cfg = json.loads(os.environ["MILABENCH_CONFIG"])
    target = Path(str(cfg["model"]["hf_path"]).replace("{milabench_data}", os.environ["MILABENCH_DIR_DATA"]))
    if (target / "config.json").exists():
        return
    snapshot_download(
        cfg["model"]["hf"],
        local_dir=target,
        token=os.environ.get("MILABENCH_HF_TOKEN") or os.environ.get("HF_TOKEN"),
        allow_patterns=["config.json", "*.py", "generation_config.json"],
    )


if __name__ == "__main__":
    main()
