#!/usr/bin/env python
"""Everything the offline runs need: the model's files and the Decision Index kit (its own venv).

argv: <server argv> -- <client argv> (benchmarks/serving's build_prepare_plan). The public suite is not downloaded
here: it is rebuilt once from its 43 upstream datasets (~7 GB, ~17 GB of work space, an hour or more) and kept in
$MILABENCH_DECISION_INDEX (default $MILABENCH_DIR_DATA/decision-index):

    decision-index suite rebuild --work work
    decision-index suite import --dir suite-0.3 --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \\
        --added-rows .../release-v2-rebuilt/added-rows.jsonl.gz --gsm8k-rows .../release-v3-rebuilt/gsm8k-rows.jsonl.gz
    decision-index suite sample --dir suite-0.3 --n 6000 --out sample-6000.jsonl.gz

(cais/hle is gated: accept its terms on the Hub first.) The suite is not redistributable: keep it on the machine.
"""

import os
import sys
from argparse import ArgumentParser

from benchmate.benchserve import tools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main as client  # noqa: E402


def main():
    argv = [arg for arg in sys.argv[1:] if arg != "--"]
    parser = ArgumentParser()
    parser.add_argument("server_model", nargs="?", default=None)
    parser.add_argument("--model", default=None)
    args, _ = parser.parse_known_args(argv)

    tools.ensure("decision-index")
    data = client.data_dir()
    missing = [p for p in ("suite-0.3", client.ROWS["quick"]) if not os.path.exists(os.path.join(data, p))]
    if missing:
        raise SystemExit(f"Decision Index suite missing in {data}: {missing}. Rebuild it once (see prepare.py).")

    model = args.server_model or args.model
    if model and not os.path.isdir(model):
        from huggingface_hub import snapshot_download

        print("model ready:", snapshot_download(model))
    print("Prepare script completed successfully")


if __name__ == "__main__":
    main()
