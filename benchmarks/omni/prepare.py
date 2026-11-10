#!/usr/bin/env python
"""Download the served model (server argv positional) into the HF cache: the runs are offline."""
import sys

from benchmate.hugginface import download_hf_model


def main(argv):
    model = next((arg for arg in argv if not arg.startswith("-") and "/" in arg), None)
    if model is None:
        raise SystemExit("prepare: no model in the server argv")
    download_hf_model(model)
    print(f"prepared {model}")


if __name__ == "__main__":
    main(sys.argv[1:])
