#!/usr/bin/env python

from argparse import ArgumentParser
from benchmate.hugginface import download_hf_dataset, download_hf_model
import subprocess
import sys


def arguments():
    argv = [arg for arg in sys.argv[1:] if arg != '--']

    parser = ArgumentParser()
    parser.add_argument('server_model', type=str, nargs='?', default=None, help='Model to use for the server')
    parser.add_argument('--model', type=str, help='Model name (client-side)')
    parser.add_argument('--dataset-name', type=str, help='Dataset name (random, hf, etc.)')
    parser.add_argument('--dataset-path', type=str, help='Path to HuggingFace dataset')
    parser.add_argument('--hf-name', type=str, help='HuggingFace dataset name')
    parser.add_argument('--hf-split', type=str, default=None, help='Dataset split to use')

    args, _ = parser.parse_known_args(argv)
    return args


def _is_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    # ROCm builds expose a hip version; flashinfer is CUDA-only.
    return bool(getattr(torch.version, "cuda", None)) and not getattr(
        torch.version, "hip", None
    )


def setup_flashinfer():
    if not _is_cuda():
        print("Skipping flashinfer setup (CUDA-only)")
        return

    commands = [
        # ["flashinfer", "clear-cache"],
        ["flashinfer", "show-config"],
        # ["flashinfer", "download-cubin"],
    ]
    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"Warning: '{' '.join(cmd)}' exited with code {result.returncode}")


def main():
    args = arguments()

    if args.hf_name:
        download_hf_dataset(args.hf_name, args.hf_split)

    model = args.model or args.server_model
    hf_aliases = {
        "Kimi-K3": "moonshotai/Kimi-K3",
    }
    model = hf_aliases.get(model, model)
    if model is None:
        raise SystemExit("prepare: no model found in server positional or --model")
    download_hf_model(model)

    setup_flashinfer()

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
