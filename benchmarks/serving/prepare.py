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


def client_passes() -> list[dict]:
    """client.passes of the shared section and the arch variant."""
    import json
    import os

    try:
        cfg = json.loads(os.environ.get("MILABENCH_CONFIG", "{}"))
    except json.JSONDecodeError:
        return []
    arch = cfg.get("system", {}).get("arch", "cuda")
    sections = [cfg.get("client") or {}, (((cfg.get("variants") or {}).get(arch) or {}).get("client") or {})]
    return [spec for section in sections for spec in section.get("passes") or []]


def eval_tasks() -> list[str]:
    """lm-eval tasks of the client passes (client.passes[].eval.tasks)."""
    tasks = []
    for spec in client_passes():
        tasks += [t for t in (spec.get("eval") or {}).get("tasks", []) if t not in tasks]
    return tasks


def _flag_value(argv, name):
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def draft_models() -> list[str]:
    """Speculative decoding drafters of the server argv: vLLM ``--speculative-config {"model": ...}``,
    SGLang ``--speculative-draft-model-path``. The server loads them offline like the target."""
    import json

    argv = sys.argv[1:]
    models = []
    if config := _flag_value(argv, "--speculative-config"):
        try:
            model = json.loads(config).get("model")
        except (json.JSONDecodeError, AttributeError):
            model = None
        if model:
            models.append(model)
    if model := _flag_value(argv, "--speculative-draft-model-path"):
        models.append(model)
    return models


def download_eval_datasets(tasks):
    """Building the tasks downloads their datasets into the HF cache (the runs are offline)."""
    import os

    from lm_eval.tasks import TaskManager, get_task_dict

    print(f"Downloading lm-eval datasets: {tasks}")
    include = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_tasks")
    get_task_dict(tasks, TaskManager(include_path=include))


def main():
    args = arguments()

    if args.hf_name:
        download_hf_dataset(args.hf_name, args.hf_split)

    if tasks := eval_tasks():
        download_eval_datasets(tasks)

    from benchmate.benchserve import prepare_text_corpus
    prepare_text_corpus(sys.argv[1:], client_passes())

    # QualityBench / SWE-bench passes: their tools, datasets and container images
    from benchmate.benchserve import prepare_passes
    prepare_passes()

    model = args.model or args.server_model
    hf_aliases = {
        "Kimi-K3": "moonshotai/Kimi-K3",
        "GLM-5.2-FP8": "zai-org/GLM-5.2-FP8",
        "GLM-5.2-MXFP4": "amd/GLM-5.2-MXFP4",
    }
    model = hf_aliases.get(model, model)
    if model is None:
        raise SystemExit("prepare: no model found in server positional or --model")
    download_hf_model(model)

    for draft in draft_models():
        print(f"Downloading draft model: {draft}")
        download_hf_model(draft)

    # setup_flashinfer()

    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
