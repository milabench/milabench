#!/usr/bin/env python

import json
import os
import argparse
import time
import sys

import torch

from voir.phase import StopProgram
from benchmate.monitor import setupvoir
from benchmate.toggles import get_observation_count
import torchcompat.core as accelerator

root = os.path.dirname(__file__)

WIKITEXT_DATASET = "Milabench/wikitext"
WIKITEXT_CONFIG = "wikitext-103-v1"
WIKITEXT_REVISION = "a79fad1"
# LLAMA official tokenizer is gated, this one is the same and public
TOKENIZER = "hf-internal-testing/llama-tokenizer"


def load_cached_wikitext(args, err):
    """Train split saved by prepare, read from the datasets cache (no network).

    Offline, datasets looks for the cache folder of config '<config>-data_dir=<config>', but the
    online load saved it as '<config>-<hash of data_dir>', so load_dataset never finds it.
    """
    import glob
    from datasets import Dataset, DatasetDict, concatenate_datasets
    from datasets import config as ds_config

    namespace, name = args.dataset_name.split("/")
    pattern = os.path.join(
        ds_config.HF_DATASETS_CACHE, f"{namespace}___{name}",
        f"{args.dataset_config_name}*", "*", "*", "*-train*.arrow"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"no cached dataset in {pattern}") from err
    builds = {os.path.dirname(f) for f in files}
    if len(builds) > 1:
        raise RuntimeError(f"several cached builds: {sorted(builds)}") from err
    print(f"offline: loading {len(files)} cached arrow files from {builds.pop()}")
    return DatasetDict({"train": concatenate_datasets([Dataset.from_file(f) for f in files])})


def load_wikitext(args):
    from datasets import load_dataset

    try:
        return load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            revision=args.dataset_rev,
            data_dir=args.dataset_config_name,
        )
    except Exception as err:
        if args.prepare:
            raise
        return load_cached_wikitext(args, err)


def available_models():
    models = dict()

    for size in ("7b", "13b", "70b"):
        models[f"llama2-{size}"] = {
            "name": f"meta-llama/Llama-2-{size}-chat-hf",
            "config": f"llama2_{size}_chat_hf.config",
        }

    return models


class WrappedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.count = 0

    def __call__(self, *args, **kwargs):
        input_ids = self.tokenizer(*args, **kwargs)

        self.count = 1
        for c in input_ids["input_ids"].shape:
            self.count *= c

        return input_ids

    def __getattr__(self, attr):
        if hasattr(self.tokenizer, attr):
            method = getattr(self.tokenizer, attr)
            return method
        else:
            raise AttributeError(
                f"'{type(self.tokenizer).__name__}' object has no attribute '{attr}'"
            )


def println(*args, **kwargs):
    print(*args, *kwargs, file=sys.stderr)


def huggingface_main(args, model, config):
    # Huggingface imported AFTER setup
    import transformers
    from transformers import LlamaForCausalLM, LlamaTokenizerFast
    from transformers.models.llama.configuration_llama import LlamaConfig
    println("Dataset")
    dataset = load_wikitext(args)

    if args.prepare:
        # The run loads the tokenizer offline: it must be in the hub cache as well
        println("Tokenizer")
        LlamaTokenizerFast.from_pretrained(TOKENIZER)
        if args.pretrained:
            # The run loads the weights offline: they must be in the hub cache.
            # safetensors only, the repo also has the same weights as .bin
            from huggingface_hub import snapshot_download

            println("Weights")
            snapshot_download(
                config["_name_or_path"],
                allow_patterns=["*.json", "*.safetensors"],
            )
        println("Prepare complete")
        return 0

    println("Tokenizer")
    # LLAMA tokenizer official tokenizer is hidden behind a login
    hf_tokenizer = LlamaTokenizerFast.from_pretrained(
        TOKENIZER,
        model_max_length=256,
        truncation=True,
    )
    # Offline, transformers 5 does not fail when the tokenizer files are missing: it returns an
    # empty tokenizer (3 tokens) that encodes every prompt to nothing, and the model then generates
    # from an empty prompt (slow, meaningless rate).
    if len(hf_tokenizer) < 1000:
        raise RuntimeError(f"{TOKENIZER} has {len(hf_tokenizer)} tokens: its files are missing from the cache (prepare)")
    tokenizer = WrappedTokenizer(hf_tokenizer)
    
    # We do not download LLAMA because it takes too long
    # we just instantiate an untrained one
    println("Model")
    device = accelerator.fetch_device(0)
    print(device)

    if args.pretrained:
        model = LlamaForCausalLM.from_pretrained(
            config["_name_or_path"], 
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = LlamaForCausalLM(LlamaConfig.from_dict(config)).to(device=device, dtype=torch.bfloat16)

    println("Pipeline")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        # 
        tokenizer=tokenizer,
    )

    in_token_count = 0
    out_token_count = 0

    start = time.time()

    log, monitor = setupvoir()
    max_obs = get_observation_count(40)

    # loader = Wrapper(dataset["train"], accelerator.Event, earlystop=60)
    loader = dataset["train"]

    println("Starting")
    count = 0
    for entry in loader:
        text = entry["text"].strip()

        # Titles
        if text == "" or text.startswith(" = ") or len(text) < 10:
            continue

        count += 1
        sequences = pipeline(
            text,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            truncation=True,
        )

        for seq in sequences:
            out_token_count += len(seq["generated_text"])

        in_token_count += tokenizer.count
        total = out_token_count + in_token_count

        elapsed = time.time() - start
        println(
            f"{elapsed =}, {total / elapsed =} {in_token_count =} {out_token_count =}"
        )

        if total > 30:
            out_token_count = 0
            in_token_count = 0
            start = time.time()

            if log is not None:
                log({"task": "train", "rate": total / elapsed, "units": "Tok/s", "time": time.time()})

        if count > max_obs:
            break

    monitor.stop()


def main():
    import torch

    models = available_models()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama2-7b", choices=models.keys())
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cache", required=True, type=str)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--dataset_name", default=WIKITEXT_DATASET)
    parser.add_argument("--dataset_config_name", default=WIKITEXT_CONFIG)
    parser.add_argument("--dataset_rev", default=WIKITEXT_REVISION)

    #
    args = parser.parse_args()
    os.environ["XDG_CACHE_HOME"] = str(args.cache)

    settings = models[args.model]
    model, config = settings["name"], settings["config"]

    with open(os.path.join(root, "config", config), "r") as file:
        config = json.load(file)

    with torch.no_grad():
        return huggingface_main(args, model, config)


if __name__ == "__main__":
    from benchmate.metrics import StopProgram
    import traceback
    try:
        main()
    except StopProgram:
        print("Early stopped")
    except Exception:
        traceback.print_exc()