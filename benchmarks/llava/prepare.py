#!/usr/bin/env python
"""Download LLaVA weights and dataset into the HF cache (no model load)."""

from benchmate.hugginface import download_hf_dataset, download_hf_model

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REVISION = "e2214c2851fadaf9241c9f9ac91dcdee51981021"
DATASET = "HuggingFaceM4/the_cauldron"
DATASET_NAME = "aokvqa"


def main():
    # Processor/tokenizer/config files live in the same repo as the weights.
    download_hf_model(MODEL_ID, revision=REVISION)
    download_hf_dataset(DATASET, split="train", name=DATASET_NAME)


if __name__ == "__main__":
    main()
