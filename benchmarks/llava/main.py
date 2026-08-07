#!/usr/bin/env python

from dataclasses import dataclass
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, LlavaForConditionalGeneration

import argklass
from benchmate.observer import BenchObserver
import torchcompat.core as compat


def apply_chat_template(texts):
    """Format one sample's conversation list into a LLaVA prompt."""
    formatted_conversation = "<image>\n"
    for conversation in texts:
        formatted_conversation += f"Human: {conversation['user'][0]}\n"
        formatted_conversation += f"Assistant: {conversation['assistant'][0]}\n"
    return formatted_conversation.strip()


def llava_collate(batch):
    """Flatten cauldron samples into lists the processor can batch."""
    images = []
    texts = []
    for item in batch:
        sample_images = item["images"]
        if isinstance(sample_images, (list, tuple)):
            images.append(sample_images[0] if len(sample_images) == 1 else sample_images)
        else:
            images.append(sample_images)
        texts.append(item["texts"])
    return {"images": images, "texts": texts}


@dataclass
class Arguments:
    batch_size: int = 10
    epochs: int = 10
    seed: int = 42
    num_workers: int = 5
    gradient_accumulation_steps: int = 1


def main():
    parser = argklass.ArgumentParser(description="llava")
    parser.add_arguments(Arguments)
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="all",
        project_dir="logs",
    )

    set_seed(args.seed)

    # Load LLaVA model and processor with device_map="auto"
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.bfloat16,
        device_map=compat.device_type,
        revision="e2214c2851fadaf9241c9f9ac91dcdee51981021"
    )
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        revision="e2214c2851fadaf9241c9f9ac91dcdee51981021"
    )

    # Load dataset and create DataLoader
    dataset = load_dataset("HuggingFaceM4/the_cauldron", "aokvqa")["train"]
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=llava_collate,
        num_workers=args.num_workers
    )

    def batch_size_fn(batch):
        return len(batch["images"])

    observer = BenchObserver(
        batch_size_fn=batch_size_fn, earlystop=70, raise_stop_program=True,
        stdout=True,
    )
    optimizer = observer.optimizer(torch.optim.AdamW(model.parameters(), lr=5e-5))
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # model = torch.compile(model,backend="hpu_backend")

    for epoch in range(args.epochs):
        for i, batch in enumerate(observer.iterate(dataloader)):
            images = batch["images"]
            prompts = [apply_chat_template(texts) for texts in batch["texts"]]

            inputs = processor(
                text=prompts, images=images, return_tensors="pt", padding=True
            )

            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

            inputs = {
                k: v.to(
                    accelerator.device,
                    dtype=torch.float32 if v.dtype == torch.float16 else v.dtype,
                )
                for k, v in inputs.items()
            }

            outputs = model(**inputs)
  
            loss = outputs.loss
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            compat.mark_step()
            optimizer.step()
            compat.mark_step()
            optimizer.zero_grad()
            observer.record_loss(loss)

    assert epoch < 2, "milabench stopped the train script before the end of training"
    assert (
        observer.step < 70
    ), "milabench stopped the train script before the end of training"


if __name__ == "__main__":
    from voir.phase import StopProgram
    from benchmate.monitor import bench_monitor

    try:
        with bench_monitor():
            main()
    except StopProgram:
        pass