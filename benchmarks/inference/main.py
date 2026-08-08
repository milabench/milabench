#!/usr/bin/env python
from dataclasses import dataclass

from datasets import load_dataset, Audio
from argklass import ArgumentParser
from argklass.arguments import argument
import torch
import torchcompat.core as accelerator
from torch.utils.data import DataLoader


whisper_defaults_generation_args = {
    # "max_new_tokens": 448,
    "num_beams": 1,
    "condition_on_prev_tokens": False,
    # zlib compression ratio threshold (in token space)
    "compression_ratio_threshold": 1.35,  
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
    'language': 'en',
}

flux_default_generation_args = {
    "height": 256,
    "width": 256,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
}

chat_default_generation_args = {

}


class InferenceBenchmark:
    def __init__(self):
        self.raise_stop = False
        self.custom_step = False

    def get_batch_size(self, item):
        return len(item)

    def prepare_voir(self, args):
        from benchmate.observer import BenchObserver
        from benchmate.monitor import bench_monitor
        
        observer = BenchObserver(
            accelerator.Event, 
            earlystop=65,
            batch_size_fn=self.get_batch_size,
            raise_stop_program=self.raise_stop,
            stdout=True,
        )

        return observer, bench_monitor

    def load_model(self, args, device):
        pass

    def transform(self, item):
        return item

    def collate(self, group):
        batch = []
        for item in group:
            batch.append(self.transform(item))
        return batch

    def load_dataset(self, observer, args):
        # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        dataset = load_dataset(
            args.dataset,
            name=args.subset,  # Subset
            split=args.split,  # Split
        )

        return observer.loader(self.dataloader(dataset, args), custom_step=self.custom_step)

    def dataloader(self, dataset, args):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collate
        )

    def run(self, pipe, batch, kwargs):
        return pipe(batch, **kwargs, batch_size=len(batch))






class WhisperBenchmark(InferenceBenchmark):

    def huggingface_pipeline(self, model, processor, device):
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            # dtype=args.dtype,
            device=device,
        )

        return pipe, {}

    def custom_pipeline(self, model, processor, device):
        def inference_pipe(batch, generate_kwargs, batch_size):
            audio = [x["array"].numpy() for x in batch]

            # Whisper encoder requires fixed 3000 mel frames (30s); padding=True
            # only pads to the longest clip in the batch.
            inputs = processor(
                audio,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
                sampling_rate=16_000,
            )

            generate_inputs = {
                "input_features": inputs.input_features.to(device).to(torch.bfloat16),
            }
            if "attention_mask" in inputs:
                generate_inputs["attention_mask"] = inputs.attention_mask.to(device)

            with torch.inference_mode(), torch.autocast(
                device_type=accelerator.device_type, dtype=torch.bfloat16
            ):
                generated_ids = model.generate(**generate_inputs, **generate_kwargs)

            return processor.batch_decode(generated_ids, skip_special_tokens=True)

        return inference_pipe, {}

    def load_model(self, args, device):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model, 
            # dtype=args.dtype, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            use_safetensors=True
        )

        model.to(device)

        processor = AutoProcessor.from_pretrained(args.model)

        kwargs = dict(args.kwargs) if args.kwargs else whisper_defaults_generation_args

        if False:
            pipe, _ = self.huggingface_pipeline(model, processor, device)
        else:
            pipe, _ = self.custom_pipeline(model, processor, device)

        # try:
        #     pipe = torch.compile(pipe, backend="inductor", mode="max-autotune")
        # except Exception as e:
        #     print("Could not compile manual pipe")

        return pipe, kwargs

    def run(self, pipe, batch, kwargs):
        if isinstance(batch, dict):
            batch = [batch]
        if not batch:
            raise ValueError("whisper batch is empty")
        return pipe(batch, generate_kwargs=kwargs, batch_size=len(batch))

    def get_batch_size(self, x):
        return len(x)

    def transform(self, item):
        audio = item["audio"]
        data = audio.get_all_samples()
        array = data.data.mean(dim=0)
        return {
            "array": array,
            "sampling_rate": data.sample_rate
        }


class FluxBenchmark(InferenceBenchmark):
    def __init__(self):
        super().__init__()
    
        self.i = 0
        self.dataset = None
        self.bs = 0
        self.raise_stop = False
        self.custom_step = True

    def get_batch_size(self, item):
        self.bs = len(item)
        return self.bs

    def load_model(self, args, device):
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            args.model, 
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device_type,
        )

        if False:
            models = {
                'transformer': pipeline.transformer,
                'scheduler': pipeline.scheduler,
                'vae': pipeline.vae,
                'text_encoder': pipeline.text_encoder,
                'text_encoder_2': pipeline.text_encoder_2,
                'tokenizer': pipeline.tokenizer,
                'tokenizer_2': pipeline.tokenizer_2,
            }

            for model in models:
                pass

        # pipeline.transformer.to(memory_format=torch.channels_last)
        # pipeline.vae.to(memory_format=torch.channels_last)
        # pipeline.transformer.enable_forward_chunking(chunk_size=1, dim=1)
        # torch.compile(pipeline.)

        # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        # pipe.enable_model_cpu_offload()

        kwargs = dict(args.kwargs) or flux_default_generation_args
        return pipe, kwargs

    def load_dataset(self, observer, args):
        base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
        num_shards = 10  # Number of webdataset tar files
        urls = [base_url.format(i=i) for i in range(num_shards)]
        dataset = load_dataset(
            "webdataset", 
            data_files={"train": urls}, 
            split="train", 
            streaming=False
        )
        self.bs = args.batch_size
        self.dataset = observer.loader(self.dataloader(dataset, args), custom_step=self.custom_step)
        return self.dataset

    def transform(self, item):
        p = item["json"]["prompt"]
        return p[:min(len(p), 70)]

    def collate(self, group):
        batch = []
        for item in group:
            batch.append(self.transform(item))
        
        self.bs = len(batch)
        return batch

    def on_step(self, pipe, step: int, timestep: int, kwargs):
        self.dataset.acc_batch_size = self.bs
        should_stop = self.dataset.step()
        return {}

    def run(self, pipe, batch, kwargs):
        if isinstance(batch, str):
            batch = [batch]
        if not batch:
            raise ValueError("txt-to-image flux batch is empty")

        # Keep metrics aligned with the prompts actually sent to the pipeline.
        self.bs = len(batch)

        call_kwargs = dict(kwargs)
        generator = call_kwargs.get("generator")
        if generator is None:
            call_kwargs["generator"] = [
                torch.Generator(accelerator.device_type).manual_seed(0)
                for _ in range(self.bs)
            ]
        elif not isinstance(generator, list):
            call_kwargs["generator"] = [generator] * self.bs

        return pipe(
            batch,
            callback_on_step_end_tensor_inputs=[],
            callback_on_step_end=self.on_step,
            **call_kwargs,
        )




class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tok_in = 0

    def __call__(self, *args, **kwargs):
        tensor = self.tokenizer(*args, **kwargs)
        shape = tensor["input_ids"].shape
        self.tok_in += shape[0] * shape[1]
        return tensor


class ChatBenchmark(InferenceBenchmark):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.tok_per_sec = True
        self.tokenizer = None

    def transform(self, item):
        return item["problem"]

    def get_batch_size(self, item):
        return len(item)

    def load_dataset(self, observer, args):
        dataset = load_dataset(
            args.dataset,
            name=args.subset,  # Subset
            split=args.split,  # Split
        )

        self.dataset = observer.loader(self.dataloader(dataset, args), custom_step=self.tok_per_sec)
        return self.dataset

    def load_model(self, args, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = TokenizerWrapper(tokenizer)
        model_device = next(model.parameters()).device

        def inference_pipe(batch, generate_kwargs, batch_size):
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generate_kwargs)

            input_lengths = inputs["attention_mask"].sum(dim=1)
            outputs = []
            for i in range(generated_ids.shape[0]):
                new_tokens = generated_ids[i, input_lengths[i] :].tolist()
                outputs.append([{"generated_token_ids": new_tokens}])
            return outputs

        kwargs = dict(args.kwargs) or chat_default_generation_args
        return inference_pipe, kwargs

    def run(self, pipe, batch, kwargs):
        outputs = pipe(batch, generate_kwargs=kwargs, batch_size=len(batch))

        if self.tok_per_sec:
            tok_out = 0
            for out in outputs:
                for o in out:
                    tok_out += len(o["generated_token_ids"])

            tok_tot = self.tokenizer.tok_in + tok_out
            self.dataset.step(tok_tot)
            self.tokenizer.tok_in = 0

        return outputs


def load_benchmark(argv):
    match argv.mode:
        case "whisper":
            return WhisperBenchmark()

        case "flux":
            return FluxBenchmark()
    
        case "chat":
            return ChatBenchmark()
    
    raise RuntimeError(f"Benchmark {argv.mode} does not exist")


def parse_kv(s):
    key, value = s.split("=", 1)
    return key, value


@dataclass
class Arguments:
    mode: str = None
    dataset: str = None
    split: str = None
    subset: str = None
    model: str = None
    batch_size: int = 16
    kwargs: list = argument(default_factory=list, nargs="+", default=[], type=parse_kv)
    dtype: str = "bfloat16"
    multi_gpu: bool = False
    prepare: bool = False


def main(argv=None):
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args(argv)

    bench = load_benchmark(args) 
    observer, monitor = bench.prepare_voir(args)
    device = accelerator.fetch_device(0)

    if args.prepare:
        dataset = bench.load_dataset(observer, args)
        pipe, kwargs = bench.load_model(args, device)
        return 0
    
    with monitor():
        with torch.no_grad():
            dataset = bench.load_dataset(observer, args)
            pipe, kwargs = bench.load_model(args, device)
            # We cannot wrap the dataset with our timed loader anymore
            # dataset = setup_dataset(args)
            # output = pipe(dataset, **kwargs, batch_size=args.batch_size)

            # Here it still works
            for batch in dataset:
                output = bench.run(pipe, batch, kwargs)


# MultiGPU Setup (?)
#   Not worth doing that might as well just launch N same bench
#   Split the batch across GPUs
#       result = pipe(texts[accelerator.process_index::accelerator.num_processes])
#   

if __name__ == "__main__":
    # milabench run --config /home/mila/d/delaunap/scratch/milabench/benchmarks/inference/dev.yaml --base /tmp/data/ --use-current-env --select whisper-transcribe-single
    # milabench run --config /home/mila/d/delaunap/scratch/milabench/benchmarks/inference/dev.yaml --base /tmp/data/ --use-current-env --select txt-to-image-gpus
    # milabench run --config dev.yaml --base /tmp/data/ --use-current-env --select llm-chat-completion


    main()
