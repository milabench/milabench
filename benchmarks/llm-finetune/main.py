#!/usr/bin/env python
"""LoRA / QLoRA fine-tuning benchmark: unsloth + TRL SFTTrainer on a pinned instruction-dataset slice.

Fixed job shape: every optimizer step trains on exactly batch_size * gradient_accumulation sequences of
seq_length tokens (packing, "wrapped" strategy), so tokens/step is a constant and runs are comparable
across models. The first --warmup_steps are not measured.

Reports (voir data events):
  rate              training tokens/s, one event per measured step (units tok/s)
  finetune_summary  once at the end: tokens/s, step time p50/p90, peak memory (torch allocator and the
                    machine's used memory: unified memory on a DGX Spark), energy over the measured window
                    (NVML energy counter, else integrated power), J/token and kWh per 1M tokens,
                    loss at the start / end and the loss sanity check
"""

import os
import statistics
import threading
import time

from argklass import ArgumentParser

import torch

if hasattr(torch, "xpu") and torch.xpu.is_available():
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_PADDING_FREE", "1")

import unsloth  # noqa: E402,F401  (must precede trl/transformers)
from unsloth import FastLanguageModel, FastModel  # noqa: E402

from datasets import load_dataset  # noqa: E402
from transformers import AutoConfig, TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from ftargs import Arguments  # noqa: E402


def push():
    from benchmate.metrics import file_push, give_push, is_running_with_voir

    return give_push() if is_running_with_voir() else file_push()


# --- power / memory sampling ------------------------------------------------------------------------


class Sampler:
    """NVML energy counter at the window's ends, plus a 4 Hz thread for power (fallback energy) and the
    machine's used memory (on unified memory the GPU's allocations are the machine's)."""

    def __init__(self, gpu_index=0, interval=0.25):
        self.interval = interval
        self.power, self.mem_used = [], []
        self.handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as exc:  # no NVML (XPU, ...): memory only
            print(f"NVML unavailable: {exc}", flush=True)
        self._stop = threading.Event()
        self._thread = None
        self.e0 = self.e1 = None
        self.t0 = self.t1 = None

    def _energy(self):
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # mJ -> J
        except Exception:
            return None

    @staticmethod
    def _meminfo_used_gib():
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, value = line.split(":", 1)
                    info[key] = int(value.split()[0])
            return (info["MemTotal"] - info["MemAvailable"]) / 2**20
        except Exception:
            return None

    def _loop(self):
        while not self._stop.is_set():
            if self.handle is not None:
                try:
                    self.power.append(self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
                except Exception:
                    pass
            used = self._meminfo_used_gib()
            if used is not None:
                self.mem_used.append(used)
            self._stop.wait(self.interval)

    def start(self):
        self.t0 = time.time()
        self.e0 = self._energy() if self.handle is not None else None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.t1 = time.time()
        self.e1 = self._energy() if self.handle is not None else None
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self):
        out = {"window_start": self.t0, "window_end": self.t1}
        seconds = (self.t1 or 0) - (self.t0 or 0)
        if self.e0 is not None and self.e1 is not None and self.e1 > self.e0:
            out["energy_j"], out["energy_source"] = self.e1 - self.e0, "nvml-counter"
        elif self.power:
            out["energy_j"], out["energy_source"] = statistics.fmean(self.power) * seconds, "power-integral"
        if self.power:
            out["power_mean_w"] = statistics.fmean(self.power)
            out["power_peak_w"] = max(self.power)
        if self.mem_used:
            out["system_mem_peak_gib"] = max(self.mem_used)
        return out


# --- step timing ------------------------------------------------------------------------------------


class StepTimer(TrainerCallback):
    def __init__(self, args, sampler, give):
        self.args = args
        self.sampler = sampler
        self.give = give
        self.tokens_per_step = args.batch_size * args.gradient_accumulation * args.seq_length
        self.step_times, self.losses = [], []
        self.tokens_seen0 = None
        self._t = None
        self.window_tokens = None

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()

    def on_step_begin(self, args, state, control, **kw):
        self._sync()
        if state.global_step == self.args.warmup_steps:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.tokens_seen0 = state.num_input_tokens_seen
            self.sampler.start()
        self._t = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        self._sync()
        dt = time.perf_counter() - self._t
        if state.global_step > self.args.warmup_steps:  # global_step was incremented before this callback
            self.step_times.append(dt)
            self.give(task="train", rate=self.tokens_per_step / dt, units="tok/s")
            if state.global_step == self.args.warmup_steps + self.args.steps:
                self.sampler.stop()
                if self.tokens_seen0 is not None and state.num_input_tokens_seen:
                    self.window_tokens = state.num_input_tokens_seen - self.tokens_seen0

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


# --- data -------------------------------------------------------------------------------------------

ROLES = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def to_messages(row):
    turns = row.get("conversations") or row.get("messages") or []
    out = []
    for t in turns:
        role = ROLES.get(t.get("from") or t.get("role"))
        content = t.get("value") if "value" in t else t.get("content")
        if role and content:
            out.append({"role": role, "content": content})
    return out


def render(tokenizer, messages):
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            # templates without a system role / strict alternation: fold the system prompt into the first turn
            if messages and messages[0]["role"] == "system" and len(messages) > 1:
                merged = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
                return tokenizer.apply_chat_template(merged + messages[2:], tokenize=False)
            raise
    # base models without a chat template (Llama-3.2-1B): a plain instruction format
    parts = [f"### {m['role'].capitalize()}:\n{m['content']}" for m in messages]
    return "\n\n".join(parts) + (tokenizer.eos_token or "")


def main():
    parser = ArgumentParser()
    parser.add_arguments(Arguments)
    args, _ = parser.parse_known_args()
    give = push()
    token = os.environ.get("HF_TOKEN")

    device = None
    if torch.cuda.is_available():
        device = f"cuda:{os.environ.get('MILABENCH_GPU_ID', '0')}"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = f"xpu:{os.environ.get('MILABENCH_GPU_ID', '0')}"

    grad_ckpt = {"unsloth": "unsloth", "true": True, "none": False}[args.grad_ckpt]

    loader = args.loader
    if loader == "auto":
        cfg = AutoConfig.from_pretrained(args.model, revision=args.model_revision, token=token, trust_remote_code=False)
        loader = "model" if hasattr(cfg, "text_config") or hasattr(cfg, "vision_config") else "language"

    t_load = time.perf_counter()
    common = dict(
        model_name=args.model,
        revision=args.model_revision,
        max_seq_length=args.seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.bfloat16,
        device_map=device,
        token=token,
        use_gradient_checkpointing=grad_ckpt,
    )
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if loader == "model":
        model, processor = FastModel.from_pretrained(**common, full_finetuning=False)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=grad_ckpt,
            random_state=args.seed,
        )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(**common)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=targets,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=grad_ckpt,
            random_state=args.seed,
        )
    load_seconds = time.perf_counter() - t_load
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights_gib = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else None

    dataset = load_dataset(args.dataset, revision=args.dataset_revision, split=f"train[:{args.num_samples}]")
    dataset = dataset.map(
        lambda row: {"text": render(tokenizer, to_messages(row))},
        remove_columns=dataset.column_names,
        desc="chat template",
    )
    dataset = dataset.filter(lambda row: len(row["text"]) > 0)

    output_dir = args.output or os.path.join(os.environ.get("MILABENCH_BASE", "/tmp/milabench"), "outputs", "llm-finetune")
    sampler = Sampler(gpu_index=int(os.environ.get("MILABENCH_GPU_ID", "0") or 0))
    timer = StepTimer(args, sampler, give)
    total_steps = args.warmup_steps + args.steps

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        callbacks=[timer],
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_steps=total_steps,
            warmup_steps=0,
            learning_rate=args.learning_rate,
            lr_scheduler_type="linear",
            logging_steps=1,
            optim=args.optim,
            weight_decay=0.01,
            bf16=True,
            seed=args.seed,
            data_seed=args.seed,
            report_to=[],
            save_strategy="no",
            output_dir=output_dir,
            include_num_input_tokens_seen="non_padding",
            dataset_num_proc=1,
            dataset_text_field="text",
            max_length=args.seq_length,
            packing=True,
            packing_strategy="wrapped",
        ),
    )

    trainer.train()
    if sampler.t1 is None:  # the run ended early (dataset exhausted): close the window anyway
        sampler.stop()

    steps = len(timer.step_times)
    seconds = sum(timer.step_times)
    nominal_tokens = steps * timer.tokens_per_step
    tokens = timer.window_tokens or nominal_tokens
    tok_s = tokens / seconds if seconds else 0.0

    losses = timer.losses
    k = min(10, max(1, len(losses) // 4))
    loss_start = statistics.fmean(losses[:k]) if losses else float("nan")
    loss_end = statistics.fmean(losses[-k:]) if losses else float("nan")
    loss_drop = (loss_start - loss_end) / loss_start if losses and loss_start else float("nan")
    loss_ok = bool(losses) and all(map(lambda x: x == x and x < 1e4, losses)) and loss_drop >= args.min_loss_drop

    power = sampler.summary()
    summary = {
        "model": args.model,
        "qlora": args.load_in_4bit,
        "loader": loader,
        "seq_length": args.seq_length,
        "sequences_per_step": args.batch_size * args.gradient_accumulation,
        "tokens_per_step": timer.tokens_per_step,
        "lora_r": args.lora_r,
        "trainable_params_m": round(trainable / 1e6, 2),
        "warmup_steps": args.warmup_steps,
        "steps": steps,
        "tokens": tokens,
        "seconds": round(seconds, 3),
        "load_seconds": round(load_seconds, 1),
        "train_tokens_per_s": round(tok_s, 1),
        "step_time_s_p50": round(statistics.median(timer.step_times), 4) if steps else None,
        "step_time_s_p90": round(sorted(timer.step_times)[int(0.9 * (steps - 1))], 4) if steps else None,
        "weights_gib": round(weights_gib, 2) if weights_gib is not None else None,
        "torch_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None,
        "loss_start": round(loss_start, 4),
        "loss_end": round(loss_end, 4),
        "loss_drop": round(loss_drop, 4),
        "loss_ok": loss_ok,
        "losses": [round(x, 4) for x in losses],
        **{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in power.items()},
    }
    if summary.get("energy_j") and tokens:
        summary["joules_per_token"] = round(summary["energy_j"] / tokens, 5)
        summary["kwh_per_1m_tokens"] = round(summary["energy_j"] / tokens * 1e6 / 3.6e6, 4)
        summary["tokens_per_joule"] = round(tokens / summary["energy_j"], 3)
    give(task="train", finetune_summary=summary)
    print(f"finetune_summary: {summary}", flush=True)

    if args.require_loss_drop and not loss_ok:
        print(f"loss check failed: start {loss_start:.4f} -> end {loss_end:.4f} (drop {loss_drop:.2%} < {args.min_loss_drop:.0%})", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
