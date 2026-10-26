#!/usr/bin/env python
"""GLM-5 training on Megatron-LM's ``pretrain`` with frozen parameters, launched by torchrun (one process per GPU).

The Megatron command line comes from the bench config (``model.argv`` + ``argv``, see benchfile.py). The rest of the
config is read here:

    model.hf_path   local HF dir with config.json (the GLM-5 spec reads the DSA indexer sizes from it)
    frozen:         which parameters do not train, see ``freeze_patterns``
    data:           {docs_per_sample: N}, random tokens packed in N documents per sample (THD layout, as the DSA
                    attention needs ``packed_seq_params``)
    placement:      node order / rank order, see placement.py (reported here, applied by benchfile.py)

Throughput is reported as ``rate`` (global tokens / step time) from the first rank, through setupvoir.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import partial

for _p in reversed(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
    if _p:
        sys.path.insert(0, _p)

import torch


def config() -> dict:
    raw = os.environ.get("MILABENCH_CONFIG")
    return json.loads(raw) if raw else {}


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def debug(*msg):
    if rank() == 0:
        print("[megatron-glm5]", *msg, file=sys.stderr, flush=True)


# --- frozen parameters -------------------------------------------------------------------------------------------

# slime's `--freeze-params-name-list experts` is a regex search on the parameter name: it also matches shared_experts
PRESETS = {
    "none": {},
    "experts": {"freeze": ["experts"]},
    # train the attention (MLA + indexer) only
    "attention-only": {"only_train": ["self_attention"]},
}
INDEXER = ("wq_b", "wk", "k_norm", "weights_proj")


def freeze_patterns(frozen: dict | None) -> dict:
    frozen = dict(frozen or {})
    spec = dict(PRESETS[frozen.get("preset", "none")])
    spec["freeze"] = [*spec.get("freeze", []), *frozen.get("freeze", [])]
    spec["only_train"] = [*spec.get("only_train", []), *frozen.get("only_train", [])]
    spec["indexer"] = bool(frozen.get("indexer", False))
    return spec


def freeze_model(model, spec: dict):
    if spec["only_train"]:
        for name, param in model.named_parameters():
            param.requires_grad = any(re.search(p, name) for p in spec["only_train"])
    for name, param in model.named_parameters():
        if any(re.search(p, name) for p in spec["freeze"]):
            param.requires_grad = False
        if spec["indexer"] and ".self_attention." in f".{name}." and name.split(".")[-2] in INDEXER:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --- model -------------------------------------------------------------------------------------------------------

def build_model_provider(spec: dict, hf_path: str):
    import gpt_builders
    from megatron.training.arguments import core_transformer_config_from_args
    from model_provider import model_provider

    from slime_plugins.models.glm5.glm5 import get_glm5_spec

    counts = [0, 0]

    def builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
        if config is None:
            config = core_transformer_config_from_args(args)
        # gpt_builder asks args.spec (an import path) for the layer spec: serve the GLM-5 one, which needs
        # (args, config, vp_stage) and the HF config for the indexer sizes
        args.hf_checkpoint = hf_path
        original = gpt_builders.import_module
        gpt_builders.import_module = lambda _spec: get_glm5_spec(args, config, vp_stage)
        previous, args.spec = args.spec, ["glm5", "get_glm5_spec"]
        try:
            model = gpt_builders.gpt_builder(args, pre_process, post_process, vp_stage, config, pg_collection)
        finally:
            gpt_builders.import_module = original
            args.spec = previous
        trainable, total = freeze_model(model, spec)
        counts[0] += trainable
        counts[1] += total
        debug(f"model chunk (vp_stage={vp_stage}): {trainable:,} trainable / {total:,} parameters "
              f"on this rank (running total {counts[0]:,} / {counts[1]:,})")
        return model

    return partial(model_provider, builder)


# --- data --------------------------------------------------------------------------------------------------------

class RandomPackedDataset(torch.utils.data.Dataset):
    """Random tokens, ``docs`` equal-length documents per sample (THD layout: cu_seqlens)."""

    def __init__(self, samples: int, seq_length: int, vocab_size: int, docs: int):
        assert seq_length % docs == 0, "seq-length must be a multiple of data.docs_per_sample"
        self.samples, self.seq, self.vocab, self.docs = samples, seq_length, vocab_size, docs

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(idx)
        tokens = torch.randint(0, self.vocab, (self.seq + 1,), generator=gen, dtype=torch.int64)
        doc_len = self.seq // self.docs
        cu = torch.arange(0, self.seq + 1, doc_len, dtype=torch.int32)
        return {
            "tokens": tokens[:-1],
            "labels": tokens[1:],
            "loss_mask": torch.ones(self.seq, dtype=torch.float32),
            "position_ids": torch.arange(doc_len).repeat(self.docs),
            "cu_seqlens": cu,
            "cu_seqlens_padded": cu.clone(),
            "max_seqlen": torch.tensor(doc_len, dtype=torch.int32),
        }


def build_dataset_provider(docs: int):
    from megatron.training import get_args

    def provider(train_val_test_num_samples, vp_stage=None):
        args = get_args()
        train = RandomPackedDataset(train_val_test_num_samples[0], args.seq_length, args.padded_vocab_size, docs)
        return train, None, None

    provider.is_distributed = True
    return provider


# --- metrics -----------------------------------------------------------------------------------------------------

def patch_train_step(log):
    import megatron.training.training as training
    from megatron.training import get_args

    original = training.train_step
    state = {"step": 0}

    def timed(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        state["step"] += 1
        if log is not None:
            margs = get_args()
            tokens = margs.global_batch_size * margs.seq_length
            now = time.time()
            log({"rate": tokens / elapsed, "units": "tokens/s", "task": "train", "time": now})
            log({"step_time": elapsed, "units": "s", "task": "train", "time": now})
            loss_dict = result[0] if isinstance(result, tuple) else {}
            if isinstance(loss_dict, dict) and "lm loss" in loss_dict:
                log({"loss": float(loss_dict["lm loss"]), "task": "train", "time": now})
        return result

    training.train_step = timed


# --- placement report --------------------------------------------------------------------------------------------

def flag(argv, name, default=1):
    return int(argv[argv.index(name) + 1]) if name in argv else default


def report_placement(cfg: dict, argv: list[str]):
    import placement

    nodes = (cfg.get("system") or {}).get("nodes") or []
    place = cfg.get("placement") or {}
    if len(nodes) < 1:
        return
    try:
        rep = placement.describe(
            nodes, place,
            tp=flag(argv, "--tensor-model-parallel-size"),
            pp=flag(argv, "--pipeline-model-parallel-size"),
            cp=flag(argv, "--context-parallel-size"),
            ep=flag(argv, "--expert-model-parallel-size"),
            etp=flag(argv, "--expert-tensor-model-parallel-size") if "--expert-tensor-model-parallel-size" in argv else None,
        )
    except Exception as exc:  # the report is informative, never fatal
        debug("placement report failed:", exc)
        return
    debug("node order:", [n["name"] for n in nodes])
    debug("placement\n" + placement.format_report(rep))


def main():
    cfg = config()
    # Megatron parses sys.argv: nothing of ours must be left in it
    argv = sys.argv[1:]

    log = monitor = None
    if rank() == 0:
        from benchmate.monitor import setupvoir

        log, monitor = setupvoir(monogpu=False, interval=0.5)
        report_placement(cfg, argv)

    spec = freeze_patterns(cfg.get("frozen"))
    debug("frozen:", spec)
    hf_path = str(cfg["model"]["hf_path"]).replace("{milabench_data}", os.environ["MILABENCH_DIR_DATA"])
    docs = int((cfg.get("data") or {}).get("docs_per_sample", 8))

    from megatron.core.enums import ModelType
    from megatron.training import pretrain
    from megatron.training.argument_utils import gpt_config_from_args, pretrain_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args
    from pretrain_gpt import forward_step, get_embedding_ranks

    args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
    full_config = pretrain_cfg_container_from_args(args, gpt_config_from_args(args))

    patch_train_step(log)
    try:
        pretrain(
            full_config,
            build_dataset_provider(docs),
            ModelType.encoder_or_decoder,
            forward_step,
            model_provider=build_model_provider(spec, hf_path),
            get_embedding_ranks=get_embedding_ranks,
        )
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
