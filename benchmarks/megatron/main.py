#!/usr/bin/env python
"""GLM-5 training on Megatron-LM's ``pretrain`` with frozen parameters, launched by torchrun (one process per GPU).

The Megatron command line comes from the bench config (``model.argv`` + ``argv``, see benchfile.py). The rest of the
config is read here:

    model.hf_path   local HF dir with config.json (the GLM-5 spec reads the DSA indexer sizes from it)
    frozen:         which parameters do not train, see ``freeze_patterns``
    data:           {docs_per_sample: N, samples: M, subset: S}, FineWeb text (benchmate.fineweb, staged by prepare.py),
                    tokenized with the model's HF tokenizer and packed in N documents per sample (THD layout, as the
                    DSA attention needs ``packed_seq_params``)
    placement:      node order / rank order, see placement.py (reported here, applied by benchfile.py)

Throughput is reported as ``rate`` (global tokens / step time, items/s) by benchmate's ManualTimedIterator (CUDA events, rank 0), through setupvoir.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

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

def configure_model(model_cfg, args, spec: dict, hf_path: str):
    """Hook the GLM-5 layer spec and the freezing into the model config.

    With a config container Megatron builds the model through ``model_cfg.get_builder_cls()`` and ignores a
    ``model_provider``: the layer spec is a callable on the config, and the freezing is a pre-wrap hook (it has
    to run before the DDP wrapper, which only allocates gradient buffers for the parameters that require grad)."""
    from slime_plugins.models.glm5.glm5 import get_glm5_spec

    # get_glm5_spec reads the DSA indexer sizes from the HF config at this path
    args.hf_checkpoint = hf_path

    def layer_spec(cfg, vp_stage=None):
        return get_glm5_spec(args, cfg.transformer, vp_stage)

    def freeze_hook(models):
        import torch.distributed as dist

        collect_ep_groups(config())
        PLACEMENT["timing"] = bool((config().get("placement") or {}).get("a2a_timing", True))
        if PLACEMENT["timing"]:
            install_a2a_timer()
        counts = [0, 0]
        for i, chunk in enumerate(models):
            trainable, total = freeze_model(chunk, spec)
            counts[0] += trainable
            counts[1] += total
            debug(f"model chunk {i}: {trainable:,} trainable / {total:,} parameters on this rank")
        # fraction over all the ranks' parameters: the replicated (non-expert) ones count once per data-parallel rank,
        # so this is a weighted figure, close to the real fraction of the model that trains
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, tuple(counts))
        trainable, total = (sum(c[k] for c in gathered) for k in (0, 1))
        if LOG is not None and total:
            fraction = trainable / total
            debug(f"params_trainable_fraction: {fraction:.4f}")
            LOG({"params_trainable_fraction": fraction, "units": "ratio", "task": "main", "time": time.time()})
        return models

    model_cfg.transformer_layer_spec = layer_spec
    model_cfg.pre_wrap_hooks.append(freeze_hook)
    return model_cfg


# --- data --------------------------------------------------------------------------------------------------------

def fineweb_dir(data: dict) -> Path:
    return Path(os.environ["MILABENCH_DIR_DATA"]) / "fineweb" / data.get("subset", "sample-10BT")


class FineWebPackedDataset(torch.utils.data.Dataset):
    """FineWeb text, ``docs`` equal-length slots per sample (THD layout: cu_seqlens).

    Each slot starts at its own FineWeb document, truncated to the slot; a document shorter than the slot is followed
    by the next ones (separated by EOS) so the slot is always full and the tensor shapes stay fixed. Documents wrap
    around when the subset is smaller than what the run consumes."""

    def __init__(self, samples: int, seq_length: int, docs: int, hf_path: str, data: dict):
        assert seq_length % docs == 0, "seq-length must be a multiple of data.docs_per_sample"
        from benchmate.fineweb import load_fineweb
        from transformers import AutoTokenizer

        self.samples, self.seq, self.docs = samples, seq_length, docs
        self.doc_len = seq_length // docs
        self.dataset = load_fineweb(fineweb_dir(data))
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        eos = self.tokenizer.eos_token_id
        self.eos = 0 if eos is None else eos

    def __len__(self):
        return self.samples

    def slot(self, first_doc: int):
        """doc_len tokens starting at document ``first_doc``."""
        tokens, i = [], first_doc
        # ~4 characters per token: tokenize no more text than the slot can hold
        while len(tokens) < self.doc_len + 1:
            text = self.dataset[i % len(self.dataset)]["text"]
            tokens += self.tokenizer(text[: 8 * self.doc_len], add_special_tokens=False)["input_ids"] + [self.eos]
            i += 1
        return tokens[: self.doc_len + 1]

    def __getitem__(self, idx):
        slots = [self.slot(idx * self.docs + j) for j in range(self.docs)]
        # inputs and labels are shifted inside each slot: no label crosses a document boundary
        tokens = torch.tensor([t for s in slots for t in s[:-1]], dtype=torch.int64)
        labels = torch.tensor([t for s in slots for t in s[1:]], dtype=torch.int64)
        cu = torch.arange(0, self.seq + 1, self.doc_len, dtype=torch.int32)
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": torch.ones(self.seq, dtype=torch.float32),
            "position_ids": torch.arange(self.doc_len).repeat(self.docs),
            "cu_seqlens": cu,
            "cu_seqlens_padded": cu.clone(),
            "max_seqlen": torch.tensor(self.doc_len, dtype=torch.int32),
        }


def build_dataset_provider(docs: int, hf_path: str, data: dict):
    from megatron.training import get_args

    def provider(train_val_test_num_samples, vp_stage=None):
        args = get_args()
        train = FineWebPackedDataset(train_val_test_num_samples[0], args.seq_length, docs, hf_path, data)
        return train, None, None

    provider.is_distributed = True
    return provider


def with_wandb(log):
    """Also push the events to wandb (benchmate's ``wandb_push``) when ``BENCHMATE_WANDB=1``.

    ``setupvoir``'s log only writes to milabench's smuggled stdout: benchmate's ``auto_push`` (which adds wandb) is not
    involved, so the opt-in flag would do nothing here. Offline nodes: ``WANDB_MODE=offline``, then ``wandb sync``."""
    from benchmate.metrics import wandb_push
    from benchmate.toggles import wandb_enabled

    if not wandb_enabled():
        return log
    push = wandb_push()

    def both(event):
        log(event)
        push(**event)

    return both


# --- placement at run time -----------------------------------------------------------------------------------------
# The node order only means something if Megatron's expert groups are the consecutive hosts we think they are:
# read the real groups and the real hostnames once the process groups exist, and time the expert all-to-all per group.

LOG = None          # milabench event sink on rank 0 (setupvoir), None elsewhere
STARTED = time.time()

PLACEMENT = {"group_of_rank": {}, "groups": [], "crosses": [], "events": [], "ep_group": None, "timing": False}


def short(host: str) -> str:
    return host.split(".")[0]


def collect_ep_groups(cfg: dict):
    """Gather (rank, hostname, expert group ranks) from every rank; rank 0 logs the groups and how many cross a switch."""
    import socket

    import torch.distributed as dist
    from megatron.core import parallel_state

    import placement

    group = parallel_state.get_expert_model_parallel_group()
    PLACEMENT["ep_group"] = group
    info = (dist.get_rank(), socket.gethostname(), tuple(dist.get_process_group_ranks(group)))
    infos = [None] * dist.get_world_size()
    dist.all_gather_object(infos, info)

    groups = sorted({ranks for _, _, ranks in infos})
    PLACEMENT["groups"] = groups
    PLACEMENT["group_of_rank"] = {r: i for i, ranks in enumerate(groups) for r in ranks}
    if dist.get_rank() != 0:
        return

    host_of_rank = {r: short(h) for r, h, _ in infos}
    nodes = (cfg.get("system") or {}).get("nodes") or []
    switch_of = {short(n): s for n, s in placement.switch_map(nodes, cfg.get("placement") or {}).items()} if nodes else {}
    node_hosts = list(dict.fromkeys(host_of_rank[r] for r in sorted(host_of_rank)))  # hosts in rank order
    debug("hosts in rank order:", node_hosts)
    expected = [short(n["name"]) for n in nodes]
    if expected and expected != node_hosts:
        debug("WARNING: the hosts run in another order than the placement (system.nodes):", expected)

    crosses = []
    for i, ranks in enumerate(groups):
        hs = list(dict.fromkeys(host_of_rank[r] for r in ranks))
        sw = {switch_of.get(h, "unknown") for h in hs}
        crosses.append(len(sw) > 1)
        debug(f"ep group {i:>3}: ranks {ranks[0]}..{ranks[-1]} ({len(ranks)}), hosts {hs}, switches {sorted(sw)}"
              f"{'  CROSSES' if len(sw) > 1 else ''}")
    PLACEMENT["crosses"] = crosses
    debug(f"expert groups crossing a switch: {sum(crosses)} / {len(groups)}")


def install_a2a_timer():
    """Time every all-to-all on the expert group (forward, backward and recompute) with CUDA events.

    An async all-to-all (``async_op``, used with a separate NCCL stream) is timed up to its ``wait()``."""
    import torch.distributed as dist

    original = dist.all_to_all_single

    class TimedHandle:
        def __init__(self, handle, start):
            self.handle, self.start = handle, start

        def wait(self, *args, **kwargs):
            result = self.handle.wait(*args, **kwargs)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            PLACEMENT["events"].append((self.start, end))
            return result

        def __getattr__(self, name):
            return getattr(self.handle, name)

    def timed(*args, **kwargs):
        group = kwargs.get("group")
        if group is None or group is not PLACEMENT["ep_group"]:
            return original(*args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original(*args, **kwargs)
        if kwargs.get("async_op"):
            return TimedHandle(result, start)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        PLACEMENT["events"].append((start, end))
        return result

    dist.all_to_all_single = timed


def report_a2a(log):
    """After a step: every rank's all-to-all time, averaged per expert group (a collective: all ranks call it)."""
    import torch.distributed as dist

    events, PLACEMENT["events"] = PLACEMENT["events"], []
    ms = sum(s.elapsed_time(e) for s, e in events) if events else 0.0
    mine = torch.tensor([ms], device="cuda")
    gathered = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, mine)
    if log is None or not PLACEMENT["groups"]:
        return
    now = time.time()
    times = [g.item() for g in gathered]
    per_group = [sum(times[r] for r in ranks) / len(ranks) for ranks in PLACEMENT["groups"]]
    for i, value in enumerate(per_group):
        log({"a2a_time": value, "units": "ms", "ep_group": i, "crosses_switch": PLACEMENT["crosses"][i],
             "task": "train", "time": now})
    for label, flag in (("crossing", True), ("aligned", False)):
        values = [v for v, c in zip(per_group, PLACEMENT["crosses"]) if c == flag]
        if values:
            log({f"a2a_time_{label}": sum(values) / len(values), "units": "ms", "task": "train", "time": now})


# --- metrics -----------------------------------------------------------------------------------------------------

def patch_train_step(log):
    import megatron.training.training as training
    from megatron.training import get_args

    import step_metrics

    steps = step_metrics.install(log)
    original = training.train_step
    state = {"step": 0}

    def timed(*args, **kwargs):
        # rate / laps / clock: StepTimer (benchmate TimedIterator). Here only what it does not know about.
        result = original(*args, **kwargs)
        state["step"] += 1
        margs = get_args()
        steps.step(margs.global_batch_size * margs.seq_length)
        if state["step"] == 1 and log is not None:
            # launch -> end of the first (warm-up) step: imports, init, model build, first step
            log({"time_to_first_step": time.time() - STARTED, "units": "s", "task": "main", "time": time.time()})
        if PLACEMENT["ep_group"] is not None and PLACEMENT["timing"]:
            report_a2a(log)
        if log is not None:
            loss_dict = result[0] if isinstance(result, tuple) else {}
            if isinstance(loss_dict, dict) and "lm loss" in loss_dict:
                log({"loss": float(loss_dict["lm loss"]), "task": "train", "time": time.time()})
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
        sizes = dict(
            tp=flag(argv, "--tensor-model-parallel-size"),
            pp=flag(argv, "--pipeline-model-parallel-size"),
            cp=flag(argv, "--context-parallel-size"),
            ep=flag(argv, "--expert-model-parallel-size"),
            etp=flag(argv, "--expert-tensor-model-parallel-size") if "--expert-tensor-model-parallel-size" in argv else None,
        )
        rep = placement.describe(nodes, place, **sizes)
        layout = placement.format_layout(nodes, place, **sizes)
    except Exception as exc:  # the report is informative, never fatal
        debug("placement report failed:", exc)
        return
    debug("node order:", [n["name"] for n in nodes])
    debug("placement\n" + placement.format_report(rep))
    debug("layout (node, switch, GPU -> rank, pipeline stage / data / expert index)\n" + layout)


def main():
    global LOG
    cfg = config()
    # Megatron parses sys.argv: nothing of ours must be left in it
    argv = sys.argv[1:]

    log = monitor = None
    if rank() == 0:
        from benchmate.monitor import setupvoir

        log, monitor = setupvoir(monogpu=False, interval=0.5)
        log = with_wandb(log)
        LOG = log
        report_placement(cfg, argv)

    spec = freeze_patterns(cfg.get("frozen"))
    debug("frozen:", spec)
    hf_path = str(cfg["model"]["hf_path"]).replace("{milabench_data}", os.environ["MILABENCH_DIR_DATA"])
    data = cfg.get("data") or {}
    docs = int(data.get("docs_per_sample", 8))

    from megatron.core.enums import ModelType
    from megatron.training import pretrain
    from megatron.training.argument_utils import gpt_config_from_args, pretrain_cfg_container_from_args
    from megatron.training.arguments import parse_and_validate_args
    from pretrain_gpt import forward_step, get_embedding_ranks

    args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
    model_cfg = configure_model(gpt_config_from_args(args), args, spec, hf_path)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)

    patch_train_step(log)
    try:
        pretrain(
            full_config,
            build_dataset_provider(docs, hf_path, data),
            ModelType.encoder_or_decoder,
            forward_step,
            get_embedding_ranks=get_embedding_ranks,
        )
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
