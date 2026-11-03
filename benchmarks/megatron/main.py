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
    checkpoint:     {dir, op: save | load, content: weights | full, mode: sync | async, step}: turned into Megatron's
                    checkpoint flags before they are parsed, timed and reported as ``checkpoint`` events (see
                    ``checkpoint_argv`` / ``patch_checkpointing``)

Throughput is reported as ``rate`` (global tokens / step time, items/s) by benchmate's ManualTimedIterator (CUDA events, rank 0), through setupvoir.
"""

from __future__ import annotations

import json
import math
import os
import re
import socket
import statistics
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

def index_share() -> bool:
    """model.index_share (default true): GLM-5's DSA cross-layer top-k sharing (index_topk_freq=4, offset 3 in the HF
    config). With it, a pipeline stage can only start on a computing layer (1, 2, 3, 7, 11, ..., 75), which rules out
    the PP25 / PP16 window layouts. false: every layer computes its own top-k (plain DSA), any layout works."""
    return bool((config().get("model") or {}).get("index_share", True))


def no_index_share(auto_config):
    """``AutoConfig`` stand-in for the glm5 plugin: the HF config with index sharing turned off."""

    class NoIndexShare:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            hf_config = auto_config.from_pretrained(*args, **kwargs)
            hf_config.index_topk_freq, hf_config.index_skip_topk_offset = 1, 0
            return hf_config

    return NoIndexShare


def configure_model(model_cfg, args, spec: dict, hf_path: str):
    """Hook the GLM-5 layer spec and the freezing into the model config.

    With a config container Megatron builds the model through ``model_cfg.get_builder_cls()`` and ignores a
    ``model_provider``: the layer spec is a callable on the config, and the freezing is a pre-wrap hook (it has
    to run before the DDP wrapper, which only allocates gradient buffers for the parameters that require grad)."""
    # model.spec: default -> Megatron's own TE layer spec (small stand-in models for smoke tests, no GLM-5 plugin)
    glm5 = (config().get("model") or {}).get("spec", "glm5") == "glm5"
    if glm5:
        from slime_plugins.models.glm5 import glm5 as glm5_plugin
        from slime_plugins.models.glm5.glm5 import get_glm5_spec

        # get_glm5_spec reads the DSA indexer sizes from the HF config at this path
        args.hf_checkpoint = hf_path
        if not index_share():
            glm5_plugin.AutoConfig = no_index_share(glm5_plugin.AutoConfig)

    def layer_spec(cfg, vp_stage=None):
        return get_glm5_spec(args, cfg.transformer, vp_stage)

    def freeze_hook(models):
        import torch.distributed as dist

        collect_ep_groups(config())
        PLACEMENT["timing"] = bool((config().get("placement") or {}).get("a2a_timing", True))
        PLACEMENT["every"] = max(int((config().get("placement") or {}).get("a2a_report_every", 10)), 1)
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
        fraction = trainable / total if total else None
        RUN["trainable_fraction"], RUN["model_initialized"] = fraction, time.time()
        if LOG is not None and total:
            debug(f"params_trainable_fraction: {fraction:.4f}")
            LOG({"params_trainable_fraction": fraction, "units": "ratio", "task": "main", "time": time.time()})
        if LOG is not None:
            LOG({"run_start": run_start(args, fraction), "task": "main", "time": time.time()})
        return models

    if glm5:
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


def start_nodemon(cfg: dict):
    """Start the node sampler beside the training processes, when ``BENCHMATE_NODEMON_DIR`` is set.

    Every rank calls this: the first on each host takes the lock and samples, the
    others return at once. It runs before Megatron is imported, so model init and
    the first steps are covered too."""
    from benchmate.nodemon import run
    from benchmate.toggles import nodemon_dir

    out = nodemon_dir()
    if out is None:
        return None
    sampler = run(out, bench=cfg.get("name", "megatron"))
    debug("nodemon:", sampler.state().value, out)
    return sampler


# --- placement at run time -----------------------------------------------------------------------------------------
# The node order only means something if Megatron's expert groups are the consecutive hosts we think they are:
# read the real groups and the real hostnames once the process groups exist, and time the expert all-to-all per group.

LOG = None          # milabench event sink on rank 0 (setupvoir), None elsewhere
STARTED = time.time()

PLACEMENT = {"group_of_rank": {}, "groups": [], "crosses": [], "events": [], "ep_group": None, "timing": False,
             "every": 10, "steps": 0, "a2a_ms": [], "a2a_steps": 0}
# for the run summary: when each phase ended on this rank (epoch seconds)
RUN = {"trainable_fraction": None, "model_initialized": None, "first_step_done": None}
WARMUP_STEPS = 1  # left out of the summary's step statistics (NCCL communicators, allocator growth)
# the loss per step, kept by one rank of the last pipeline stage (``loss_reporter``): (step, loss tensor, grad norm, time)
LOSS = {"reporter": False, "steps": []}


def loss_reporter() -> bool:
    """Only the last pipeline stage gets the loss from ``train_step`` (averaged over data parallel there): its first
    data-parallel / tensor / context rank keeps it, rank 0 reports it at the end (``push_summary``)."""
    from megatron.core import parallel_state as ps

    return (ps.is_pipeline_last_stage(ignore_virtual=True) and ps.get_data_parallel_rank() == 0
            and ps.get_tensor_model_parallel_rank() == 0 and ps.get_context_parallel_rank() == 0)


def short(host: str) -> str:
    return host.split(".")[0]


def gpu_uuid():
    """UUID of this rank's GPU (what the node monitor identifies GPUs by), None if unavailable."""
    try:
        return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    except Exception:
        return None


def collect_ep_groups(cfg: dict):
    """Gather (rank, hostname, expert group ranks) from every rank; rank 0 logs the groups and how many cross a switch."""
    import socket

    import torch.distributed as dist
    from megatron.core import parallel_state

    import placement

    group = parallel_state.get_expert_model_parallel_group()
    PLACEMENT["ep_group"] = group
    info = (dist.get_rank(), socket.gethostname(), tuple(dist.get_process_group_ranks(group)), gpu_uuid())
    infos = [None] * dist.get_world_size()
    dist.all_gather_object(infos, info)

    groups = sorted({ranks for _, _, ranks, _ in infos})
    PLACEMENT["groups"] = groups
    PLACEMENT["group_of_rank"] = {r: i for i, ranks in enumerate(groups) for r in ranks}
    if dist.get_rank() != 0:
        return

    host_of_rank = {r: short(h) for r, h, _, _ in infos}
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

    # recorded in the run data with run_start (scripts/analysis/FORMAT.md, `placement`): the crossing count must not depend
    # on a2a_timing being on. crossing / groups / gpu_uuids / mode are additions to the spec's fields.
    place = cfg.get("placement") or {}
    mode = place.get("nodes", "as-is") if isinstance(place.get("nodes", "as-is"), str) else "list"
    leaves = {}
    for h in node_hosts:
        leaves.setdefault(switch_of.get(h, "unknown"), []).append(h)
    snapshot = os.environ.get("TT_TOPOLOGY")
    PLACEMENT["record"] = {
        "hosts": node_hosts,
        "family": os.environ.get("TT_FAMILY") or FAMILY.get(mode),
        "ep_groups": [list(dict.fromkeys(host_of_rank[r] for r in ranks)) for ranks in groups],
        "leaves": leaves,
        "excluded_nodes": json.loads(os.environ.get("TT_EXCLUDED") or "{}"),
        "health_check": os.environ.get("TT_HEALTH_DIR") or None,
        "topology_snapshot_epoch": int(os.path.getmtime(snapshot)) if snapshot and os.path.isfile(snapshot) else None,
        "crossing": sum(crosses),
        "groups": len(groups),
        "gpu_uuids": {r: u for r, _, _, u in sorted(infos)},
        "mode": mode,
    }


# placement.nodes -> FORMAT.md's placement.family, when the launcher does not set TT_FAMILY
FAMILY = {"aligned": "EP", "crossed": "EP", "mixed": "EP", "leaf": "ALL", "topology": "ALL", "as-is": "slurm"}


PRECISIONS = {"tensorwise": "fp8-tensorwise", "blockwise": "fp8-blockwise", "mxfp8": "mxfp8", "delayed": "fp8-delayed"}


def precision(args) -> tuple[str, dict | None]:
    """(workload.precision, workload.precision_detail): bf16 | fp8-tensorwise | fp8-blockwise | mxfp8, and how it runs."""
    if not getattr(args, "fp8", None):
        return "bf16", None
    recipe = str(getattr(args, "fp8_recipe", None) or "delayed").split(".")[-1]
    try:
        import transformer_engine

        te = transformer_engine.__version__
    except Exception:
        te = None
    major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (None, None)
    return PRECISIONS.get(recipe, f"fp8-{recipe}"), {
        "format": args.fp8, "recipe": recipe, "fp8_param_gather": bool(getattr(args, "fp8_param_gather", False)),
        "moe_router_padding_for_quantization": bool(getattr(args, "moe_router_padding_for_quantization", False)),
        "block_scaling_fp32_scales": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES") == "1",
        "compute_capability": f"sm_{major}{minor}" if major is not None else None, "transformer_engine": te,
        "scope": "TransformerEngine linear / grouped GEMMs; the GLM-5 plugin's DSA indexer and sparse MLA stay bf16",
        "master_weights": "fp32 (bf16 weights, quantized copies per step)",
    }


def check_precision(args):
    """MXFP8 needs Blackwell (sm_100+): stop every rank before the model is built rather than deep inside TE."""
    if getattr(args, "fp8", None) and str(getattr(args, "fp8_recipe", "")).split(".")[-1] == "mxfp8":
        major, minor = torch.cuda.get_device_capability()
        if major < 10:
            raise SystemExit(f"[megatron-glm5] --fp8-recipe mxfp8 needs a Blackwell GPU (sm_100+), this one is "
                             f"sm_{major}{minor}: use tensorwise or blockwise here")


def pp_split(args) -> list[int] | str:
    """Layers per pipeline stage (first to last), or the explicit --pipeline-model-parallel-layout."""
    if getattr(args, "pipeline_model_parallel_layout", None):
        return str(args.pipeline_model_parallel_layout)
    pp, layers = args.pipeline_model_parallel_size, args.num_layers
    first = getattr(args, "decoder_first_pipeline_num_layers", None)
    last = getattr(args, "decoder_last_pipeline_num_layers", None)
    middle = pp - (first is not None) - (last is not None)
    rest = layers - (first or 0) - (last or 0)
    per = rest // middle if middle > 0 else 0
    return [*([first] if first is not None else []), *([per] * middle), *([last] if last is not None else [])][:pp]


def layout_record(args) -> dict:
    """``parallelism`` additions (scripts/trillium/FORMAT.md): the layout name, model copies (PP x EP x ETP GPUs each =
    expert data parallel), layers per stage (``pp_split``, an addition) and micro-batches per pipeline and step."""
    tp, pp, ep = args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.expert_model_parallel_size
    etp = getattr(args, "expert_tensor_parallel_size", None) or tp
    dp = getattr(args, "data_parallel_size", 1) or 1
    return {
        "layout": f"pp{pp}-ep{ep}" + (f"-tp{tp}" if tp > 1 else ""),
        "pp_schedule": "1F1B" if pp > 1 else None,  # no virtual pipeline stages
        "expert_dp": args.world_size // (pp * ep * etp),
        "copies": args.world_size // (pp * ep * etp),
        "pp_split": pp_split(args),
        "microbatches_per_pipeline": args.global_batch_size // (args.micro_batch_size * dp),
    }


def run_start(args, trainable_fraction):
    """``data: {run_start: {run, placement, workload, parallelism}}`` (scripts/analysis/FORMAT.md, section 1): once, after init, so a
    run that fails later keeps its identity. The launcher's TT_* variables fill run.* and placement.family."""
    env = os.environ.get
    run = {
        "cluster": env("CC_CLUSTER"),
        "campaign": env("TT_CAMPAIGN"),
        "variant": env("TT_VARIANT"),
        "experiment": env("TT_EXPERIMENT"),
        "repetition": int(env("TT_REPETITION")) if env("TT_REPETITION") else None,
        "slurm_job_id": env("SLURM_JOB_ID"),
        "slurm_step_id": env("SLURM_STEP_ID"),
        "slurm_reservation": env("SLURM_JOB_RESERVATION"),
        "exclusive_cluster": env("TT_EXCLUSIVE") == "1",
    }

    def dtype(name):
        value = getattr(args, name, None)
        return str(value).replace("torch.", "") if value is not None else None

    workload = {}
    try:
        cfg = config()
        hf_path = str(cfg["model"]["hf_path"]).replace("{milabench_data}", os.environ["MILABENCH_DIR_DATA"])
        try:
            with open(Path(hf_path) / "config.json") as f:
                layers_full = json.load(f).get("num_hidden_layers")
        except OSError:
            layers_full = None
        try:
            import subprocess

            megatron = Path(__file__).parent / "Megatron-LM"
            commit = subprocess.run(["git", "-C", str(megatron), "rev-parse", "HEAD"], capture_output=True, text=True,
                                    timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            commit = None
        dp = getattr(args, "data_parallel_size", 1) or 1
        grad_accum = args.global_batch_size // (args.micro_batch_size * dp)
        precise = getattr(args, "use_precision_aware_optimizer", False)
        workload = {
            "model": Path(cfg["model"].get("hf", "GLM-5")).name, "framework": "megatron", "framework_commit": commit,
            "torch_version": torch.__version__,
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())) if torch.cuda.is_available() else None,
            "n_layers": args.num_layers, "n_layers_full": layers_full,
            "params_trainable_fraction": trainable_fraction,
            "params_total": None, "params_active": None,
            "seq_len": args.seq_length, "microbatch_sequences": args.micro_batch_size, "grad_accum": grad_accum,
            "tokens_per_gpu_per_step": args.micro_batch_size * args.seq_length * grad_accum,
            "global_batch_tokens": args.global_batch_size * args.seq_length,
            "precision": precision(args)[0], "precision_detail": precision(args)[1],
            # the gradient buffers' dtype is the one reduced across data parallel
            "grad_reduce_dtype": "bf16" if getattr(args, "grad_reduce_in_bf16", False)
                                 or "bfloat16" in str(getattr(args, "main_grads_dtype", "")) else "fp32",
            "seed": getattr(args, "seed", None),
            "optimizer": "AdamW" if args.optimizer == "adam" and args.weight_decay > 0 else args.optimizer,
            "optimizer_state_dtype": dtype("exp_avg_dtype") if precise else "fp32",
            "optimizer_offload_fraction": getattr(args, "optimizer_offload_fraction", 0.0)
                                          if getattr(args, "optimizer_cpu_offload", False) else 0.0,
            "activation_checkpointing": args.recompute_granularity or "none", "compile": "none",
            "moe": {
                "n_experts": args.num_experts, "top_k": args.moe_router_topk,
                "n_shared_experts": 1 if getattr(args, "moe_shared_expert_intermediate_size", None) else 0,
                "ep": args.expert_model_parallel_size,
                "routing": "aux-loss-free" if getattr(args, "moe_router_enable_expert_bias", False)
                           else args.moe_router_load_balancing_type,
                "capacity_factor": getattr(args, "moe_expert_capacity_factor", None),
                "dispatcher": args.moe_token_dispatcher_type,
            } if getattr(args, "num_experts", None) else None,
        }
    except Exception as exc:  # the record is informative, never fatal
        debug("run_start workload failed:", exc)
    try:
        parallelism = layout_record(args)
    except Exception as exc:
        debug("run_start parallelism failed:", exc)
        parallelism = {}
    return {"run": run, "placement": PLACEMENT.get("record"), "workload": workload, "parallelism": parallelism}


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
    """Every rank's all-to-all time per step, averaged per expert group (a collective: all ranks call it).

    Covers the steps since the last report (the events accumulate), so the figure is a per-step mean. The warm-up step is
    dropped by the caller: it includes NCCL's lazy setup of the expert communicator."""
    import torch.distributed as dist

    events, PLACEMENT["events"] = PLACEMENT["events"], []
    steps, PLACEMENT["steps"] = max(PLACEMENT["steps"], 1), 0
    ms = sum(s.elapsed_time(e) for s, e in events) / steps if events else 0.0
    mine = torch.tensor([ms], device="cuda")
    gathered = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, mine)
    if log is None or not PLACEMENT["groups"]:
        return
    now = time.time()
    times = [g.item() for g in gathered]
    per_group = [sum(times[r] for r in ranks) / len(ranks) for ranks in PLACEMENT["groups"]]
    if len(PLACEMENT["a2a_ms"]) != len(per_group):
        PLACEMENT["a2a_ms"] = [0.0] * len(per_group)
    PLACEMENT["a2a_ms"] = [total + value * steps for total, value in zip(PLACEMENT["a2a_ms"], per_group)]
    PLACEMENT["a2a_steps"] += steps
    for i, value in enumerate(per_group):
        log({"a2a_time": value, "units": "ms", "ep_group": i, "crosses_switch": PLACEMENT["crosses"][i],
             "task": "train", "time": now})
    for label, flag in (("crossing", True), ("aligned", False)):
        values = [v for v, c in zip(per_group, PLACEMENT["crosses"]) if c == flag]
        if values:
            log({f"a2a_time_{label}": sum(values) / len(values), "units": "ms", "task": "train", "time": now})


# --- metrics -----------------------------------------------------------------------------------------------------

def report_peak_memory(log):
    """Peak GPU memory of this rank after the first step. ``log`` is every rank's own sink (benchmate.monitor.metric_log):
    rank 0 -> milabench, the other ranks -> their rankstore file (merged afterwards by ``LocalMetricStore.read``).

    Per rank, with its pipeline stage: the last stage (FP32 logits) is the one most likely to run out of memory."""
    from megatron.core import parallel_state

    gib = 2**30
    allocated = torch.cuda.max_memory_allocated() / gib
    reserved = torch.cuda.max_memory_reserved() / gib
    stage = parallel_state.get_pipeline_model_parallel_rank()
    log({"peak_memory_allocated": allocated, "peak_memory_reserved": reserved, "pp_stage": stage, "units": "GiB",
         "task": "main", "time": time.time()})


def patch_straggler(rank_log, state, steps):
    """Per-rank straggler data, with ``--log-straggler``. Megatron's detector gathers the ranks' numbers on rank 0 and only
    prints the min / max rank; every rank computes its own in ``elapsed()`` (GPU time of the model's GEMM kernels, the
    slowest ``get_batch``, GPU temperature / power / utilization / clock), so each rank logs them there: no extra
    collective, and the rank files keep all the ranks instead of only the extremes."""
    from megatron.core.utils import StragglerDetector
    from megatron.training import get_args

    original = StragglerDetector.elapsed

    def elapsed(self):
        result = original(self)
        if not self._off:
            gemm, data, temp, power, util, clock = result
            interval = max(get_args().log_interval, 1)
            rtt = gemm / interval
            flops = getattr(self, "_mb_flops", 0.0)  # of the interval, given to report()
            # Etpt, as the detector computes it: TFLOP/s of this rank, from the forward only, hence / amp
            etpt = flops / interval / (rtt * 10**9 * self.world) / self.amp if rtt > 0 else 0.0
            rank_log({
                "straggler_gemm_ms": rtt,  # Rtt: per step, over the steps since the last report
                "straggler_etpt": etpt,
                "straggler_data_ms": data,  # slowest get_batch of the interval
                "straggler_temp": temp, 
                "straggler_power": power, 
                "straggler_util": util, 
                "straggler_clock": clock,
                "step": state["step"], "task": "train", "time": time.time(),
            })
        return result

    report = StragglerDetector.report

    def timed_report(self, total_flops=0.0, *args, **kwargs):
        self._mb_flops = total_flops
        # the report syncs with the GPU and gathers on rank 0, after the step: keep it out of the next step's time
        result = report(self, total_flops, *args, **kwargs)
        steps.restart_clock()
        return result

    StragglerDetector.elapsed = elapsed
    StragglerDetector.report = timed_report


def patch_train_step(log):
    import megatron.training.training as training
    from megatron.training import get_args

    import step_metrics
    from benchmate.monitor import metric_log

    steps = step_metrics.install(log)
    rank_log = log if log is not None else metric_log()  # rank 0: milabench (+ wandb), other ranks: rankstore (+ wandb)
    if log is not None:
        from benchmate.metrics import wandb_push
        from benchmate.rankstore import add_sink
        from benchmate.toggles import wandb_all_ranks, wandb_enabled

        # rank 0's step rows go to its rankstore file, not through ``log``: mirror them to its wandb run as metric_log
        # does for the other ranks (wandb_push returns the process's one run)
        if wandb_enabled() and wandb_all_ranks():
            add_sink(wandb_push())
    original = training.train_step
    state = {"step": 0}
    patch_straggler(rank_log, state, steps)

    def timed(*args, **kwargs):
        # rate / laps / clock: StepTimer (benchmate TimedIterator). Here only what it does not know about.
        if state["step"] == 0:
            LOSS["reporter"] = loss_reporter()
        steps.rows.begin()
        steps.rows.time_optimizer(args[3] if len(args) > 3 else kwargs.get("optimizer"))
        result = original(*args, **kwargs)
        state["step"] += 1
        margs = get_args()
        if LOSS["reporter"]:
            # the last stage's loss (already averaged over data parallel by train_step), kept on the device: read once
            # the step's events have completed (rank rows) and at the end (summary), no extra GPU sync here
            loss_dict = result[0] if isinstance(result, tuple) else {}
            if isinstance(loss_dict, dict) and "lm loss" in loss_dict:
                grad_norm = result[5] if isinstance(result, tuple) and len(result) > 5 else None
                loss = loss_dict["lm loss"].detach()
                LOSS["steps"].append((margs.curr_iteration + 1, loss, grad_norm, time.time()))
                steps.rows.set_loss(loss, grad_norm)
        # every rank of a data-parallel replica runs the replica's tokens (micro-batch x sequence x grad accumulation)
        steps.rows.end(state["step"], margs.global_batch_size * margs.seq_length // (getattr(margs, "data_parallel_size", 1) or 1))
        steps.step(margs.global_batch_size * margs.seq_length)
        if state["step"] == 1 and log is not None:
            # launch -> end of the first (warm-up) step: imports, init, model build, first step
            log({"time_to_first_step": time.time() - STARTED, "units": "s", "task": "main", "time": time.time()})
        if state["step"] == 1:
            RUN["first_step_done"] = time.time()
            report_peak_memory(rank_log)  # every rank
        if PLACEMENT["ep_group"] is not None and PLACEMENT["timing"]:
            if state["step"] == 1:
                PLACEMENT["events"] = []  # warm-up: NCCL creates the expert communicator at its first collective
            else:
                PLACEMENT["steps"] += 1
            # every N steps, and at the last step so the end of the run is counted too
            if state["step"] > 1 and (state["step"] % PLACEMENT["every"] == 0 or state["step"] == margs.train_iters):
                # the report syncs with the GPU and runs a collective: keep it out of the measured step time
                report_a2a(log)
                steps.restart_clock()
        return result

    training.train_step = timed
    return steps


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))] if values else None


def rank_summary(rows):
    """This rank's part of the run summary, from its step rows (warm-up left out)."""
    from megatron.core import parallel_state

    measured = [r for r in rows.written if r["batch_id"] > WARMUP_STEPS]
    steps = [r["elapsed"] for r in measured]
    waits = [r["next"] for r in measured if "next" in r]

    def median(key):
        values = [r[key] for r in measured if key in r]
        return round(statistics.median(values), 3) if values else None

    return {
        "rank": int(os.environ.get("RANK", 0)),
        "host": short(socket.gethostname()),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "gpu_uuid": gpu_uuid(),
        "pp_stage": parallel_state.get_pipeline_model_parallel_rank(),
        "step_median": round(statistics.median(steps), 5) if steps else None,
        "step_p90": round(percentile(steps, 0.9), 5) if steps else None,
        "step_max": round(max(steps), 5) if steps else None,
        "fwd_ms_median": median("fwd_ms"),
        "bwd_ms_median": median("bwd_ms"),
        "fwd_bwd_ms_median": median("fwd_bwd_ms"),
        "optimizer_ms_median": median("optimizer_ms"),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "data_wait_sum_s": round(sum(waits), 6) if waits else None,
        "data_wait_max_s": round(max(waits), 6) if waits else None,
        # not in per_rank: startup and data loading are summarized across the ranks
        "_process_start": STARTED,
        "_model_initialized": RUN["model_initialized"],
        "_first_step_done": RUN["first_step_done"],
        "_data_wait_median": statistics.median(waits) if waits else None,
        "_measured": (measured[0]["time"], measured[-1]["time"], len(measured)) if measured else None,
        # the loss reporter's steps (end of the run: the device is done, reading the tensors costs nothing)
        "_losses": [(step, float(loss), None if grad is None else float(grad), t) for step, loss, grad, t in LOSS["steps"]],
    }


# dense BF16 spec per GPU (MFU in BF16 units for every recipe, scripts/trillium/FORMAT.md)
PEAK_FLOPS = {"B200": 2.25e15, "H100": 989e12}


def peak_flops():
    name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    return next((v for k, v in PEAK_FLOPS.items() if k in name), None)


def mfu_inputs(args):
    try:
        from megatron.training.training import num_floating_point_operations

        tokens = args.global_batch_size * args.seq_length
        flops = num_floating_point_operations(args, args.global_batch_size) / tokens
    except Exception as exc:
        debug("flops per token failed:", exc)
        flops = None
    return {
        "flops_per_token": flops,
        "peak_flops_per_gpu": peak_flops(),
        "formula": (
            "Megatron's num_floating_point_operations (megatron/training/training.py) for one global batch / "
            "global batch tokens. It assumes every weight gets its gradient: frozen parameters skip theirs, so it "
            "overestimates the FLOPs of frozen runs (params_trainable_fraction < 1). "
            "mfu = flops_per_token * tokens_per_s / (peak * gpus)"
        ),
    }


def push_summary(log, steps):
    """``data: {summary: {...}}`` (scripts/analysis/FORMAT.md, ``milabench.pretrain.v1``), rank 0, after training.
    A collective: every rank calls it, once ``pretrain`` has returned (process groups still up)."""
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.training import get_args

    args = get_args()  # Megatron's own, with the derived sizes
    try:
        local = rank_summary(steps.rows)
    except Exception as exc:  # every rank must reach the gather
        debug("rank summary failed:", exc)
        local = {"rank": int(os.environ.get("RANK", 0))}
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, local)
    if log is None:
        return

    def spread(key):
        values = [r[key] for r in ranks if r.get(key) is not None]
        return {"min": min(values), "max": max(values)} if values else None

    t0 = min((r["_process_start"] for r in ranks if r.get("_process_start")), default=STARTED)

    def since_start(key):
        values = [r[key] - t0 for r in ranks if r.get(key) is not None]
        return {"min_s": min(values), "max_s": max(values)} if values else None

    info = run_start(args, RUN["trainable_fraction"])
    hosts = list(dict.fromkeys(r["host"] for r in sorted(ranks, key=lambda r: r["rank"]) if r.get("host")))
    dp = ps.get_data_parallel_world_size()
    waits = [r["_data_wait_median"] for r in ranks if r.get("_data_wait_median") is not None]
    window = local.get("_measured")
    a2a_steps = PLACEMENT["a2a_steps"]
    summary = {
        "schema": "milabench.pretrain.v1",
        "run": info["run"],
        "workload": info["workload"],
        "parallelism": {
            "gpus": dist.get_world_size(),
            "nodes": len(hosts),
            # Megatron's data parallelism replicates the weights; the distributed optimizer shards its states only
            "dp_replicate": dp,
            "dp_shard": 1,
            "distributed_optimizer": bool(getattr(args, "use_distributed_optimizer", False)),
            "expert_dp": ps.get_expert_data_parallel_world_size(),
            "tp": ps.get_tensor_model_parallel_world_size(),
            "pp": ps.get_pipeline_model_parallel_world_size(),
            "vpp": getattr(args, "virtual_pipeline_model_parallel_size", None),
            "cp": ps.get_context_parallel_world_size(),
            "ep": ps.get_expert_model_parallel_world_size(),
            **info["parallelism"],
        },
        "placement": info["placement"] or {"hosts": hosts},
        "measurement": {
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": window[2] if window else 0,
            "measure_seconds_budget": None,
            "measurement_start_epoch": RUN["first_step_done"],
            "measurement_end_epoch": window[1] if window else None,
            "rate_definition": (
                "per step: global tokens / step time; step time from device events on rank 0 (benchmate "
                "ManualTimedIterator); official value = median of the per-step rates"
            ),
        },
        "mfu_inputs": mfu_inputs(args),
        "memory_gib": {"peak_allocated": spread("peak_alloc_gib"), "peak_reserved": spread("peak_reserved_gib")},
        "startup": {
            "process_start_epoch": spread("_process_start"),
            "model_initialized": since_start("_model_initialized"),
            "first_step_done": since_start("_first_step_done"),
            "measurement_start": since_start("_first_step_done"),
            "compile_cache": "none",
            "note": "seconds since the earliest rank's process start (main.py import)",
        },
        "data_loading": {
            "median_s_per_step": max(waits) if waits else None,
            "max_s_per_step": max((r["data_wait_max_s"] for r in ranks if r.get("data_wait_max_s") is not None),
                                  default=None),
            "definition": (
                "host time blocked in the train loader's next, measured steps; median per rank, worst rank "
                "reported. Only the first and last pipeline stages have a loader"
            ),
        },
        "per_rank": [{k: v for k, v in r.items() if not k.startswith("_")} for r in sorted(ranks, key=lambda r: r["rank"])],
        "per_ep_group": [
            {"hosts": hosts_of_group, "a2a_s_per_step": total / a2a_steps / 1000, "crosses_switch": crosses}
            for hosts_of_group, total, crosses in zip(
                (info["placement"] or {}).get("ep_groups") or [], PLACEMENT["a2a_ms"], PLACEMENT["crosses"])
        ] if a2a_steps else None,
        "rank_metrics": {
            "dir": os.path.dirname(steps.rows.path) if steps.rows.path else None,
            "files": "metrics.<bench>.<host>.<rank>.jsonl (benchmate rankstore)",
            "rows": "worker and config (benchmate.rankmetrics.v1) headers, then one row per step on every rank "
                    "(batch_id, elapsed, wall, next, rate, local_tokens, fwd_ms / bwd_ms or fwd_bwd_ms, optimizer_ms, "
                    "mem_*_gib)",
        },
        "deviations": [] if index_share() else [
            {"id": "dsa_index_share", "text": "DSA index sharing off: every layer computes its own top-k"}],
        "estimates": [],
    }
    # the loss per step, from the last pipeline stage (one rank keeps it, see loss_reporter)
    for step, loss, grad_norm, t in sorted(item for r in ranks for item in r.get("_losses") or []):
        log({"loss": loss, "grad_norm": grad_norm, "step": step, "task": "train", "time": t})
    log({"summary": summary, "task": "train", "time": time.time()})


# --- checkpointing (Trillium F1) ----------------------------------------------------------------------------------
# ``checkpoint: {dir, op: save | load, content: weights | full, mode: sync | async, step}``. One save per run (at
# ``step``, default the last one), timed on rank 0 with barriers around it: ``blocked_s`` is the time training is
# stopped, ``total_s`` the time until the write is complete (async: when Megatron finalizes it). A checksum of the
# model weights is written beside the checkpoint and compared after the load: slime's Megatron patch lets a load
# skip missing tensors silently.

CKPT = {"cfg": None, "saved": False}
CHECKSUM_FILE = "milabench_checksum.json"


def drop(argv: list[str], name: str, value: bool = False) -> list[str]:
    """``argv`` without ``name`` (and its value)."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == name:
            skip = value
            continue
        out.append(a)
    return out


LIST_FLAGS = ("--recompute-modules",)


def split_lists(argv: list[str]) -> list[str]:
    """The config writes a list as one comma-separated value (milabench's assemble_options): Megatron's list flags
    (nargs) want one token per item."""
    out = []
    for i, a in enumerate(argv):
        out.extend(a.split(",") if i > 0 and argv[i - 1] in LIST_FLAGS else [a])
    return out


def checkpoint_argv(cfg: dict, argv: list[str]) -> list[str]:
    """The Megatron flags of ``checkpoint:`` (added before the arguments are parsed)."""
    ck = cfg.get("checkpoint")
    if not ck:
        return argv
    ck = {"content": "weights", "mode": "sync", **ck}
    ck["dir"] = os.path.expandvars(str(ck["dir"]))
    if "$" in ck["dir"]:
        raise SystemExit(f"[megatron-glm5] checkpoint.dir {ck['dir']}: environment variable not set")
    assert ck["op"] in ("save", "load") and ck["content"] in ("weights", "full") and ck["mode"] in ("sync", "async"), ck
    for name in ("--save", "--load", "--save-interval", "--ckpt-format"):
        argv = drop(argv, name, value=True)
    argv = [*argv, "--ckpt-format", "torch_dist"]
    if ck["op"] == "save":
        step = int(ck.get("step") or flag(argv, "--train-iters"))
        ck["step"] = step
        argv += ["--save", ck["dir"], "--save-interval", str(step)]
        if ck["content"] == "weights":
            argv += [f for f in ("--no-save-optim", "--no-save-rng") if f not in argv]
        else:
            argv = drop(drop(argv, "--no-save-optim"), "--no-save-rng")
        if ck["mode"] == "async":
            # without the persistent worker Megatron turns async off (a warning only); nvrx (the default) is not installed
            argv += ["--async-save", "--use-persistent-ckpt-worker", "--async-strategy", "mcore"]
    else:
        argv += ["--load", ck["dir"], "--ckpt-fully-parallel-load", "--exit-on-missing-checkpoint", "--no-load-rng"]
        if ck["content"] == "weights":
            argv += ["--finetune", "--no-load-optim"]
        else:
            # a full load resumes at the saved iteration: --train-iters counts the steps to run after it
            try:
                with open(os.path.join(ck["dir"], "latest_checkpointed_iteration.txt")) as f:
                    ck["step"] = int(f.read().strip())
            except (OSError, ValueError):
                ck["step"] = 0
            iters = flag(argv, "--train-iters")
            argv = [*drop(argv, "--train-iters", value=True), "--train-iters", str(ck["step"] + iters)]
    CKPT["cfg"] = ck
    return argv


def fs_type(path: str) -> str | None:
    """Filesystem type of ``path`` (longest mount point of /proc/mounts that contains it)."""
    best, kind = "", None
    real = os.path.realpath(path)
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                mount = parts[1]
                if (real == mount or real.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
                    best, kind = mount, parts[2]
    except OSError:
        pass
    return kind


def dir_usage(path: str) -> tuple[int, int]:
    """(bytes, files) under ``path``."""
    size = files = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                size += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return size, files


def weights_checksum(model) -> list[float]:
    """[sum, sum of squares] of this rank's model weights, in float64, in slices (no full-size copy)."""
    total = torch.zeros(2, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        for chunk in model:
            for _, p in sorted(chunk.named_parameters(), key=lambda kv: kv[0]):
                for part in p.detach().reshape(-1).split(2**26):
                    part = part.double()
                    total[0] += part.sum()
                    total[1] += (part * part).sum()
    return total.tolist()


def gather_checksums(model) -> list:
    import torch.distributed as dist

    mine = weights_checksum(model)
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, mine)
    return every


def checkpoint_event(ck: dict, start: float, blocked_end: float, end: float, step: int, checksum_match=None,
                     saved_by=None):
    """``data: {checkpoint: {...}}`` on rank 0 (scripts/trillium/FORMAT.md, "Checkpoint events")."""
    if LOG is None:
        return
    size, files = dir_usage(ck["dir"])
    # times on rank 0 between barriers: the slowest rank's
    event = {
        "op": ck["op"], "step": step,
        "repetition": int(os.environ["TT_REPETITION"]) if os.environ.get("TT_REPETITION") else None,
        "mode": ck["mode"] if ck["op"] == "save" else "sync", "content": ck["content"], "format": "torch_dist",
        "bytes": size, "files": files, "blocked_s": blocked_end - start, "total_s": end - start,
        "path": ck["dir"], "filesystem": fs_type(ck["dir"]), "start_epoch": start, "end_epoch": end,
        "checksum_match": checksum_match,
        "saved_by": saved_by,  # load: {mode, run} of the save that wrote the checkpoint
    }
    debug("checkpoint:", json.dumps(event))
    LOG({"checkpoint": event, "task": "train", "time": time.time()})


def patch_checkpointing(steps):
    """Time the save (``save_checkpoint_and_time``, async finalize) and the load (``load_checkpoint``)."""
    import torch.distributed as dist
    import megatron.training.checkpointing as checkpointing
    import megatron.training.training as training

    ck = CKPT["cfg"]
    if ck is None:
        return

    if ck["op"] == "save":
        save = training.save_checkpoint_and_time
        schedule = checkpointing.schedule_async_save
        pending = {}

        def finished(step, start, blocked_end, checksums):
            if dist.get_rank() == 0:
                with open(os.path.join(ck["dir"], CHECKSUM_FILE), "w") as f:
                    json.dump({"step": step, "content": ck["content"], "mode": ck["mode"],
                               "run": os.environ.get("MILABENCH_RUN_NAME"), "ranks": checksums}, f)
            checkpoint_event(ck, start, blocked_end, time.time(), step)

        def timed_schedule(request):
            # the last finalize function: runs on every rank once all of them have written (maybe_finalize_async_save)
            # (pending is read then: blocked_end is only known once the save call has returned)
            request.add_finalize_fn(lambda: finished(pending["step"], pending["start"], pending["blocked_end"],
                                                     pending["checksums"]))
            return schedule(request)

        def timed_save(iteration, model, *args, **kwargs):
            if CKPT["saved"]:  # one checkpoint per run: Megatron's end-of-training save is skipped
                debug(f"checkpoint already saved: no save at iteration {iteration}")
                return None
            CKPT["saved"] = True
            checksums = gather_checksums(model)
            dist.barrier()
            start = time.time()
            pending.update(step=iteration, start=start, checksums=checksums)
            result = save(iteration, model, *args, **kwargs)
            dist.barrier()
            blocked_end = time.time()
            pending["blocked_end"] = blocked_end
            if ck["mode"] == "sync":
                finished(iteration, start, blocked_end, checksums)
            steps.restart_clock()
            return result

        checkpointing.schedule_async_save = timed_schedule
        training.save_checkpoint_and_time = timed_save
    else:
        load = training.load_checkpoint

        def timed_load(model, *args, **kwargs):
            dist.barrier()
            start = time.time()
            result = load(model, *args, **kwargs)
            dist.barrier()
            end = time.time()
            checksums = gather_checksums(model)
            match, step, saved_by = None, ck.get("step"), None
            try:
                with open(os.path.join(ck["dir"], CHECKSUM_FILE)) as f:
                    saved = json.load(f)
                match, step = saved["ranks"] == checksums, saved["step"]
                saved_by = {"mode": saved.get("mode"), "run": saved.get("run")}
            except OSError:
                debug(f"no {CHECKSUM_FILE} in {ck['dir']}: the load cannot be checked")
            checkpoint_event(ck, start, end, end, step, match, saved_by)
            if match is False:
                raise RuntimeError("the loaded weights differ from the saved ones (checksum): a partial load?")
            return result

        training.load_checkpoint = timed_load


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
    sampler = start_nodemon(cfg)
    # Megatron parses sys.argv: nothing of ours must be left in it
    argv = sys.argv[1:]

    # one node monitor per node (local rank 0): rank 0's goes to milabench, the others' to their rankstore file
    # (benchmate.monitor.metric_log). Only rank 0 reports the run itself (steps, loss, placement).
    log = monitor = None
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        from benchmate.monitor import setupvoir

        node_log, monitor = setupvoir(monogpu=False, interval=0.5)
        if rank() == 0:
            log = with_wandb(node_log)
            LOG = log
            report_placement(cfg, argv)

    gpus = (cfg.get("placement") or {}).get("gpus_per_node")
    if gpus and os.environ.get("LOCAL_WORLD_SIZE") and int(os.environ["LOCAL_WORLD_SIZE"]) != int(gpus):
        # the layout check and the placement report assume gpus_per_node processes per node
        debug(f"WARNING: {os.environ['LOCAL_WORLD_SIZE']} processes on this node, placement.gpus_per_node is {gpus}")

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

    sys.argv = [sys.argv[0], *checkpoint_argv(cfg, split_lists(argv))]
    args = parse_and_validate_args(args_defaults={"tokenizer_type": "NullTokenizer"})
    check_precision(args)
    model_cfg = configure_model(gpt_config_from_args(args), args, spec, hf_path)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)

    steps = patch_train_step(log)
    patch_checkpointing(steps)
    try:
        pretrain(
            full_config,
            build_dataset_provider(docs, hf_path, data),
            ModelType.encoder_or_decoder,
            forward_step,
            get_embedding_ranks=get_embedding_ranks,
        )
        steps.flush()
        push_summary(log, steps)
    finally:
        steps.flush()
        if sampler is not None:
            sampler.stop("bench done")
        if monitor is not None:
            monitor.stop()


def kill_everywhere():
    """An exception on one rank leaves the others blocked in a receive until the job's time limit (seen with the first
    forward failing on one stage): stop the benchmark on every node of the allocation, the job goes on to its next entry."""
    import subprocess

    nodes = os.environ.get("SLURM_JOB_NUM_NODES")
    if nodes and int(nodes) > 1:
        subprocess.run(["srun", "--overlap", f"--nodes={nodes}", f"--ntasks={nodes}", "--ntasks-per-node=1", "pkill", "-KILL",
                        "-f", "megatron/main.py"], timeout=120, check=False)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stderr.flush()
        kill_everywhere()
        raise
