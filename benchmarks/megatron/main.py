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
    # model.spec: default -> Megatron's own TE layer spec (small stand-in models for smoke tests, no GLM-5 plugin)
    glm5 = (config().get("model") or {}).get("spec", "glm5") == "glm5"
    if glm5:
        from slime_plugins.models.glm5.glm5 import get_glm5_spec

        # get_glm5_spec reads the DSA indexer sizes from the HF config at this path
        args.hf_checkpoint = hf_path

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
             "every": 10, "steps": 0}


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


def run_start(args, trainable_fraction):
    """``data: {run_start: {run, placement, workload}}`` (scripts/analysis/FORMAT.md, section 1): once, after init, so a
    run that fails later keeps its identity. The launcher's TT_* variables fill run.* and placement.family."""
    env = os.environ.get
    run = {
        "cluster": env("CC_CLUSTER"),
        "campaign": env("TT_CAMPAIGN"),
        "variant": env("TT_VARIANT"),
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
            "precision": "fp8" if getattr(args, "fp8", None) else "bf16",
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
    return {"run": run, "placement": PLACEMENT.get("record"), "workload": workload}


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
    original = training.train_step
    state = {"step": 0}
    patch_straggler(rank_log, state, steps)

    def timed(*args, **kwargs):
        # rate / laps / clock: StepTimer (benchmate TimedIterator). Here only what it does not know about.
        result = original(*args, **kwargs)
        state["step"] += 1
        margs = get_args()
        steps.step(margs.global_batch_size * margs.seq_length)
        if state["step"] == 1 and log is not None:
            # launch -> end of the first (warm-up) step: imports, init, model build, first step
            log({"time_to_first_step": time.time() - STARTED, "units": "s", "task": "main", "time": time.time()})
        if state["step"] == 1:
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
        if log is not None:
            loss_dict = result[0] if isinstance(result, tuple) else {}
            if isinstance(loss_dict, dict) and "lm loss" in loss_dict:
                log({"loss": float(loss_dict["lm loss"]), "task": "train", "time": time.time()})
        return result

    training.train_step = timed
    return steps


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

    steps = patch_train_step(log)
    try:
        pretrain(
            full_config,
            build_dataset_provider(docs, hf_path, data),
            ModelType.encoder_or_decoder,
            forward_step,
            get_embedding_ranks=get_embedding_ranks,
        )
    finally:
        steps.flush()
        if sampler is not None:
            sampler.stop("bench done")
        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
