"""Inference servers (OpenAI-compatible API) run by milabench next to a client workload.

Each backend is its own command; ``InferenceServer`` picks one from the configured backend, and
``deploy()`` lays it out on the run's nodes::

    server = InferenceServer(server_pack, *server_argv, backend="atom")
    service, stop = server.deploy(multinode="dp")
    Service(service, SequenceCommand(eval_client, perf_client), ready=server.ready, stop=stop)

``Service`` waits for ``ready()`` before starting the clients and stops the server once they are done.
Everything backend specific (launcher, flags, environment, multi-node layout) stays in the server command.
"""

from __future__ import annotations

import asyncio
import copy
import os
import threading
import re
import json
import glob
import urllib.request
from typing import List

from ..utils import select_nodes
from . import ListCommand, SingleCmdCommand, CmdCommand, clone_with, count_command_errors, max_node_count, node_address
from .executors import get_or_create_warden


class FirstExit(ListCommand):
    """Run the commands together; done as soon as one of them exits.

    A multi-node server is down once any node's server is: when a worker node's engine dies, the coordinator
    stays up but can no longer answer, and a plain ``ListCommand`` would wait for it forever (the clients with
    it). ``Service`` stops the workload when this returns and kills what is left of the service."""

    async def execute(
        self, phase="run", timeout=False, timeout_delay=600,
        warden=None, resource_cleaner=True, with_gpu_warden=True, **kwargs
    ):
        run_kwargs = {**self._kwargs, **kwargs, "phase": phase, "timeout": timeout, "timeout_delay": timeout_delay}
        with get_or_create_warden(warden, with_gpu_warden=with_gpu_warden, resource_cleaner=resource_cleaner) as warden:
            tasks = [asyncio.create_task(executor.execute(**run_kwargs, warden=warden)) for executor in self.executors]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        # still running: their processes are killed by the caller (Service); keep the tasks referenced until then
        self._pending = pending
        return sum(count_command_errors(task.result() if not task.exception() else task.exception()) for task in done)


class ServerCommand(SingleCmdCommand):
    """Base inference server: ``launcher() + filter_argv(server_argv)``, run from the pack's venv.

    Arguments:
        pack: pack the server runs on (tag it ``nolog``: it is stopped by a signal)
        *server_argv: backend arguments (model, parallelism, ...)
        host/port: where the clients reach the server, for ``ready()``
    """

    name = None

    def __init__(self, pack, *server_argv, host="127.0.0.1", port=8000, **kwargs) -> None:
        # setsid: the server spawns engine/worker processes, they are stopped as a group
        super().__init__(pack, **{"setsid": True, **kwargs})
        self.server_argv = list(server_argv)
        self.host = host
        self.port = port
        self.nnodes = 1
        self.ray_head = None

    @property
    def arch(self) -> str:
        return self.pack.config.get("system", {}).get("arch", "cuda")

    def model_dirs(self) -> List[str]:
        """The HF cache folders of the models this server loads (the model, its drafter)."""
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        models = [m for m in (_server_model(self.server_argv),) if m]
        spec = _flag([str(a) for a in self.server_argv], "--speculative-config")
        if spec:
            try:
                models.append(json.loads(spec).get("model"))
            except (json.JSONDecodeError, AttributeError):
                pass
        if draft := _flag([str(a) for a in self.server_argv], "--speculative-draft-model-path"):
            models.append(draft)
        return [os.path.join(hub, "models--" + m.replace("/", "--")) for m in models if m]

    async def execute(self, *args, **kwargs):
        evict_page_cache()
        # while it loads and runs, keep its own weights out of the page cache (unified memory: the cache of
        # a 60 GB checkpoint just read is what the driver's next allocations fail on)
        stop = threading.Event()
        dirs = ":".join(self.model_dirs())
        if os.environ.get("MILABENCH_EVICT_PAGE_CACHE") and dirs:
            def keep_evicting():
                # its own weights every 5 s; every 60 s all of MILABENCH_EVICT_PAGE_CACHE (datasets, the other
                # models' files the client or a previous step read: ~23 GiB of /data was cached on a GB10)
                tick = 0
                while not stop.wait(5):
                    tick += 1
                    evict_page_cache(None if tick % 12 == 0 else dirs)
            threading.Thread(target=keep_evicting, daemon=True).start()
        try:
            return await super().execute(*args, **kwargs)
        finally:
            stop.set()

    @property
    def python(self) -> str:
        return f"{self.pack.dirs.venv}/bin/python"

    def launcher(self) -> List:
        raise NotImplementedError()

    def filter_argv(self, argv) -> List:
        """Drop the arguments this backend does not accept"""
        return argv

    def env(self) -> dict:
        """Backend specific environment; the benchmark's ``env`` config takes precedence"""
        env = {}
        if self.arch == "cuda":
            cache = self.pack.dirs.cache
            env["FLASHINFER_CACHE_DIR"] = str(cache / "flashinfer")
            env["FLASHINFER_CUBIN_DIR"] = str(cache / "flashinfer" / "cubins")
        return env

    def dp_flags(self, nnodes, dp_local, master_ip, rank=0, master_port=29550) -> List:
        """Flags of the server owning DP ranks ``rank .. rank + dp_local`` of a multi-node data parallel group.

        ``rank == 0`` is the coordinator, which serves the HTTP endpoint; the others only run engines.
        """
        raise NotImplementedError(f"multi-node data parallel is not supported by {self.name}")

    def data_parallel(self, nnodes, dp_local, master_ip, rank=0) -> ServerCommand:
        """Copy of this server joining a multi-node data parallel group"""
        server = copy.copy(self)
        server._kwargs = dict(self._kwargs)
        server.server_argv = [*self.server_argv, *self.dp_flags(nnodes, dp_local, master_ip, rank=rank)]
        server.nnodes = nnodes
        return server

    def nodes(self) -> List[str]:
        """Addresses of the run's nodes, the main node first"""
        from .srun import _main_node

        config = self.pack.config
        main = _main_node(config)
        nodes = select_nodes(config["system"]["nodes"], max_node_count(config))
        others = [n for n in nodes if n is not main and node_address(n) != node_address(main)]
        return [node_address(main)] + [node_address(n) for n in others]

    def deploy(self, multinode=None, dp_local=1):
        """``(service, stop)`` running this server on the run's nodes (``stop`` may be None).

        ``multinode="dp"``: one server per node, joined by data parallelism (see ``data_parallel_group``).
        Backends can support other layouts (vLLM: one engine over the nodes with Ray).
        """
        nodes = self.nodes()
        if len(nodes) <= 1:
            return self, None
        if multinode == "dp":
            return self.data_parallel_group(nodes, dp_local)
        raise NotImplementedError(f"{self.name}: no multi-node layout {multinode!r}")

    def data_parallel_group(self, nodes, dp_local=1):
        """Coordinator on the main node, ``benchmate.server_worker`` running an engine-only server on the others.

        The worker servers never exit on their own, and the ssh fallback of ``milabench slurm srun`` does not
        forward signals: the returned stop command kills them by pidfile."""
        from .srun import SrunExceptMain

        coordinator = self.data_parallel(len(nodes), dp_local, nodes[0])
        # server_worker fills in the node's rank
        engine = self.data_parallel(len(nodes), dp_local, nodes[0], rank="@DP_RANK@")

        config = self.pack.config
        pack = self.pack.copy(clone_with(config, {"tag": [*config["tag"], "dpworker"]}))
        pidfile = f"{self.pack.dirs.runs}/server-worker.{config['name']}.pid"
        options = engine.options
        kwargs = {"env": options["env"], "cwd": options.get("cwd")}
        worker = CmdCommand(
            pack, self.python, "-m", "benchmate.server_worker",
            "--nodes", ",".join(nodes),
            "--dp-local", str(dp_local),
            "--pidfile", pidfile,
            "--", *engine.argv(),
            **kwargs,
        )
        stop = CmdCommand(pack, self.python, "-m", "benchmate.server_worker", "--stop", "--pidfile", pidfile, **kwargs)
        return FirstExit(coordinator, SrunExceptMain(worker)), SrunExceptMain(stop)

    def extra_argv(self) -> List:
        """Flags this backend adds to the configured ones (memory sizing)"""
        return []

    def _argv(self, **kwargs) -> List:
        return [*self.launcher(), *cap_memory_fraction(self.filter_argv(self.server_argv)), *self.extra_argv()]

    @property
    def options(self):
        options = super().options
        return {
            **options,
            "env": {**self.env(), **options.get("env", {}), **self.pack.config.get("env", {})},
        }

    def ready_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1/models"

    async def ready(self) -> bool:
        return await asyncio.to_thread(_http_ok, self.ready_url())


def evict_page_cache(paths: str | None = None) -> int:
    """Drop the page cache of the files under ``MILABENCH_EVICT_PAGE_CACHE`` (``:``-separated folders),
    before a server starts; returns the bytes advised away.

    On unified memory the page cache of the previous models' weights (tens of GB) is memory CUDA reports as
    used: vLLM refuses to start when its ``--gpu-memory-utilization`` share is above the free memory, and
    every model a run loads adds its weights to the cache. Dropping the whole cache needs root
    (``/proc/sys/vm/drop_caches``); advising the kernel the files' pages are not needed
    (``posix_fadvise(DONTNEED)``) does not, and only the model caches matter."""
    paths = paths if paths is not None else os.environ.get("MILABENCH_EVICT_PAGE_CACHE", "")
    advised = 0
    for root in filter(None, paths.split(":")):
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                try:
                    if os.path.islink(path):
                        continue
                    fd = os.open(path, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    advised += os.fstat(fd).st_size
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return advised


# --- KV cache sized for the benchmark (unified memory) ---------------------------------------------------
#
# vLLM reserves --gpu-memory-utilization of the device whatever the model: an 8B model takes ~90 GiB of a
# 120 GiB DGX Spark, KV cache for a million tokens it never uses. On unified memory that is the machine's
# memory, and the NVIDIA driver's own allocations then fail (NV_ERR_NO_MEMORY) and the machine freezes.
# With MILABENCH_KV_CACHE_FIT set, vLLM gets --kv-cache-memory-bytes: the most tokens the benchmark's passes
# keep in flight, times the model's KV bytes per token (from its config.json), with a margin.

QUALITYBENCH_CONCURRENCY = 16  # qualitybench run --concurrency default
KV_MARGIN = 1.5
RUNTIME_OVERHEAD = 12 * 2**30
KV_FLOOR = 8 * 2**30  # hybrid models' recurrent state per sequence, drafters, the allocator's blocks
# KV cache when the weights already fill the memory cap (75 GB models under 0.65 of a 120 GiB Spark): one
# sequence's attention KV is not enough, vLLM also counts the linear-attention / Mamba state and its blocks
KV_MIN = 2 * 2**30


def _merged_section(config: dict, name: str) -> dict:
    arch = config.get("system", {}).get("arch", "cuda")
    shared = dict(config.get(name) or {})
    variant = dict(((config.get("variants") or {}).get(arch) or {}).get(name) or {})
    merged = {**shared, **variant}
    merged["argv"] = {**(shared.get("argv") or {}), **(variant.get("argv") or {})}
    return merged


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tokens_in_flight(config: dict, max_model_len: int, max_num_seqs: int) -> int:
    """The most tokens the benchmark's client passes keep in the server's KV cache at once."""
    client = _merged_section(config, "client")
    base = client["argv"]
    passes = client.get("passes") or [{"argv": {}}]
    need = 0
    for spec in passes:
        if "quality" in spec:
            # a QualityBench request: a prompt of a few thousand tokens, 2048 tokens of answer, or up to
            # thinking_max_tokens with thinking on (max_tokens: a floor on either)
            quality = spec.get("quality") or {}
            thinking = quality.get("thinking") or []
            thinking = [thinking] if isinstance(thinking, (str, bool)) else thinking
            answer = int(quality.get("max_tokens") or 2048)
            if any(t in ("on", True) for t in thinking):
                answer = max(answer, int(quality.get("thinking_max_tokens", 32768)))
            need = max(need, min(QUALITYBENCH_CONCURRENCY, max_num_seqs) * min(4096 + answer, max_model_len))
            continue
        if any(k in spec for k in ("eval", "swebench", "terminalbench", "bfcl")):
            # agents (SWE-bench, Terminal-Bench): one request per worker at a time, each up to a full context.
            # Without this they fell to the generic case below (max_num_seqs full contexts): a Terminal-Bench
            # server took the whole memory cap, and its task containers' builds ran the machine out (2026-11-24)
            if "eval" in spec:
                conc = (spec.get("eval") or {}).get("num_concurrent", 64)
            elif "swebench" in spec:
                conc = (spec.get("swebench") or {}).get("workers", 4)
            elif "terminalbench" in spec:
                conc = (spec.get("terminalbench") or {}).get("workers", 4)
            else:
                conc = (spec.get("bfcl") or {}).get("threads", 16)
            need = max(need, min(int(conc), max_num_seqs) * max_model_len)
            continue
        argv = {**base, **(spec.get("argv") or {})}
        conc = min(int(_num(argv.get("--max-concurrency"), max_num_seqs)), max_num_seqs)
        if argv.get("--dataset-name") == "prefix_repetition":
            per_seq = sum(_num(argv.get(k)) for k in ("--prefix-repetition-prefix-len", "--prefix-repetition-suffix-len",
                                                         "--prefix-repetition-output-len"))
        elif argv.get("--random-input-len") or argv.get("--random-output-len"):
            ratio = _num(argv.get("--random-range-ratio"), 0.0)
            per_seq = _num(argv.get("--random-input-len"), 1024) * (1 + ratio) + _num(argv.get("--random-output-len"), 128)
        else:
            per_seq = max_model_len  # datasets of unknown length: a full context each
        need = max(need, conc * min(per_seq, max_model_len))
    return int(min(max(need, max_model_len), max_num_seqs * max_model_len))


def _hf_config(model: str) -> dict | None:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for path in sorted(glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots", "*", "config.json"))):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
    return None


def kv_layout(hf: dict, kv_dtype: str | None) -> dict | None:
    """KV cache bytes per token of a model, from its config.json: ``full`` for the layers that keep the whole
    context, ``sliding`` for those that keep only the last ``window`` tokens (Gemma 3/4, gpt-oss).

    Attention layers only: the linear-attention and Mamba layers of hybrid models keep a fixed state per
    sequence instead. Layers that reuse another layer's KV (Gemma 3n/4 ``num_kv_shared_layers``) have none.
    Gemma 4's full-attention layers have their own head size and KV head count (``global_head_dim``,
    ``num_global_key_value_heads``)."""
    cfg = hf
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(hf.get(key), dict) and hf[key].get("num_hidden_layers"):
            cfg = hf[key]
    layers = cfg.get("num_hidden_layers")
    heads = cfg.get("num_attention_heads")
    if not layers or not heads:
        return None
    elem = 1 if str(kv_dtype or "").startswith("fp8") else 2
    window = cfg.get("sliding_window") or None
    full, sliding = layers, 0
    if isinstance(cfg.get("layer_types"), list):
        kinds = [str(t) for t in cfg["layer_types"]]
        if shared := int(cfg.get("num_kv_shared_layers") or 0):
            kinds = kinds[:-shared]
        attention = [t for t in kinds if "attention" in t and "linear" not in t]
        sliding = sum(1 for t in attention if "sliding" in t) if window else 0
        full = (len(attention) - sliding) if attention else layers
    elif isinstance(cfg.get("hybrid_override_pattern"), str):
        full = cfg["hybrid_override_pattern"].count("*") or layers
    elif cfg.get("full_attention_interval"):
        full = max(1, layers // int(cfg["full_attention_interval"]))
    if cfg.get("kv_lora_rank"):  # MLA: one latent vector per layer
        return {"full": int(full * (cfg["kv_lora_rank"] + cfg.get("qk_rope_head_dim", 0)) * elem),
                "sliding": 0, "window": None}
    kv_heads = cfg.get("num_key_value_heads") or heads
    head_dim = cfg.get("head_dim") or cfg.get("hidden_size", 0) // heads
    if not head_dim:
        return None
    global_heads = cfg.get("num_global_key_value_heads") or kv_heads
    global_dim = cfg.get("global_head_dim") or head_dim
    return {"full": int(2 * full * global_heads * global_dim * elem),
            "sliding": int(2 * sliding * kv_heads * head_dim * elem), "window": window if sliding else None}


def kv_bytes_per_token(hf: dict, kv_dtype: str | None) -> int | None:
    """KV cache bytes per token of a sequence shorter than the sliding window (every attention layer)."""
    layout = kv_layout(hf, kv_dtype)
    return layout and layout["full"] + layout["sliding"]


def kv_bytes(layout: dict, tokens: int, sequences: int) -> int:
    """KV cache of ``tokens`` spread over ``sequences``: sliding-window layers keep at most ``window`` tokens
    of each sequence."""
    kept = min(tokens, sequences * layout["window"]) if layout["window"] else tokens
    return int(tokens * layout["full"] + kept * layout["sliding"])


def _server_model(server_argv: List) -> str | None:
    argv = [str(a) for a in server_argv]
    for name in ("--model", "--model-path"):
        if value := _flag(argv, name):
            return value
    for i, arg in enumerate(argv):
        if arg.startswith("-"):
            continue
        if i and argv[i - 1].startswith("--") and "=" not in argv[i - 1]:
            continue  # a flag's value
        if re.fullmatch(r"[\w.-]+/[\w.-]+", arg):
            return arg
    return None


def weights_bytes(model: str) -> int:
    """The model's weights on disk (top-level safetensors of its snapshot: not original/ or metal/ copies)."""
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snapshots = sorted(glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots", "*")))
    if not snapshots:
        return 0
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(snapshots[-1], "*.safetensors")))


def memory_total() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return 0


def runtime_overhead(weights: int) -> int:
    """What a vLLM server holds beside its weights and KV cache: activations, CUDA graphs, the drafter, the
    multimodal encoder cache, torch's allocator slack. Not proportional to the weights (a big MoE's
    activations are those of its few active experts): ~16 GiB measured on a GB10 next to Gemma-4-31B
    (dense, vision encoder, MTP drafter), 4 GiB was the old guess."""
    return RUNTIME_OVERHEAD


def kv_cache_budget(config: dict, server_argv: List) -> int | None:
    """--kv-cache-memory-bytes for this benchmark, or None to keep vLLM's own sizing (model config unknown).

    What the passes keep in flight (times a margin), within the machine's memory cap (the memory fraction,
    less the weights and the runtime's overhead, see runtime_overhead): a smaller KV cache makes vLLM queue or preempt
    requests, never fail, as long as one full-length sequence fits."""
    model = _server_model(server_argv)
    hf = _hf_config(model) if model else None
    if not hf:
        return None
    layout = kv_layout(hf, _flag(server_argv, "--kv-cache-dtype"))
    if not layout:
        return None
    max_model_len = int(_num(_flag(server_argv, "--max-model-len"), 0)) or int(
        (hf.get("text_config") or hf).get("max_position_embeddings", 32768))
    max_num_seqs = int(_num(_flag(server_argv, "--max-num-seqs"), 256))
    tokens = tokens_in_flight(config, max_model_len, max_num_seqs)
    tp = max(1, int(_num(_flag(server_argv, "--tensor-parallel-size"), 1)))
    budget = max(kv_bytes(layout, tokens, max_num_seqs) * KV_MARGIN, KV_FLOOR)
    fraction = _num(os.environ.get("MILABENCH_GPU_MEMORY_FRACTION_MAX"), 0.9) or 0.9
    weights = weights_bytes(model)
    cap = fraction * memory_total() * tp - weights - runtime_overhead(weights)
    one_sequence = kv_bytes(layout, max_model_len, 1) * 1.1
    return int(max(min(budget, cap), one_sequence, KV_MIN) / tp)


# share of the device memory a server may reserve up front (vLLM, SGLang)
MEMORY_FRACTION_FLAGS = ("--gpu-memory-utilization", "--mem-fraction-static")
# vLLM-Omni: per-stage engine settings as JSON ({"<stage>": {"gpu_memory_utilization": 0.4, ...}, ...})
STAGE_OVERRIDES_FLAG = "--stage-overrides"


def _cap_stage_overrides(value, cap: float):
    """The ``--stage-overrides`` JSON with every stage's ``gpu_memory_utilization`` clamped to *cap*.

    Returned unchanged (same object) when nothing is above the cap or the value is not a JSON object."""
    try:
        stages = json.loads(str(value))
    except ValueError:
        return value
    if not isinstance(stages, dict):
        return value
    changed = False
    for stage in stages.values():
        if not isinstance(stage, dict) or "gpu_memory_utilization" not in stage:
            continue
        try:
            fraction = float(stage["gpu_memory_utilization"])
        except (TypeError, ValueError):
            continue
        if fraction > cap:
            stage["gpu_memory_utilization"] = cap
            changed = True
    return json.dumps(stages) if changed else value


def cap_memory_fraction(argv: List) -> List:
    """Clamp the servers' memory fraction to ``MILABENCH_GPU_MEMORY_FRACTION_MAX``, a property of the machine.

    On unified memory (DGX Spark, APUs) the "GPU memory" is the system RAM: a server reserving 90% of it
    leaves too little for the OS, the client and its containers, and the machine swaps until it freezes
    instead of failing with an out-of-memory error. The benchmark configs stay hardware neutral; the machine
    sets the ceiling (e.g. 0.75 on a Spark). vLLM-Omni's ``--stage-overrides`` JSON: each stage's
    ``gpu_memory_utilization`` is clamped the same way (per stage; stages sharing a device add up)."""
    cap = os.environ.get("MILABENCH_GPU_MEMORY_FRACTION_MAX")
    if not cap:
        return list(argv)
    cap = float(cap)
    out = []
    for i, arg in enumerate(argv):
        arg = str(arg)
        prev = str(argv[i - 1]) if i else ""
        if prev in MEMORY_FRACTION_FLAGS:
            out.append(str(min(float(arg), cap)))
            continue
        if prev == STAGE_OVERRIDES_FLAG:
            out.append(_cap_stage_overrides(argv[i], cap))
            continue
        name, sep, value = arg.partition("=")
        if sep and name in MEMORY_FRACTION_FLAGS:
            out.append(f"{name}={min(float(value), cap)}")
            continue
        if sep and name == STAGE_OVERRIDES_FLAG:
            capped = _cap_stage_overrides(value, cap)
            out.append(argv[i] if capped is value else f"{name}={capped}")
            continue
        out.append(argv[i])
    return out


def _flag(argv, name, default=None):
    """Last value of ``name`` in argv (``name value`` or ``name=value``)"""
    for i, arg in enumerate(argv):
        arg = str(arg)
        if arg == name and i + 1 < len(argv):
            default = argv[i + 1]
        elif arg.startswith(name + "="):
            default = arg.split("=", 1)[1]
    return default


def _http_ok(url) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.status == 200
    except OSError:
        return False


class VLLMServer(ServerCommand):
    name = "vllm"

    def launcher(self):
        return [f"{self.pack.dirs.venv}/bin/vllm", "serve"]

    def extra_argv(self):
        argv = []
        # page cache eviction (unified memory) only works on pages no process maps: vLLM's default
        # safetensors loading mmaps every shard, so the checkpoint's whole cache (tens of GB) stays resident
        # during the load, next to the weights copied out of it, and the driver's next allocation fails.
        # eager: one shard at a time read() into RAM, its cache dropped by the eviction thread
        if os.environ.get("MILABENCH_EVICT_PAGE_CACHE") and not _flag(self.server_argv, "--safetensors-load-strategy"):
            argv += ["--safetensors-load-strategy", "eager"]
        # unified memory: a KV cache sized for the benchmark's passes, not a fraction of the machine
        if os.environ.get("MILABENCH_KV_CACHE_FIT") and not _flag(self.server_argv, "--kv-cache-memory-bytes"):
            budget = kv_cache_budget(self.pack.config, self.server_argv)
            argv += ["--kv-cache-memory-bytes", str(budget)] if budget else []
        return argv

    def env(self):
        env = super().env()
        if self.arch == "rocm":
            env["VLLM_TARGET_DEVICE"] = "rocm"
            env["VLLM_DISABLE_FLASHINFER"] = "1"
        # atom registers vLLM platform/model plugins that swap vLLM's own models for ATOM's whenever atom is installed
        env["ATOM_DISABLE_VLLM_PLUGIN"] = "1"
        if "milabench_sharded" in [str(a) for a in self.server_argv]:
            # benchmate's vLLM loader: per-rank weight shards, written on the first run (node-local)
            env["VLLM_MILABENCH_SHARD_DIR"] = os.path.join(
                str(self.pack.dirs.data), "vllm-shards", self.pack.config["name"]
            )
        if self.ray_head:
            # vLLM only forwards VLLM_*, NCCL_*, HF_*, ... to its Ray actors, and `ray start` on the workers
            # (over ssh) does not see this env either: without it, worker nodes load ATOM's model while the
            # head loads vLLM's
            known = {**self.pack.make_env(), **env, **self.pack.config.get("env", {})}
            env["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"] = ",".join(
                name
                for name in ("ATOM_DISABLE_VLLM_PLUGIN", "XDG_CACHE_HOME", "TORCH_HOME", "GLOO_SOCKET_IFNAME")
                if name in known
            )
            # the engine runs on the main node: same address as the Ray head (rayrun sets it for the actors)
            env["VLLM_HOST_IP"] = self.ray_head
        return env

    def uses_ray(self) -> bool:
        return _flag(self.server_argv, "--distributed-executor-backend") == "ray"

    def deploy(self, multinode=None, dp_local=1):
        """Without ``multinode="dp"``: one engine over every node with Ray (TP inside the nodes, PP across)"""
        nodes = self.nodes()
        if multinode == "dp" or (len(nodes) <= 1 and not self.uses_ray()):
            return super().deploy(multinode, dp_local)

        from .ray import RayCluster

        server = copy.copy(self)
        server._kwargs = dict(self._kwargs)
        if not self.uses_ray():
            server.server_argv = [*self.server_argv, "--distributed-executor-backend", "ray"]
        server.ray_head = nodes[0]
        cluster = RayCluster(server)
        # `ray stop` is the last step of RayCluster's sequence, which a stopped server cuts short
        return cluster, cluster.stop_executor()

    def dp_flags(self, nnodes, dp_local, master_ip, rank=0, master_port=29550):
        flags = [
            "--data-parallel-size", str(nnodes * dp_local),
            "--data-parallel-size-local", str(dp_local),
            "--data-parallel-address", master_ip,
            "--data-parallel-rpc-port", str(master_port),
        ]
        if rank != 0:
            flags += ["--data-parallel-start-rank", str(rank), "--headless"]
        return flags


class AtomServer(ServerCommand):
    """ROCm's ATOM. ``entry`` replaces ATOM's own entry point (e.g. a script applying compatibility shims)."""

    name = "atom"

    # vLLM flags ATOM rejects, so one server config can be shared by both backends
    VLLM_ONLY_FLAGS = frozenset(
        {
            "--dtype",
            "--tensor-parallel-size",
            "--distributed-executor-backend",
            "--tokenizer-mode",
            "--config_format",
            "--load_format",
            "--tool-call-parser",
            "--enable-auto-tool-choice",
            "--kv-cache-dtype",
        }
    )

    def __init__(self, pack, *server_argv, entry=None, **kwargs) -> None:
        super().__init__(pack, *server_argv, **kwargs)
        self.entry = entry

    def launcher(self):
        if self.entry:
            return [self.python, str(self.entry)]
        return [self.python, "-m", "atom.entrypoints.openai.api_server"]

    def filter_argv(self, argv):
        filtered = []
        i = 0
        while i < len(argv):
            arg = str(argv[i])
            if "=" in arg:
                if arg.split("=", 1)[0] in self.VLLM_ONLY_FLAGS:
                    i += 1
                    continue
            elif arg in self.VLLM_ONLY_FLAGS:
                i += 2 if i + 1 < len(argv) and not str(argv[i + 1]).startswith("-") else 1
                continue
            filtered.append(argv[i])
            i += 1
        return filtered

    def env(self):
        env = super().env()
        # flydsl (aiter's kernel DSL) serializes through MLIR's ROCDL target, which needs a ROCm tree with
        # amdgcn/bitcode and llvm/bin/ld.lld; the torch rocm-sdk wheels lay theirs out differently
        # (an empty ROCM_PATH, as exported by env.sh, counts as unset)
        env["ROCM_PATH"] = os.environ.get("ROCM_PATH") or "/opt/rocm"
        if self.nnodes > 1:
            # aiter's custom all-gather (IPC) is intra-node: it is disabled on a DP group that spans nodes,
            # and ATOM's DP-sharded lm_head then asserts on it. ATOM's opt-out falls back to RCCL.
            env["ATOM_USE_CUSTOM_ALL_GATHER"] = "0"
        return env

    def dp_flags(self, nnodes, dp_local, master_ip, rank=0, master_port=29550,
                 # ATOM's engine sockets take master_port + 100 + 3 * rank: keep the rendezvous port below that range
                 base_port=29450):
        # engine-only when the DP rank is not 0
        return [
            "--data-parallel-size", str(nnodes * dp_local),
            "--data-parallel-size-local", str(dp_local),
            "--data-parallel-rank", str(rank),
            "--data-parallel-master-ip", master_ip,
            "--data-parallel-master-port", str(master_port),
            "--data-parallel-base-port", str(base_port),
        ]


class OmniServer(VLLMServer):
    """vLLM-Omni: ``vllm serve <model> --omni``, diffusion (image / video / audio) and TTS models behind
    OpenAI-style endpoints (``/v1/images/generations``, ``/v1/videos``, ``/v1/audio/speech``)"""

    name = "vllm-omni"

    def extra_argv(self):
        return []

    def _argv(self, **kwargs) -> List:
        argv = cap_memory_fraction(self.filter_argv(self.server_argv))
        return [*self.launcher(), *argv, *([] if "--omni" in argv else ["--omni"])]


class SGLangServer(ServerCommand):
    name = "sglang"

    def launcher(self):
        return [self.python, "-m", "sglang.launch_server"]


class TRTLLMServer(ServerCommand):
    """NVIDIA's TensorRT-LLM (``trtllm-serve``), the CUDA counterpart of ATOM"""

    name = "trtllm"

    def launcher(self):
        return [f"{self.pack.dirs.venv}/bin/trtllm-serve"]


class SystemOneServer(ServerCommand):
    """Decision models behind ``POST /v1/systemone``: benchmarks/decision's engine_server.py in the pack's venv
    (the model's own engine / runtime, see its --adapter)"""

    name = "systemone"

    def launcher(self):
        return [self.python, os.path.join(str(self.pack.dirs.code), "engine_server.py")]


class RawServer(ServerCommand):
    """Server launched by an explicit command line (``server.command`` in the benchmark config)"""

    name = "command"

    def __init__(self, pack, command, *server_argv, **kwargs) -> None:
        super().__init__(pack, *server_argv, **kwargs)
        self.command = list(command)

    def launcher(self):
        return self.command


SERVER_COMMANDS = {cls.name: cls for cls in (VLLMServer, OmniServer, AtomServer, SGLangServer, TRTLLMServer, SystemOneServer)}


def InferenceServer(pack, *server_argv, backend="vllm", command=None, **kwargs) -> ServerCommand:
    """Server command of *backend*, or of an explicit *command* line"""
    if command:
        kwargs.pop("entry", None)
        return RawServer(pack, command, *server_argv, **kwargs)
    try:
        cls = SERVER_COMMANDS[backend]
    except KeyError as exc:
        raise ValueError(f"Unknown inference server backend: {backend} (known: {sorted(SERVER_COMMANDS)})") from exc
    if cls is not AtomServer:
        kwargs.pop("entry", None)
    return cls(pack, *server_argv, **kwargs)
