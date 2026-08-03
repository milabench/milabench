from collections import defaultdict
import contextvars
import multiprocessing
import os
from copy import deepcopy
import time
from dataclasses import dataclass, field

import numpy as np
import yaml
from voir.instruments.gpu import get_gpu_info
from cantilever.core.statstream import StatStream

from .syslog import syslog
from .system import CPUOptions, SizerOptions, system_global, option
from .validation.validation import ValidationLayer

ROOT = os.path.dirname(__file__)



default_scaling_folder = os.path.join(ROOT, "..", "config", "scaling")
default_scaling_config = os.path.join(default_scaling_folder, "default.yaml")

gpu_name_to_file = {
    "AMD Instinct MI325 OAM": "MI325",
    "AMD Instinct MI355X": "MI355X",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA L40S": "L40S"
}


def gpu_name():
    try:
        info = get_gpu_info()
        values = list(info["gpus"].values())
        n = values[0]["product"]
        return gpu_name_to_file.get(n, n)
    except:
        return None


def gpu_capacity():
    try:
        info = get_gpu_info()
        values = list(info["gpus"].values())
        return values[0]["memory"]["total"]
    except:
        return None

def get_scaling_config():
    name = gpu_name()

    specialized = os.path.join(default_scaling_folder, f"{name}.yaml")

    if name is None or not os.path.exists(specialized):
        return default_scaling_config
    
    return specialized


metric_prefixes = {
    "T": (12, 4),
    "G": (9, 3),
    "M": (6, 2),
    "k": (3, 1),
    "h": (2, None),
    "da": (1, None),
    "d": (-1, None),
    "c": (-2, None),
    "m": (-3, None),
    "u": (-6, None),
    "n": (-9, None),
    "p": (-12, None),
}


def to_octet(value: str) -> float:
    for p, (vm, vb) in metric_prefixes.items():
        if f"{p}iB" in value or f"{p}io" in value:
            return float(value[: -(len(p) + 2)]) * 1024**vb

        if f"{p}B" in value or f"{p}o" in value:
            return float(value[: -(len(p) + 1)]) * 10**vm

    if "io" in value:
        return float(value.replace("io", ""))

    if "o" in value:
        return float(value.replace("o", ""))

    return float(value)


def observation_memory(obs):
    """Return observation memory in octets, preferring allocator stats.

    Preference: ``torchmem`` then ``jaxmem`` then NVML ``memory``.
    """
    for key in ("torchmem", "jaxmem"):
        if key in obs and obs[key]:
            return to_octet(obs[key])
    return to_octet(obs["memory"])


def _max_allocated_mib(payload):
    """Max ``max_allocated`` across a torchmem/jaxmem per-device payload."""
    if not payload:
        return None
    return max(data.get("max_allocated", 0) for data in payload.values())


def _obs_mib_int(obs, key):
    """Parse ``\"1234 MiB\"`` style observation fields; return None if missing."""
    value = obs.get(key)
    if not value:
        return None
    return int(str(value).split(" ")[0])


def _obs_pair_octets(obs):
    """Return ``(nvml_memory, allocator_memory)`` in octets, or ``None``.

    Allocator memory is ``torchmem`` when present, else ``jaxmem``.
    """
    mem = obs.get("memory")
    alloc = obs.get("torchmem") or obs.get("jaxmem")
    if not mem or not alloc:
        return None
    return to_octet(mem), to_octet(alloc)


def _fixed_overhead(observations):
    """Median NVML−allocator gap (octets), or ``None`` if no paired observations."""
    gaps = []
    for obs in observations:
        pair = _obs_pair_octets(obs)
        if pair is None:
            continue
        mem, alloc = pair
        gaps.append(max(mem - alloc, 0.0))
    if not gaps:
        return None
    return float(np.median(gaps))


def _fit_torchmem_vs_batch(observations):
    """Fit ``allocator ≈ α·batch_size + β``.

    Returns ``(α, β)`` in octets-per-batch / octets, or ``None`` if fewer than
    two paired points or non-positive slope.
    """
    batches = []
    allocs = []
    for obs in observations:
        pair = _obs_pair_octets(obs)
        if pair is None:
            continue
        _, alloc = pair
        batches.append(float(obs["batch_size"]))
        allocs.append(alloc)
    if len(batches) < 2:
        return None
    alpha, beta = np.polyfit(batches, allocs, deg=1)
    if alpha <= 0:
        return None
    return float(alpha), float(beta)


class Sizer:
    """Automatically scale the batch size to match GPU spec"""

    def __init__(self, sizer=None, config=None):
        self.path = config
        self.sizer_override = sizer
        
        
        if config is None:
            config = SizerOptions.instance().config
            
            if config is None:
                config = get_scaling_config()

        self.scaling_config = {}
        if os.path.exists(config):
            with open(config, "r") as sconf:
                self.scaling_config = yaml.safe_load(sconf)
        else:
            print(config, "does not exist")

    @property
    def options(self):
        if self.sizer_override:
            return self.sizer_override
        return SizerOptions()

    def benchscaling(self, benchmark):
        # key
        if isinstance(benchmark, str):
            return self.scaling_config.get(benchmark)

        # benchmark config
        if isinstance(benchmark, dict) and "name" in benchmark:
            return self.scaling_config.get(benchmark["name"])

        # pack
        return self.scaling_config.get(benchmark.config["name"])

    def _scaling_v1(self, config):
        data = list(sorted(config["model"].items(), key=lambda x: x[0]))

        mem = [to_octet(v[1]) for v in data]
        size = [float(v[0]) for v in data]

        return mem, size
    
    def _scaling_v2(self, config):
        data = config.get("observations", [])

        data = list(sorted(data, key=lambda x: x["batch_size"]))

        mem = [observation_memory(v) for v in data]
        size = [float(v["batch_size"]) for v in data]
        perf = [float(v["perf"]) for v in data]

        return mem, size, perf

    def get_capacity(self, capacity):
        if self.options.capacity is not None:
            capacity = self.options.capacity

        if capacity == "All":
            capacity = f"{gpu_capacity()} MiB"

        if isinstance(capacity, str):
            capacity = to_octet(capacity)

        return capacity

    def _finalize_batch_size(self, newsize_f):
        """Round predicted batch size and apply multiple/power constraints."""
        newsize_i = int(newsize_f)

        if newsize_i <= 0:
            return 1

        if (newsize_f - newsize_i) > 0.5:
            newsize_i += 1

        final_size = newsize_i

        if self.options.multiple:
            final_size = (newsize_i // self.options.multiple) * self.options.multiple

        if self.options.power:
            final_size = int(self.options.power) ** int(np.log2(newsize_i))

        return max(final_size, 1)

    def _auto_size_legacy(self, mem, size, capacity, bench_name):
        """Single-series fit: ``batch_size ≈ a·mem + b`` (v1 / fallback)."""
        if len(mem) <= 1:
            syslog(f"Not enough data for {bench_name}")
            return 1

        model = np.poly1d(np.polyfit(mem, size, deg=1))
        newsize_f = model(capacity)
        final_size = self._finalize_batch_size(newsize_f)
        syslog(
            "auto_size legacy path for {}: capacity={} predicted={} final={}",
            bench_name,
            capacity,
            newsize_f,
            final_size,
        )
        return final_size

    def _auto_size_fixed_torchmem(self, observations, capacity, bench_name):
        """Predict BS from fixed NVML overhead + linear torchmem vs batch.

        ``total(B) = fixed + (α·B + β)``; solve for capacity. Returns ``None``
        when there is not enough paired data or the slope is non-positive.
        """
        fixed = _fixed_overhead(observations)
        fit = _fit_torchmem_vs_batch(observations)
        if fixed is None or fit is None:
            return None

        alpha, beta = fit
        newsize_f = (capacity - fixed - beta) / alpha
        final_size = self._finalize_batch_size(newsize_f)
        syslog(
            "auto_size fixed+torchmem for {}: fixed={} alpha={} beta={} "
            "capacity={} predicted={} final={}",
            bench_name,
            fixed,
            alpha,
            beta,
            capacity,
            newsize_f,
            final_size,
        )
        return final_size

    def auto_size(self, benchmark, capacity):
        capacity = self.get_capacity(capacity)

        if capacity is None:
            syslog("Capacity is missing")
            return None

        config = self.benchscaling(benchmark)

        if not config:
            return 1

        if isinstance(benchmark, str):
            bench_name = benchmark
        elif isinstance(benchmark, dict):
            bench_name = benchmark.get("name", "?")
        else:
            bench_name = benchmark.config["name"]

        # v1 scaling maps have no torchmem column — keep legacy fit.
        if "model" in config:
            mem, size = self._scaling_v1(config)
            return self._auto_size_legacy(mem, size, capacity, bench_name)

        observations = config.get("observations", [])
        result = self._auto_size_fixed_torchmem(observations, capacity, bench_name)
        if result is not None:
            return result

        mem, size, _ = self._scaling_v2(config)
        return self._auto_size_legacy(mem, size, capacity, bench_name)

    def optimized(self, benchmark, capacity):
        # Old V1 format
        config = self.benchscaling(benchmark)

        if "model" in config:
            return config["optimized"]

        # Look for the best batch size
        capacity = self.get_capacity(capacity)

        if capacity is None:
            syslog("Capacity is missing")
            return None

        # Prefer NVML ``memory`` when present so fit checks match auto_size capacity.
        data = config["observations"]
        data = list(sorted(data, key=lambda x: x["perf"], reverse=True))

        for obs in data:
            if obs.get("memory"):
                used_mem = to_octet(obs["memory"])
            else:
                used_mem = observation_memory(obs)
            if used_mem < capacity:
                return int(obs["batch_size"])

        return None

    def size(self, benchmark, capacity):
        config = self.benchscaling(benchmark)

        if self.options.size is not None:
            return self.options.size

        if self.options.optimized:
            return self.optimized(benchmark, capacity)

        if self.options.auto:
            return self.auto_size(benchmark, capacity)

        syslog("Could not find auto scale the batch size")
        return None


sizer_global = contextvars.ContextVar("sizer_global", default=None)


def batch_sizer() -> Sizer:
    return Sizer()
    # sizer = sizer_global.get()
    # if sizer is None:
    #     sizer_global.set(Sizer())
    #     return batch_sizer()
    # return sizer


def get_batch_size(config, start_event):
    try:
        sizer = batch_sizer()
        return sizer.find_batch_size(config, start_event)
    except:
        return 'NA'


def suggested_batch_size(pack):
    sizer = batch_sizer()

    system = system_global.get()
    capacity = system.get("gpu", dict()).get("capacity")

    if capacity is None:
        capacity = f"{gpu_capacity()} MiB"

    return sizer.size(pack, capacity)


def compact_dump():
    # This is to create a compact yaml that is still readable
    from yaml.representer import SequenceNode, ScalarNode

    class CustomDumper(yaml.SafeDumper):
        
        def represent_sequence(self, tag, sequence, flow_style=None):
            value = []
            node = SequenceNode(tag, value, flow_style=flow_style)

            if self.alias_key is not None:
                self.represented_objects[self.alias_key] = node
            best_style = True

            for item in sequence:
                node_item = self.represent_data(item)
                node_item.flow_style = True

                if not (isinstance(node_item, ScalarNode) and not node_item.style):
                    best_style = False

                value.append(node_item)

            if flow_style is None:
                if self.default_flow_style is not None:
                    node.flow_style = self.default_flow_style
                else:
                    node.flow_style = best_style
            return node

    return CustomDumper

@dataclass
class BenchStats:
    benchname: str
    active_count: int = 0
    rc: list = field(default_factory=list)
    early_stopped: list = field(default_factory=list)
    batch_size: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    perf: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    cpu: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    max_usage: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    torchmem_usage: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))
    jaxmem_usage: StatStream = field(default_factory=lambda: StatStream(drop_first_obs=0))

    def max_memory_usage(self):
        for stream in (self.torchmem_usage, self.jaxmem_usage, self.max_usage):
            if stream.current_count != 0:
                return stream.max
        return float("-inf")

    def has_stopped_early(self):
        return len(self.early_stopped) > 0 and self.early_stopped[-1]

class MemoryUsageExtractor(ValidationLayer):
    """Extract max memory usage per benchmark to populate the memory model"""

    def __init__(self):
        sizer = Sizer()

        self.filepath = SizerOptions.instance().save

        if self.filepath and os.path.exists(self.filepath):
            with open(self.filepath, "r") as fp:
                self.memory = yaml.safe_load(fp) or {}
        elif SizerOptions.instance().save == SizerOptions.instance().config:
            self.memory = deepcopy(sizer.scaling_config)
        else:
            self.memory = {}
        
        if self.memory.get("version", 1.0) <= 1.0:
            self.convert()
            self.memory["version"] = 2.0

        self.benchname = None
        self._benchstat = {}
        global on_batch_size_set, on_cpu_count_set


        # TODO: currently this is okay but we might have to find a way to make
        # this class only remove its callback
        on_batch_size_set = [self.on_batch_size_set]
        on_cpu_count_set = [self.on_cpu_count_set]

    def convert(self):
        # TODO: this could be handled seemlessly on the loading part
        for bench, config in self.memory.items():
            if bench == "version":
                continue
        
            model = config.pop("model", None)

            if model is not None:
                obs = []
                for k, v in model.items():
                    obs.append({"batch_size": k, "memory": v})
    
                config["observations"] = obs

    def benchstat(self, name):
        if self.benchname != name:
            self._benchstat[name] = BenchStats(name)
            self.benchname = name
        return self._benchstat[name]

    def on_cpu_count_set(self, pack, _, value):
        self.benchstat(pack.config["name"]).cpu += value

    def on_batch_size_set(self, pack, _, value):
        self.benchstat(pack.config["name"]).batch_size += value

    def on_start(self, entry):
        if self.filepath is None:
            return

        self.benchstat(entry.pack.config["name"]).active_count += 1

    def on_data(self, entry):
        if self.filepath is None:
            return

        if entry.data is None:
            return

        stat = self.benchstat(entry.pack.config["name"])

        # Legacy in-loop JAX peak (purejaxrl); fold into jaxmem column
        if memorypeak := entry.data.get("memory_peak"):
            stat.jaxmem_usage += memorypeak

        if gpudata := entry.data.get("gpudata"):
            for device, data in gpudata.items():
                usage, total = data.get("memory", [0, 1])
                stat.max_usage += usage

        if torchmem := entry.data.get("torchmem"):
            if (usage := _max_allocated_mib(torchmem)) is not None:
                stat.torchmem_usage += usage

        if jaxmem := entry.data.get("jaxmem"):
            if (usage := _max_allocated_mib(jaxmem)) is not None:
                stat.jaxmem_usage += usage

        if rate := entry.data.get("rate"):
            stat.perf += rate
 
    def on_stop(self, entry):
        stat = self.benchstat(entry.pack.config["name"])
        stat.early_stopped.append(True)

    def on_end(self, entry):
        if self.filepath is None:
            return

        stats = self.benchstat(entry.pack.config["name"])

        # Only update is successful
        rc = entry.data["return_code"]
        if rc == 0 or stats.has_stopped_early():
            rc = 0

        stats.active_count -= 1
        stats.rc.append(rc)

        if stats.batch_size.current_count <= 0 and int(stats.batch_size.avg) == 0:
            syslog("MemoryUsageExtractor: Skipping missing batch_size {}", entry)
            return

        if stats.max_memory_usage() == float("-inf"):
            syslog("MemoryUsageExtractor: Missing memory info {}", entry)
            return

        if stats.active_count <= 0:
            if sum(stats.rc) == 0:
                self.push_observation(stats)
            else:
                syslog("MemoryUsageExtractor: Could not add scaling data because of a failure {}", stats.benchname)

            try:
                self.save()
            except Exception as err:
                print(f"MemoryUsageExtractor: Could not save scaling file because of {err}")

    def push_observation(self, stats):
        config = self.memory.setdefault(stats.benchname, dict())
        observations = config.setdefault("observations", [])

        obs = {
            "cpu": int(stats.cpu.avg),
            "batch_size": int(stats.batch_size.avg),
            "perf": float(f"{stats.perf.avg:.2f}"),
            "time": int(time.time())
        }

        # NVML / gpudata stays in ``memory``; allocator stats get their own columns
        if stats.max_usage.current_count > 0:
            obs["memory"] = f"{int(stats.max_usage.max)} MiB"
        if stats.torchmem_usage.current_count > 0:
            obs["torchmem"] = f"{int(stats.torchmem_usage.max)} MiB"
        if stats.jaxmem_usage.current_count > 0:
            obs["jaxmem"] = f"{int(stats.jaxmem_usage.max)} MiB"

        observations.append(obs)
        config["observations"] = list(sorted(observations, key=lambda x: x["batch_size"]))

    def save(self):
        if self.filepath is not None:
            with open(self.filepath, "w") as file:
                yaml.dump(self.memory, file, Dumper=compact_dump(), width=float("inf"))

    def report(self, *args, **kwargs):
        for name, stats in self._benchstat.items():
            if stats.active_count > 0:
                syslog("MemoryUsageExtractor: Could not add scaling data because bench never ended {}", stats.benchname)

        self.save()

def arch_to_device(arch):
    device_types = [
        "cpu",
        "cuda",
        "ipu",
        "xpu",
        "mkldnn",
        "opengl", "opencl", "ideep", "hip", "ve",
        "fpga", "maia", "xla", "lazy", "vulkan", "mps", "meta",
        "hpu", "mtia", "privateuseone"
    ]
    arch_to_device = {t:t for t in device_types}
    arch_to_device["rocm"] = "cuda"
    return arch_to_device.get(arch, "cpu")


on_cpu_count_set = []
on_batch_size_set = []

def broadcast(delegates, *args, **kwargs):
    for fun in delegates:
        try:
            fun(*args, **kwargs)
        except Exception as err:
            print(f"Error during broadcasting {fun} {err}")


def new_argument_resolver(pack):
    system_config = system_global.get()
    if system_config is None:
        system_config = {}

    context = deepcopy(system_config)

    arch = context.get("arch", "cpu")
    device_count_used = 1
    device_count_system = len(get_gpu_info()["gpus"])

    if hasattr(pack, "config"):
        device_count_used = len(pack.config.get("devices", [0]))

    if device_count_used <= 0:
        device_count_used = 1

    ccl = {"hpu": "hccl", "cuda": "nccl", "rocm": "rccl", "xpu": "ccl", "cpu": "gloo"}

    cpu_opt = CPUOptions.instance()

    def cpu(value, default):
        newvalue = default

        if cpu_opt.enabled:
            newvalue = value
        
        broadcast(on_cpu_count_set, pack, default, newvalue)
        return newvalue

    
    def mult(a, b):
        return a * b
    
    gpu_opt = SizerOptions.instance()
    def batch_resize(default):
        val = default

        if gpu_opt.enabled:
            if (gpu_opt.add is not None or gpu_opt.mult is not None):
                val = max(1, int(default * (gpu_opt.mult or 1)) + (gpu_opt.add or 0))
            else:
                val = suggested_batch_size(pack)
                assert val is not None

        broadcast(on_batch_size_set, pack, default, val)
        return val

    def clamp(x, mn=cpu_opt.cpu_min, mx=cpu_opt.cpu_max):
        return min(max(x, mn), mx)

    total_cpu = cpu_opt.total_count or multiprocessing.cpu_count()
    total_available = total_cpu - cpu_opt.reserved_cores

    context["cpu_count"] = total_available
    context["gpu_count"] = device_count_system
    context["cpu_per_gpu"] = total_available // max(device_count_system, 1)
    context["n_worker"] = clamp(context["cpu_per_gpu"])

    if cpu_opt.n_workers is not None:
        context["n_worker"] = cpu_opt.n_workers

    context["arch"] = arch
    context["device_name"] = arch_to_device(arch)
    context["ccl"] = ccl.get(arch, "gloo")

    context["milabench_base"] = option("base", str, default="")
    dirs = vars(pack.dirs)
    context["milabench_venv"] = dirs.get('venv', "")
    context["milabench_code"] = dirs.get('code', "")
    context["milabench_extra"] = dirs.get('extra', "")
    context["milabench_data"] = dirs.get('data', "")
    context["milabench_runs"] = dirs.get('runs', "")
    context["milabench_cache"] = dirs.get('cache', "")
    context["milabench_name"] = pack.config.get("name", None)
    context["benchmark_folder"] = pack.config.get('definition', None)

    def expr(ex):
        return ex

    def auto_eval(arg):
        try:
            newvalue: str = str(arg).format(**context)

            # Handles the case where argument=value
            finalize_val = lambda x: x
            if "=" in newvalue:
                name, newvalue = newvalue.split("=", maxsplit=1)
                finalize_val = lambda x: f"{name}={x}"

            if newvalue.startswith("auto") or newvalue.startswith("expr") :
                newvalue = str(eval(newvalue, {"auto": cpu, "expr": expr, "auto_batch": batch_resize}, {}))
            
            return finalize_val(newvalue)
        except KeyError as err:
            syslog("Couldn't resolve {} because of {}", arg, err)
            return arg

    return auto_eval


def resolve_placeholder(pack, value):
    resolver = new_argument_resolver(pack)
    return resolver(value)


def resolve_argv(pack, argv):
    resolver = new_argument_resolver(pack)
    argv = list(argv)
    for i, arg in enumerate(argv):
        argv[i] = resolver(arg)
    return argv




def deduplicate_observation(scaling):
    deduplicated_scaling = {}

    for bench, data in scaling.items():
        if bench == "version":
            deduplicated_scaling[bench] = data
            continue

        observations = data.get("observations", [])
        duplicate_sets = defaultdict(list)

        for obs in observations:
            index = (obs["batch_size"], obs["cpu"])
            duplicate_sets[index].append(obs)
        
        newobs = []

        # Add back unique observation
        for key in list(duplicate_sets.keys()):
            if len(duplicate_sets[key]) == 1:
                data = duplicate_sets.pop(key)[0]

                if data["perf"] > 0:
                    newobs.append(data)

        # Merge duplicates
        while len(duplicate_sets) > 0:
            key, data = duplicate_sets.popitem()

            memory_stat = StatStream(0)
            torchmem_stat = StatStream(0)
            jaxmem_stat = StatStream(0)
            perf_stat = StatStream(0)
            lastest_time = 0

            for obs in data:
                perf = obs["perf"]

                if perf > 0:
                    if (mem := _obs_mib_int(obs, "memory")) is not None:
                        memory_stat += mem
                    if (tm := _obs_mib_int(obs, "torchmem")) is not None:
                        torchmem_stat += tm
                    if (jm := _obs_mib_int(obs, "jaxmem")) is not None:
                        jaxmem_stat += jm
                    perf_stat += perf
                    lastest_time = max(lastest_time, obs["time"])

            # Prefer NVML for merge similarity; fall back to allocator columns
            merge_mem = memory_stat
            if merge_mem.current_count == 0:
                merge_mem = torchmem_stat if torchmem_stat.current_count else jaxmem_stat

            should_generate_aggregate = (
                (perf_stat.current_count > 1 and merge_mem.current_count > 1) and 
                (merge_mem.avg > 0 and merge_mem.sd / merge_mem.avg < 0.1) and 
                (perf_stat.avg > 0   and perf_stat.sd   / perf_stat.avg   < 0.1)
            )
            should_generate_single = perf_stat.current_count == 1 and perf_stat.avg > 0 and merge_mem.current_count == 1

            if should_generate_aggregate or should_generate_single:
                # If observation are similar-ish merge them into one
                merged = {
                    "batch_size": key[0], 
                    "cpu": key[1],
                    "perf": int(perf_stat.avg * 100) / 100,
                    "time": int(lastest_time)
                }
                if memory_stat.current_count > 0:
                    merged["memory"] = f"{int(memory_stat.avg)} MiB"
                if torchmem_stat.current_count > 0:
                    merged["torchmem"] = f"{int(torchmem_stat.avg)} MiB"
                if jaxmem_stat.current_count > 0:
                    merged["jaxmem"] = f"{int(jaxmem_stat.avg)} MiB"
                newobs.append(merged)
            else:
                if (not should_generate_aggregate) and perf_stat.avg > 0 and merge_mem.avg > 0:
                    syslog("{}: could not merge observation, significant differences because (Mem: {:.2f} < 0.1) and (Perf: {:.2f} < 0.1)",
                         bench, merge_mem.sd / merge_mem.avg, perf_stat.sd / perf_stat.avg)
                
                for obs in data:
                    if obs["perf"] > 0:
                        newobs.append(obs)

        # make sure observations are sorted
        newobs = list(sorted(newobs, key=lambda x: x["batch_size"]))

        deduplicated_scaling[bench] = {
            "observations": newobs
        }

    return deduplicated_scaling


def deduplicate_scaling_file(filepath):
    with open(filepath, "r") as fp:
        memory = yaml.safe_load(fp) or {}

    newmem = deduplicate_observation(memory)

    with open(f"{filepath}.new.yml", "w") as fp:
        yaml.dump(newmem, fp, Dumper=compact_dump(), width=float("inf"))



def scaling_to_csv(filepath):
    import csv

    with open(filepath, "r") as fp:
        memory = yaml.safe_load(fp) or {}


    with open("scaling.csv", "w") as file:
        writer = csv.writer(file)
        row_count = 0

        for k, items in memory.items():
            if k == "version":
                continue
        

            rows = items["observations"]
            
            for row in rows:
                row["bench"] = k

                sorted_row = sorted(list(row.items()), key=lambda x: x[0])

                value_row = list(map(lambda x: x[1], sorted_row))

                if row_count == 0:
                    header_row = list(map(lambda x: x[0], sorted_row))
                    writer.writerow(header_row)

                writer.writerow(value_row)
                row_count += 1
    



def merge_scaling_files(*files):
    all_data = defaultdict(lambda: {"observations": []})

    for file in files:
        with open(file, "r") as fp:
            data = yaml.safe_load(fp) or {}

        for k, items in data.items():
            if k == "version":
                continue

            rows = all_data[k]["observations"]
            
            rows.extend(items["observations"])

            all_data[k]["observations"] = list(sorted(rows, key=lambda x: x["batch_size"]))


    newmem = deduplicate_observation(dict(all_data))

    with open("merged.yaml", "w") as fp:
        yaml.dump(dict(newmem), fp, Dumper=compact_dump(), width=float("inf"))
    

if __name__ == "__main__":
    import sys
    # filepath = "/home/testroot/milabench/config/scaling/MI325.yaml"
    # scaling_to_csv(filepath)
    # deduplicate_scaling_file(filepath)

    merge_scaling_files(*sys.argv[1:])
