import json
import os
import subprocess
import traceback
import datetime

import cpuinfo
import voir.instruments.gpu as gpu



from ._version import __commit__, __date__, __tag__
from .utils import error_guard


def _get_gpu_info():
    try:
        smi = gpu.select_backend()

        return {
            "arch": smi.arch,
            "gpus": smi.get_gpus_info(),
            "system": smi.system_info(),
        }
    except Exception:
        traceback.print_exc()
        return {}


@error_guard({})
def fetch_torch_version(pack):
    import milabench.scripts.torchversion as torchversion

    cwd = pack.dirs.code
    exec_env = pack.full_env(dict())
    
    result = subprocess.run(
        [str(x) for x in ["python", torchversion.__file__]],
        env=exec_env,
        cwd=cwd,
        capture_output=True,
    )

    try:
        return json.loads(result.stdout)
    except json.decoder.JSONDecodeError:
        print(result.stdout)
        return {}

# One entry per venv and code dir, not per pack: a multi-node run has one pack per node, copies of one config, and
# each computation took ~12.5 s (a GitHub query that times out offline, a torch import), one node after the other,
# before any process started: 666 s at 50 nodes (Trillium job 1256251)
_metadata_cache = {}


def machine_metadata(pack=None):
    """Retrieve machine metadata"""
    # fetch_torch_version reads the code dir and the pack's env; a pack may have no venv dir (tests' MockPack)
    key = None if pack is None else (str(getattr(pack.dirs, "venv", None)), str(pack.dirs.code))
    if key not in _metadata_cache:
        _metadata_cache[key] = _machine_metadata(pack)
    return _metadata_cache[key]


machine_metadata.cache_clear = _metadata_cache.clear


@error_guard({})
def _machine_metadata(pack=None):
    from .scripts.vcs import retrieve_git_versions
    import milabench.scripts.torchversion as torchversion

    uname = os.uname()
    gpus = _get_gpu_info()
    cpu = cpuinfo.get_cpu_info()

    if pack is None:
        torchv = torchversion.get_pytorch_version()
    else:
        torchv = fetch_torch_version(pack)

    return {
        "cpu": {
            "count": os.cpu_count(),
            "brand": cpu.pop("brand_raw", "<unknown>"),
        },
        "os": {
            "sysname": uname.sysname,
            "nodename": uname.nodename,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "accelerators": gpus,
        "date": datetime.datetime.utcnow().timestamp(),
        "milabench": retrieve_git_versions(
            __tag__,
            __commit__,
            __date__,
            include_pr=True,
        ),
        "pytorch": torchv,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(machine_metadata(), indent=2))
