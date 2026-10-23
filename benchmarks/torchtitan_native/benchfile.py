"""Native TorchTitan pre-training throughput benchmark (Qwen3.8).

Separate from ``benchmarks/torchtitan`` (Hugging Face modeling backend, older
TorchTitan pin): this pack tracks a TorchTitan commit with native Qwen3.8
support and the PyTorch nightly it requires, in its own install group.
"""

from __future__ import annotations

import os

from milabench.commands import (
    PackCommand,
    TorchrunAllGPU,
    TorchrunAllNodes,
    VoirCommand,
    max_node_count,
)
from milabench.commands.srun import (
    ForeachSrun,
    SrunCommand,
    TorchrunSrun,
    _node_hostlist_token,
)
from milabench.utils import select_nodes
from milabench.pack import Package, install_benchmate


def default_cache_dir() -> str:
    import getpass

    return f"/tmp/{getpass.getuser()}/tt_native_cache"


class NativeTorchrun(TorchrunAllGPU):
    """Always go through benchrun/torchrun, even on a single GPU."""

    def should_wrap(self):
        return True


class NativeTorchrunSrun(TorchrunSrun):
    """Main node locally; workers through one ``srun -x main``.

    Same as milabench's ``TorchrunSrun`` but always wrapped by torchrun.
    No ``WorkingDir`` wrapper: its ``env -`` clears the environment, which
    drops voir's ``DATA_FD`` and this pack's ``make_env`` variables.
    Workers get ``--node-rank`` from ``SLURM_NODEID`` (see benchrun).
    """

    def __init__(self, executor, *args, **kwargs) -> None:
        # Static rendezvous waits for every node: fail within a few minutes
        # if one never joins (torchrun default: 600 s). Configurable per
        # bench with `rdzv_timeout` (seconds).
        rdzv_timeout = int(executor.pack.config.get("rdzv_timeout", 300))
        base_exec = TorchrunAllNodes.make_base_executor(
            NativeTorchrun,
            executor,
            f"--rdzv-conf=timeout={rdzv_timeout}",
            *args,
            **kwargs,
        )
        ForeachSrun.__init__(self, base_exec)

    @property
    def executors(self):
        # Workers on exactly the selected nodes (``srun -w``), not on every
        # other node of the allocation (``srun -x main``): a run may use fewer
        # nodes than allocated (e.g. n3 inside a 4-node job, or a node subset
        # chosen for a topology test).
        config = self.executor.pack.config
        nodes = select_nodes(config["system"]["nodes"], max_node_count(config))
        if len(nodes) <= 1:
            return [self.single_node()]
        workers = ",".join(_node_hostlist_token(n) for n in nodes[1:])
        return [
            self.main_executor(),
            SrunCommand(self.worker_executor(), nodelist=workers),
        ]


class TorchtitanNative(Package):
    base_requirements = "requirements.in"
    prepare_script = None
    main_script = "main.py"

    async def install(self):
        # The torch nightly index is declared inside requirements.in, which the
        # TOML/legacy install paths do not handle; install it directly.
        # milabench installs without build isolation, so seed build backends
        # first (torchtitan's git build needs setuptools).
        await self.pip_install("setuptools", "wheel", build_isolation=True)
        await self.pip_install("-r", self.dirs.code / "requirements.in")
        await install_benchmate(self)

    async def prepare(self):
        # Random tokens, random init: nothing to download or stage.
        pass

    def make_env(self):
        env = super().make_env()
        code = str(self.dirs.code)
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = code if not prev else f"{code}{os.pathsep}{prev}"
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("VOIR_PLAIN_SMUGGLE", "1")
        # Compile caches on node-local disk: home is read-only on some compute
        # nodes (Trillium), and the shared filesystem is slow for many ranks
        # writing small files. The path must not depend on the job: cache
        # entries store absolute paths, so a pre-warmed archive only works if
        # it is restored at the same location (not under $SLURM_TMPDIR).
        cache = os.environ.get("TT_NATIVE_CACHE_DIR") or default_cache_dir()
        env.setdefault("TRITON_CACHE_DIR", f"{cache}/triton")
        env.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{cache}/inductor")
        # Compute nodes have no internet access.
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")
        return env

    def build_run_plan(self):
        plan = VoirCommand(PackCommand(self, lazy=True))
        return NativeTorchrunSrun(plan).use_stdout()


__pack__ = TorchtitanNative
