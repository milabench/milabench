"""Dummy torchrun + srun multi-node benchmark.

Main node runs locally and owns milabench metrics. Workers are launched once
via ``srun -x <main>`` and tagged ``nolog`` so only node 0's rates count.
"""

from milabench.commands import PackCommand, TorchrunAllGPU, TorchrunAllNodes
from milabench.commands.srun import ForeachSrun
from milabench.pack import Package


class TorchrunAlways(TorchrunAllGPU):
    """Always wrap with benchrun/torchrun (even for a single GPU/CPU per node)."""

    def should_wrap(self):
        return True

    def device_count(self):
        # At least one process per node so CPU/gloo smoke tests still launch.
        return max(len(self.pack.config.get("devices", [])), 1)

    def _argv(self, **kwargs):
        argv = super()._argv(**kwargs)
        nproc = f"--nproc-per-node={self.device_count()}"
        return [nproc if a == "--nproc-per-node=gpu" else a for a in argv]


class TorchrunSrunAlways(ForeachSrun):
    """Same layout as ``TorchrunSrun``, but always launches through torchrun."""

    def __init__(self, executor, *args, **kwargs) -> None:
        base_exec = TorchrunAllNodes.make_base_executor(
            TorchrunAlways,
            executor,
            *args,
            **kwargs,
        )
        super().__init__(base_exec)


class Torchsrun(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self):
        # PackCommand (no VoirCommand): metrics via BenchObserver(stdout=True).
        # TorchrunSrunAlways: main local + workers via ``srun -x main``.
        return TorchrunSrunAlways(PackCommand(self, lazy=True)).use_stdout()


__pack__ = Torchsrun


def main():
    """Debug helper: print the resolved run plan."""
    config = {
        "name": "torchsrun",
        "definition": ".",
        "plan": {"method": "njobs", "n": 1},
        "num_machines": 2,
        "argv": ["--steps", "20", "--sleep", "0.01"],
        "tag": [],
        "dirs": {
            "code": ".",
            "extra": "extra",
            "cache": "cache",
            "venv": "env",
            "base": "base",
            "data": "data",
            "runs": "runs",
        },
        "system": {
            "self": {"name": "n0", "ip": "n0", "main": True, "user": "u", "hostname": "n0"},
            "nodes": [
                {"name": "n0", "ip": "n0", "main": True, "user": "u", "hostname": "n0"},
            ] + [{"name": f"n{i}", "ip": f"n{i}", "main": False, "user": "u", "hostname": f"n{i}"} for i in range(1, 20)],
        },
    }
    plan = Torchsrun(config).build_run_plan()
    for pack, argv, _ in plan.commands():
        print("")
        print(pack.config.get("tag", []))
        print("    ", " ".join(argv))


if __name__ == "__main__":
    main()
