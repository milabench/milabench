from milabench.commands import PackCommand
from milabench.commands.ray import RayCluster
from milabench.pack import Package


class RaySmoke(Package):
    """Ray cluster placement smoke test.

    Independent of the vllm benchmark: vllm/platforms/cpu.py forces the
    local "mp" distributed executor whenever world_size > 1 on CPU, so
    vllm never actually exercises the multi-node Ray cluster milabench
    builds. This runs directly against Ray to prove the cluster itself
    genuinely spreads work across every allocated node.
    """

    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self):
        return RayCluster(PackCommand(self, lazy=True))


__pack__ = RaySmoke
