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
        # main.py reports metrics via benchmate.monitor.setupvoir(), which
        # smuggles JSON into stdout as escape sequences for milabench to
        # decode back into `data` events -- but only when the command is
        # launched with use_stdout=True (DATA_FD=1 + a Decoder on stdout,
        # see voir.proc.Multiplexer.start). Without it, milabench reads
        # stdout as plain lines and these leak into the log as raw
        # escape-sequence text instead of being scored. Same pattern as
        # benchmarks/llama and benchmarks/flops, which also call
        # setupvoir() directly.
        return RayCluster(PackCommand(self, lazy=True).use_stdout())


__pack__ = RaySmoke
