import os

import milabench.commands as cmd
from milabench.pack import Package


class LLMFinetune(Package):
    """LoRA / QLoRA fine-tuning (unsloth + TRL SFTTrainer), voir-instrumented.

    Shares the ``unsloth`` install group (venv) with benchmarks/unsloth. Unlike that pack, the run goes
    through voir, so milabench samples GPU memory / power (gpudata) for the report.
    """

    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        token = os.environ.get("HF_TOKEN") or os.environ.get("MILABENCH_HF_TOKEN")
        if token:
            env["HF_TOKEN"] = token
        return env

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()


__pack__ = LLMFinetune
