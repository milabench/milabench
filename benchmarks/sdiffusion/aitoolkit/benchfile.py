import os

from milabench.pack import Package


class SDiffusionPack(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def build_run_plan(self):
        from milabench.commands import PackCommand

        if "HF_TOKEN" in os.environ or "MILABENCH_HF_TOKEN" in os.environ:
            os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", os.environ.get("MILABENCH_HF_TOKEN"))

        main = self.dirs.code / self.main_script
        plan = PackCommand(self, *self.argv, lazy=True)
        return plan.use_stdout()


__pack__ = SDiffusionPack
