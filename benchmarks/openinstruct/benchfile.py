from milabench.commands import AccelerateAllNodes, PackCommand, VoirCommand
from milabench.pack import Package

SOURCE_DIR = "src"
REPO_URL = "https://github.com/allenai/open-instruct"
BRANCH = "d96645078084659a6979e65491827ed8d17ecd63"


class OpenInstruct(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    @property
    def working_directory(self):
        # Upstream paths (output/, configs/) are relative to the cloned tree.
        return self.dirs.code / SOURCE_DIR

    async def install(self):
        await super().install()

        source_destination = self.dirs.code / SOURCE_DIR
        if not source_destination.exists():
            source_destination.clone_subtree(REPO_URL, BRANCH)

        for script in source_destination.glob("open_instruct/*.py"):
            try:
                script.chmod(script.stat().st_mode | 0o111)
            except OSError:
                pass

        from compat import patch_upstream_model_utils

        patch_upstream_model_utils(source_destination / "open_instruct" / "model_utils.py")

        await self.pip_install("-e", str(source_destination), "--no-deps")

    def build_run_plan(self):
        plan = VoirCommand(PackCommand(self, lazy=True))
        tags = self.config.get("tags") or []
        if "monogpu" in tags and "multigpu" not in tags:
            return plan.use_stdout()
        return AccelerateAllNodes(plan).use_stdout()


__pack__ = OpenInstruct
