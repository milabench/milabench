from milabench.commands import AccelerateAllNodes, PackCommand
from milabench.pack import Package

SOURCE_DIR = "src"
REPO_URL = "https://github.com/allenai/open-instruct"
BRANCH = "d96645078084659a6979e65491827ed8d17ecd63"


class OpenInstruct(Package):
    base_requirements = "requirements.in"
    prepare_script = None
    main_script = "src/open_instruct/finetune.py"

    @property
    def working_directory(self):
        return self.dirs.code / SOURCE_DIR

    async def install(self):
        await super().install()

        source_destination = self.dirs.code / SOURCE_DIR
        if not source_destination.exists():
            source_destination.clone_subtree(REPO_URL, BRANCH)

        # Upstream open-instruct expects an editable install of its own tree.
        await self.pip_install("-e", str(source_destination), "--no-deps")

    def build_run_plan(self):
        plan = PackCommand(self, *self.argv, lazy=True)
        return AccelerateAllNodes(plan).use_stdout()


__pack__ = OpenInstruct
