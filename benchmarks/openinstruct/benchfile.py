from milabench.commands import (
    AccelerateAllNodes,
    AccelerateLaunchCommand,
    CmdCommand,
    DockerRunCommand,
    ListCommand,
    SSHCommand,
)
from milabench.pack import Package

SOURCE_DIR = "src"
REPO_URL = "https://github.com/allenai/open-instruct"
BRANCH = "d96645078084659a6979e65491827ed8d17ecd63"

class OpenInstruct(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "src/open_instruct/finetune.py"

    async def install(self):
        await super().install()

        source_destination = self.dirs.code / SOURCE_DIR
        if not source_destination.exists():
            source_destination.clone_subtree(
                REPO_URL, BRANCH
            )
        
        # This package has the ugliest package management ever
        await self.pip_install("-e", str(source_destination), "--no-deps")

    def build_run_plan(self):
        from milabench.commands import PackCommand
        main = self.dirs.code / self.main_script
        plan = PackCommand(self, *self.argv, lazy=True, cwd=main.parent.parent)

        return AccelerateAllNodes(plan).use_stdout()


__pack__ = OpenInstruct



