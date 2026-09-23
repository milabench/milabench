from milabench.pack import Package
import milabench.commands as cmd


class ComfyUI(Package):
    base_requirements = "requirements.in"
    main_script = "main.py"

    def build_run_plan(self):
        main = self.dirs.code / self.main_script
        pack = cmd.PackCommand(self, *self.argv, lazy=True)
        return cmd.VoirCommand(pack, cwd=main.parent).use_stdout()


__pack__ = ComfyUI
