from milabench.commands import AccelerateAllNodes, PackCommand, VoirCommand
from milabench.pack import Package

SOURCE_DIR = "src"
REPO_URL = "https://github.com/allenai/open-instruct"
BRANCH = "d96645078084659a6979e65491827ed8d17ecd63"


class OpenInstruct(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "src/open_instruct/finetune.py"

    @property
    def working_directory(self):
        return self.dirs.code / SOURCE_DIR

    async def install(self):
        await super().install()

        source_destination = self.dirs.code / SOURCE_DIR
        if not source_destination.exists():
            source_destination.clone_subtree(REPO_URL, BRANCH)

        # Ensure scripts are executable (PackCommand may exec the path directly).
        for script in source_destination.glob("open_instruct/*.py"):
            try:
                script.chmod(script.stat().st_mode | 0o111)
            except OSError:
                pass

        # Upstream open-instruct expects flash_4; older ai2-olmo-core may lack it.
        self._patch_attn_backends(source_destination / "open_instruct" / "model_utils.py")

        # Upstream open-instruct expects an editable install of its own tree.
        await self.pip_install("-e", str(source_destination), "--no-deps")

    @staticmethod
    def _patch_attn_backends(path):
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        if "getattr(AttentionBackendName, _name, None)" in text:
            return
        needle = (
            "_OLMO_CORE_TO_HF_ATTN: dict[AttentionBackendName, str] = {\n"
            '    AttentionBackendName.flash_4: "flash_attention_4",\n'
            '    AttentionBackendName.flash_3: "flash_attention_3",\n'
            '    AttentionBackendName.flash_2: "flash_attention_2",\n'
            '    AttentionBackendName.torch: "sdpa",\n'
            '    AttentionBackendName.te: "sdpa",\n'
            "}\n"
        )
        if needle not in text:
            return
        replacement = '''_OLMO_CORE_TO_HF_ATTN: dict[AttentionBackendName, str] = {}
for _name, _hf in (
    ("flash_4", "flash_attention_4"),
    ("flash_3", "flash_attention_3"),
    ("flash_2", "flash_attention_2"),
    ("torch", "sdpa"),
    ("te", "sdpa"),
):
    _backend = getattr(AttentionBackendName, _name, None)
    if _backend is not None:
        _OLMO_CORE_TO_HF_ATTN[_backend] = _hf
'''
        path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    def build_run_plan(self):
        # VoirCommand supplies `voir`/`python` so finetune.py need not be +x alone.
        plan = VoirCommand(PackCommand(self, lazy=True))
        tags = self.config.get("tags") or []
        # NJobs stamps all GPUs onto the pack *after* build_run_plan, so pin/monogpu
        # mutations here are overwritten. For monogpu smoke, skip AccelerateAllNodes
        # and rely on HIP_VISIBLE_DEVICES from the pack env for a single-process run.
        if "monogpu" in tags and "multigpu" not in tags:
            return plan.use_stdout()
        return AccelerateAllNodes(plan).use_stdout()


__pack__ = OpenInstruct
