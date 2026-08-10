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
        self._patch_finetune_skip_model_save(source_destination / "open_instruct" / "finetune.py")
        self._patch_finetune_torchvision_shim(source_destination / "open_instruct" / "finetune.py")
        self._patch_finetune_deepspeed_scheduler(source_destination / "open_instruct" / "finetune.py")
        self._patch_finetune_rate_metric(source_destination / "open_instruct" / "finetune.py")

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

    @staticmethod
    def _patch_finetune_skip_model_save(path):
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        if "skip_model_save" in text:
            return
        field_needle = (
            "    clean_checkpoints_at_end: bool = field(\n"
            '        default=True, metadata={"help": "Whether to clean up all previous checkpoints at the end of the run."}\n'
            "    )\n"
        )
        field_replacement = (
            field_needle
            + "    skip_model_save: bool = field(\n"
            '        default=False, metadata={"help": "Skip writing the final merged model to output_dir."}\n'
            "    )\n"
        )
        if field_needle not in text:
            return
        text = text.replace(field_needle, field_replacement, 1)
        save_needle = (
            "    if args.output_dir is not None:\n"
            "        save_with_accelerate(\n"
        )
        save_replacement = (
            "    if args.output_dir is not None and not args.skip_model_save:\n"
            "        save_with_accelerate(\n"
        )
        if save_needle not in text:
            return
        path.write_text(text.replace(save_needle, save_replacement, 1), encoding="utf-8")

    @staticmethod
    def _patch_finetune_torchvision_shim(path):
        """datasets 4.x imports VideoReader; ROCm torchvision omits it."""
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        marker = "_milabench_torchvision_videoreader_shim"
        if marker in text:
            return
        needle = 'os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA\n'
        shim = (
            'os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA\n'
            "try:\n"
            "    from torchvision.io import VideoReader  # noqa: F401\n"
            "except ImportError:\n"
            "    import torchvision.io as _tv_io\n"
            "\n"
            "    class VideoReader:  # milabench: dummy for datasets 4.x on ROCm\n"
            "        pass\n"
            "\n"
            f"    _tv_io.VideoReader = VideoReader  # {marker}\n"
        )
        if needle not in text:
            return
        path.write_text(text.replace(needle, shim, 1), encoding="utf-8")

    @staticmethod
    def _patch_finetune_deepspeed_scheduler(path):
        """DeepSpeed ZeRO-3 reshapes optimizer param groups after prepare()."""
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        marker = "_milabench_deepspeed_scheduler_rebuild"
        if marker in text:
            return
        needle = (
            "    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(\n"
            "        model, optimizer, train_dataloader, lr_scheduler\n"
            "    )\n"
        )
        replacement = (
            needle
            + "\n"
            "    if accelerator.state.deepspeed_plugin is not None:\n"
            "        lr_scheduler = _create_scheduler(\n"
            "            args, optimizer, num_training_steps_for_scheduler\n"
            f"        )  # {marker}\n"
        )
        if needle not in text:
            return
        path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")

    @staticmethod
    def _patch_finetune_rate_metric(path):
        """Emit voir/milabench rate metrics during training logs."""
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8")
        marker = "_milabench_rate_metric"
        if marker in text:
            return
        needle = (
            "                    if args.with_tracking:\n"
            "                        accelerator.log(metrics_to_log, step=completed_steps)\n"
        )
        replacement = (
            "                    if accelerator.is_main_process:\n"
            "                        print(\n"
            '                            json.dumps(\n'
            "                                {\n"
            '                                    "task": "train",\n'
            '                                    "rate": total_tokens\n'
            "                                    / (time.perf_counter() - start_time),\n"
            '                                    "units": "items/s",\n'
            '                                    "time": time.time(),\n'
            "                                }\n"
            "                            ),\n"
            "                            flush=True,\n"
            f"                        )  # {marker}\n"
            "                    if args.with_tracking:\n"
            "                        accelerator.log(metrics_to_log, step=completed_steps)\n"
        )
        if needle not in text:
            return
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
