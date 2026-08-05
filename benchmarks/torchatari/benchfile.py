from milabench.pack import Package


class Torchatari(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        env = super().make_env()
        rom_dir = self.dirs.data / "atari_roms"
        env["ALE_ROM_FOLDER"] = str(rom_dir)
        env["ALE_ROMS"] = str(rom_dir)
        return env


__pack__ = Torchatari
