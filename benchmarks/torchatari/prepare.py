#!/usr/bin/env python
"""Stage Atari ROMs under MILABENCH_DIR_DATA for envpool / ale-py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _data_dir() -> Path:
    return Path(os.environ["MILABENCH_DIR_DATA"])


def _marker(rom_dir: Path) -> Path:
    return rom_dir / ".milabench_roms_ready"


def install_roms(rom_dir: Path) -> None:
    rom_dir.mkdir(parents=True, exist_ok=True)
    if _marker(rom_dir).exists() and any(rom_dir.glob("*.bin")):
        print(f"Atari ROMs already present at {rom_dir}")
        return

    env = os.environ.copy()
    # Prefer a stable, milabench-owned ROM location.
    env["ALE_ROM_FOLDER"] = str(rom_dir)
    env["ALE_ROMS"] = str(rom_dir)

    print(f"Downloading Atari ROMs into {rom_dir}")
    attempts = [
        [sys.executable, "-m", "AutoROM", "--accept-license", "-v", str(rom_dir)],
        [sys.executable, "-m", "AutoROM.accept_rom_license"],
        [sys.executable, "-c", "import ale_py; print(ale_py.__file__)"],
    ]
    last_err = None
    for cmd in attempts:
        try:
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, env=env)
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_err = exc
            print(f"Attempt failed: {exc}")
    else:
        # Final fallback: import path used by ale-py / gymnasium wrappers.
        try:
            import AutoROM  # noqa: F401

            subprocess.run(
                [sys.executable, "-m", "AutoROM", "--accept-license"],
                check=True,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not download Atari ROMs; install AutoROM / ale-py ROMs"
            ) from (last_err or exc)

    _marker(rom_dir).write_text("ok\n", encoding="utf-8")
    print(f"Atari ROMs ready at {rom_dir}")


def main() -> None:
    data = _data_dir()
    data.mkdir(parents=True, exist_ok=True)
    rom_dir = data / "atari_roms"
    install_roms(rom_dir)
    print("=" * 60)
    print("Prepare script completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()
