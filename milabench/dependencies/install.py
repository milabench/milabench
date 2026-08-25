"""Install-time requirement generation and pip argument building."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .platforms import PlatformConfig, deps_need_vllm
from .pin import (
    _append_vllm_index_args,
    _build_constraints_content,
    _build_index_args,
    get_constraint_file,
)
from .requirements import resolve_benchmark


@dataclass
class InstallArgs:
    """Complete set of arguments needed to install a benchmark's deps."""

    requirements_file: Path
    constraint_file: Path | None
    index_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    platform_constraint_file: Path | None = None
    _temp_files: list[Path] = field(default_factory=list, repr=False)

    def as_pip_args(self) -> list[str]:
        """Build the full pip install argument list."""
        args = ["-r", str(self.requirements_file)]
        # Platform policy (compat matrix / backend.constraints) before pin lockfile
        if self.platform_constraint_file and self.platform_constraint_file.exists():
            args.extend(["-c", str(self.platform_constraint_file)])
        if self.constraint_file and self.constraint_file.exists():
            args.extend(["-c", str(self.constraint_file)])
        args.extend(self.index_args)
        return args

    def cleanup(self):
        """Remove temporary files."""
        for f in self._temp_files:
            f.unlink(missing_ok=True)


def get_index_args(
    platform_config: PlatformConfig,
    backend: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Build index URL arguments for pip/uv install.

    Args:
        platform_config: Loaded platform configuration.
        backend: Backend name (cuda, rocm, etc.).
        overrides: Variable overrides from CLI.

    Returns:
        List of --index-url, --extra-index-url, --find-links arguments.
    """
    return _build_index_args(platform_config, backend, overrides)


def install_args(
    benchmark_path: Path | str,
    platform_config: PlatformConfig,
    backend: str,
    pin_dir: Path,
    overrides: dict[str, str] | None = None,
    unpinned: bool = False,
) -> InstallArgs:
    """Generate the complete install arguments for a benchmark.

    This is the main entry point called from pack.py install().

    Args:
        benchmark_path: Path to the benchmark directory.
        platform_config: Loaded platform configuration.
        backend: Backend name (cuda, rocm, etc.).
        pin_dir: Path to .pin/ directory.
        overrides: CLI variable overrides (e.g. {"cuda": "130", "torch": "2.12.0"}).
        unpinned: If True, skip pin lockfile (NGC/dev mode). Platform policy
            constraints from platforms.toml are still applied.

    Returns:
        InstallArgs with paths and arguments ready for pip install.
    """
    benchmark_path = Path(benchmark_path)
    all_overrides = _merge_backend_override(backend, platform_config, overrides)

    # Resolve dependencies to flat list
    resolved_deps = resolve_benchmark(
        benchmark_path, backend, platform_config, all_overrides
    )

    # Exact vLLM map applies only when this benchmark requests vllm
    vllm_mapping = None
    if deps_need_vllm(resolved_deps):
        vllm_mapping = platform_config.resolve_vllm(
            backend, all_overrides, required=True
        )

    # Write temp requirements file
    temp_req = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=f"milabench-{benchmark_path.name}-",
        delete=False,
    )
    for dep in resolved_deps:
        temp_req.write(f"{dep}\n")
    temp_req.close()
    temp_req_path = Path(temp_req.name)
    temp_files = [temp_req_path]

    # Platform policy constraints (compat matrix, backend.constraints, vLLM map)
    platform_constraint_file = None
    platform_lines = _build_constraints_content(
        platform_config, backend, all_overrides
    )
    if vllm_mapping is not None:
        platform_lines.append(vllm_mapping.as_constraint())
        platform_lines.extend(vllm_mapping.constraints)
    if platform_lines:
        temp_cons = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            prefix=f"milabench-{benchmark_path.name}-platform-",
            delete=False,
        )
        temp_cons.write("\n".join(platform_lines) + "\n")
        temp_cons.close()
        platform_constraint_file = Path(temp_cons.name)
        temp_files.append(platform_constraint_file)

    # Find matching pin lockfile
    constraint_file = None
    if not unpinned:
        import platform as plat

        variables = platform_config.resolve_vars(all_overrides)
        backend_version = variables.get(backend, "")
        torch_version = variables.get("torch", "")
        arch = plat.machine()  # x86_64, aarch64, etc.

        if backend_version and torch_version:
            # Try arch-less constraint file first (current default)
            constraint_file = get_constraint_file(
                pin_dir, backend, backend_version, torch_version
            )
            if not constraint_file.exists():
                # Fall back to arch-specific file (legacy)
                constraint_file = get_constraint_file(
                    pin_dir, backend, backend_version, torch_version, arch
                )
                if not constraint_file.exists():
                    constraint_file = None

        # The lockfile may pin a version the mapped vLLM wheel rejects
        # (nvidia-cudnn-frontend==1.27 vs vllm 0.19.1's <1.19). Drop those
        # exact pins so the mapping's ranges in the platform constraint file win.
        if (
            constraint_file is not None
            and vllm_mapping is not None
            and vllm_mapping.extra_constraint_names()
        ):
            filtered = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                prefix="milabench-vllm-pin-",
                dir=str(constraint_file.parent),
                delete=False,
            )
            filtered.close()
            filtered_path = Path(filtered.name)
            _drop_packages_from_constraint_file(
                constraint_file,
                filtered_path,
                vllm_mapping.extra_constraint_names(),
            )
            constraint_file = filtered_path
            temp_files.append(filtered_path)

    # Build index args (+ vLLM source when mapped)
    index_args = get_index_args(platform_config, backend, all_overrides)
    if vllm_mapping is not None:
        _append_vllm_index_args(index_args, vllm_mapping)

    return InstallArgs(
        requirements_file=temp_req_path,
        constraint_file=constraint_file,
        platform_constraint_file=platform_constraint_file,
        index_args=index_args,
        _temp_files=temp_files,
    )


def _drop_packages_from_constraint_file(
    src: Path,
    dest: Path,
    drop: list[str],
) -> None:
    """Copy a uv pin file, omitting ``pkg==`` blocks whose names are in drop.

    Written next to ``src`` so relative ``-c constraints.common.txt`` includes
    still resolve.
    """
    drop_names = {name.lower() for name in drop}
    skipping = False
    kept: list[str] = []
    for line in src.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "-", "--")) and "==" in stripped:
            name = stripped.split("==", 1)[0].strip().lower()
            skipping = name in drop_names
        if not skipping:
            kept.append(line)
    dest.write_text("\n".join(kept) + "\n")


def _merge_backend_override(
    backend: str,
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None,
) -> dict[str, str]:
    """Merge the backend name into overrides so it's available as a variable."""
    merged = dict(overrides) if overrides else {}
    # The backend version comes from either the override or the default vars
    # e.g., if backend="cuda" and vars has cuda="130", that's the default
    return merged
