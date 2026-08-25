"""Pin matrix iteration and constraint file generation via uv pip compile."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .platforms import (
    PlatformConfig,
    VllmMapping,
    _normalize_overrides,
    deps_need_vllm,
)
from .requirements import (
    has_toml_requirements,
    load_benchmark_requirements,
    SUPPORTED_BACKENDS,
)


def constraint_filename(backend: str, backend_version: str, torch_version: str, arch: str = "") -> str:
    """Generate the constraint filename for a (backend, torch, arch) combination.

    Convention: constraints.{backend}{version}.torch{torch_nodots}.{arch}.txt
    If arch is empty, omit it (backward compat / single-arch setups).

    Examples:
        constraint_filename("cuda", "130", "2.12.0", "x86_64") → "constraints.cuda130.torch2120.x86_64.txt"
        constraint_filename("rocm", "7.1", "2.10.0", "aarch64") → "constraints.rocm71.torch2100.aarch64.txt"
        constraint_filename("cpu", "", "2.12.0", "x86_64") → "constraints.cpu.torch2120.x86_64.txt"
        constraint_filename("cuda", "130", "2.12.0") → "constraints.cuda130.torch2120.txt"
    """
    torch_nodots = torch_version.replace(".", "")
    backend_ver_nodots = backend_version.replace(".", "")
    base = f"constraints.{backend}{backend_ver_nodots}.torch{torch_nodots}"
    if arch:
        return f"{base}.{arch}.txt"
    return f"{base}.txt"


def get_constraint_file(
    pin_dir: Path,
    backend: str,
    backend_version: str,
    torch_version: str,
    arch: str = "",
) -> Path:
    """Get the path to the constraint file for a given combination.

    Args:
        pin_dir: The .pin/ directory.
        backend: Backend name (cuda, rocm, etc.).
        backend_version: Backend version (130, 7.1, etc.).
        torch_version: Torch version (2.12.0, etc.).
        arch: CPU architecture (x86_64, aarch64). If empty, omitted from filename.

    Returns:
        Path to the constraint file.
    """
    return pin_dir / constraint_filename(backend, backend_version, torch_version, arch)


def _collect_all_toml_deps(
    benchmarks_dir: Path,
    backend: str,
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Collect all resolved dependencies from all TOML-enabled benchmarks for a backend."""
    all_deps = []
    seen = set()

    for bench_dir in sorted(benchmarks_dir.iterdir()):
        if not bench_dir.is_dir():
            continue
        if bench_dir.name.startswith(("_", ".")):
            continue
        if bench_dir.name == "retired":
            continue
        if not has_toml_requirements(bench_dir):
            continue

        reqs = load_benchmark_requirements(bench_dir)
        if not reqs.is_enabled(backend):
            continue

        deps = reqs.get_dependencies(backend)
        variables = platform_config.resolve_vars(overrides)

        for dep in deps:
            try:
                resolved = dep.format(**variables)
            except KeyError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                all_deps.append(resolved)

    return all_deps


# Wheel indexes that only exist for some torch×backend combos:
#   - GitHub "expanded_assets" 404s when that release was never published
#     (milabench/wheels does not build every combo).
#   - data.pyg.org 403s for combos PyG never shipped (e.g. cu129).
_OPTIONAL_FIND_LINKS_RES = (
    re.compile(r"^https?://github\.com/[^/]+/[^/]+/releases/expanded_assets/"),
    re.compile(r"^https?://data\.pyg\.org/whl/"),
)
_find_links_availability_cache: dict[str, bool] = {}


def _is_optional_find_links_url(url: str) -> bool:
    """True for find-links that should be omitted when the URL is missing."""
    return any(pattern.match(url) for pattern in _OPTIONAL_FIND_LINKS_RES)


def _find_links_url_available(url: str, timeout: float = 10.0) -> bool:
    """Return True if ``url`` responds without HTTP 404.

    Results are cached for the process lifetime so pin/install across many
    benchmarks does not re-probe the same release page.
    """
    if url in _find_links_availability_cache:
        return _find_links_availability_cache[url]

    import urllib.error
    import urllib.request

    available = False
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "milabench"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            available = 200 <= getattr(resp, "status", 200) < 400
    except urllib.error.HTTPError as e:
        # Some hosts disallow HEAD; retry GET on 405/403 before giving up.
        if e.code in (403, 405):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "milabench"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    available = 200 <= getattr(resp, "status", 200) < 400
            except Exception:
                available = False
        else:
            available = False
    except Exception:
        available = False

    _find_links_availability_cache[url] = available
    return available


def _build_index_args(
    platform_config: PlatformConfig,
    backend: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Build --index-url, --extra-index-url, --find-links arguments.

    Combo-specific wheel indexes (GitHub ``releases/expanded_assets/...``,
    ``data.pyg.org``) are only included when the page exists. Missing ones fall
    back to the normal package indexes instead of failing resolution.
    """
    backend_config = platform_config.get_backend(backend)
    variables = platform_config.resolve_vars(overrides)
    args = []

    idx = backend_config.indexes
    if idx.index_url:
        resolved_url = idx.index_url.format(**variables)
        args.extend(["--index-url", resolved_url])

    for url in idx.extra_index_url:
        resolved_url = url.format(**variables)
        args.extend(["--extra-index-url", resolved_url])

    for url in idx.find_links:
        resolved_url = url.format(**variables)
        if _is_optional_find_links_url(resolved_url) and not _find_links_url_available(
            resolved_url
        ):
            print(
                f"Skipping missing find-links (not published): {resolved_url}",
                flush=True,
            )
            continue
        args.extend(["--find-links", resolved_url])

    return args


def _build_constraints_content(
    platform_config: PlatformConfig,
    backend: str,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    """Build constraint lines from platform_config [backend.constraints] and [compat.*]."""
    backend_config = platform_config.get_backend(backend)
    variables = platform_config.resolve_vars(overrides)
    lines = []

    for pkg, version_spec in backend_config.constraints.items():
        resolved_spec = version_spec.format(**variables)
        lines.append(f"{pkg}{resolved_spec}")

    # Append compat-derived constraints (scoped to the active backend)
    lines.extend(_resolve_compat_constraints(platform_config, overrides, backend=backend))

    return lines


def _normalize_backend_version(raw: str) -> str:
    """Normalize compact backend versions for PEP 440 comparison.

    "130" -> "13.0", "126" -> "12.6". Already-dotted values pass through.
    """
    if "." in raw or not raw:
        return raw
    return f"{raw[:-1]}.{raw[-1]}"


# Accelerator keys that identify the active pin/install backend.
_COMPAT_ACCEL_KEYS = ("cuda", "rocm", "hpu", "xpu", "cpu")


def _resolve_compat_constraints(
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None = None,
    backend: str | None = None,
) -> list[str]:
    """Evaluate [compat.*] rules against known pin-time variables.

    Returns constraint lines like 'torchao<0.18' for each package
    whose first matching rule applies.

    When ``backend`` is set (e.g. ``"rocm"``), only that accelerator's version
    is visible to conditions — plus ``torch``. Defaults for other accelerators
    in ``[vars]`` are ignored so rules like ``torch>=2.12,rocm>=7`` apply only
    while pinning/installing ROCm, not CUDA.
    """
    from packaging.version import Version

    if not platform_config.compat:
        return []

    variables = platform_config.resolve_vars(overrides)
    if backend:
        keys = ["torch"]
        if backend in _COMPAT_ACCEL_KEYS:
            keys.append(backend)
    else:
        keys = ["torch", *_COMPAT_ACCEL_KEYS]

    known_versions: dict[str, Version] = {}
    for key in keys:
        raw = variables.get(key, "")
        if not raw:
            continue
        normalized = _normalize_backend_version(raw) if key != "torch" else raw
        try:
            known_versions[key] = Version(normalized)
        except Exception:
            pass

    lines = []
    for pkg, entry in platform_config.compat.items():
        for rule in entry.rules:
            if _compat_conditions_match(rule.conditions, known_versions):
                try:
                    constraint = rule.constraint.format(**variables)
                except KeyError as e:
                    raise ValueError(
                        f"Variable {e} used in [compat.{pkg}] constraint "
                        f"'{rule.constraint}' is not defined in platforms.toml "
                        f"[vars] or CLI overrides. Available: {list(variables.keys())}"
                    ) from e
                lines.append(f"{pkg}{constraint}")
                break  # first match wins
    return lines


def _compat_conditions_match(conditions_str: str, known: dict) -> bool:
    """Check if all conditions in 'torch>=2.11,cuda>=13' match known versions."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    for cond in conditions_str.split(","):
        cond = cond.strip()
        match = re.match(r"([a-zA-Z0-9_-]+)(.*)", cond)
        if not match:
            return False
        name, spec_str = match.group(1), match.group(2)
        if name not in known:
            return False
        if known[name] not in SpecifierSet(spec_str):
            return False
    return True


def _requirement_name(line: str) -> str:
    """Return the distribution name from a requirement/constraint line."""
    return re.split(r"[<>=!~;\[]", line.strip(), maxsplit=1)[0].strip().lower()


def _is_torch_requirement(line: str) -> bool:
    """True if a requirements/constraint line pins the ``torch`` distribution."""
    return _requirement_name(line) == "torch"


def _is_vllm_requirement(line: str) -> bool:
    """True if a requirements/constraint line refers to the ``vllm`` distribution."""
    return _requirement_name(line) == "vllm"


def _append_vllm_index_args(args: list[str], mapping: VllmMapping) -> list[str]:
    """Append the verified vLLM source from the exact mapping table."""
    args.extend(mapping.as_index_args())
    return args


def _vllm_wheel_note_lines(mapping: VllmMapping) -> list[str]:
    """Comment block recording the install-time vLLM wheel (not a lock pin)."""
    lines = [
        "# vllm is not solved in this lockfile; install uses the mapped wheel:",
        f"#   {mapping.as_constraint()}",
    ]
    if mapping.find_links:
        lines.append(f"#   --find-links {mapping.find_links}")
    if mapping.extra_index_url:
        lines.append(f"#   --extra-index-url {mapping.extra_index_url}")
    return lines


def _append_vllm_wheel_note(path: Path, mapping: VllmMapping) -> None:
    """Insert the vLLM wheel note after the pin-file header comments."""
    note = "".join(f"{line}\n" for line in _vllm_wheel_note_lines(mapping))
    lines = path.read_text().splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            insert_at = i
            break
        insert_at = i + 1
    path.write_text("".join(lines[:insert_at]) + note + "".join(lines[insert_at:]))


def _torch_minor_pin(torch_version: str) -> str:
    """Return a minor-version torch constraint for the pin matrix entry.

    For ``2.12.1`` → ``torch>=2.12,<2.13`` so newer patches in that minor
    can resolve, while still blocking the next minor (e.g. 2.13).

    Used only when the backend has no local version tag (hpu/xpu). CUDA,
    ROCm, and CPU pins must use ``_torch_pin`` so uv cannot pick an
    untagged PyPI wheel (PEP 440: ``2.10.0`` > ``2.10.0+cu130``).
    """
    parts = torch_version.split(".")
    if len(parts) < 2:
        raise ValueError(f"Expected torch major.minor[.patch], got {torch_version!r}")
    major, minor = int(parts[0]), int(parts[1])
    return f"torch>={major}.{minor},<{major}.{minor + 1}"


def _torch_local_label(backend: str, backend_version: str) -> str | None:
    """Local version tag for a PyTorch wheel on this backend.

    Matches the tags on download.pytorch.org: ``cu130``, ``rocm7.1``, ``cpu``.
    """
    if backend == "cuda" and backend_version:
        return f"cu{backend_version}"
    if backend == "rocm" and backend_version:
        return f"rocm{backend_version}"
    if backend == "cpu":
        return "cpu"
    return None


def _torch_pin(torch_version: str, backend: str, backend_version: str) -> str:
    """Constraint that keeps uv on the backend's torch wheel.

    A minor range like ``torch>=2.10,<2.11`` also matches the untagged
    PyPI build. That wheel depends on ``nvidia-*-cu12`` and, with
    ``--index-strategy unsafe-best-match``, wins over ``2.10.0+cu130``.
    """
    local = _torch_local_label(backend, backend_version)
    if local:
        return f"torch=={torch_version}+{local}"
    return _torch_minor_pin(torch_version)


# Torch's default (untagged) wheel pulls these. They must not appear in a
# CUDA 13.x pin next to ``nvidia-cuda-runtime==13.*`` / ``nvidia-*-cu13``.
# ``nvidia-cutlass-dsl-libs-cu12`` is a different naming scheme and is OK.
_CUDA13_FORBIDDEN_CU12 = frozenset(
    {
        "nvidia-cublas-cu12",
        "nvidia-cuda-cupti-cu12",
        "nvidia-cuda-nvrtc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cufft-cu12",
        "nvidia-cufile-cu12",
        "nvidia-curand-cu12",
        "nvidia-cusolver-cu12",
        "nvidia-cusparse-cu12",
        "nvidia-cusparselt-cu12",
        "nvidia-nccl-cu12",
        "nvidia-nvjitlink-cu12",
        "nvidia-nvshmem-cu12",
        "nvidia-nvtx-cu12",
    }
)


def _iter_pinned_packages(text: str):
    """Yield ``(name, version)`` for each ``pkg==ver`` pin line."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        if "==" not in stripped:
            continue
        name, version = stripped.split("==", 1)
        yield name.strip().lower(), version.strip()


def _validate_pinned_constraint_file(
    path: Path,
    backend: str,
    backend_version: str,
    torch_version: str,
) -> None:
    """Reject a pin that resolved the wrong torch wheel or mixed CUDA stacks."""
    from packaging.version import Version

    pins = dict(_iter_pinned_packages(path.read_text()))
    torch_ver = pins.get("torch")
    if not torch_ver:
        raise RuntimeError(f"Pin file {path} has no torch== pin")

    local = _torch_local_label(backend, backend_version)
    if local:
        expected = f"{torch_version}+{local}"
        if torch_ver != expected:
            raise RuntimeError(
                f"Pin file {path} resolved torch=={torch_ver}, "
                f"expected torch=={expected}. Untagged PyPI torch wheels "
                f"pull the default CUDA 12 NVIDIA stack and mix with "
                f"+{local} torchvision/vllm."
            )

    if backend == "cuda" and backend_version:
        try:
            cuda_ver = Version(_normalize_backend_version(backend_version))
        except Exception:
            cuda_ver = None
        if cuda_ver is not None and cuda_ver >= Version("13"):
            mixed = sorted(name for name in pins if name in _CUDA13_FORBIDDEN_CU12)
            if mixed:
                raise RuntimeError(
                    f"Pin file {path} mixes CUDA 12 NVIDIA packages {mixed} "
                    f"into a cuda={backend_version} pin. Usually caused by "
                    f"resolving untagged torch=={torch_version} from PyPI."
                )


# Guards `_ensure_build_backends` so we only seed the ambient env once per
# process, no matter how many combinations pin_all iterates over.
_BUILD_BACKENDS_SEEDED = False


def _ensure_build_backends(platform_config: PlatformConfig) -> None:
    """Install ``[build] requires`` into the ambient environment before compiling.

    ``pin_combination`` runs ``uv pip compile --no-build-isolation`` so that
    torch-ecosystem source dists (e.g. the ``torchtitan`` git dep) build against
    the torch already present instead of re-downloading it into an isolated
    build env. The trade-off is that their build backends (setuptools, wheel,
    hatchling, …) must already be importable in *this* environment, otherwise
    ``prepare_metadata_for_build_wheel`` fails with e.g.
    ``ModuleNotFoundError: No module named 'setuptools'``.

    Install-time solves this via ``install_requires`` in ``pack.py``; pinning
    needs the same pre-seed. Configured through platforms.toml's ``[build]
    requires`` (``PlatformConfig.build_requires``). No-op when unset/empty or
    once already seeded for this process.
    """
    global _BUILD_BACKENDS_SEEDED
    if _BUILD_BACKENDS_SEEDED:
        return

    build_requires = getattr(platform_config, "build_requires", None) or []
    if not build_requires:
        _BUILD_BACKENDS_SEEDED = True
        return

    print(
        f"Seeding build backends (--no-build-isolation): {', '.join(build_requires)}",
        flush=True,
    )
    result = subprocess.run(
        ["uv", "pip", "install", *build_requires],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install [build] requires before pinning "
            f"({', '.join(build_requires)}):\n{result.stderr}"
        )

    _BUILD_BACKENDS_SEEDED = True


async def pin_combination(
    platform_config: PlatformConfig,
    backend: str,
    backend_version: str,
    torch_version: str,
    benchmarks_dir: Path,
    pin_dir: Path,
    from_scratch: bool = False,
    extra_compile_args: list[str] | None = None,
    arch: str = "",
) -> Path:
    """Pin one (backend, torch, arch) combination.

    Collects all TOML-declared deps for the given backend, runs uv pip compile,
    and writes the shared constraint file.

    Args:
        platform_config: Loaded platform configuration.
        backend: Backend name (cuda, rocm, cpu).
        backend_version: Backend version (130, 7.1, "").
        torch_version: Torch version (2.12.0).
        benchmarks_dir: Path to benchmarks/ directory.
        pin_dir: Path to .pin/ directory.
        from_scratch: If True, delete existing constraint file first.
        extra_compile_args: Additional args for uv pip compile.
        arch: CPU architecture (x86_64, aarch64). If empty, omitted from filename.

    Returns:
        Path to the generated constraint file.
    """
    overrides = {backend: backend_version, "torch": torch_version}
    output_file = get_constraint_file(pin_dir, backend, backend_version, torch_version, arch)

    if from_scratch and output_file.exists():
        output_file.unlink()

    pin_dir.mkdir(parents=True, exist_ok=True)

    # `uv pip compile --no-build-isolation` (below) builds source dists using
    # this environment's build backends, so make sure they're present first.
    _ensure_build_backends(platform_config)

    # Collect all deps from TOML-enabled benchmarks
    all_deps = _collect_all_toml_deps(
        benchmarks_dir, backend, platform_config, overrides
    )

    if not all_deps:
        raise RuntimeError(
            f"No TOML dependencies found for backend={backend}, "
            f"version={backend_version}, torch={torch_version}"
        )

    # Never compile vLLM into the shared lockfile. Its extras pull jax/nvidia
    # stacks that mix CUDA generations into every benchmark pin. Install uses
    # the exact wheel from platforms.toml [vllm.*] instead.
    vllm_mapping = None
    if deps_need_vllm(all_deps):
        vllm_mapping = platform_config.resolve_vllm(
            backend, overrides, required=False
        )
        all_deps = [dep for dep in all_deps if not _is_vllm_requirement(dep)]
        if vllm_mapping is None:
            print(
                f"Skipping vLLM for {backend}={backend_version} torch={torch_version} "
                f"(no exact mapping in platforms.toml)",
                flush=True,
            )
        else:
            print(
                f"vLLM omitted from shared pin; install uses "
                f"{vllm_mapping.as_constraint()}",
                flush=True,
            )
        if not all_deps:
            raise RuntimeError(
                f"No TOML dependencies left for backend={backend}, "
                f"version={backend_version}, torch={torch_version} "
                f"after omitting vLLM from the shared pin"
            )

    # Build index URL arguments (vLLM find-links are install-only)
    index_args = _build_index_args(platform_config, backend, overrides)

    # Build platform constraint lines (vLLM version is install-only)
    constraint_lines = _build_constraints_content(platform_config, backend, overrides)

    # Exact local-version pin (torch==2.10.0+cu130). A minor range also matches
    # untagged PyPI torch, which PEP 440 ranks above +cu130 and which pulls
    # nvidia-*-cu12 into a CUDA 13 constraint file.
    torch_pin = _torch_pin(torch_version, backend, backend_version)
    constraint_lines = [
        line for line in constraint_lines if not _is_torch_requirement(line)
    ]
    constraint_lines.insert(0, torch_pin)

    # Write temporary input files
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="toml-deps-", delete=False
    ) as deps_file:
        deps_file.write(f"# Auto-generated from requirements.toml files\n")
        deps_file.write(f"# Backend: {backend}={backend_version}, torch={torch_version}\n")
        for dep in all_deps:
            deps_file.write(f"{dep}\n")
        deps_path = Path(deps_file.name)

    constraints_path = None
    if constraint_lines:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="toml-constraints-", delete=False
        ) as cons_file:
            for line in constraint_lines:
                cons_file.write(f"{line}\n")
            constraints_path = Path(cons_file.name)

    try:
        cmd = [
            "uv", "pip", "compile",
            "--no-build-isolation",
            "--index-strategy", "unsafe-best-match",
            "-o", str(output_file),
            *index_args,
        ]

        if constraints_path:
            cmd.extend(["-c", str(constraints_path)])

        if extra_compile_args:
            cmd.extend(extra_compile_args)

        cmd.append(str(deps_path))

        # Add header comment to output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(benchmarks_dir.parent),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"uv pip compile failed for {backend}={backend_version} torch={torch_version}:\n"
                f"{result.stderr}"
            )

        # Post-process: strip index URLs from output (we inject at install time)
        _strip_index_urls_from_constraint_file(output_file, backend, backend_version, torch_version, arch)
        _validate_pinned_constraint_file(
            output_file, backend, backend_version, torch_version
        )
        if vllm_mapping is not None:
            _append_vllm_wheel_note(output_file, vllm_mapping)

    finally:
        deps_path.unlink(missing_ok=True)
        if constraints_path:
            constraints_path.unlink(missing_ok=True)

    return output_file


def _strip_index_urls_from_constraint_file(
    path: Path,
    backend: str,
    backend_version: str,
    torch_version: str,
    arch: str = "",
) -> None:
    """Remove --index-url/--extra-index-url/--find-links lines and normalize temp paths.

    Constraint files should be pure version pins. Index URLs are injected at install time.
    Temp file paths (/tmp/toml-deps-XXXX.txt, /tmp/toml-constraints-XXXX.txt) are
    normalized to stable names so re-pinning doesn't produce noisy git diffs.
    """
    lines = path.read_text().splitlines()
    filtered = []
    header_added = False

    label = f"{backend}={backend_version}" if backend_version else backend
    arch_label = f" arch={arch}" if arch else ""
    header_line = f"# Pinned with: {label} torch={torch_version}{arch_label}"

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("--index-url", "--extra-index-url", "--find-links")):
            continue
        if stripped.startswith("# This file was autogenerated"):
            if not header_added:
                filtered.append(header_line)
                filtered.append(f"# Generated by: milabench pin")
                header_added = True
            continue
        # Remove the uv pip compile command line (contains machine-specific paths)
        if stripped.startswith("#") and "uv pip compile" in stripped:
            continue
        # Normalize temp file paths to stable names
        line = re.sub(r"/tmp/toml-deps-[a-z0-9_]+\.txt", "requirements.in", line)
        line = re.sub(r"/tmp/toml-constraints-[a-z0-9_]+\.txt", "constraints.in", line)
        filtered.append(line)

    if not header_added:
        filtered.insert(0, header_line)
        filtered.insert(1, f"# Generated by: milabench pin")

    path.write_text("\n".join(filtered) + "\n")


def _filter_combinations(
    combinations: list[tuple[str, str, str, str]],
    overrides: dict[str, str] | None,
) -> list[tuple[str, str, str, str]]:
    """Keep only combos matching CLI ``--set`` keys (cuda, torch, backend, …).

    ``--set`` is otherwise ignored by discovery, which is why
    ``--set cuda=130 --set torch=2.10.0`` used to pin the full matrix.
    Only keys the user passed are filters; platforms.toml ``[vars]`` defaults
    do not restrict the matrix.
    """
    if not overrides:
        return combinations

    normalized = _normalize_overrides(overrides)
    torch_filter = normalized.get("torch") if "torch" in overrides else None

    accel_filters: dict[str, str] = {}
    if "backend" in overrides:
        for key in _COMPAT_ACCEL_KEYS:
            if key in normalized:
                accel_filters[key] = normalized[key]
    for key in _COMPAT_ACCEL_KEYS:
        if key in overrides:
            accel_filters[key] = normalized[key]

    if torch_filter is None and not accel_filters:
        return combinations

    def keep(combo: tuple[str, str, str, str]) -> bool:
        backend, backend_version, torch_version, _arch = combo
        if torch_filter is not None and torch_version != torch_filter:
            return False
        if accel_filters:
            if backend not in accel_filters:
                return False
            wanted = accel_filters[backend]
            if wanted and backend_version != wanted:
                return False
        return True

    filtered = [combo for combo in combinations if keep(combo)]
    if not filtered:
        wanted = " ".join(f"{k}={v}" for k, v in overrides.items())
        raise RuntimeError(
            f"No pin combinations match --set {wanted}. "
            "Check that this (backend, torch) pair exists on the torch wheel index."
        )
    return filtered


def _get_combinations(
    platform_config: PlatformConfig,
    overrides: dict[str, str] | None = None,
) -> list[tuple[str, str, str, str]]:
    """Get pin combinations from discovery (preferred) or static matrix (legacy).

    Discovery fetches the PyTorch wheel index at runtime to find available
    (backend, backend_version, torch_version, arch) tuples.

    When ``overrides`` contains ``--set`` keys (``cuda``, ``torch``,
    ``backend``, …), only matching combinations are returned.

    Returns:
        List of (backend_name, backend_version, torch_version, arch) tuples.
        For legacy matrix, arch is "" (single-arch mode).
    """
    if platform_config.discovery is not None:
        from .discovery import discover_combinations, print_discovered_combinations
        from .discovery import DiscoveryConfig as DiscConfig

        dc = DiscConfig(
            torch_index=platform_config.discovery.torch_index,
            torch_min=platform_config.discovery.torch_min,
            backends=platform_config.discovery.backends,
            python=platform_config.discovery.python,
            platforms=platform_config.discovery.platforms,
            latest_patch_only=platform_config.discovery.latest_patch_only,
        )
        combos = _filter_combinations(discover_combinations(dc), overrides)
        print_discovered_combinations(combos)
        return combos

    if platform_config.pin_matrix is not None:
        # Legacy matrix doesn't have arch — use empty string
        combos = [
            (b, v, t, "") for b, v, t in platform_config.pin_matrix.combinations()
        ]
        return _filter_combinations(combos, overrides)

    raise RuntimeError(
        "No [pin.discovery] or [pin.matrix] section in platforms.toml. "
        "Cannot determine which combinations to pin."
    )


async def pin_all(
    platform_config: PlatformConfig,
    benchmarks_dir: Path,
    pin_dir: Path,
    from_scratch: bool = False,
    extra_compile_args: list[str] | None = None,
    dry_run: bool = False,
    overrides: dict[str, str] | None = None,
) -> list[Path]:
    """Pin all discovered or matrix-defined combinations.

    Prefers [pin.discovery] (fetches the torch wheel index at runtime).
    Falls back to [pin.matrix] (static cartesian product) if discovery is not configured.

    Args:
        platform_config: Loaded platform configuration.
        benchmarks_dir: Path to benchmarks/ directory.
        pin_dir: Path to .pin/ directory.
        from_scratch: If True, delete existing constraint files first.
        extra_compile_args: Additional args for uv pip compile.
        dry_run: If True, print discovered combinations without pinning.
        overrides: CLI ``--set`` filters (e.g. ``{"cuda": "130", "torch": "2.10.0"}``).
            Restricts which combinations are pinned; omitted keys stay unconstrained.

    Returns:
        List of generated constraint file paths (empty if dry_run).
    """
    combinations = _get_combinations(platform_config, overrides=overrides)

    if not combinations:
        raise RuntimeError("No valid combinations found to pin.")

    if dry_run:
        return []

    # Deduplicate by (backend, version, torch) — arch doesn't affect resolution
    seen = set()
    unique_combinations = []
    for backend_name, backend_version, torch_version, arch in combinations:
        key = (backend_name, backend_version, torch_version)
        if key not in seen:
            seen.add(key)
            unique_combinations.append((backend_name, backend_version, torch_version))

    print(f"\nPinning {len(unique_combinations)} unique combinations "
          f"(deduplicated from {len(combinations)} arch variants)\n")

    results = []
    errors = []
    delay_between_pins = float(os.environ.get("MILABENCH_PIN_DELAY", "2"))

    for i, (backend_name, backend_version, torch_version) in enumerate(unique_combinations):
        label = f"{backend_name}={backend_version}" if backend_version else backend_name
        print(f"Pinning [{i+1}/{len(unique_combinations)}]: {label} torch={torch_version}")
        try:
            path = await pin_combination(
                platform_config=platform_config,
                backend=backend_name,
                backend_version=backend_version,
                torch_version=torch_version,
                benchmarks_dir=benchmarks_dir,
                pin_dir=pin_dir,
                from_scratch=from_scratch,
                extra_compile_args=extra_compile_args,
            )
            results.append(path)
            print(f"  → {path}")
        except Exception as exc:
            errors.append((label, torch_version, exc))
            print(f"  ✗ FAILED: {exc}")

        if delay_between_pins > 0 and i < len(unique_combinations) - 1:
            await asyncio.sleep(delay_between_pins)

    if errors:
        print(f"\n{len(errors)}/{len(unique_combinations)} combinations failed:")
        for label, tv, exc in errors:
            print(f"  - {label} torch={tv}: {exc}")

    if len(results) > 1:
        _extract_common_constraints(results, pin_dir)

    return results


@dataclass
class _PinBlock:
    """A package pin with its associated '# via' annotations (which follow it)."""
    package_line: str
    via_comments: list[str] = field(default_factory=list)

    @property
    def package_name(self) -> str:
        """Extract package name (before ==) for comparison."""
        return self.package_line.split("==")[0].strip()

    def as_text(self) -> str:
        """Render the block as text (package line + trailing via comments)."""
        return "\n".join([self.package_line] + self.via_comments)


def _parse_constraint_file(path: Path) -> tuple[list[str], list[_PinBlock]]:
    """Parse a uv pip compile constraint file into header lines and pin blocks.

    In uv pip compile output, each package entry looks like:
        package-name==1.2.3
            # via
            #   some-dep
            #   another-dep

    The '# via' comments FOLLOW the package they describe.

    Returns:
        (file_header, blocks) where file_header is the top-of-file comments
        (before the first package) and blocks are the package entries with
        their trailing via comments.
    """
    lines = path.read_text().splitlines()
    file_header: list[str] = []
    blocks: list[_PinBlock] = []
    current_block: _PinBlock | None = None
    found_first_package = False

    for line in lines:
        stripped = line.strip()
        is_via_comment = stripped.startswith("#") and found_first_package

        if not found_first_package:
            if stripped and not stripped.startswith("#"):
                found_first_package = True
                current_block = _PinBlock(package_line=stripped)
            else:
                file_header.append(line)
        elif not stripped or is_via_comment:
            if current_block is not None:
                current_block.via_comments.append(line)
        else:
            if current_block is not None:
                blocks.append(current_block)
            current_block = _PinBlock(package_line=stripped)

    if current_block is not None:
        blocks.append(current_block)

    return file_header, blocks


def _extract_common_constraints(constraint_files: list[Path], pin_dir: Path) -> None:
    """Extract packages common to ALL constraint files into constraints.common.txt.

    Each individual file is then rewritten to contain only its unique packages
    plus a `-c constraints.common.txt` reference. Comments (# via annotations)
    are kept together with their package.
    """
    common_file = pin_dir / "constraints.common.txt"

    # Parse all files into blocks
    file_data: dict[Path, tuple[list[str], list[_PinBlock]]] = {}
    for path in constraint_files:
        file_data[path] = _parse_constraint_file(path)

    # Build package_line → block mapping per file, and find common package lines
    all_package_sets: list[set[str]] = []
    for path, (header, blocks) in file_data.items():
        all_package_sets.append({b.package_line for b in blocks})

    common_lines = set.intersection(*all_package_sets)

    if not common_lines:
        print("\nNo common packages across constraint files.")
        return

    total_per_file = len(all_package_sets[0])
    print(f"\nExtracted {len(common_lines)}/{total_per_file} common packages "
          f"→ {common_file.name}")

    # For the common file, take blocks from the first file (comments are representative)
    first_header, first_blocks = file_data[next(iter(constraint_files))]
    common_blocks = sorted(
        [b for b in first_blocks if b.package_line in common_lines],
        key=lambda b: b.package_name.lower(),
    )

    # Write common file
    common_parts = [
        "# Common pinned packages shared across all backend/torch combinations",
        "# Auto-generated by: milabench pin",
        "",
    ]
    for block in common_blocks:
        common_parts.append(block.as_text())
    common_file.write_text("\n".join(common_parts) + "\n")

    # Rewrite each constraint file with only its unique packages + reference to common
    for path, (header, blocks) in file_data.items():
        unique_blocks = sorted(
            [b for b in blocks if b.package_line not in common_lines],
            key=lambda b: b.package_name.lower(),
        )
        # Keep header but strip any old -c references
        clean_header = [h for h in header if not h.strip().startswith("-c ")]

        parts = clean_header + [f"-c {common_file.name}", ""]
        for block in unique_blocks:
            parts.append(block.as_text())
        path.write_text("\n".join(parts) + "\n")
