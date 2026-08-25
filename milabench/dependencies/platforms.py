"""Load and validate platforms.toml, resolve variables."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


_BACKEND_PREFIXES = {
    "cu": "cuda",
    "cuda": "cuda",
    "rocm": "rocm",
    "cpu": "cpu",
    "hpu": "hpu",
    "xpu": "xpu",
}


def _normalize_overrides(overrides: dict[str, str]) -> dict[str, str]:
    """Normalize CLI overrides.

    Handles:
      - backend=cu130 → cuda=130
      - backend=rocm7.2 → rocm=7.2
      - backend=cpu → (no version needed)
      - torch=2.10 → torch=2.10.0 (ensure 3-part version)
    """
    result = dict(overrides)

    # Parse "backend" shorthand into backend_type + version
    if "backend" in result:
        raw = result.pop("backend")
        match = re.match(r"([a-zA-Z]+)(.*)", raw)
        if match:
            prefix, version = match.group(1).lower(), match.group(2)
            backend_name = _BACKEND_PREFIXES.get(prefix, prefix)
            if version:
                result.setdefault(backend_name, version)
        else:
            result.setdefault("cpu", "")

    # Normalize torch version to 3 parts (2.10 → 2.10.0)
    if "torch" in result:
        parts = result["torch"].split(".")
        if len(parts) == 2:
            result["torch"] = f"{result['torch']}.0"

    return result


@dataclass
class VllmMapping:
    """Exact vLLM package version and wheel source for one platform pair."""

    version: str
    find_links: str | None = None
    extra_index_url: str | None = None
    # Extra constraints the rest of the tree must honor so the mapped wheel
    # can install without compiling vLLM into the shared lockfile.
    constraints: list[str] = field(default_factory=list)

    def as_constraint(self) -> str:
        return f"vllm=={self.version}"

    def as_index_args(self) -> list[str]:
        args: list[str] = []
        if self.extra_index_url:
            args.extend(["--extra-index-url", self.extra_index_url])
        if self.find_links:
            args.extend(["--find-links", self.find_links])
        return args

    def extra_constraint_names(self) -> list[str]:
        """Package names from :attr:`constraints` (e.g. nvidia-cudnn-frontend)."""
        names: list[str] = []
        for line in self.constraints:
            name = re.split(r"[<>=!~;\[]", line.strip(), maxsplit=1)[0].strip().lower()
            if name:
                names.append(name)
        return names


@dataclass
class PlatformConfig:
    """Parsed platforms.toml with resolved variable interpolation."""

    vars: dict[str, str] = field(default_factory=dict)
    pin_matrix: PinMatrix | None = None
    discovery: DiscoveryConfig | None = None
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    compat: dict[str, CompatEntry] = field(default_factory=dict)
    # Build backends pre-seeded into every install_group venv before
    # installing actual requirements (see install_requires() in pack.py).
    # None (as opposed to []) means "[build] wasn't in platforms.toml" --
    # callers fall back to their own hardcoded default in that case, so an
    # explicit `requires = []` can still mean "seed nothing".
    build_requires: list[str] | None = None
    # backend → {(torch, backend_version): VllmMapping}
    vllm_maps: dict[str, dict[tuple[str, str], VllmMapping]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def resolve_vars(self, overrides: dict[str, str] | None = None) -> dict[str, str]:
        """Return vars dict with CLI overrides applied.

        Normalizes overrides first:
          - backend=cu130 → cuda=130
          - backend=rocm7.2 → rocm=7.2
          - torch=2.10 → torch=2.10.0

        Also injects derived variables:
          - cuda_major: first 2 chars of cuda version (e.g. "130" → "13", "126" → "12")
          - torch_short: major.minor of torch (e.g. "2.10.0" → "2.10")

        Explicit ``vllm=...`` overrides are rejected; vLLM is selected by the
        exact ``(backend, backend_version, torch)`` mapping in platforms.toml.
        """
        overrides = _normalize_overrides(overrides) if overrides else {}
        if "vllm" in overrides:
            raise ValueError(
                "vLLM is selected by the (backend, torch) mapping in platforms.toml; "
                "do not pass --set vllm=... . Set torch and cuda/rocm instead."
            )

        resolved = dict(self.vars)
        resolved.update(overrides)

        # Derived variables
        if "cuda" in resolved and "cuda_major" not in resolved:
            resolved["cuda_major"] = resolved["cuda"][:2]

        # torch_short: major.minor for release tags (2.10.0 → 2.10)
        if "torch" in resolved and "torch_short" not in resolved:
            parts = resolved["torch"].split(".")
            if len(parts) >= 2:
                resolved["torch_short"] = f"{parts[0]}.{parts[1]}"
            else:
                resolved["torch_short"] = resolved["torch"]

        return resolved

    def lookup_vllm(
        self,
        backend: str,
        backend_version: str,
        torch_version: str,
    ) -> VllmMapping:
        """Return the exact vLLM mapping for ``(backend, backend_version, torch)``.

        Raises:
            ValueError: if the pair is not in the mapping tables.
        """
        backend_maps = self.vllm_maps.get(backend, {})
        key = (torch_version, backend_version)
        mapping = backend_maps.get(key)
        if mapping is not None:
            return mapping

        supported = self.supported_vllm_pairs(backend)
        supported_txt = ", ".join(supported) if supported else "(none)"
        raise ValueError(
            f"No vLLM mapping for {backend}={backend_version} torch={torch_version}. "
            f"Supported {backend} pairs: {supported_txt}"
        )

    def try_lookup_vllm(
        self,
        backend: str,
        backend_version: str,
        torch_version: str,
    ) -> VllmMapping | None:
        """Like :meth:`lookup_vllm` but returns None when the pair is unmapped."""
        return self.vllm_maps.get(backend, {}).get((torch_version, backend_version))

    def resolve_vllm(
        self,
        backend: str,
        overrides: dict[str, str] | None = None,
        *,
        required: bool = True,
    ) -> VllmMapping | None:
        """Resolve the exact vLLM mapping for the active backend/torch pair."""
        variables = self.resolve_vars(overrides)
        backend_version = variables.get(backend, "")
        torch_version = variables.get("torch", "")
        if not backend_version or not torch_version:
            if required:
                raise ValueError(
                    f"vLLM mapping requires backend version and torch; got "
                    f"{backend}={backend_version!r} torch={torch_version!r}"
                )
            return None
        if required:
            return self.lookup_vllm(backend, backend_version, torch_version)
        return self.try_lookup_vllm(backend, backend_version, torch_version)

    def supported_vllm_pairs(self, backend: str | None = None) -> list[str]:
        """Human-readable supported ``torch,backend_version`` pairs."""
        backends = [backend] if backend else sorted(self.vllm_maps)
        lines: list[str] = []
        for name in backends:
            for torch_ver, backend_ver in sorted(self.vllm_maps.get(name, {})):
                if backend:
                    lines.append(f"{torch_ver},{backend_ver}")
                else:
                    lines.append(f"{name}:{torch_ver},{backend_ver}")
        return lines

    def resolve_string(self, template: str, overrides: dict[str, str] | None = None) -> str:
        """Interpolate {var} placeholders in a string."""
        variables = self.resolve_vars(overrides)
        return template.format(**variables)

    def get_backend(self, name: str) -> BackendConfig:
        if name not in self.backends:
            # Return a default backend config (pypi only, no constraints)
            # This allows backends discovered from the index that aren't
            # explicitly configured in platforms.toml to still work.
            return BackendConfig(name=name)
        return self.backends[name]


def deps_need_vllm(deps: list[str]) -> bool:
    """True if any dependency line requests the ``vllm`` package."""
    for dep in deps:
        name = re.split(r"[<>=!~;\[]", dep.strip(), maxsplit=1)[0].strip().lower()
        if name == "vllm":
            return True
    return False


@dataclass
class BackendConfig:
    """Configuration for a single backend (cuda, rocm, etc.)."""

    name: str
    indexes: IndexConfig = field(default_factory=lambda: IndexConfig())
    constraints: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, OverrideConfig] = field(default_factory=dict)


@dataclass
class IndexConfig:
    """Index URL configuration for a backend."""

    index_url: str = "https://pypi.org/simple"
    extra_index_url: list[str] = field(default_factory=list)
    find_links: list[str] = field(default_factory=list)


@dataclass
class OverrideConfig:
    """Override configuration for a package that needs special install handling."""

    package: str
    install: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class CompatRule:
    """A single condition -> constraint mapping from [compat.*]."""

    conditions: str  # e.g. "torch>=2.11,cuda>=13"
    constraint: str  # e.g. "<0.18"


@dataclass
class CompatEntry:
    """Compat rules for one package."""

    package: str
    rules: list[CompatRule] = field(default_factory=list)


@dataclass
class DiscoveryConfig:
    """Configuration for index-based discovery from platforms.toml [pin.discovery]."""

    torch_index: str = "https://download.pytorch.org/whl/torch/"
    torch_min: str = "2.10"
    backends: list[str] = field(default_factory=lambda: ["cuda", "rocm", "cpu"])
    python: str | None = None
    platforms: list[str] | None = None  # e.g. ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64"]
    latest_patch_only: bool = True


@dataclass
class PinMatrix:
    """Defines which (backend, torch) combinations to pin (legacy static matrix)."""

    torch: list[str] = field(default_factory=list)
    backends: dict[str, list[str]] = field(default_factory=dict)
    exclude: list[dict[str, str]] = field(default_factory=list)

    def combinations(self) -> list[tuple[str, str, str]]:
        """Yield valid (backend_name, backend_version, torch_version) tuples."""
        combos = []
        for torch_ver in self.torch:
            for backend_name, backend_versions in self.backends.items():
                for backend_ver in backend_versions:
                    if not self._is_excluded(backend_name, backend_ver, torch_ver):
                        combos.append((backend_name, backend_ver, torch_ver))
        return combos

    def _is_excluded(self, backend_name: str, backend_ver: str, torch_ver: str) -> bool:
        for exc in self.exclude:
            match = True
            if backend_name in exc and exc[backend_name] != backend_ver:
                match = False
            if backend_name not in exc:
                match = False
            if "torch" in exc and exc["torch"] != torch_ver:
                match = False
            if match:
                return True
        return False


def _find_platforms_toml(start_path: Path | None = None) -> Path:
    """Locate platforms.toml, searching from milabench repo root."""
    if start_path is None:
        start_path = Path(__file__).parent.parent.parent

    candidates = [
        start_path / "platforms.toml",
        Path(os.environ.get("MILABENCH_CONFIG_DIR", "")) / "platforms.toml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"platforms.toml not found. Searched: {[str(c) for c in candidates]}"
    )


def _parse_vllm_maps(raw: dict[str, Any]) -> dict[str, dict[tuple[str, str], VllmMapping]]:
    """Parse ``[vllm.cuda]`` / ``[vllm.rocm]`` exact mapping tables."""
    result: dict[str, dict[tuple[str, str], VllmMapping]] = {}
    for backend, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        # Skip legacy reverse maps if someone still has them
        if backend in {"torch", "untagged_cuda"}:
            continue
        backend_maps: dict[tuple[str, str], VllmMapping] = {}
        for key, value in entries.items():
            if not isinstance(value, dict):
                continue
            parts = [p.strip() for p in str(key).split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"[vllm.{backend}] key {key!r} must be 'torch,backend_version' "
                    f"(e.g. '2.10.0,130')"
                )
            torch_ver, backend_ver = parts
            version = value.get("version")
            if not version:
                raise ValueError(
                    f"[vllm.{backend}] entry {key!r} requires a 'version' field"
                )
            find_links = value.get("find-links") or value.get("find_links")
            extra = value.get("extra-index-url") or value.get("extra_index_url")
            if not find_links and not extra:
                raise ValueError(
                    f"[vllm.{backend}] entry {key!r} requires 'find-links' "
                    f"or 'extra-index-url'"
                )
            raw_constraints = value.get("constraints") or []
            if isinstance(raw_constraints, str):
                raw_constraints = [raw_constraints]
            backend_maps[(torch_ver, backend_ver)] = VllmMapping(
                version=str(version),
                find_links=str(find_links) if find_links else None,
                extra_index_url=str(extra) if extra else None,
                constraints=[str(c) for c in raw_constraints],
            )
        if backend_maps:
            result[str(backend)] = backend_maps
    return result


def load_platform_config(
    path: Path | str | None = None,
    overrides: dict[str, str] | None = None,
) -> PlatformConfig:
    """Load and parse platforms.toml.

    Args:
        path: Explicit path to platforms.toml. If None, auto-detected.
        overrides: CLI variable overrides (e.g. {"cuda": "130", "torch": "2.12.0"}).

    Returns:
        PlatformConfig with all sections parsed.
    """
    if path is None:
        path = _find_platforms_toml()
    else:
        path = Path(path)

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = PlatformConfig(raw=raw)

    # Parse [vars]
    config.vars = {k: str(v) for k, v in raw.get("vars", {}).items()}

    # Parse [build] (build backends pre-seeded before installing requirements)
    build_raw = raw.get("build", {})
    if "requires" in build_raw:
        config.build_requires = [str(x) for x in build_raw["requires"]]

    # Parse [pin.discovery] (preferred) or [pin.matrix] (legacy)
    pin_raw = raw.get("pin", {})
    discovery_raw = pin_raw.get("discovery", {})
    matrix_raw = pin_raw.get("matrix", {})

    if discovery_raw:
        # platforms can be a list or a single string
        platforms_val = discovery_raw.get("platforms")
        if isinstance(platforms_val, str):
            platforms_val = [platforms_val]

        config.discovery = DiscoveryConfig(
            torch_index=discovery_raw.get("torch_index", "https://download.pytorch.org/whl/torch/"),
            torch_min=str(discovery_raw.get("torch_min", "2.10")),
            backends=discovery_raw.get("backends", ["cuda", "rocm", "cpu"]),
            python=discovery_raw.get("python"),
            platforms=platforms_val,
            latest_patch_only=discovery_raw.get("latest_patch_only", True),
        )
    elif matrix_raw:
        config.pin_matrix = PinMatrix(
            torch=matrix_raw.get("torch", []),
            backends=matrix_raw.get("backends", {}),
            exclude=matrix_raw.get("exclude", []),
        )

    # Parse backend sections (cuda, rocm, hpu, xpu, cpu)
    backend_names = {"cuda", "rocm", "hpu", "xpu", "cpu"}
    for name in backend_names:
        if name not in raw:
            continue

        section = raw[name]
        backend = BackendConfig(name=name)

        # Indexes
        idx = section.get("indexes", {})
        if idx:
            backend.indexes = IndexConfig(
                index_url=idx.get("index-url", "https://pypi.org/simple"),
                extra_index_url=idx.get("extra-index-url", []),
                find_links=idx.get("find-links", []),
            )

        # Constraints
        backend.constraints = section.get("constraints", {})

        # Overrides
        overrides_raw = section.get("overrides", {})
        for pkg_name, override_data in overrides_raw.items():
            backend.overrides[pkg_name] = OverrideConfig(
                package=pkg_name,
                install=override_data.get("install"),
                env=override_data.get("env", {}),
            )

        config.backends[name] = backend

    # Parse [vllm.cuda] / [vllm.rocm] exact maps
    vllm_raw = raw.get("vllm", {})
    if isinstance(vllm_raw, dict):
        config.vllm_maps = _parse_vllm_maps(vllm_raw)

    # Parse [compat.*] sections
    compat_raw = raw.get("compat", {})
    for pkg_name, rules_dict in compat_raw.items():
        if not isinstance(rules_dict, dict):
            continue
        entry = CompatEntry(package=pkg_name)
        for conditions_key, constraint_value in rules_dict.items():
            entry.rules.append(CompatRule(
                conditions=conditions_key,
                constraint=str(constraint_value),
            ))
        config.compat[pkg_name] = entry

    return config
