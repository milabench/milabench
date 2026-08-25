"""Compare pin lockfiles (vLLM vs torch) for version conflicts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_INCLUDE_RE = re.compile(r"^-c\s+(\S+)")
_REQ_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]+\])?\s*(.*)$"
)
_VLLM_PREFIX = "constraints.vllm."


def default_pin_dir() -> Path:
    """Repo ``.pin/`` next to the ``milabench`` package."""
    return Path(__file__).resolve().parents[2] / ".pin"


def normalize_name(name: str) -> str:
    """PEP 503-style name: lowercase, ``[-_.]`` collapsed to ``-``."""
    name = name.split("[", 1)[0].strip().lower()
    return re.sub(r"[-_.]+", "-", name)


def parse_requirement_line(line: str) -> tuple[str, str] | None:
    """Return ``(normalized_name, spec)`` or ``None`` for comments / pip options."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    match = _REQ_RE.match(stripped)
    if not match:
        return None
    return normalize_name(match.group(1)), match.group(2).strip()


def load_pins(path: Path, *, _seen: set[Path] | None = None) -> dict[str, str]:
    """Resolve a lockfile into ``{package: spec}``, following ``-c`` includes.

    Pins in the file override the same package from an included file.
    """
    path = path.resolve()
    seen = _seen if _seen is not None else set()
    if path in seen:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Constraint file not found: {path}")
    seen.add(path)

    pins: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        include = _INCLUDE_RE.match(stripped)
        if include:
            target = Path(include.group(1))
            if not target.is_absolute():
                target = path.parent / target
            for name, spec in load_pins(target, _seen=seen).items():
                pins.setdefault(name, spec)
            continue
        parsed = parse_requirement_line(stripped)
        if parsed is None:
            continue
        name, spec = parsed
        pins[name] = spec
    return pins


def specs_conflict(left: str, right: str) -> bool:
    """True when two requirement specs cannot be satisfied together."""
    left = left.strip()
    right = right.strip()
    if left == right:
        return False
    if not left or not right:
        return False
    if left.startswith("==") and right.startswith("=="):
        return left[2:].strip() != right[2:].strip()
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return True

    def _exact(spec: str) -> str | None:
        if spec.startswith("=="):
            return spec[2:].strip()
        return None

    left_exact = _exact(left)
    right_exact = _exact(right)
    try:
        if left_exact is not None and right_exact is None:
            return Version(left_exact) not in SpecifierSet(right)
        if right_exact is not None and left_exact is None:
            return Version(right_exact) not in SpecifierSet(left)
    except InvalidVersion:
        return True
    return True


def display_spec(spec: str) -> str:
    """Pretty-print a pin (drop a leading ``==``)."""
    spec = spec.strip()
    if spec.startswith("=="):
        return spec[2:].strip()
    return spec or "(unpinned)"


@dataclass(frozen=True)
class PinConflict:
    package: str
    left: str
    right: str


@dataclass(frozen=True)
class PinMatch:
    package: str
    spec: str


@dataclass(frozen=True)
class PinDiff:
    left: Path
    right: Path
    conflicts: list[PinConflict]
    same: list[PinMatch]
    only_left: list[str]
    only_right: list[str]

    @property
    def shared(self) -> int:
        return len(self.conflicts) + len(self.same)

    @property
    def left_total(self) -> int:
        return self.shared + len(self.only_left)

    @property
    def right_total(self) -> int:
        return self.shared + len(self.only_right)

    @property
    def agreement(self) -> float:
        return len(self.same) / self.shared if self.shared else 1.0

    @property
    def incompatible(self) -> bool:
        return bool(self.conflicts)


def compare_pins(
    left: dict[str, str],
    right: dict[str, str],
    *,
    left_path: Path,
    right_path: Path,
) -> PinDiff:
    shared = set(left) & set(right)
    conflicts: list[PinConflict] = []
    same: list[PinMatch] = []
    for name in sorted(shared):
        if specs_conflict(left[name], right[name]):
            conflicts.append(PinConflict(name, left[name], right[name]))
        else:
            same.append(PinMatch(name, left[name]))
    return PinDiff(
        left=left_path,
        right=right_path,
        conflicts=conflicts,
        same=same,
        only_left=sorted(set(left) - set(right)),
        only_right=sorted(set(right) - set(left)),
    )


def compare_files(left: Path, right: Path) -> PinDiff:
    return compare_pins(
        load_pins(left),
        load_pins(right),
        left_path=left,
        right_path=right,
    )


def matching_torch_lockfile(vllm_path: Path) -> Path | None:
    """``constraints.vllm.cuda130.torch2100.txt`` → sibling torch lockfile."""
    name = vllm_path.name
    if not name.startswith(_VLLM_PREFIX):
        return None
    sibling = vllm_path.with_name("constraints." + name[len(_VLLM_PREFIX) :])
    return sibling if sibling.is_file() else None


def matching_vllm_lockfile(torch_path: Path) -> Path | None:
    """``constraints.cuda130.torch2100.txt`` → sibling vLLM lockfile."""
    name = torch_path.name
    if not name.startswith("constraints.") or name.startswith(_VLLM_PREFIX):
        return None
    if name == "constraints.common.txt":
        return None
    sibling = torch_path.with_name("constraints.vllm." + name[len("constraints.") :])
    return sibling if sibling.is_file() else None


def list_vllm_lockfiles(pin_dir: Path) -> list[Path]:
    return sorted(p for p in pin_dir.glob("constraints.vllm.*.txt") if p.is_file())


def list_torch_lockfiles(pin_dir: Path) -> list[Path]:
    files = []
    for path in sorted(pin_dir.glob("constraints.*.txt")):
        if not path.is_file():
            continue
        if path.name.startswith(_VLLM_PREFIX):
            continue
        if path.name == "constraints.common.txt":
            continue
        files.append(path)
    return files


def default_pairs(pin_dir: Path) -> list[tuple[Path, Path]]:
    """Each vLLM lockfile paired with its same-(backend, torch) torch pin."""
    pairs = []
    for vllm in list_vllm_lockfiles(pin_dir):
        other = matching_torch_lockfile(vllm)
        if other is not None:
            pairs.append((vllm, other))
    return pairs


def _filename_token(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("cu") and value[2:].replace(".", "").isdigit():
        value = value[2:]
    return value.replace(".", "")


def filter_pairs(
    pairs: list[tuple[Path, Path]],
    *,
    cuda: str | None = None,
    torch: str | None = None,
    backend: str = "cuda",
) -> list[tuple[Path, Path]]:
    """Keep pairs whose filenames match a CUDA (or other backend) + torch version."""
    filtered = pairs
    if cuda:
        token = f"{backend}{_filename_token(cuda)}"
        filtered = [
            (left, right)
            for left, right in filtered
            if token in left.name or token in right.name
        ]
    if torch:
        token = f"torch{_filename_token(torch)}"
        filtered = [
            (left, right)
            for left, right in filtered
            if token in left.name or token in right.name
        ]
    return filtered


def pair_for_versions(
    pin_dir: Path,
    *,
    cuda: str,
    torch: str,
    backend: str = "cuda",
) -> tuple[Path, Path]:
    """Resolve the vLLM + torch lockfiles for one (cuda, torch) pair."""
    from .pin import get_constraint_file

    vllm = get_constraint_file(
        pin_dir, backend, cuda, torch, group="vllm"
    )
    other = get_constraint_file(pin_dir, backend, cuda, torch)
    if vllm.is_file() and other.is_file():
        return vllm, other

    matches = filter_pairs(default_pairs(pin_dir), cuda=cuda, torch=torch, backend=backend)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No vLLM/torch lockfile pair for {backend}={cuda} torch={torch} in {pin_dir}"
        )
    names = ", ".join(left.name for left, _ in matches)
    raise FileNotFoundError(
        f"Ambiguous {backend}={cuda} torch={torch}; matches: {names}. "
        "Pass a more specific torch version (e.g. 2.10.0)."
    )


def all_pairs(pin_dir: Path) -> list[tuple[Path, Path]]:
    """Every vLLM lockfile against every torch lockfile."""
    vllm_files = list_vllm_lockfiles(pin_dir)
    torch_files = list_torch_lockfiles(pin_dir)
    return [(left, right) for left in vllm_files for right in torch_files]


def _pct(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:.0f}%"


def format_diff(
    diff: PinDiff,
    *,
    unique: bool = False,
    same: bool = False,
) -> str:
    same_n = len(diff.same)
    diff_n = len(diff.conflicts)
    lines = [
        f"## {diff.left.name}  vs  {diff.right.name}",
        f"   vLLM {diff.left_total} packages, other {diff.right_total} packages",
        (
            f"   {diff.shared} shared: {same_n} same ({_pct(same_n, diff.shared)}), "
            f"{diff_n} different ({_pct(diff_n, diff.shared)})"
        ),
        (
            f"   {len(diff.only_left)} only in vLLM, "
            f"{len(diff.only_right)} only in other"
        ),
    ]
    if diff.conflicts:
        name_w = max(len(c.package) for c in diff.conflicts)
        left_w = max(len(display_spec(c.left)) for c in diff.conflicts)
        name_w = max(name_w, len("package"))
        left_w = max(left_w, len("vllm"))
        lines.append("")
        lines.append("   different shared packages:")
        lines.append(f"   {'package':<{name_w}}  {'vllm':<{left_w}}  other")
        for conflict in diff.conflicts:
            lines.append(
                f"   {conflict.package:<{name_w}}  "
                f"{display_spec(conflict.left):<{left_w}}  "
                f"{display_spec(conflict.right)}"
            )
    if same and diff.same:
        name_w = max(len(m.package) for m in diff.same)
        name_w = max(name_w, len("package"))
        lines.append("")
        lines.append("   same shared packages:")
        for match in diff.same:
            lines.append(f"   {match.package:<{name_w}}  {display_spec(match.spec)}")
    if unique:
        if diff.only_left:
            lines.append("")
            lines.append(f"   only in {diff.left.name} ({len(diff.only_left)}):")
            for name in diff.only_left:
                lines.append(f"     {name}")
        if diff.only_right:
            lines.append("")
            lines.append(f"   only in {diff.right.name} ({len(diff.only_right)}):")
            for name in diff.only_right:
                lines.append(f"     {name}")
    return "\n".join(lines)
