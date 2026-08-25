"""Compare vLLM pin lockfiles to other constraint files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from argklass.arguments import argument
from argklass.command import Command

from ...dependencies.compare import (
    all_pairs,
    compare_files,
    default_pairs,
    default_pin_dir,
    filter_pairs,
    format_diff,
    list_torch_lockfiles,
    matching_torch_lockfile,
    matching_vllm_lockfile,
    pair_for_versions,
)


def _resolve_pairs(args) -> list[tuple[Path, Path]]:
    pin_dir = Path(args.pin_dir).expanduser() if args.pin_dir else default_pin_dir()
    files = [Path(f).expanduser() for f in (args.files or [])]
    cuda = getattr(args, "cuda", None)
    torch = getattr(args, "torch", None)
    backend = getattr(args, "backend", None) or "cuda"

    if len(files) >= 2:
        left = files[0]
        return [(left, other) for other in files[1:]]

    if len(files) == 1:
        path = files[0]
        if path.name.startswith("constraints.vllm."):
            if args.all:
                return [(path, other) for other in list_torch_lockfiles(path.parent)]
            sibling = matching_torch_lockfile(path)
            if sibling is None:
                raise FileNotFoundError(
                    f"No matching torch lockfile for {path.name} in {path.parent}"
                )
            return [(path, sibling)]
        sibling = matching_vllm_lockfile(path)
        if sibling is None:
            raise FileNotFoundError(
                f"No matching vLLM lockfile for {path.name} in {path.parent}"
            )
        return [(sibling, path)]

    if not pin_dir.is_dir():
        raise FileNotFoundError(f"Pin directory not found: {pin_dir}")

    if cuda and torch:
        return [pair_for_versions(pin_dir, cuda=cuda, torch=torch, backend=backend)]

    pairs = all_pairs(pin_dir) if args.all else default_pairs(pin_dir)
    pairs = filter_pairs(pairs, cuda=cuda, torch=torch, backend=backend)
    if not pairs:
        wanted = []
        if cuda:
            wanted.append(f"{backend}={cuda}")
        if torch:
            wanted.append(f"torch={torch}")
        hint = f" for {' '.join(wanted)}" if wanted else ""
        raise FileNotFoundError(
            f"No vLLM lockfiles with matching torch pins{hint} in {pin_dir}"
        )
    return pairs


class PinDiff(Command):
    """Show packages whose pins disagree between vLLM and other lockfiles."""

    name = "pin-diff"

    # fmt: off
    @dataclass
    class Arguments:
        """Show packages whose pins disagree between vLLM and other lockfiles."""
        files   : list[str] = argument(default=[], nargs="*")  # Lockfiles (default: each vLLM vs its torch sibling)
        pin_dir : str       = None   # .pin directory (default: repo .pin/)
        cuda    : str       = None   # CUDA version (e.g. 130) — with --torch selects that pair
        torch   : str       = None   # PyTorch version (e.g. 2.10.0)
        backend : str       = None   # Backend name when not cuda (default: cuda)
        all     : bool      = False  # Compare each vLLM lockfile to every torch lockfile
        same    : bool      = False  # Also list shared packages that pin the same version
        unique  : bool      = False  # Also list packages present on only one side
    # fmt: on

    @staticmethod
    def execute(args):
        try:
            pairs = _resolve_pairs(args)
        except FileNotFoundError as exc:
            print(exc)
            return 1

        any_conflict = False
        for i, (left, right) in enumerate(pairs):
            if i:
                print()
            diff = compare_files(left, right)
            print(format_diff(diff, unique=args.unique, same=args.same))
            any_conflict = any_conflict or diff.incompatible
        return 1 if any_conflict else 0


COMMANDS = PinDiff
