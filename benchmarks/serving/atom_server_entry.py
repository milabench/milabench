"""ATOM OpenAI server entrypoint with milabench aiter compatibility patches."""

from __future__ import annotations

from atom_aiter_compat import ensure_atom_aiter_compat
from patch_aiter_for_atom import patch_aiter_shuffle

# Worker subprocesses import aiter directly; patch the wheel on disk first.
patch_aiter_shuffle()


def main() -> None:
    ensure_atom_aiter_compat()
    from atom.entrypoints.openai.api_server import main as atom_main

    atom_main()


if __name__ == "__main__":
    main()
