#!/usr/bin/env python3

import os
import subprocess
import sys
from contextlib import contextmanager

import torch.distributed.run as distrun
import torch.distributed.elastic.multiprocessing.api as elastic
import torch.distributed.elastic.multiprocessing.subprocess_handler as sub


class NewSubprocessHandler(sub.SubprocessHandler):
    def _popen(self, args, env) -> subprocess.Popen:
        kwargs = {}

        if fd := os.getenv("DATA_FD"):
            kwargs["pass_fds"] = [int(fd)]

        return subprocess.Popen(
            args=args,
            env=env,
            stdout=self._stdout,
            stderr=self._stderr,
            **kwargs,
        )


def get_subprocess_handler(*args, **kwargs):
    return NewSubprocessHandler(*args, **kwargs)


@contextmanager
def forward_voir_file():
    """Overrides torchruns way of creating a new process so we can forward our file desctriptor"""
    old_handle = elastic.get_subprocess_handler
    old_handler = elastic.SubprocessHandler

    elastic.get_subprocess_handler = get_subprocess_handler
    elastic.SubprocessHandler = NewSubprocessHandler

    yield

    elastic.get_subprocess_handler = old_handle
    elastic.SubprocessHandler = old_handler


def _split_at_script_sep(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split torchrun options from script args at ``--`` (if present)."""
    try:
        sep = argv.index("--")
    except ValueError:
        return list(argv), []
    return list(argv[:sep]), list(argv[sep:])


def _argv_has_option(argv, *names: str) -> bool:
    for arg in argv:
        for name in names:
            if arg == name or arg.startswith(f"{name}="):
                return True
    return False


def _rdzv_backend(argv) -> str:
    """Return the rendezvous backend from argv (torchrun default: static)."""
    for i, arg in enumerate(argv):
        for name in ("--rdzv-backend", "--rdzv_backend"):
            if arg == name and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return "static"


def _max_nnodes(argv) -> int:
    """Return the maximum node count from ``--nnodes`` (torchrun default: 1).

    torchrun accepts either ``N`` or a ``MIN:MAX`` range; we use MAX so a
    single-node job (``1`` or ``1:1``) is detected regardless of form.
    """
    value = None
    for i, arg in enumerate(argv):
        if arg == "--nnodes" and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--nnodes="):
            value = arg.split("=", 1)[1]
    if value is None:
        return 1
    try:
        return int(str(value).split(":")[-1])
    except ValueError:
        return 1


def _slurm_node_rank() -> int:
    """Rank for static rdzv when milabench launches workers via ``srun -x main``.

    Main is started locally with an explicit ``--node-rank=0``. Worker tasks in
    the excluded-main srun step get ``SLURM_NODEID`` 0..N-2, so we map them to
    global ranks 1..N-1.
    """
    nodeid = os.environ.get("SLURM_NODEID")
    if nodeid is None or nodeid == "":
        return 0
    return int(nodeid) + 1


def maybe_inject_node_rank(argv):
    """If static rdzv and ``--node-rank`` is absent, derive it from Slurm.

    Inserts ``--node-rank`` among torchrun options (before ``--``), never as a
    script argument after the separator.

    For a single-node job (``--nnodes`` max of 1) the rank is always 0: the job
    runs inside a Slurm allocation where ``SLURM_NODEID`` may be 0, but there is
    no rank-0 worker to wait for, so deriving a nonzero rank would hang static
    rendezvous.
    """
    argv = list(argv)
    torch_args, rest = _split_at_script_sep(argv)
    if _rdzv_backend(torch_args or argv) != "static":
        return argv
    if _argv_has_option(torch_args, "--node-rank", "--node_rank"):
        return argv
    if _max_nnodes(torch_args or argv) <= 1:
        node_rank = 0
    else:
        node_rank = _slurm_node_rank()
    torch_args.append(f"--node-rank={node_rank}")
    return torch_args + rest


def _format_cmd(argv) -> str:
    return " ".join(str(a) for a in argv)


def run(args):
    with forward_voir_file():
        distrun.run(args)


def main(args=None):
    argv = list(args) if args is not None else sys.argv[1:]
    argv = maybe_inject_node_rank(argv)
    print(f"[benchrun] {_format_cmd(argv)}", file=sys.stderr, flush=True)
    run(distrun.parse_args(argv))


if __name__ == "__main__":
    main()
