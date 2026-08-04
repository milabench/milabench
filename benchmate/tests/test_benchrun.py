"""Unit tests for benchrun static-rdzv ``--node-rank`` injection."""

import sys
from unittest.mock import MagicMock

# benchrun imports torch at module load; stub so these tests run without torch.
for _name in (
    "torch",
    "torch.distributed",
    "torch.distributed.run",
    "torch.distributed.elastic",
    "torch.distributed.elastic.multiprocessing",
    "torch.distributed.elastic.multiprocessing.api",
    "torch.distributed.elastic.multiprocessing.subprocess_handler",
):
    sys.modules.setdefault(_name, MagicMock())

from benchmate.benchrun import (  # noqa: E402
    _rdzv_backend,
    _slurm_node_rank,
    maybe_inject_node_rank,
)


STATIC_ARGV = [
    "--nnodes=3",
    "--rdzv-backend=static",
    "--rdzv-endpoint=127.0.0.1:29400",
    "--nproc-per-node=1",
    "--",
    "main.py",
]


def _torch_node_rank(argv):
    """Parse ``--node-rank`` from torchrun options (before ``--`` only)."""
    sep = argv.index("--") if "--" in argv else len(argv)
    torch_args = argv[:sep]
    for i, arg in enumerate(torch_args):
        if arg.startswith("--node-rank="):
            return int(arg.split("=", 1)[1])
        if arg == "--node-rank" and i + 1 < len(torch_args):
            return int(torch_args[i + 1])
        if arg.startswith("--node_rank="):
            return int(arg.split("=", 1)[1])
    raise AssertionError(f"no --node-rank before -- in {argv}")


def _with_main_node_rank(argv, rank=0):
    """Plan shape for the main node: explicit ``--node-rank`` before ``--``."""
    argv = list(argv)
    sep = argv.index("--")
    return argv[:sep] + [f"--node-rank={rank}"] + argv[sep:]


class TestRdzvBackend:
    def test_defaults_to_static(self):
        assert _rdzv_backend(["--nnodes=2"]) == "static"

    def test_equals_form(self):
        assert _rdzv_backend(["--rdzv-backend=c10d"]) == "c10d"

    def test_separate_arg(self):
        assert _rdzv_backend(["--rdzv-backend", "c10d"]) == "c10d"

    def test_underscore_form(self):
        assert _rdzv_backend(["--rdzv_backend=etcd"]) == "etcd"


class TestSlurmNodeRank:
    def test_unset_is_zero(self, monkeypatch):
        monkeypatch.delenv("SLURM_NODEID", raising=False)
        assert _slurm_node_rank() == 0

    def test_empty_is_zero(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "")
        assert _slurm_node_rank() == 0

    def test_worker_step_ids_are_offset(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        assert _slurm_node_rank() == 1
        monkeypatch.setenv("SLURM_NODEID", "1")
        assert _slurm_node_rank() == 2


class TestMaybeInjectNodeRank:
    def test_injects_zero_without_slurm(self, monkeypatch):
        monkeypatch.delenv("SLURM_NODEID", raising=False)
        out = maybe_inject_node_rank(STATIC_ARGV)
        assert "--node-rank=0" in out
        assert out.index("--node-rank=0") < out.index("--")
        assert out[out.index("--"):] == ["--", "main.py"]

    def test_injects_slurm_nodeid_plus_one(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        out = maybe_inject_node_rank(STATIC_ARGV)
        assert out[out.index("--") - 1] == "--node-rank=1"

        monkeypatch.setenv("SLURM_NODEID", "1")
        out = maybe_inject_node_rank(STATIC_ARGV)
        assert out[out.index("--") - 1] == "--node-rank=2"

    def test_skips_when_node_rank_already_set_equals(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        # Already a torchrun option (before --)
        argv = [
            "--nnodes=3",
            "--rdzv-backend=static",
            "--node-rank=0",
            "--",
            "main.py",
        ]
        assert maybe_inject_node_rank(argv) == argv

    def test_skips_when_node_rank_already_set_separate(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = ["--node-rank", "0", *STATIC_ARGV]
        assert maybe_inject_node_rank(argv) == argv

    def test_skips_underscore_node_rank(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = ["--node_rank=3", *STATIC_ARGV]
        assert maybe_inject_node_rank(argv) == argv

    def test_ignores_node_rank_after_script_sep(self, monkeypatch):
        """A script arg named --node-rank must not block torchrun injection."""
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = [*STATIC_ARGV, "--node-rank=99"]
        out = maybe_inject_node_rank(argv)
        assert out[out.index("--") - 1] == "--node-rank=1"
        assert out[-1] == "--node-rank=99"

    def test_skips_c10d(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = [
            "--nnodes=3",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:29400",
        ]
        assert maybe_inject_node_rank(argv) == argv

    def test_default_backend_is_static(self, monkeypatch):
        monkeypatch.delenv("SLURM_NODEID", raising=False)
        argv = ["--nnodes=1", "--nproc-per-node=1"]
        out = maybe_inject_node_rank(argv)
        assert out == [*argv, "--node-rank=0"]

    def test_does_not_mutate_input(self, monkeypatch):
        monkeypatch.delenv("SLURM_NODEID", raising=False)
        original = list(STATIC_ARGV)
        maybe_inject_node_rank(original)
        assert original == STATIC_ARGV


class TestThreeNodeJobRanks:
    """Main + ``srun -x main`` workers must cover ranks 0..nnodes-1 exactly once.

    Production layout (TorchrunSrun / TorchrunSrunAlways):
    - Main runs locally with plan argv ``--node-rank=0`` (before ``--``).
      Outer Slurm still sets ``SLURM_NODEID=0`` on that process.
    - Workers are one ``srun -x main`` step; step-local ``SLURM_NODEID`` is
      0..N-2 and benchrun maps those to global ranks 1..N-1.
    """

    def test_full_job_assigns_ranks_0_1_2(self, monkeypatch):
        # Main: outer job NODEID=0 must NOT become rank 1.
        monkeypatch.setenv("SLURM_NODEID", "0")
        main_argv = _with_main_node_rank(STATIC_ARGV, 0)
        main_out = maybe_inject_node_rank(main_argv)
        assert _torch_node_rank(main_out) == 0
        assert main_out.index("--node-rank=0") < main_out.index("--")
        assert main_out.count("--node-rank=0") == 1

        # Workers: step-local ids 0, 1 → global 1, 2 (no plan --node-rank).
        worker_ranks = []
        for step_id in (0, 1):
            monkeypatch.setenv("SLURM_NODEID", str(step_id))
            out = maybe_inject_node_rank(STATIC_ARGV)
            assert out.index(f"--node-rank={step_id + 1}") < out.index("--")
            worker_ranks.append(_torch_node_rank(out))

        ranks = [_torch_node_rank(main_out), *worker_ranks]
        assert ranks == [0, 1, 2]
        assert sorted(set(ranks)) == [0, 1, 2]

    def test_main_without_explicit_rank_collides_under_outer_slurm(self, monkeypatch):
        """Documents why ``main_executor`` must pin ``--node-rank=0``."""
        monkeypatch.setenv("SLURM_NODEID", "0")
        # Same argv workers get — injection alone would steal rank 1 from a worker.
        assert _torch_node_rank(maybe_inject_node_rank(STATIC_ARGV)) == 1
