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
        assert out[-1] == "--node-rank=0"
        assert out[:-1] == STATIC_ARGV

    def test_injects_slurm_nodeid_plus_one(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        out = maybe_inject_node_rank(STATIC_ARGV)
        assert out[-1] == "--node-rank=1"

        monkeypatch.setenv("SLURM_NODEID", "1")
        out = maybe_inject_node_rank(STATIC_ARGV)
        assert out[-1] == "--node-rank=2"

    def test_skips_when_node_rank_already_set_equals(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = [*STATIC_ARGV, "--node-rank=0"]
        assert maybe_inject_node_rank(argv) == argv

    def test_skips_when_node_rank_already_set_separate(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = ["--node-rank", "0", *STATIC_ARGV]
        assert maybe_inject_node_rank(argv) == argv

    def test_skips_underscore_node_rank(self, monkeypatch):
        monkeypatch.setenv("SLURM_NODEID", "0")
        argv = ["--node_rank=3", *STATIC_ARGV]
        assert maybe_inject_node_rank(argv) == argv

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
