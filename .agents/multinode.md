# Multi-node runs

Read this before running or debugging any `*-nodes` benchmark, or when editing the system file.

## System file (`$MILABENCH_SYSTEM`)

```yaml
system:
  arch: rocm
  env:                       # forwarded to every node
    NCCL_SOCKET_IFNAME: <iface>
    NCCL_IB_HCA: <hca list>
  nodes:
    - {name: node1, ip: 10.0.0.1, port: 5000, main: true, user: me}
    - {name: node2, ip: 10.0.0.2, main: false, user: me}
```

- `main: true` on the node milabench runs from. It must SSH to the workers without a password,
  and each worker's host key must be in `~/.ssh/known_hosts`.
- **Put every NCCL/RCCL variable the fabric needs under `system.env`.** Values in `~/.bashrc` usually
  sit after the non-interactive guard, so SSH-launched workers never see them, and cross-node
  collectives then hang silently. Find them with `grep NCCL ~/.bashrc`.

## What gets copied to the workers

- milabench rsyncs `$MILABENCH_BASE` (the venvs) to the workers, with `--delete` for the venv,
  so `pip list` should match on every node.
- **It does not copy the code.** Workers run their own milabench (and voir) checkout. Sync it after
  **every** code change; a stale worker gives deadlocks and garbage rates:
  ```bash
  rsync -a --exclude wandb/ --exclude __pycache__/ milabench/ <user>@<worker>:<same path>/milabench/
  ```

## Launchers

- torchrun-style benchmarks: `benchrun` on every node, rank from `SLURM_NODEID` (`commands/srun.py`).
  Launchers that bypass benchrun (e.g. `tune run`) get `PET_NODE_RANK` from `SlurmNodeRank`.
- vLLM `*-nodes` benchmarks: `RayCluster` (`milabench/commands/ray.py`) starts Ray on every node,
  then runs the server on the main node with `--distributed-executor-backend ray`.
  - Prefer TP = GPUs per node and PP = number of nodes: only the activations between stages cross the network.
  - vLLM only forwards `VLLM_*`, `NCCL_*`, `HF_*`, ... to its Ray actors, and `ray start` on the workers does
    not see milabench's env. Other variables must be listed in `VLLM_RAY_EXTRA_ENV_VARS_TO_COPY`
    (`benchmarks/vllm/benchfile.py` does it for `ATOM_DISABLE_VLLM_PLUGIN`).
- `milabench slurm srun -- <cmd>` runs a command on every node (native `srun` when there is one, SSH otherwise).

## Cleaning up after a hung or aborted run

Worker ranks and Ray daemons are not cleaned up when the main node gives up, and they keep the worker GPUs busy.

```bash
milabench tools warden --dry-run   # list leftovers on every node of $MILABENCH_SYSTEM (or --system <file>)
milabench tools warden             # SIGTERM them, then SIGKILL after --timeout (10 s)
```

- It catches your processes under `$MILABENCH_BASE` or `milabench/benchmarks/`, the `milabench` command,
  `ray::`/`VLLM::`/raylet/gcs_server, and anything holding a GPU (benchmate's GPU warden).
- It refuses (exit 3) while a `milabench run/prepare/install` is alive on any node. `--force` kills that
  too, so never pass it during a run you want to keep.
- Workers don't need the new code synced: the node-side script is sent over SSH.
