# D2-D4 network sweeps

`DESIGN.md` group D, output in the formats of `FORMAT.md` section 3. One process per GPU
(`srun --ntasks-per-node=<gpus>`), PyTorch `torch.distributed` on NCCL from the bench venv
(`$SCRATCH/tt38/output/venv/torchtitan_native`, read-only; torch nightly with NCCL 2.30.7 from pip),
so the same scripts run on Rorqual (4 x H100, 1 HCA per node) and Trillium (8 x B200).

| script | experiment | output |
|---|---|---|
| `pairs.py` | D2: every node pair, round-robin rounds of disjoint pairs, a round at once (`--sequential`: one pair at a time) | `pairs.jsonl` (`trillium.pairs.v1`) |
| `collectives.py` | D3: all-reduce, all-gather, reduce-scatter, all-to-all on the first k nodes, k = 1, 2, 4, 8, 13, 26, 52, 1 GiB x 2 repetitions; size sweep 1 KiB - 4 GiB at 2 and all nodes | `collectives.jsonl` (`trillium.collectives.v1`, `D3`) |
| `bisection.py` | D4: random halves, random disjoint pairs across them, all at once; 3 cuts | `collectives.jsonl` (`D4`) |
| `netlib.py` | process group from Slurm, the pair exchange, timing, JSONL writer | |
| `net.sbatch` | runs the topology dump then the chosen experiments in one allocation | `logs/` |

Measurement:

- **Pair (D2, D4)**: GPU i of node a and GPU i of node b send each other `--size` bytes (1 GiB) at once
  (`batch_isend_irecv`), 10 iterations after 3 warm-up, on all GPUs: every GPU drives its own nearest
  HCA. `bw_gbs` = bytes one node sends per second = GPUs x size / slowest GPU's time per iteration
  (one direction, while the other direction runs too). `lat_us` = one-way time of an 8-byte ping-pong
  (200 round trips, GPU-timed, includes NCCL kernel launch), median over the GPU pairs.
- **Collectives (D3)**: nccl-tests conventions: size = a rank's largest buffer (rounded down to a
  multiple of the ranks), time = mean over ranks (max kept as `time_max_us`), algbw = size / time,
  busbw = algbw x (n-1)/n (x 2 for all-reduce). fp32, out-of-place. The nodes not in the group idle.
- `logs/topo/<host>.txt`: `nvidia-smi topo -m` (GPU <-> HCA affinity) and every IB port's rate and state.
  `logs/d2.log` etc.: each experiment's stdout (progress per round / per collective).

## Commands

Submit from the repository root (`$SLURM_SUBMIT_DIR/scripts/trillium/net` holds the scripts).

```bash
# Trillium, all 52 nodes: D2, D3, D4 into $SCRATCH/trillium/nccl/<job>/
sbatch --nodes=52 --time=1:00:00 --output=$SCRATCH/trillium/nccl/%x-%j.out scripts/trillium/net/net.sbatch
sbatch --nodes=52 ... scripts/trillium/net/net.sbatch d2          # one experiment
NET_D2_ARGS=--sequential sbatch --nodes=52 --time=... scripts/trillium/net/net.sbatch d2

# Rorqual test (4 nodes): override account, partition and GPUs; results under NET_ROOT
sbatch --account=rrg-bengioy-ad_gpu --partition=gpubase_bynode_b1 --gpus-per-node=h100:4 --mem=0 \
    --nodes=4 --time=0:20:00 --output=$SCRATCH/tri-dev/net/%x-%j.out \
    --export=ALL,NET_ROOT=$SCRATCH/tri-dev/net,NET_D3_ARGS="--sweep-nodes 2",NET_D4_ARGS="--cuts 1" \
    scripts/trillium/net/net.sbatch d2 d3 d4
```

Environment: `NET_ROOT` (results root, default `$SCRATCH/trillium`), `NET_OUT` (exact directory),
`NET_PY` (python), `NET_MODULES` (default `cuda/12.9`), `NET_D{2,3,4}_ARGS`, `NCCL_DEBUG` (default WARN).
Arguments after `--` go to every script.

## Durations

Measured on Rorqual, 4 nodes x 4 H100 (job 25476067, 2026-11-03): whole job 1 min 49 s; D2 43 s
(3 rounds of ~3 s, the rest is srun + torch import + NCCL init), D3 47 s (1, 2, 4 nodes, size sweep
at 2), D4 14 s (1 cut). Trillium at 52 nodes x 8 B200, *estimates* (not measured):

| | estimate | basis |
|---|---|---|
| D2, concurrent | ~10 min | 51 rounds x 3-6 s (1 GiB x 13 at ~90 GB/s per node, NCCL connection set-up for each new peer, 416-rank gathers) + ~2 min start-up |
| D2, `--sequential` | ~1 h | 1,326 pairs x 2-3 s |
| D3 | ~15 min | 7 node counts x 2 repetitions x 4 collectives + 2 size sweeps (up to 4 GiB all-to-all at 52 nodes: ~0.4 s per iteration) + a communicator per node count |
| D4 | ~2 min | 3 cuts x ~5 s + start-up |

`net.sbatch` asks for 1 h; shorten it once D2 has run once.

## Rorqual test (job 25476067, 4 nodes, 1 HDR HCA = 25 GB/s per node)

- D2: 18.7-22.2 GB/s per node one direction (median 21.4) with both directions at once; latency
  24.2-25.1 us one way. That latency is NCCL send/recv through torch (kernel launch included), far
  above the IB wire latency (~2 us, `ib_write_lat`): it compares pairs, not fabrics.
- D3, 1 GiB: 1 node (NVLink) all-reduce 334-337 GB/s bus BW, all-gather / reduce-scatter 310,
  all-to-all 300; 2 nodes all-reduce 37.5 (NCCL tree: bus BW can exceed the NIC), all-gather /
  reduce-scatter 18-21, all-to-all 8.6-9.0; 4 nodes 19-21, all-to-all 6.3-6.8 (= ~20 GB/s per node
  across the NIC).
- D4, 2 + 2 nodes: median 20.9 GB/s per node, min 19.7.
