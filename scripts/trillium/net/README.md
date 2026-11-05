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
  (`batch_isend_irecv`), 10 iterations after 3 warm-up, on all GPUs (`--gpus`: those only). `bw_gbs` =
  bytes one node sends per second = GPUs x size / slowest GPU's time per iteration (one direction, while
  the other direction runs too). `lat_us` = one-way time of an 8-byte ping-pong (200 round trips,
  GPU-timed, includes NCCL kernel launch), median over the GPU pairs. **On B200 use `--gpus 0,2,4,6`** to
  measure the fabric: see "Trillium" below.
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
NET_D2_ARGS=--sequential sbatch --export=ALL --nodes=52 --time=... scripts/trillium/net/net.sbatch d2

# Rorqual test (4 nodes): override account, partition and GPUs; results under NET_ROOT
sbatch --account=rrg-bengioy-ad_gpu --partition=gpubase_bynode_b1 --gpus-per-node=h100:4 --mem=0 \
    --nodes=4 --time=0:20:00 --output=$SCRATCH/tri-dev/net/%x-%j.out \
    --export=ALL,NET_ROOT=$SCRATCH/tri-dev/net,NET_D3_ARGS="--sweep-nodes 2",NET_D4_ARGS="--cuts 1" \
    scripts/trillium/net/net.sbatch d2 d3 d4
```

Trillium's `sbatch` is a shell function that adds `--export=NONE --get-user-env`: without `--export=ALL`
the variables below do not reach the job. `--time` must be at least 15 minutes.

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

## Trillium (2026-11-04, 8 x B200, NCCL 2.30.7)

- NICs: the job environment sets `NCCL_IB_HCA==mlx5_0:1:0,mlx5_1:1:1,mlx5_4:1:2,mlx5_9:1:3`, the four 400 Gb/s
  ports. Port counters confirm it: traffic on these four, evenly, none on `mlx5_2`, `3`, `5`-`8`, `10`.
  `nvidia-smi topo -m`, same on 25 nodes: GPUs 0, 2, 4, 6 each share a PCIe switch with one of them; 1, 3, 5, 7 with none.
- Send/recv (D2) on all 8 GPUs does not measure the fabric: GPUs 1, 3, 5, 7 stripe over all four NICs (PXN to send)
  and, all four active, get 4.5 GB/s each (each alone: 37-45). So 45 GB/s per node on 2-4 nodes, whatever
  `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_NCHANNELS_PER_NET_PEER`, `NCCL_NET_SHARED_BUFFERS`, `NCCL_BUFFSIZE`,
  `NCCL_MAX_P2P_PEERS` (`NCCL_P2P_PXN_LEVEL=0`: 35), and 10.4-11.5 GB/s on a 25-node job, also one pair at a time
  (job 1239737), so that drop comes with the size of the job, not from the traffic. All-gather on the same NICs: 192.
- `--gpus 0,2,4,6` (each on its own NIC) on 2-4 nodes: 148-164 GB/s per node, repeatable to 0.3%. Not yet run above 4 nodes.
- Alltoall goes through send/recv too: 10 GB/s bus BW at 2 nodes, 4.3 at 25.
- Durations, 25 nodes: D3 2 min 38 s (job 1239634); D2 5 min 9 s on all 8 GPUs (job 1239660, 25 rounds of 10.7 s).
