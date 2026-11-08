# Kimi-K3 inference on Trillium's B200 nodes (K3-INF)

vLLM serving Kimi-K3 (`moonshotai/Kimi-K3`, 1.56 TB: MXFP4 routed experts, BF16 rest) on 2 B200 nodes, with the
client passes of the MI355X study (`config/kimi_study.yaml`, `_kimi`), so the two vendors compare pass for pass.
`scripts/analysis/kimi_study.py` makes the tables.

| file | what |
|---|---|
| `serve.sh ENTRY NODES TIME [NAME=value ...]` | submits one entry of `config/kimi_study.yaml` (`nw-k3inf-<entry>`) |
| `job.sh` | the job: power check on every node, `milabench slurm system`, `milabench run --select ENTRY` |
| `truncate_k3.py SNAPSHOT OUT [N]` | the first N decoder layers of the checkpoint (symlinks + edited config and index) |

Entries (`config/kimi_study.yaml`, B200 section):

- `kimi-vllm-2n-tp8pp2`: TP8 inside each node, PP2 across the two (Ray). The study's server flags (`_kimi_vllm`) and
  all its passes.
- `kimi-vllm-2n-tp8pp2-bringup`: the same server, passes `conc1` and `conc64` only (the step 2 milestone).

## Install (login node, once)

`$B=$SCRATCH/trillium/new/k3-inf` is `MILABENCH_BASE`. vLLM 0.32.0 on torch 2.13.0 + cu130 (`platforms.toml`,
`[vllm.cuda]`): Kimi-K3 needs vLLM >= 0.27.0, the MI355X study ran 0.32.0.

```bash
B=$SCRATCH/trillium/new/k3-inf; mkdir -p $B/logs; cd $B
git clone -b trillium-new https://github.com/milabench/milabench.git src
unset PYTHONPATH PIP_CONFIG_FILE; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$B/uv-cache
uv venv --python-preference only-managed --python 3.12 --seed src/.venv
VIRTUAL_ENV=$B/src/.venv uv pip install -e ./src "huggingface_hub[hf_xet]" nvidia-ml-py   # pynvml: voir's cuda backend
export PATH=$B/src/.venv/bin:$PATH MILABENCH_BASE=$B MILABENCH_GPU_ARCH=cuda
milabench install --config src/config/kimi_study.yaml --select kimi-vllm-2n-tp8pp2-bringup --set torch=2.13.0 cuda=130
src/scripts/trillium/kimi/kernels.sh   # FlashInfer's cubins and jit-cache for the offline nodes (19 min, see the script)
# the checkpoint, one large download at a time on the login node (about 80 MB/s, 5.5 h). It fetches files in name order:
# tokenizer and modeling files come last; fetch them by name first for a truncated test. The login node kills a process
# after 1 h of CPU time (ulimit -t 3600): hf uses ~35% of a core, so it dies after ~3 h; run it again, it skips the
# complete files (not the partial ones: delete the stale *.incomplete)
nice -n 19 ionice -c3 hf download moonshotai/Kimi-K3 --cache-dir $B/data/hub --max-workers 8
```

Without `kernels.sh`, flashinfer refuses to import (no flashinfer-cubin, or a mismatched one), or the nodes try to
download cubins. vLLM finds no nvcc (`CUDA_HOME` unset), so FlashInfer JIT is off: only the jit-cache's sm100a modules.

## Run

Always with an explicit entry (`--select`): the config's other enabled entries are ROCm ones (ATOM).

```bash
cd $SCRATCH/trillium/new/k3-inf && git -C src pull --ff-only
# truncated model (4 layers), same entry: the data directory points at a copy of the HF cache layout
S=$(ls -d data/hub/models--moonshotai--Kimi-K3/snapshots/*); T=trunc4/hub/models--moonshotai--Kimi-K3
mkdir -p $T/refs; basename $S | tr -d '\n' > $T/refs/main
python3 -I src/scripts/trillium/kimi/truncate_k3.py $S $T/snapshots/$(basename $S) 4
NODELIST=trig[0147-0148] src/scripts/trillium/kimi/serve.sh kimi-vllm-2n-tp8pp2-bringup 2 0:30:00 MILABENCH_DIRS_DATA=$PWD/trunc4
# full model
NODELIST=trig[0147-0148] src/scripts/trillium/kimi/serve.sh kimi-vllm-2n-tp8pp2-bringup 2 1:00:00
```

Output: `slurm/nw-k3inf-<entry>.<job>.out`, run data `runs/<entry>.<job>/` (one `.data` per pass).

## Measured

4-layer model (`trunc4`), `kimi-vllm-2n-tp8pp2-bringup`, job 1261631, `trig[0101-0102]`, 2026-11-07: PASS in 1558 s.
vLLM picked `FLASHINFER_TRTLLM_MXFP4_MXFP8` for the MXFP4 experts (activations quantized to MXFP8; the checkpoint is
W4A16), the native fused KDA decode and FlashKDA prefill, FLASHINFER_MLA decode with FA2 prefill (FA4 does not take
head size 192). Weights 19 s (PP0) and 46 s (PP1) for 54.5 GiB, cold. Engine init 1209 s, almost all of it FlashInfer's
autotune (MXFP4 MoE 22 + 8 profiles at 25-40 s each, then MLA decode and BF16 GEMM). vLLM saves it
(`cache/vllm/flashinfer_autotune_cache/<vllm config hash>`, one file per rank) and loads it at the next start (job 1262548:
"Loaded 35 configs"), but the MXFP4 MoE op tunes again: its cached keys carry values that look per process. Every start
pays the 20-25 min. conc1 60/60, conc64 128/128 requests; numbers of a 4-layer model, not of Kimi-K3.

Full model with random weights, `kimi-vllm-2n-tp8pp2-dummy`, job 1261890, `trig[0101-0102]`: PASS in 1888 s. Weights
95.7-96.1 GiB per GPU (PP0 47 layers, PP1 46), CUDA graphs 3.8 GiB in 32 s, 51.4-53.5 GiB left for the KV cache:
3.6M tokens, 27 requests at the full 133,120. Engine init 1499 s, the autotune. conc1 TPOT 8.70 ms, conc64 1467
output tok/s, TPOT p50 21.4 ms: random weights route at random, not Kimi-K3 numbers.

Clean end (`job.sh`): after the run, or at USR1 5 min before `--time`, `ray stop --force` and a kill of what is left of
vLLM and Ray on every node, then a wait until `nvidia-smi --query-compute-apps` is empty. Job 1262548 (4-layer model,
20 min, ended in the autotune): both nodes "GPUs free", neither drained. milabench delays a first SIGTERM; the second
one, sent by hand there, stopped it: the trap now sends both (not run since).

**Kimi-K3, 2 nodes, `kimi-vllm-2n-tp8pp2-bringup`, job 1262824, `trig[0101-0102]`, 2026-11-07 22:45-23:27: PASS**
(milabench 2509 s, job 42:04, both nodes "GPUs free", neither drained). Weights 688 s (PP0) and 651 s (PP1): each stage
reads its own ~47 shards, ~1.1 GB/s per node from VAST, cold. Engine init 1434 s: MoE autotune 22 profiles in 11-13 min
plus 8 in 4-5 min, BF16 GEMM 4.5 min, graphs 27 s. MoE backend `FLASHINFER_TRTLLM_MXFP4_MXFP8` (vLLM 0.32's default on
B200: activations in MXFP8). 1k in, 256 out, random tokens:

| pass | output tok/s (per GPU) | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | power W, 2 nodes | J / output tok |
|---|---|---|---|---|---|---|---|
| conc1 | 111.3 (7.0) | 556 | 85 / 88 | 8.69 / 8.70 | 8.68 / 8.89 | 7014 | 63.0 |
| conc64 | 1304.7 (81.5) | 6523 | 1881 / 11689 | 26.7 / 64.1 | 20.6 / 30.1 | 9165 | 7.0 |

Power: main node's GPUs only (voir), times 2 (`scripts/analysis/kimi_study.py`). conc64's p99 TTFT is the first burst:
64 prompts of 1024 tokens at once, 16,384 batched tokens per step (also in job 1261890).

Not tested yet: the study's other passes and the accuracy pass; `-notune` (no FlashInfer autotune, against this run);
prefix caching, DCP and speculative decoding stay off as in the MI355X study.
