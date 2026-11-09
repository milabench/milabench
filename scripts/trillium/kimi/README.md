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

- `kimi-vllm-2n-tp8pp2`: TP8 inside each node, PP2 across the two (Ray). The study's server flags (`_kimi_vllm`), a
  throwaway `warmup` pass, the study's 20 performance passes unchanged, then its `eval` pass (`prepend` / `append` in
  `client.passes`, `milabench/merge.py`). `max_duration` 3.5 h (2.2-2.7 h estimated).
- `kimi-vllm-2n-tp8pp2-bringup`: the same server, passes `warmup`, `conc1` and `conc64` (the step 2 milestone).
- `kimi-vllm-2n-tp8pp2-bringup-notune`: the same without FlashInfer's autotune (`--no-enable-flashinfer-autotune`, as
  vLLM's Kimi-K3 recipes).
- `kimi-vllm-2n-tp8pp2-dummy`: random weights (memory, graphs).

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

Before the study entry (its `eval` pass: lm-eval GSM8K, ARC-Challenge, GPQA diamond, MMLU-Pro), stage the eval
datasets into the HF cache that the offline nodes read (`data/`), on a data-transfer node (`ssh tri-dm1`; transfers
over 10 GB go there, these are small). `prepare` also checks the checkpoint against the Hub and fetches nothing that is
there:

```bash
ssh tri-dm1
B=$SCRATCH/trillium/new/k3-inf; unset PYTHONPATH PIP_CONFIG_FILE
export PATH=$B/src/.venv/bin:$PATH MILABENCH_BASE=$B MILABENCH_GPU_ARCH=cuda
milabench prepare --config $B/src/config/kimi_study.yaml --select kimi-vllm-2n-tp8pp2
```

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

| pass | output tok/s (per GPU) | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | power W, 2 nodes (est.) | J / output tok (est.) |
|---|---|---|---|---|---|---|---|
| conc1 | 111.3 (7.0) | 556 | 85 / 88 | 8.69 / 8.70 | 8.68 / 8.89 | 7014 | 63.0 |
| conc64 | 1304.7 (81.5) | 6523 | 1881 / 11689 | 26.7 / 64.1 | 20.6 / 30.1 | 9165 | 7.0 |

Power: main node's GPUs only (voir), times 2 (`scripts/analysis/kimi_study.py`). conc64's p99 TTFT is one engine stall,
not queueing: vLLM's JIT monitor logs Triton compiling `layer_norm_gated_fwd_kernel` in the pass's first second, ITL max
9.8 s, 22 of 128 requests at 11.4-11.7 s TTFT, the others at 2.32 s or less (also in job 1261890; conc1 had a CuTeDSL
compile of `FlashAttentionForwardSm100` in its first request). Since then: a throwaway `warmup` pass (64 x 1024 in, 64
out) before the measured passes, and Triton's cache in `cache/triton`, kept across jobs (`job.sh`).

**Autotune on and off**, both with the warm-up, at `79b0af94`, 2026-11-08, run side by side: `-bringup` job 1263424 on
`trig[0101-0102]` (weights 101-110 s: the checkpoint still in those nodes' page cache from job 1262824), `-notune` job
1263425 on `trig[0103-0104]` (cold, 674-693 s). Engine init 1441 s against 96 s. Every JIT the monitor logs falls in
the warm-up, none in conc1 or conc64. Both PASS, nodes "GPUs free", none drained; `ulimit -c` 0 in the batch tasks of all
four nodes and in their raylet and Ray workers (`/proc/<pid>/limits`).

| pass | autotune | output tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | power W, 2 nodes (est.) | J / output tok (est.) |
|---|---|---|---|---|---|---|---|
| conc1 | on | 111.25 | 85 / 89 | 8.69 / 8.70 | 8.68 / 8.84 | 7016 | 63.1 |
| conc1 | off | 105.22 | 102 / 106 | 9.14 / 9.14 | 9.14 / 9.19 | 6871 | 65.3 |
| conc64 | on | 2026.35 | 1458 / 2809 | 26.5 / 30.1 | 20.6 / 24.5 | 11884 | 5.9 |
| conc64 | off | 1479.77 | 2410 / 5987 | 28.9 / 47.1 | 24.6 / 27.1 | 10036 | 6.8 |

Repeat of the same config on the same nodes (1262824 and 1263424, conc1): 111.28 and 111.25 tok/s, TPOT 8.69 both.
Autotune buys 5.7% at conc1 and 37% at conc64 (TTFT p50 1.65x), far beyond that: the study runs with autotune and pays
its ~24 min at every start. The two runs used different node pairs; the campaign's E1 measured under 1% between them.

### The study on 2 nodes: K3-INF-01

`kimi-vllm-2n-tp8pp2`, job 1263799, `trig[0145-0146]`, 2026-11-08 02:45:55-04:49:02 (2:03:07), milabench `79b0af94`, exit
0 after 7370 s; both nodes "GPUs free", neither drained; `ulimit -c` 0 on both. Weights 676 s (PP0) and 669 s (PP1),
engine init 1443 s, ready 39 min after the job started; the 21 passes 52 min (3126 s, 2430 s of it measured), the eval
31 min. Estimated in the row: ready ~37 min, passes ~1 h, eval 35-60 min, 2.2-2.7 h in all: each part at or under it.
0 failed requests in every pass. Raw data: `runs/kimi-vllm-2n-tp8pp2.1263799/`; the tables from
`scripts/analysis/kimi_study.py` (`passes.csv`, `itl_by_context.csv`, `accuracy.csv`), recomputed from the `.data` files for
conc64, conc512 and ctx128k-c1 (throughput = requests x output length / duration, TTFT median from the per-request
samples, power = the main node's mean over the pass window, x2).

Per GPU = /16, per node = /2. Power and J / token are estimates: the main node's 8 GPUs, times 2; the PP1 node was never
sampled (voir samples the main node only).

| pass | in / out | output tok/s (/GPU, /node) | total tok/s (/GPU, /node) | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | W, 2 nodes (est.) | J / out tok (est.) |
|---|---|---|---|---|---|---|---|---|
| warmup (throwaway) | 1024 / 64 | 305 (19, 153) | 5189 (324, 2594) | 1907 / 12063 | 182.4 / 202.7 | 21.1 / 534.8 | 6020 | 19.7 |
| conc1 | 1024 / 256 | 111.1 (6.9, 56) | 556 (35, 278) | 85 / 88 | 8.70 / 8.71 | 8.69 / 8.90 | 7022 | 63.2 |
| conc4 | 1024 / 256 | 326.7 (20.4, 163) | 1634 (102, 817) | 256 / 288 | 11.27 / 12.04 | 11.26 / 11.47 | 9549 | 29.2 |
| conc16 | 1024 / 256 | 876.3 (54.8, 438) | 4381 (274, 2191) | 693 / 721 | 15.63 / 17.96 | 15.55 / 16.09 | 9729 | 11.1 |
| conc64 | 1024 / 256 | 2002.6 (125.2, 1001) | 10013 (626, 5007) | 1462 / 2968 | 26.42 / 30.11 | 20.88 / 23.56 | 11972 | 6.0 |
| conc128 | 1024 / 256 | 2243.0 (140.2, 1121) | 11215 (701, 5607) | 2837 / 9338 | 44.76 / 61.15 | 27.18 / 450.2 | 11104 | 5.0 |
| conc256 | 1024 / 256 | 2865.0 (179.1, 1433) | 14325 (895, 7163) | 2698 / 14761 | 70.72 / 93.22 | 37.80 / 1342 | 11501 | 4.0 |
| conc512 | 1024 / 256 | 3466.4 (216.6, 1733) | 17332 (1083, 8666) | 4176 / 21833 | 131.6 / 143.3 | 51.25 / 1390 | 11632 | 3.4 |
| conc1024 | 1024 / 256 | 3451.8 (215.7, 1726) | 17259 (1079, 8629) | 40131 / 59875 | 140.0 / 145.9 | 50.39 / 1397 | 11558 | 3.3 |
| ctx1k-c1 | 1024 / 128 | 107.5 (6.7, 54) | 968 (60, 484) | 85 / 88 | 8.70 / 8.73 | 8.69 / 8.99 | 7152 | 66.5 |
| ctx8k-c1 | 8192 / 128 | 84.2 (5.3, 42) | 5475 (342, 2738) | 399 / 402 | 8.82 / 8.83 | 8.82 / 9.01 | 7678 | 91.1 |
| ctx32k-c1 | 32768 / 128 | 55.0 (3.4, 28) | 14138 (884, 7069) | 1168 / 1176 | 9.13 / 9.14 | 9.11 / 9.48 | 9171 | 166.7 |
| ctx64k-c1 | 65536 / 128 | 39.1 (2.4, 20) | 20043 (1253, 10021) | 2087 / 2100 | 9.37 / 9.38 | 9.34 / 9.85 | 10616 | 271.7 |
| ctx128k-c1 | 131072 / 128 | 21.5 (1.3, 11) | 22017 (1376, 11009) | 4555 / 4607 | 11.06 / 12.06 | 9.67 / 28.92 | 11394 | 530.5 |
| ctx1k-c16 | 1024 / 128 | 766.3 (47.9, 383) | 6896 (431, 3448) | 692 / 717 | 15.58 / 20.27 | 15.51 / 16.04 | 9755 | 12.7 |
| ctx8k-c16 | 8192 / 128 | 220.5 (13.8, 110) | 14331 (896, 7165) | 3501 / 11821 | 37.24 / 120.7 | 13.68 / 1350 | 9016 | 40.9 |
| ctx32k-c16 | 32768 / 128 | 78.0 (4.9, 39) | 20041 (1253, 10021) | 11009 / 18809 | 115.4 / 246.4 | 17.95 / 1471 | 9895 | 126.9 |
| ctx64k-c16 | 65536 / 128 | 44.0 (2.7, 22) | 22571 (1411, 11285) | 14714 / 46235 | 249.9 / 359.4 | 20.25 / 1620 | 10705 | 243.3 |
| ctx128k-c16 | 131072 / 128 | 19.5 (1.2, 10) | 19987 (1249, 9994) | 30368 / 101805 | 594.7 / 786.8 | 25.14 / 1965 | 10451 | 535.9 |
| decodelong-c1 | 1024 / 4096 | 114.3 (7.1, 57) | 143 (9, 71) | 87 / 88 | 8.73 / 8.73 | 8.73 / 8.79 | 6905 | 60.4 |
| decodelong-c16 | 1024 / 4096 | 968.9 (60.6, 484) | 1211 (76, 606) | 699 / 720 | 16.42 / 16.57 | 16.33 / 17.18 | 9883 | 10.2 |

Saturation: output 2003 tok/s at conc64, 2865 at conc256, 3466 at conc512 and 3452 at conc1024. The server runs at most
512 sequences (`--max-num-seqs`, the study's flag): conc1024 queues, its TTFT p50 is 40 s. The p99 ITL of 0.45-2 s from
conc128 up, and in the c16 context passes, is decode waiting behind 16,384-token prefill chunks (chunked prefill).

Cost of context length, one request (c1): TTFT 85 ms at 1k, 399 at 8k, 1168 at 32k, 2087 at 64k, 4555 at 128k (prefill
20.5k tok/s at 8k, 28-31k from 32k: close to linear in the prompt); TPOT 8.70, 8.82, 9.13, 9.37, 11.06 ms (+27% at 128k).
In decodelong-c1 the ITL stays at 8.70, 8.73, 8.75 ms over context positions 1-2k, 2-4k, 4-8k (`itl_by_context.csv`): 69
of the 93 layers are KDA, with a fixed-size state. At 16 concurrent requests the prompts of the others dominate: TTFT p50
0.69, 3.5, 11.0, 14.7, 30.4 s from 1k to 128k.

Accuracy (lm-eval through the server, `eval` pass, temperature 0, up to 8192 generated tokens; ± stderr):

| GSM8K (1319) | ARC-Challenge (1172) | GPQA diamond (198) | MMLU-Pro (2002 of 12,032) |
|---|---|---|---|
| 97.5 ± 0.4 | 98.5 ± 0.4 | 84.8 ± 2.6 | 87.7 ± 0.7 |

MoE backend `FLASHINFER_TRTLLM_MXFP4_MXFP8`: activations in MXFP8 (vLLM 0.32's default on B200), the checkpoint is W4A16.

Not tested yet: prefix caching, DCP and speculative decoding stay off as in the MI355X study.

## SemiAnalysis InferenceX (formerly InferenceMAX): cited, not ours

Read 2026-11-08. Configs: `github.com/SemiAnalysisAI/InferenceX`, `inferencex-e2e/configs/nvidia-master.yaml` at
`79a44b90` and `amd-master.yaml` at `9b45b789` (the last commits before 2026-10-28, when Kimi-K3.1-Pro replaced Kimi-K3).
Results: `https://inferencex.semianalysis.com/run/kimi-k3-on-b200` and `.../run/kimi-k3-on-mi355x` (the pages say their
data may be modified or out of date). The dashboard's metric is published tokens/s per GPU at a per-user speed.

| | B200 | MI355X |
|---|---|---|
| entry | `kimik3-fp4-b200-dynamo-vllm-agentic-dspark` | `kimik3-fp4-mi355x-vllm-agentic-mtp`, `kimik3-fp4-mi355x-atom-agentic-mtp` |
| engine | Dynamo + vLLM, `vllm/vllm-openai:v0.32.0` | vLLM ROCm nightly (`df6603f0`); ATOM `nightly_202610131505` |
| GPUs, layout | 16 (2 nodes): TP8 x PP2, DCP8 | 8 (1 node): TP8, DCP8; DRAM KV offload from conc 4-14 up |
| features | FP4 (MXFP4) experts, MTP / DSpark speculative decoding, prefix caching | the same, plus KV offload |
| workload | AgentX only: multi-turn agentic-coding trace replay, contexts to 1M tokens, no fixed ISL/OSL | AgentX only |
| concurrency | 1, 4, 8, 14, 24, 32, 48, 56, 72, 80, 96 | vLLM 1-60, ATOM 1-72 |
| runs | up to 2026-10-26 | 2026-10-14 to 2026-10-27 |
| tok/s/GPU at 30 / 50 / 75 / 100 / 150 tok/s/user | 9,605 / 6,223 / 3,809 / 2,677 / 1,653 | 14,754 / 10,918 / 7,085 / 5,035 / 2,580 (best: vLLM) |

Mapping to our passes: none matches. InferenceX runs Kimi-K3 only on AgentX, with prefix caching (cached prompt
tokens count) and speculative decoding; our passes are random tokens of fixed length with both off (as the MI355X study).
Only the layout and engine line up on B200 (2 x 8 B200, TP8 x PP2, vLLM 0.32.0). Matching AgentX would be a separate
server entry (prefix caching, DSpark draft, DCP8) and the `agentx-harness` replay as the client, not a pass.
Fixed ISL/OSL on both B200 and MI355X exists only for DeepSeek-R1, at 8192 in / 1024 out, concurrency 4 to 256
(TRT-LLM, SGLang, Dynamo on B200; SGLang, ATOM on MI355X): the study has no 8k/1k pass (its 8k passes generate 128).
To set the scenario beside theirs: 8k/1k passes at concurrency 4, 16, 64 and 256 (a different model on their side).

## 4 nodes (step 4)

The same server over four nodes: `kimi-vllm-4n-tp8pp4`, TP8 inside each node, PP4 across them, the study's passes
unchanged. The one server change beyond the layout: `--max-num-seqs` 1024 instead of 512.

Why: the 2-node server's limit is its 512-sequence cap. conc512 gives 3466 output tok/s, conc1024 3452 with a TTFT p50
of 40 s (queueing). Memory is not the limit at that cap only: its KV pool (vLLM's "3,606,273 tokens", log line 562, is
27.09 x the 133,120 max-model-len, not tokens of 1,280-token requests) was 26.7% in use with 256 sequences running and
53.4% with 512: full at about 960 such requests, about 1.2M tokens.

`--max-num-seqs` for PP4, from the KV capacity (estimates):
- weights per GPU: 95.7-96.1 GiB at PP2 (47 and 46 layers), about 48 GiB at PP4 ([23,23,24,23] layers, vLLM's split);
- KV memory per GPU: 51.4-53.5 GiB at PP2, so about 100 GiB at PP4 (the same budget, about 1.9x);
- the KV and KDA state a GPU keeps per sequence is for its own layers only: half as many at PP4;
- capacity: about 960 x 1.9 x 2 = 3,650 sequences, or 960 x 1.9 = 1,820 if the per-sequence state did not shrink
  (review: the last stage binds, at 7 MLA layers, about 1.86x: about 3,400);
- 1024 (the study's highest concurrency, conc1024) then fills about 30% of the pool, 56% on the memory-only bound.

Estimates, to check in the bring-up: conc1 TPOT about the same, 8.7 ms plus two more inter-node hops per token (the
same weights read per token, over 4 stages instead of 2); saturated output about 2x the 2-node 3466 tok/s, with up to
1024 sequences in flight over twice the GPUs. Power on 4 nodes is extrapolated: `scripts/analysis/kimi_study.py`
multiplies the main node by the node count (x4), three nodes unsampled.

Bring-up, `kimi-vllm-4n-tp8pp4-bringup` (warm-up, conc1, conc64, conc1024: the changed flag at work), in a 1 h job: each node loads its ~24 shards, about
390 GB instead of 780 (~6 min cold at the ~1.1 GB/s per node of job 1263799). The autotune does not hit the cache: the
key is the vLLM config hash, new for PP4, and the MoE entries miss at every start anyway (job 1262548). Its per-GPU
shapes are the TP8 ones of 2 nodes, so expect the same ~24 min. Ready after ~33 min, done after ~46 (estimates);
`job.sh` stops the run cleanly 5 min before `--time`.

**Bring-up, job 1264956**, `trig[0117-0120]`, 2026-11-08 05:59:03-06:45:24 (46:21), `fb31a435`, exit 0 after 2762 s; all
four nodes "GPUs free", none drained; `ulimit -c` 0 in the batch tasks, raylets and Ray workers of all four. vLLM:
"Hidden layers were unevenly partitioned: [23,23,24,23]"; weights 46.3 / 47.8 / 49.88 / 48.05 GiB per GPU (PP0-PP3);
KV memory 99.7-105.4 GiB; "GPU KV cache size: 13,042,751 tokens, Maximum concurrency for 133,120 tokens per request:
97.98x" (2 nodes: 27.09x). Each rank still reads all 96 shards: loads 721-759 s, not the ~6 min estimated. Engine init
1296 s (autotune), ready at 38.2 min. Every JIT falls in the warm-up (last at 06:37:58, conc1 from 06:38:24).

| pass | done / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | 2 nodes, output tok/s |
|---|---|---|---|---|---|---|---|
| conc1 | 60 / 0 | 109.0 | 545 | 85 / 87 | 8.87 / 8.88 | 8.87 / 8.95 | 111.1 (TPOT 8.70) |
| conc64 | 128 / 0 | 1012.2 | 5061 | 1883 / 20123 | 22.7 / 93.9 | 18.1 / 18.9 | 2002.6 |
| conc1024 | 2048 / 0 | 3865.3 | 19327 | 6368 / 59048 | 213.6 / 284.9 | 52.6 / 2826 | 3451.8 |

The estimates did not hold beyond conc1. conc1024 gains 12%, not 2x: with 1024 sequences admitted (KV usage 29.5%,
about the 30% estimated) the engine is prefill-bound. It logs 22-24k prompt tok/s while generation falls to ~330 tok/s,
then bursts of 15-17k generation tok/s between prefill waves. conc64 halves: for its first ~25 s the engine logs
1.8-2.6k prompt tok/s and almost no generation (22 requests with TTFT of about 20 s), with no JIT or other event logged.
Not explained; the 2.5 h study run on 4 nodes would show whether it comes back.

Dropped: two 2-node replicas behind one endpoint (vLLM DP2). vLLM's fused MoE takes its size from the DP group, so the
experts would be sharded over DP x TP = 16 GPUs across both replicas, with the engines in lockstep: not two
independent servers (review, 2026-11-08).

### The study on 4 nodes: K3-INF-02

`kimi-vllm-4n-tp8pp4`, job 1265338, `trig[0139-0142]`, 2026-11-08 06:49:40-08:36:36 (1:46:56), checkout `fb31a435`,
milabench exit 0, clean Ray stop, no drains. Weights 704-749 s, engine init 1300 s, ready at 07:27:27 (37.8 min).
0 failed requests in 21 passes. Raw data: `runs/kimi-vllm-4n-tp8pp4.1265338/`; tables from `scripts/analysis/kimi_study.py`.

Per GPU = /32, per node = /4. Power and J / token are extrapolated, and uncertain: the main node drew 553 W per GPU at
conc1024 on 4 nodes against 722 W on 2 nodes, so its x4 may not stand for the other stages. The main node's 8 GPUs, times 4; the other three
nodes were never sampled.

| pass | in / out | output tok/s (/GPU, /node) | total tok/s (/GPU, /node) | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | W, 4 nodes (extrap.) | J / out tok (extrap.) |
|---|---|---|---|---|---|---|---|---|
| warmup (throwaway) | 1024 / 64 | 910.6 (28.5, 228) | 15480 (484, 3870) | 2138 / 3372 | 37.05 / 62.12 | 17.43 / 687.1 | 14677 | 16.1 |
| conc1 | 1024 / 256 | 108.8 (3.4, 27) | 544 (17, 136) | 86 / 90 | 8.89 / 8.90 | 8.89 / 8.96 | 11046 | 101.5 |
| conc4 | 1024 / 256 | 353.0 (11.0, 88) | 1765 (55, 441) | 237 / 285 | 10.47 / 11.09 | 10.36 / 10.55 | 14442 | 40.9 |
| conc16 | 1024 / 256 | 873.0 (27.3, 218) | 4365 (136, 1091) | 683 / 694 | 15.74 / 18.03 | 15.67 / 16.16 | 14277 | 16.4 |
| conc64 | 1024 / 256 | 1111.5 (34.7, 278) | 5557 (174, 1389) | 1900 / 2853 | 27.23 / 83.49 | 19.54 / 20.66 | 13658 | 12.3 |
| conc128 | 1024 / 256 | 2017.6 (63.0, 504) | 10088 (315, 2522) | 3232 / 15947 | 31.67 / 81.82 | 22.88 / 28.21 | 16333 | 8.1 |
| conc256 | 1024 / 256 | 2482.9 (77.6, 621) | 12415 (388, 3104) | 4951 / 26752 | 62.68 / 131.1 | 28.37 / 894.6 | 16385 | 6.6 |
| conc512 | 1024 / 256 | 4037.6 (126.2, 1009) | 20188 (631, 5047) | 11286 / 22191 | 81.91 / 124.0 | 39.67 / 2712 | 18890 | 4.7 |
| conc1024 | 1024 / 256 | 4292.5 (134.1, 1073) | 21462 (671, 5366) | 7789 / 44224 | 210.7 / 235.2 | 49.32 / 2831 | 17707 | 4.1 |
| ctx1k-c1 | 1024 / 128 | 105.4 (3.3, 26) | 948 (30, 237) | 86 / 89 | 8.89 / 8.89 | 8.89 / 8.96 | 11201 | 106.3 |
| ctx8k-c1 | 8192 / 128 | 82.7 (2.6, 21) | 5374 (168, 1343) | 404 / 407 | 9.01 / 9.01 | 9.01 / 9.12 | 11730 | 141.9 |
| ctx32k-c1 | 32768 / 128 | 58.7 (1.8, 15) | 15086 (471, 3771) | 999 / 1006 | 9.31 / 9.31 | 9.30 / 9.49 | 13630 | 232.2 |
| ctx64k-c1 | 65536 / 128 | 46.6 (1.5, 12) | 23900 (747, 5975) | 1536 / 1547 | 9.54 / 9.55 | 9.53 / 9.76 | 15917 | 341.7 |
| ctx128k-c1 | 131072 / 128 | 32.0 (1.0, 8) | 32774 (1024, 8194) | 2748 / 2758 | 9.88 / 9.89 | 9.87 / 10.19 | 18840 | 589.2 |
| ctx1k-c16 | 1024 / 128 | 767.8 (24.0, 192) | 6910 (216, 1728) | 684 / 697 | 15.57 / 20.24 | 15.56 / 16.19 | 14377 | 18.7 |
| ctx8k-c16 | 8192 / 128 | 216.9 (6.8, 54) | 14096 (440, 3524) | 3288 / 15759 | 29.04 / 131.9 | 12.02 / 114.4 | 13700 | 63.2 |
| ctx32k-c16 | 32768 / 128 | 116.6 (3.6, 29) | 29977 (937, 7494) | 8780 / 15706 | 69.16 / 129.8 | 14.75 / 1906 | 16955 | 145.4 |
| ctx64k-c16 | 65536 / 128 | 59.6 (1.9, 15) | 30600 (956, 7650) | 17483 / 30651 | 133.6 / 257.3 | 20.54 / 2016 | 17226 | 288.8 |
| ctx128k-c16 | 131072 / 128 | 31.3 (1.0, 8) | 32131 (1004, 8033) | 32210 / 75027 | 271.3 / 631.2 | 25.30 / 2250 | 18266 | 582.7 |
| decodelong-c1 | 1024 / 4096 | 111.5 (3.5, 28) | 139 (4, 35) | 88 / 88 | 8.95 / 8.95 | 8.95 / 9.04 | 10913 | 97.9 |
| decodelong-c16 | 1024 / 4096 | 985.1 (30.8, 246) | 1231 (38, 308) | 733 / 742 | 16.09 / 16.24 | 16.16 / 16.64 | 15025 | 15.3 |

Output tok/s, 2 nodes (K3-INF-01) against 4: conc1 111.1 / 108.8, conc64 2003 / 1112, conc128 2243 / 2018, conc256
2865 / 2483, conc512 3466 / 4038, conc1024 3452 / 4292; ctx-c16 from 1k to 128k 766 / 768, 221 / 217, 78 / 117, 44 / 60,
20 / 31; ctx128k-c1 21.5 / 32.0 (TTFT 4555 / 2748 ms).

Against the ~2x estimated for 4 nodes, the measured gain is +16% at conc512 and +24% at conc1024 (ratio 1.244). The
bring-up measured 3865 tok/s at conc1024 on 4 nodes (job 1264956), this run 4292, 11% apart: that spread limits how firm
the +24% is.

Accuracy, 2 / 4 nodes, ± stderr: GSM8K (1319) 97.5 ± 0.4 / 98.0 ± 0.4, ARC-C (1172) 98.5 ± 0.4 / 98.5 ± 0.3, GPQA-D
(198) 84.8 ± 2.6 / 83.8 ± 2.6, MMLU-Pro (2002) 87.7 ± 0.7 / 86.8 ± 0.8. MMLU-Pro's -0.95 is within the stderr of the
difference (about 1.05), but paired on the same questions 47 were right only on 2 nodes and 28 only on 4 (z about 2.2),
and 4 nodes has 61 unparsed answers against 45 (`samples_milabench_mmlu_pro_*.jsonl`). Observed; no cause claimed.

**Engine stalls.** vLLM logs its throughput every 10 s. Intervals with requests running and no token at all:
- 4 nodes: conc64, 07:33:21-31 (64 running); conc128, 07:34:11-21 (128 running). These two are the only zero windows.
- 2 nodes: none.
- The bring-up's conc64 (job 1264956, 06:41:16-26) was a near-stall, not a zero window: 1.8-2.6k prompt and 3.8-28
  generation tok/s with 64 running.

Near zero (under 10% of the pass's output rate in both prompt and generation): none in either run. conc256 on 4 nodes
starts slowly, generation at 17-29 tok/s over its first 30 s while prefill runs at 2.6-14.6k tok/s (2 nodes: 10-15k).
With the stalled interval taken out (output tokens / (duration - 10 s), derived, not measured; a lower bound, since a
zero window means at least 10 s without a token): conc64 1682 tok/s,
still 16% under 2 nodes' 2003; conc128 2915 tok/s, 30% over 2 nodes' 2243. Nothing else in the log at those times: no
Ray, NCCL, preemption or scheduler message, only a CuTeDSL "Argument aux_data ... cannot be converted to a JitArgument"
warning that the 2-node runs print in the same pass without stalling. The cause is not known.

**JIT in a measured pass, both runs:** CuTeDSL compiles `FlashPrepareScheduler` and `FlashAttentionForwardCombine` at the
start of ctx8k-c16 (2 nodes 03:56:23-29, 4 nodes 07:55:52-56:03), which the warm-up (1k prompts) does not cover: part of
ctx8k-c16's p99 TTFT (11.8 s and 15.8 s). Not fixed.

**Node metrics:** these jobs run no `benchmate.nodemon`, so a stall of the engine cannot be told from a node or fabric
event, and the power is extrapolated from one node. Adding it to `job.sh`, as the GLM-5, slime and K3-TRN jobs run it:
`BENCHMATE_NODEMON_DIR=$MILABENCH_BASE/nodemon/<run>`, then before `milabench run` one background
`srun --overlap --propagate=CORE --ntasks-per-node=1 $MILABENCH_BASE/venv/torch/bin/python -m benchmate.nodemon --bench $ENTRY`
(one sampler per host, its lock), and in the clean end a SIGTERM to that step and a wait, before the GPU check. About 8
lines; `pynvml` is in the venv. Not added yet.

## K3-XN: Kimi-K3 across nodes

Experts spread over the nodes (part 1), DeepEP's needs from the node (part 2), prefill and decode on separate nodes
(part 3), against K3-INF-01/02. Its own base, `$X=$SCRATCH/trillium/new/k3-xn` (`MILABENCH_BASE`): its checkout `src/`
and milabench CLI venv `src/.venv`, and links to K3-INF's vLLM venv, checkpoint and FlashInfer cubins, which it reads
only. Jobs are named `nw-k3-xn-<entry>` (`JOB_PREFIX`).

```bash
X=$SCRATCH/trillium/new/k3-xn K=$SCRATCH/trillium/new/k3-inf; mkdir -p $X/slurm $X/runs $X/extra $X/cache/{flashinfer,vllm}; cd $X
git clone -b trillium-new https://github.com/milabench/milabench.git src
unset PYTHONPATH PIP_CONFIG_FILE; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$X/uv-cache UV_LINK_MODE=copy
uv venv --python-preference only-managed --python 3.12 --seed src/.venv
VIRTUAL_ENV=$X/src/.venv uv pip install -e ./src "huggingface_hub[hf_xet]" nvidia-ml-py
ln -s $K/venv venv; ln -s $K/data data; ln -s $(readlink $K/cache/flashinfer/cubins) cache/flashinfer/cubins
# Triton's cache seeded with K3-INF's: 16 ranks on 2 nodes compiling the same kernel into an empty cache on VAST raced
# (FileNotFoundError on the kernel's .json, Inductor error on ranks 9, 10, 13 on trig0146, job 1269990); every kernel that run
# compiled has the same hash in K3-INF's cache
mkdir -p cache/triton; cp -an $K/cache/triton/. cache/triton/
```

### Part 2: driver check

`/opt/slurm/bin/.sbatch --export=NONE --get-user-env --nodelist=<node> src/scripts/trillium/kimi/driver_check.sh` (1 node,
5 min, reads only). Job 1269881, `trig0147`, 2026-11-08 18:36, 33 s:

- `/proc/driver/nvidia/params`: `EnableStreamMemOPs: 1`, `RegistryDwords: "PeerMappingOverride=1"` (from
  `/etc/modprobe.d/nvidia.conf`), what NVSHMEM's IBGDA needs. Driver 595.91.07 (open kernel module).
- Modules: `nvidia_peermem` loaded; `gdrdrv` not loaded, no `/dev/gdrdrv` (GDRCopy's kernel module). memlock unlimited.
- HCAs: NCCL uses 4 x 400 Gb/s InfiniBand, `mlx5_0`, `mlx5_1`, `mlx5_4`, `mlx5_9` (adaptive routing on), each `PIX` to
  one GPU (0, 2, 4, 6); GPUs 1, 3, 5, 7 reach them through the PCIe host bridge (`NODE`).
- NCCL 2.29.7 in the vLLM venv (torch 2.13.0 built with it). With NVLink and shared memory off, NCCL reports
  `DMA-BUF is available` and `GPU Direct RDMA Enabled` on all 4 HCAs, for its IB transport and for its GIN GDAKI backend.
- vLLM 0.32's `deepep_v2` backend needs DeepEP v2 (`ElasticBuffer`) and NCCL >= 2.30.4 at run time
  (`vllm/utils/import_utils.py`): the venv's 2.29.7 is too old, a software change (pip `nvidia-nccl-cu13`, or the
  `nccl/2.30.7` and `2.31.2` modules), not a driver one. DeepEP is not installed, and its sources are not on the cluster:
  its own limits on top-k (K3: 16) and experts (896) are not read. vLLM's DeepEP low-latency path takes hidden sizes
  2048-8192 from a list: K3's latent 3,584 is rounded up to 4,096.

### DeepEP (decided 2026-11-08: the driver settings are there)

DeepEP's current main (V2.5, 2026-09-29) renamed the API vLLM 0.32's `deepep_v2` calls (`deep_ep.ElasticBuffer`) and
needs NCCL >= 2.32.3; `a56d615` (2026-09-16) is the last commit with it, and needs NCCL >= 2.30.4 (vLLM checks the same).
Its V2 kernels run on NCCL Gin and are compiled at run time (DeepJIT: nvcc and cuobjdump from `CUDA_HOME`); the build
also compiles V1's legacy kernels against the venv's NVSHMEM 3.4.5. Limits in its code, against K3: top-k <= 32 (16),
experts <= 2,048 (896), experts per rank <= 256 and dividing evenly (56 at EP16, 28 at EP32), hidden bytes aligned (7,168).

`deepep_build.sh` (1 node; the login-node downloads at its top): `venv/deepep`, a copy of K3-INF's venv (1,164 s for
16 GB, 185k files), NCCL 2.30.7 and nixl 1.5.2 (`nixl-cu13` and its shim) installed into it, DeepEP built for sm_100 with
CUDA 13.2 from cvmfs (not its module, which loads NCCL 2.29.7 too). Builds 1270208, 1270384, 1270408 failed on the
environment: `ccache` writing to the read-only `$HOME`, then `-lcuda` and `-l:libnccl.so.2` not found by cvmfs's ld
(`LIBRARY_PATH` with the venv's NCCL and CUDA's stub); 1270433 built (46 s), then: `ElasticBuffer` present, run-time
NCCL 2.30.7 (torch still reports its build's 2.29.7), vLLM's `deep_ep_v2_unavailable_reason()` None, `nixl_agent`
imports. Its 1-node test stopped on Triton's cache in `$HOME` (`deep_ep.utils.math` is `torch.compile`-d): both scripts
now keep it node-local. `gdrdrv` was not needed for any of this.

`deepep_test.sh` (DeepEP's `tests/elastic/test_ep.py` across the job's nodes, at K3's shape), job 1270478,
`trig[0146-0147]`, 3:26, both cases exit 0 with the test's correctness checks: 16 ranks, hidden 3,584, top-16 of 896,
FP8 dispatch, BF16 combine. Per rank (SO: over the network, SU: within the node):

| tokens per rank | dispatch | combine |
|---|---|---|
| 128 | 120 us (8 GB/s SO, 41 GB/s SU) | 88-139 us (12-20 GB/s SO) |
| 4,096 | 1.15 ms (28 GB/s SO, 142 GB/s SU) | 2.07 ms (29 GB/s SO, 147 GB/s SU) |

What the test's GB/s count (`tests/elastic/test_ep.py`): logical payload bytes per token (FP8 data, scales, top-k
indices and weights), over the dispatch or combine kernel's time, one direction. SO: tokens times the nodes holding one of
their experts, the token's own node included unless `--ignore-local-traffic`; SU: tokens received, from this GPU too. With
top-16 over 2 nodes nearly every token touches both nodes, so job 1270478's 28-29 GB/s SO counts the local half. Job
1271226 (`trig[0105-0106]`, `--ignore-local-traffic`, now in `deepep_test.sh`): dispatch at 4,096 tokens 14 GB/s SO per GPU,
124-125 GB/s SU, 1.15 ms (the same kernel time); combine 14-37 GB/s SO, 0.80-2.07 ms by rank; at 128 tokens dispatch 4 GB/s
SO, 121-125 us, combine 6-10 GB/s SO, 90-141 us. 14 GB/s sent per GPU is about 56% of the 25 GB/s per GPU of 4 x 400 Gb/s
HCAs for 8 GPUs (est., one direction). The job's kernel log has 128 `NVRM: GPU<n> refcntRequestReference_IMPL: Failed to enter state 1` lines,
with the test's profiler (`profiler_start` / `profiler_stop`); none in the vLLM or build jobs on the same nodes.

**vLLM on `deepep_v2`, 2 nodes, no autotune.** vLLM picks `DeepEPV2All2AllManager`, `DeepEPV2PrepareAndFinalize`,
`TrtLlmMxfp4ExpertsModular`; its NCCL Gin check passes.
- Job 1270544, `trig[0146-0147]`, 20:10-20:25, FAILED at start: weights 128.31 GiB, CUDA graphs 6.41 GiB (all-gather
  backend: 3.09), KV memory 5.0 GiB (13.61), "max_num_seqs (512) exceeds available Mamba cache blocks (250)". Weights
  60 s on trig0146 (warm), 447 s on trig0147: a cold load with the EP filter (TP8 x PP2: 669-676 s).
- Job 1270725, `-deepep-shard-bringup-notune` (`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1`; since then the setting of every
  EP16 entry, both backends, and this one is `-deepep-bringup-notune`), `trig[0146-0147]`, 20:27-20:52,
  COMPLETED, both nodes "GPUs free". **trig0146 drained during the run (ECC):** `health.chk.gpu.ecc.error` from 20:40:32;
  GPU 2 (3A:00.0) 109 corrected DRAM errors, none uncorrected, the other GPUs none (one read through the job, 20:43).
  Weights 107.41 GiB (-20.9, the estimate), KV 26.47 GiB, "1,004,792 tokens, 7.55x" per engine (all-gather: 517,438,
  3.89x). Init 300 s (CUDA graphs 189 + 70 s). GSM8K first 200: 196 / 200 (K3-INF-01: 197 on the same; 2 right only
  there, 1 only here), answers 748 characters median (751), none empty or garbled.

| pass | output tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | all-gather (1270053) | TP8 x PP2 notune (1263425) |
|---|---|---|---|---|---|---|
| conc1 | 30.99 | 208 / 390 | 31.55 / 31.57 | 31.55 / 31.68 | 29.95, TPOT 32.88 | 105.22, TPOT 9.14 |
| conc64 | 655.54 | 2120 / 23803 | 50.39 / 134.6 | 45.86 / 48.64 | 1093.07, TPOT 45.35 | 1479.77, TPOT 28.9 |

Which DeepEP mode (vLLM 0.32, `prepare_finalize/deepep_v2.py`, `all2all_utils.py`): DeepEP V2 has no separate
low-latency and high-throughput kernels; vLLM picks its "decode mode" (`do_expand=False`, `do_cpu_sync=False`: CUDA-graph
safe, receive buffers sized to the DP-padded batch rounded up to a power of 2, one JIT kernel per size) whenever CUDA
graphs are on (`use_cudagraph = not enforce_eager`), for the engine's whole life: prefill batches use it too. Its "prefill
mode" runs only with `--enforce-eager`. The low-latency kernels are V1's (`--all2all-backend deepep_low_latency`,
NVSHMEM): they assert `num_topk <= 11` in dispatch and combine (`kNumMaxTopK = 11`, `csrc/kernels/legacy/internode_ll.cu`)
and take hidden sizes from a list without 3,584; vLLM would also need a batched-format expert kernel (Marlin, not
TRTLLM). Not runnable for K3's top-16: not tried.

conc1 is the same on both backends. Estimated from DeepEP's test (about 120 us per dispatch and 90-140 us per combine at
128 tokens per rank, over 92 MoE layers): about 24 ms of all-to-all per decode step; the profile below measured 19.8 ms. conc64 has a 23.8 s p99 TTFT (a stall, cause not determined; this layout logs no engine throughput lines), and
DeepEP and the sharded shared experts change together here: their shares are not separable, since DeepEP without the
sharding does not start.

**Both backends, sharded shared experts, `-diag` bring-ups, 2026-11-08 21:05-21:32** (the bring-up's passes, then a conc1
and a conc64 pass under vLLM's torch profiler, 10 iterations after 20, per-kernel time per rank;
`profile_summary.py` sums a trace by kind, divided by its steps). DeepEP job 1271240 on `trig[0101-0102]`, all-gather job
1271244 on `trig[0103-0104]`, same time, different node pairs. Both COMPLETED, nodes "GPUs free".

| | all-gather (1271244) | DeepEP (1271240) |
|---|---|---|
| weights / KV per GPU | 127.13 / 14.79 GiB | 107.41 / 26.47 GiB |
| KV per engine | 562,563 tokens, 4.23x | 1,004,792 tokens, 7.55x |
| conc1 tok/s, TPOT p50 | 29.70, 33.18 ms | 30.99, 31.51 ms |
| conc64 tok/s, TTFT p50 / p99, ITL p50 | 1056.30, 1306 / 6450 ms, 45.24 ms | 665.77, 2093 / 23070 ms, 45.76 ms |
| conc64 again (`conc64-profile`) | 1072.07, 1423 / 4189 ms, 43.08 ms | 961.98, 2053 / 2724 ms, 45.85 ms |
| GSM8K first 200 | 197 (K3-INF-01: 197) | 196 |

Kernel time per decode step, rank 0, profiled (the profiler slows conc1 to 25.43 tok/s on all-gather):

| ms per step | all-gather conc1 | all-gather conc64 | DeepEP conc1 | DeepEP conc64 |
|---|---|---|---|---|
| all-to-all between the nodes | 13.6 (all-gather 9.6, reduce-scatter 4.0) | 20.0 (14.1, 6.0) | 19.8 (dispatch 10.0, combine 9.7) | 23.8 (10.7, 13.1) |
| dense GEMMs | 10.0 | 12.1 | 7.0 | 9.8 |
| MoE expert GEMMs | 6.7 | 6.9 | 1.6 | 7.1 |
| TP collectives in the node | 1.4 | 2.0 | 6.6 | 3.7 |
| routing, attention, other | 6.4 | 6.6 | 6.0 | 7.1 |
| sum | 38.1 | 47.7 | 40.9 | 51.5 |

Measured, not estimated: the all-to-all is 36-48% of a decode step on both backends (one rank; the profiler's own
overhead included), DeepEP's dispatch and combine cost more than NCCL's all-gather and reduce-scatter at these batch sizes,
and the sharded shared experts move time from dense GEMMs to the node's TP collectives. (b): the conc64 drop is two
things. A one-time stall: the first conc64 on DeepEP waits ~20 s more for prefill, its repeat does not (p99 TTFT 2.7 s),
in both DeepEP runs (1270725, 1271240); inferred, not logged: DeepEP JIT-compiling its kernels for batch sizes the warm-up
never made (vLLM rounds the size up to a power of 2 and DeepEP compiles one kernel per size, `deepep_v2.py`; the warm-up's
64 equal requests never mix prefill and decode). The study entries now run a second warm-up shaped like conc64. Then a
steady 10% (962 against 1072 tok/s on the repeat), in prefill: TTFT p50 2.05 against 1.42 s, decode ITL about the same.
vLLM's `--enable-logging-iteration-details` logged nothing here (it needs the engine's stats logging, off in these DP
servers; no throughput lines either).

### Part 1: the experts over 2 nodes (EP16)

`kimi-vllm-2n-dp2tp8-ep16` (`config/kimi_study.yaml`): one TP8 engine per node joined by vLLM's data parallelism
(`multinode: dp`, `benchmate.server_worker` on the second node, no Ray), `--enable-expert-parallel`: vLLM's
`FusedMoEParallelConfig.make` sets EP = DP x TP = 16, 56 whole experts of the 896 on each GPU. K3's model takes its
sequence-parallel MoE path (PP 1, TP > 1, DP > 1, EP: `vllm/models/kimi_k3/nvidia/model.py`): attention and shared
experts stay TP8 in the node, the latent down-projection runs before the dispatch, so tokens travel at 3,584.
All-to-all `allgather_reducescatter` (vLLM 0.32's default, NCCL through the EP group's PyNccl; no driver setting):
each MoE layer all-gathers the 16 ranks' tokens (quantized to MXFP8 first) and reduce-scatters the outputs. DP16 x TP1
(attention replicated on every GPU) does not fit: about 90 GB of experts plus 114 GB of BF16 weights per GPU (est.).
`-bringup-notune`: warm-up, conc1, conc64 without FlashInfer's autotune, against `kimi-vllm-2n-tp8pp2-bringup-notune`.

```bash
cd $SCRATCH/trillium/new/k3-xn && git -C src pull --ff-only
JOB_PREFIX=nw-k3-xn NODELIST=trig[0145-0146] MILABENCH_BASE=$PWD src/scripts/trillium/kimi/serve.sh kimi-vllm-2n-dp2tp8-ep16-bringup-notune 2 1:00:00
```

`milabench slurm srun -x <main>` had to be fixed for this (`milabench/cli/slurm/srun.py`): it looked for srun's
`--nodes` in the whole command line and found `benchmate.server_worker --nodes`, so srun asked for both nodes with one
excluded (job 1269960, "Only allocated 1 nodes asked for 2"). It now reads only the options before `--`.

**Bring-up, job 1270053**, `trig[0145-0146]`, 2026-11-08 19:03-19:19 (16:19), `92eb7390`, milabench exit 0 after 963 s,
both nodes "GPUs free", neither drained. vLLM picked `FLASHINFER_TRTLLM_MXFP4_MXFP8` in its modular form
(`MoEPrepareAndFinalizeNaiveDPEPModular`, `TrtLlmMxfp4ExpertsModular`; K3-INF: the monolithic one), FlashKDA prefill, native
fused KDA decode, FLASHINFER_MLA; PyNccl for the `dp` and `ep` groups; "EP weight filter: ... loading 56/896 experts".
Weights 53-54 s, warm: the page cache of these nodes held the checkpoint (190-200 s in job 1269990 minutes before, its
prefetch done in 4-7 s); a cold load is not measured. 128.31 GiB per GPU: every GPU holds all 93 layers' TP8 share, the
latent projections and, under sequence parallelism, the shared and dense MLPs whole (`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT`
shards them; vLLM meant it for decode instances). KV memory 13.61 GiB per GPU (TP8 x PP2: 51.4-53.5), "GPU KV cache size:
517,438 tokens, Maximum concurrency for 133,120 tokens per request: 3.89x" per engine (TP8 x PP2: 27.09x): about 150
requests of 1,280 tokens per engine, 300 in all (est.; TP8 x PP2 about 960). Engine init 146 s, CUDA graphs 68 s
(3.09 GiB), engine ready at 19:08:17, 5.3 min after the job started.

| pass | done / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | TP8 x PP2 notune (1263425) |
|---|---|---|---|---|---|---|---|
| warmup | 64 / 0 | 485.8 | 8259 | | | | |
| conc1 | 60 / 0 | 29.95 | 150 | 151 / 366 | 32.88 / 32.98 | 32.87 / 33.19 | 105.22 tok/s, TPOT 9.14, TTFT 102 |
| conc64 | 128 / 0 | 1093.07 | 5465 | 1264 / 6308 | 45.35 / 64.52 | 42.65 / 45.09 | 1479.77 tok/s, TPOT 28.9, TTFT 2410 / 5987 |

Both without FlashInfer's autotune. One request decodes 3.6 times slower than on TP8 x PP2; at 64 concurrent, 26% less
output. Not profiled: the share of the all-gather and reduce-scatter (92 MoE layers, 16 ranks over 2 nodes) against the
replicated shared experts, which every GPU now reads whole, is not known. With equal sizes per rank vLLM calls NCCL's
all-gather and reduce-scatter (`cuda_communicator.py`). After the passes milabench's loggers raised `Bad file descriptor`
("1 still alive after end of benchmark", the DP worker); the pass `.data` files are complete.

**The 128.31 GiB per GPU, from the checkpoint's tensor headers** (GiB: routed experts 1347.12, attention 67.43, shared
experts 22.64, latent projections 8.81, embed and lm_head 2.19 each, layer 0's dense MLP 1.35, router 1.10; vision 0.83,
not loaded). Per GPU, with the layouts as vLLM builds them:

| part | TP8 x PP2 | DP2 x TP8 EP16 | why |
|---|---|---|---|
| routed experts | 84.20 (/16) | 84.20 (/16) | 56 of 896 per GPU either way: no second copy |
| attention | 4.21 (/16) | 8.43 (/8) | TP8 of every layer instead of half |
| shared experts | 1.42 (/16) | 22.64 (whole) | sequence-parallel MoE builds them with `disable_tp` (`KimiMLP`) |
| latent projections | 4.40 (/2) | 8.81 (whole) | `ReplicatedLinear`, every layer instead of half |
| router, dense MLP, embed, lm_head | about 0.9 | 3.0 | |
| sum | about 95.1 (measured 95.7-96.1) | 127.1 (measured 128.31) | |

So +32 GiB: 21.2 from the shared experts held whole, 9.2 from attention, latent projections and router over all layers
(the non-expert weights replicated over DP2), 1.2 more. Not a configuration choice with this backend:
`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1` shards the shared experts only with `deepep_v2` or `deep_gemm_mega_moe`
(`can_shard_sequence_parallel_shared_expert`); with `allgather_reducescatter` it shards layer 0's dense MLP only (1.35).

Not run yet: the autotune (its time on EP shapes), the study's other passes.

**Known limit (review, 2026-11-08):** with K3-INF's flags (kept, like for like) the KV memory holds about 150 requests
of 1,280 tokens per engine, about 300 in all (est.), and 3.89 x max-model-len per engine. So conc256 and up, and
ctx128k-c16 (16 x 131k tokens against two engines of 517k), queue on KV memory here where K3-INF-01 did not: they do not
compare like for like with K3-INF-01. ctx64k-c16 (16 x 65.6k) is at capacity.
`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1` (off by default) TP-shards layer 0's dense MLP and, with `deepep_v2` or
`deep_gemm_mega_moe` only, the shared experts (`KimiMLP`: all-gather of the TP group's tokens, partial GEMMs,
reduce-scatter): about 21 GiB per GPU back for the KV cache with DeepEP, 1.2 with `allgather_reducescatter` (est.).

### Part 3: prefill and decode on separate nodes (read, not run)

vLLM 0.32's `NixlConnector` moves Mamba-type state with the attention KV (`kv_connector/v1/nixl/base_worker.py`), and
K3's KDA layers are that type: `KimiK3DeltaAttention` is a `GatedDeltaNetAttention`, `MambaSpec` with `mamba_type`
GDN_ATTN, a conv state (4,608 x 3 at TP8) and a recurrent state (12 x 128 x 128). NIXL's GDN split
(`ssm_conv_transfer_utils.py`) cuts the conv state into q, k, v of 1,536 each at TP8, the order KDA splits it in. MLA
layers: one region per layer, TP may differ between the sides. Conditions, from the code:

- `VLLM_SSM_CONV_STATE_LAYOUT=DS` on both sides (asserted). K3's native fused KDA decode (picked by K3-INF) and FlashKDA
  prefill take that layout; FlashInfer's fused KDA decode does not, and is not picked.
- No pipeline parallelism on either side: `NotImplementedError` for Mamba/SSM hybrids with PP > 1. K3-INF-01's TP8 x
  PP2 cannot be the prefill or the decode instance; each 2-node instance has to be part 1's DP2 x TP8 EP16 (NIXL's
  side-channel port is offset per DP rank) or TP16.
- The hybrid KV cache manager on (the default). FlashKDA's prefill checkpoint block is used only in `align` cache mode
  (prefix caching), off here.
- `nixl` is not installed in the venv: a download, into a new venv beside K3-INF's.
