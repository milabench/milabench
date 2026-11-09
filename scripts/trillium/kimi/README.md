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
mkdir -p venv data; ln -s $K/venv/torch venv/torch; ln -s $(readlink $K/cache/flashinfer/cubins) cache/flashinfer/cubins
# the checkpoint linked, the eval datasets copied: the datasets library writes lock files into its cache (review 2)
ln -s $K/data/hub data/hub; cp -a $K/data/datasets $K/data/modules data/
# Triton's cache seeded with K3-INF's (job.sh copies it to each node at the job's start and keeps nothing back, since
# review 2): 16 ranks on 2 nodes compiling the same kernel into an empty cache on VAST raced
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

**vLLM's low-latency DeepEP path cannot run Kimi-K3.** `--all2all-backend deepep_low_latency` uses DeepEP V1's
low-latency kernels, which accept at most 11 experts per token (`kNumMaxTopK = 11`, asserted in dispatch and combine,
`csrc/kernels/legacy/internode_ll.cu`); K3 routes each token to 16. The path K3 runs on is `deepep_v2` (DeepEP V2, top-k
up to 32), which vLLM 0.32 runs in one mode for decode and prefill alike (below).

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
- Job 1270725, `-deepep-shard-bringup-notune` (`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1`; since then set in every EP16
  entry, and this one is `-deepep-bringup-notune`), `trig[0146-0147]`, 20:27-20:52,
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

**Both backends, `-diag` bring-ups, 2026-11-08 21:05-21:32** (the bring-up's passes, then a conc1
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

**The two entries differ in more than the all-to-all.** `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1` is set in every EP16
entry, but vLLM shards the shared experts only under `deepep_v2` (or `deep_gemm_mega_moe`,
`can_shard_sequence_parallel_shared_expert`, `vllm/models/kimi_k3/nvidia/model.py:159`): with `allgather_reducescatter`
it shards layer 0's dense MLP only, and the shared experts stay whole on every GPU. So the all-gather server holds 127.13
GiB of weights per GPU against DeepEP's 107.41, and 4.23x max-model-len of KV per engine against 7.55x. And the
all-gather server runs K3-INF's venv (NCCL 2.29.7), the DeepEP one `venv/deepep` (NCCL 2.30.7). Both differences stand
in every comparison of the two backends here, K3-XN-01's included.

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
and DeepEP's sharded shared experts move time from dense GEMMs to the node's TP collectives. (b): the conc64 drop is two
things. A one-time stall: the first conc64 on DeepEP waits ~20 s more for prefill, its repeat does not (p99 TTFT 2.7 s),
in both DeepEP runs (1270725, 1271240); inferred, not logged: DeepEP JIT-compiling its kernels for batch sizes the warm-up
never made (vLLM rounds the size up to a power of 2 and DeepEP compiles one kernel per size, `deepep_v2.py`; the warm-up's
64 equal requests never mix prefill and decode). The study entries now run a second warm-up shaped like conc64. DeepEP's JIT cache is node-local and deleted after each
job (`job.sh`), so every job compiles again, and a batch size that neither warm-up made can still stall one pass's p99
TTFT once. Then a
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

### K3-XN-01: the study on EP16, both backends, 2 nodes

`kimi-vllm-2n-dp2tp8-ep16` (all-gather) and `kimi-vllm-2n-dp2tp8-ep16-deepep` (DeepEP), checkout `add6000c`, submitted by
the experiment coordinator: job 1271952 on `trig[0109-0110]`, 2026-11-08 22:44:40-2026-11-09 01:05:31 (2:20:51), and job
1271953 on `trig[0111-0112]`, 22:44:40-01:19:36 (2:34:56); milabench exit 0 after 8431 and 9274 s, 18 passes each, no
failed request. Engine up 23.2 and 25.8 min after the job's start (weights 390-438 and 338-378 s, cold; engine init with
the autotune 774-822 and 978 s); the serving passes 74.2 and 82.9 min (60.3 min of passes and 13.9 of gaps between them,
all-gather); the eval 42.5 and 45.4 min. Weights, KV and the other differences between the two entries: above. Tables
from `scripts/analysis/kimi_study.py` over the three run folders (`passes.csv`, `accuracy.md`; plots not made: no
`altair` in that venv).

| pass | all-gather: out tok/s, TTFT p50 / p99 ms, TPOT p50 ms | DeepEP | K3-INF-01, TP8 x PP2 (1263799) | AG / PP | DeepEP / PP |
|---|---|---|---|---|---|
| warmup | 491.8, 1607 / 3361, 105.8 | 440.5, 2205 / 4149, 111.7 | 305.2, 1907 / 12063, 182.4 |  |  |
| warmup-mixed | 1030.9, 1234 / 4568, 56.6 | 648.7, 2214 / 21771, 61.9 |  |  |  |
| conc1 | 31.0, 150 / 296, 31.8 | 31.3, 206 / 370, 31.3 | 111.1, 85 / 88, 8.7 | 0.28 | 0.28 |
| conc4 | 119.6, 356 / 903, 32.0 | 100.8, 361 / 17054, 34.1 | 326.7, 256 / 288, 11.3 | 0.37 | 0.31 |
| conc16 | 451.2, 492 / 1338, 33.7 | 383.1, 1132 / 1180, 38.1 | 876.3, 693 / 721, 15.6 | 0.51 | 0.44 |
| conc64 | 1304.6, 1254 / 1892, 43.9 | 1179.5, 2141 / 2741, 45.5 | 2002.6, 1462 / 2968, 26.4 | 0.65 | 0.59 |
| conc128 | 2046.3, 1903 / 3213, 53.7 | 1901.9, 3137 / 4961, 53.8 | 2243.0, 2837 / 9338, 44.8 | 0.91 | 0.85 |
| conc256 | 2839.0, 2035 / 10098, 77.1 | 1847.1, 3360 / 13812, 103.1 | 2865.0, 2698 / 14761, 70.7 | 0.99 | 0.64 |
| ctx1k-c1 | 30.5, 150 / 295, 31.8 | 30.6, 206 / 370, 31.2 | 107.5, 85 / 88, 8.7 | 0.28 | 0.28 |
| ctx8k-c1 | 27.8, 574 / 717, 31.8 | 26.8, 845 / 995, 31.4 | 84.2, 399 / 402, 8.8 | 0.33 | 0.32 |
| ctx32k-c1 | 22.3, 1662 / 1694, 32.1 | 19.5, 2528 / 2538, 31.8 | 55.0, 1168 / 1176, 9.1 | 0.41 | 0.35 |
| ctx64k-c1 | 18.2, 2939 / 2972, 32.3 | 14.6, 4701 / 4718, 32.0 | 39.1, 2087 / 2100, 9.4 | 0.46 | 0.37 |
| ctx128k-c1 | 12.6, 6002 / 6023, 32.7 | 9.4, 9495 / 9525, 32.4 | 21.5, 4555 / 4607, 11.1 | 0.59 | 0.44 |
| ctx1k-c16 | 418.8, 654 / 1221, 34.5 | 352.1, 781 / 1142, 39.2 | 766.3, 692 / 717, 15.6 | 0.55 | 0.46 |
| ctx8k-c16 | 260.5, 1682 / 5929, 42.7 | 191.3, 3132 / 8036, 56.6 | 220.5, 3501 / 11821, 37.2 | 1.18 | 0.87 |
| ctx32k-c16 | 123.9, 4393 / 12123, 94.5 | 87.4, 6644 / 19206, 131.8 | 78.0, 11009 / 18809, 115.4 | 1.59 | 1.12 |
| decodelong-c1 | 31.4, 197 / 437, 31.8 | 31.9, 279 / 555, 31.3 | 114.3, 87 / 88, 8.7 | 0.27 | 0.28 |
| decodelong-c16 | 485.7, 664 / 1141, 32.8 | 439.1, 676 / 1127, 36.3 | 968.9, 699 / 720, 16.4 | 0.50 | 0.45 |

Accuracy (the study's `eval` pass: GSM8K in full, 1,319, not the first 200):

| | GSM8K | ARC-Challenge | GPQA diamond | MMLU-Pro (2002) |
|---|---|---|---|---|
| all-gather | 98.0 | 98.1 | 87.4 | 87.2 |
| DeepEP | 97.6 | 98.3 | 84.3 | 87.2 |
| K3-INF-01 | 97.5 | 98.5 | 84.8 | 87.7 |

Not run on EP16 (KV memory): conc512, conc1024, ctx64k-c16, ctx128k-c16. EP16 had a second warm-up (`warmup-mixed`),
K3-INF-01 one.

The two EP16 servers differ in more than the all-to-all: the DeepEP one shards the shared experts (107.41 against 127.13
GiB of weights per GPU, KV 7.55x against 4.23x per engine), runs NCCL 2.30.7 against 2.29.7, and has `CUDA_HOME` set, so
FlashInfer's JIT can run only there. Every difference below between the two is between these two setups, one run each.

On the same 2 nodes EP16 gives less than TP8 x PP2 at nearly every pass: one request decodes at 31-32 against 8.7 ms
per token (conc1 0.28x; the all-to-all is a third of a step, profile above), conc64 0.65x and 0.59x. Among the passes
run, it gives more only with long prompts at 16 concurrent: ctx8k-c16 1.18x and ctx32k-c16 1.59x on all-gather (1.12x on
DeepEP), and the same at conc256 on all-gather (0.99x). The DeepEP server (sharded shared experts, NCCL 2.30.7,
`CUDA_HOME` set) gives less except at conc1, ctx1k-c1 and decodelong-c1 (+0.5 to +1.5%); one run each. Its conc4 and
conc256 include an engine-wide stall (16.5, 17.1 s): without them, about 0.94x and 0.86x of all-gather (reviewer's
estimate).

Anomalies, from the raw files:
- DeepEP conc4, TTFT p99 17.1 s: requests 1, 2 and 3 waited 17.1 s, and request 0 stopped 16.5 s after its first token,
  so one engine-wide stall at the pass's start; the other 56 had a median TTFT of 0.36 s.
- DeepEP conc256: the engine stalled 17.1 s, starting 37.2 s into the pass, with 251 requests in flight; the five late
  requests (448-511, up to 20.5 s) are its tail.
- Both stalls, cause not logged: 1271953's log has no FlashInfer or DeepEP JIT, compile or autotune line, nor anything
  else, in either window (23:21:46 for conc4; 23:28:55-23:29:15 for conc256: the progress bar stands at 256/512 from
  23:29:00 to 23:29:15). The only server lines there are the DP coordinator's "Received stats for out-of-order step"
  warnings, which the all-gather job logs too (225 against 352) without stalling. FlashInfer's JIT, possible only on the
  DeepEP server, stays a candidate.
- conc256, TTFT p99 10.1 s (all-gather) and 13.8 s (DeepEP): requests 128-255 (by index) wait for the first half's
  prefill, on both backends (medians 6.6-7.3 and 6.1-10.4 s), then 1.6-3.4 s; K3-INF-01 shows the same first-burst wait.
- `JSONDecodeError`, all-gather job only during the run: 3 lines came split in the stream (in conc16, conc128 and
  ctx1k-c1) and were dropped. Each held a voir telemetry sample (`gpudata` or `torchmem`) and a client record: a
  throughput-window sample in conc16 and conc128, one request's ITL in ctx1k-c1. At the end, in both jobs, milabench's
  `_parse_report` (`milabench/common.py`) failed on 8 lines of `<entry>.0.server.nolog.dpworker.data` (the second node's
  server: FlashInfer's autotune progress bars, stored as Python repr, not JSON). Every pass `.data` file parses; no pass
  summary lost.
- The all-gather main node's GPU 0 drew 25% more than the others at conc1 (experiment coordinator's read; not checked
  here).

### Part 1 on 4 nodes: EP32

`kimi-vllm-4n-dp4tp8-ep32`: EP16's server over 4 nodes, DP4 x TP8, 28 experts per GPU (all-gather all-to-all, shared
experts whole as on EP16's all-gather server). `-bringup-notune`: warm-up, conc1, conc64, conc1024 (256 per engine), GSM8K.

**Bring-up, job 1271715**, `trig[0101-0102,0107-0108]`, 2026-11-08 22:05:40-22:39:39 (33:59), milabench exit 0, all four
nodes "GPUs free". Weights 85.01 GiB per GPU (EP16 all-gather: 127.13), KV 56.25 GiB, "2,138,945 tokens, 16.07x" per
engine, four engines (K3-INF-02's one engine: 97.98x). Loads 43-81 s (warm), 512-549 s (`trig0107-0108`, cold); engine
init 157-194 s (DP0, DP1: 626-663 s, waiting for the cold nodes); CUDA graphs 57 s, 8.24 GiB.

| pass | done / failed | output tok/s | TTFT p50 / p99 ms | TPOT p50 ms | ITL p50 ms | K3-INF-02 bring-up, TP8 x PP4, autotuned (1264956) |
|---|---|---|---|---|---|---|
| warmup | 64 / 0 | 736.9 | | | | |
| conc1 | 60 / 0 | 19.08 | 226 / 526 | 51.65 | 51.64 | 109.0, TPOT 8.87 |
| conc64 | 128 / 0 | 994.40 | 1808 / 1856 | 59.21 | 56.78 | 1012.2, TPOT 22.7 |
| conc1024 | 953 / 1095 | 49.02 | 17184 / 23962 | 1077.7 | 975.6 | 3865.3 |

conc1024 failed: at 22:39:04 a worker of DP3 (`trig0108`, GPU 0) hit "CUDA error: an illegal memory access", reported
at a later stream sync, so the kernel is not known; its engine died and the remaining requests failed; GSM8K did not run.
The job's kernel log shows no Xid, but that log is the batch host's (`trig0101`): `trig0108` was not checked. Review 3
found no configuration limit exceeded (256 per engine under 512, decode within the capture sizes, KV 45% used); what is
new is 65,536 tokens gathered per MoE layer, so routing skew or a race are the candidates (medium-low confidence). Before
that the pass ran at about 1 s per decode step.

**It repeats: software.** `kimi-vllm-4n-dp4tp8-ep32-conc1024` (the warm-up and conc1024 only), job 1271958,
`trig[0101-0103,0107]` (no `trig0108`), 2026-11-08 22:45:40-22:56:10, from a second base (`$SCRATCH/trillium/new/k3-xn2`,
its own checkout, while K3-XN-01 ran from `k3-xn/src`): "CUDA error: an illegal memory access" at 22:55:12 on
`Worker_DP1_TP0_EP8` (`trig0102`, GPU 0), and again exactly 953 requests done and 1095 failed, TPOT p50 1082 ms, TTFT p50
17.2 s; all four nodes "GPUs free". The same pass on other GPUs and nodes fails the same way: not `trig0108`.

**With `CUDA_LAUNCH_BLOCKING=1` it does not fail.** `kimi-vllm-4n-dp4tp8-ep32-conc1024-blocking`, job 1272035, the same
nodes, 22:58:24-23:22:18: conc1024 2048 done, 0 failed, 1404.86 output tok/s, TTFT p50 32.3 s, TPOT p50 604 ms, ITL p50
121 ms (p99 8.2 s); nodes "GPUs free". So no kernel is named: nothing faults when every launch waits for the one before.
Its ITL is an eighth of the failing runs' (975 ms before their fault), although blocking only slows a correct run down.
Inferred, not shown: a race between concurrent launches (K3's model runs the router, the routed down-projection and the
shared experts on an auxiliary stream; `VLLM_DISABLE_SHARED_EXPERTS_STREAM` turns one of those off), whose bad state makes
the steps slow, then a kernel read out of bounds; review 3's "routing skew or a race". A `dmesg` watcher (one task per
node, on the first CUDA error in the log) was armed; it had nothing to read. Not a configuration limit (review 3).

**EP32 on `deepep_v2`**, `kimi-vllm-4n-dp4tp8-ep32-deepep-bringup-notune`, job 1272182, `trig[0104-0106,0113]`,
2026-11-08 23:27:53-23:46:31 (18:38), from `k3-xn2`; a first try, job 1272042, was cancelled at conc1: its 60 requests at
28.75 s each come to about 29 min (estimated from its first 15), which would not have left time for the rest (now 8). Weights 65.31 GiB per GPU, KV 58.69 GiB, 2,229,948 tokens per
engine. All four nodes "GPUs free".

| pass | done / failed | output tok/s | TTFT p50 ms | TPOT p50 ms | ITL p50 ms | all-gather EP32 (1271715) |
|---|---|---|---|---|---|---|
| conc1 | 8 / 0 | 8.88 | 515 | 110.32 | 110.35 | 19.08, TPOT 51.65 |
| conc64 | 128 / 0 | 368.57 | 3728 | 128.94 | 128.83 | 994.40, TPOT 59.21 |
| conc1024 | 953 / 1095 | 96.71 | | 1895.58 | 1887.49 | 953 / 1095, TPOT 1077.7 |

On 4 nodes DeepEP is slower than the all-gather, not faster: conc1 8.88 against 19.08 tok/s (TPOT 110 against 52 ms),
conc64 369 against 994 tok/s; not profiled. No more DeepEP on 4 nodes (coordinator, 2026-11-08). And conc1024 failed the
same way: "CUDA error: an illegal memory access" at 23:45:48 on `Worker_DP2_TP0_EP16` (`trig0106`,
GPU 0), again exactly 953 done and 1095 failed. The `dmesg` watcher read only the batch host, `trig0104` (23:45:56: no Xid, only `knvlinkSendInbandData`
lines): `srun --jobid` from the login node without `-N` took one node; fixed in the watcher (`$X2/dmesg_watch.sh`), so
`trig0106`'s log is lost. 

**SW-25 (`scripts/trillium/ISSUES.md`), the EP32 conc1024 crash:** three crashes (jobs 1271715, 1271958, 1272182), two
all-to-all backends, three node sets, always a TP0 worker, always exactly 953 requests done and 1,095 failed, none under
`CUDA_LAUNCH_BLOCKING=1` (job 1272035). The kernel cannot be named. Inferred, not determined: a timing-dependent fault on
a path both backends share. The node check of `trig0108` (job 1271974) found nothing.

**With at most 128 sequences per engine it passes.** `kimi-vllm-4n-dp4tp8-ep32-seqs128` (`--max-num-seqs 128`, a
configuration option, not a fix), job 1272263, `trig[0101-0103,0107]`, 2026-11-08 23:51:17-2026-11-09 00:01:17 (10:00),
all four nodes "GPUs free"; the `dmesg` watcher (fixed) had nothing to read. KV 60.22 GiB per GPU, 2,290,115 tokens per
engine. conc1024 (512 in flight, the rest queued): 2048 done, 0 failed, 4167.15 output tok/s, TTFT p50 / p99 27.6 / 43.2 s,
TPOT p50 93.3 ms, ITL p50 70.8 ms; conc64 after it: 128 done, 1052.22 tok/s, TPOT p50 53.9 ms. So the 256-per-engine
batch is implicated (SW-25 still not explained). K3-INF-02's bring-up gave 3865 tok/s at conc1024, autotuned, one engine
of 1024 sequences; this run had no autotune. `kimi-vllm-4n-dp4tp8-ep32-s128` is EP32's study entry at this setting:
K3-XN-01's passes in its order, then conc512 and conc1024 after the eval, so that a recurrence of SW-25 loses only those
two (run as K3-XN-03, below). It runs FlashInfer's autotune, as K3-XN-01 and K3-INF-02; the s128 pass above (1272263) and
every EP32 run before K3-XN-03 ran without it. Not determined; inferred, not
measured: with the all-gather all-to-all every GPU receives every engine's tokens at each MoE layer, up to 4 x 16,384
during prefill. One request decodes at 51.65 ms against EP16's 33 ms on 2 nodes: the all-gather now spans 32 ranks on 4
nodes. Next: the same with `CUDA_LAUNCH_BLOCKING=1`, or on `deepep_v2`.

### K3-XN-03: the study on EP32 at 128 sequences per engine, 4 nodes

`kimi-vllm-4n-dp4tp8-ep32-s128`, checkout `5b986c1b`, submitted by the experiment coordinator: job 1272792 on
`trig[0109-0112]`, 2026-11-09 01:30:19-04:39:22 (3:09:03), at the same time as K3-XN-02's two jobs (other nodes);
milabench exit 0 after 11,324 s, 20 passes and the eval, no failed request; all four nodes "GPUs free"; the `dmesg`
watcher saw no CUDA error. Weights 85.01 GiB per GPU, loaded in 44 s by two engines and 137-154 s by the other two
(cold); FlashInfer's autotune 01:36:38-01:45:20 (8.7 min); engines up at 01:45:52, 15.6 min after the start. KV 60.22
GiB per GPU, 2,290,115 tokens per engine (17.20x), as 1272263; CUDA graphs 28-29 s, 4.5-4.8 GiB. Eval 60.9 min
(K3-XN-01 all-gather: 42.5). Tables from `kimi_study.py` over 1272792, K3-INF-02's 1265338 and K3-XN-01's 1271952.

| pass | EP32 s128: out tok/s, TTFT p50 / p99 ms, TPOT p50 ms | K3-INF-02, TP8 x PP4, 4 nodes (1265338) | K3-XN-01 all-gather, EP16, 2 nodes (1271952) | EP32 / K3-INF-02 | EP32 / EP16 |
|---|---|---|---|---|---|
| warmup | 446.9, 5347 / 5374, 59.6 | 910.6, 2138 / 3372, 37.0 | 491.8, 1607 / 3361, 105.8 |  |  |
| warmup-mixed | 1004.4, 1820 / 1918, 58.4 |  | 1030.9, 1234 / 4568, 56.6 |  |  |
| conc1 | 19.4, 292 / 632, 50.5 | 108.8, 86 / 90, 8.9 | 31.0, 150 / 296, 31.8 | 0.18 | 0.63 |
| conc4 | 75.2, 604 / 1026, 50.9 | 353.0, 237 / 285, 10.5 | 119.6, 356 / 903, 32.0 | 0.21 | 0.63 |
| conc16 | 296.8, 636 / 1002, 50.9 | 873.0, 683 / 694, 15.7 | 451.2, 492 / 1338, 33.7 | 0.34 | 0.66 |
| conc64 | 1024.7, 1132 / 2002, 58.0 | 1111.5, 1900 / 2853, 27.2 | 1304.6, 1254 / 1892, 43.9 | 0.92 | 0.79 |
| conc128 | 1354.8, 1960 / 7040, 86.6 | 2017.6, 3232 / 15947, 31.7 | 2046.3, 1903 / 3213, 53.7 | 0.67 | 0.66 |
| conc256 | 2864.2, 2994 / 5399, 76.8 | 2482.9, 4951 / 26752, 62.7 | 2839.0, 2035 / 10098, 77.1 | 1.15 | 1.01 |
| conc512 | 3843.1, 3027 / 10553, 91.7 | 4037.6, 11286 / 22191, 81.9 |  | 0.95 |  |
| conc1024 | 4993.9, 27825 / 34495, 94.2 | 4292.5, 7789 / 44224, 210.7 |  | 1.16 |  |
| ctx1k-c1 | 19.0, 291 / 601, 50.5 | 105.4, 86 / 89, 8.9 | 30.5, 150 / 295, 31.8 | 0.18 | 0.62 |
| ctx8k-c1 | 18.0, 720 / 1068, 50.5 | 82.7, 404 / 407, 9.0 | 27.8, 574 / 717, 31.8 | 0.22 | 0.65 |
| ctx32k-c1 | 14.7, 2167 / 4279, 50.7 | 58.7, 999 / 1006, 9.3 | 22.3, 1662 / 1694, 32.1 | 0.25 | 0.66 |
| ctx64k-c1 | 12.6, 3667 / 3691, 50.9 | 46.6, 1536 / 1547, 9.5 | 18.2, 2939 / 2972, 32.3 | 0.27 | 0.70 |
| ctx128k-c1 | 9.4, 7044 / 7078, 51.3 | 32.0, 2748 / 2758, 9.9 | 12.6, 6002 / 6023, 32.7 | 0.30 | 0.75 |
| ctx1k-c16 | 285.1, 642 / 993, 50.5 | 767.8, 684 / 697, 15.6 | 418.8, 654 / 1221, 34.5 | 0.37 | 0.68 |
| ctx8k-c16 | 212.4, 1701 / 5808, 60.5 | 216.9, 3288 / 15759, 29.0 | 260.5, 1682 / 5929, 42.7 | 0.98 | 0.82 |
| ctx32k-c16 | 126.4, 5008 / 12047, 78.7 | 116.6, 8780 / 15706, 69.2 | 123.9, 4393 / 12123, 94.5 | 1.08 | 1.02 |
| decodelong-c1 | 19.8, 346 / 887, 50.5 | 111.5, 88 / 88, 8.9 | 31.4, 197 / 437, 31.8 | 0.18 | 0.63 |
| decodelong-c16 | 314.2, 822 / 994, 50.7 | 985.1, 733 / 742, 16.1 | 485.7, 664 / 1141, 32.8 | 0.32 | 0.65 |

Accuracy (the `eval` pass, as K3-XN-01's): EP32 GSM8K 97.6, ARC-Challenge 98.1, GPQA diamond 84.3, MMLU-Pro 87.2;
K3-INF-02 98.0, 98.5, 83.8, 86.8: all within 0.5 points (GPQA's stderr: 2.6 points, 198 questions). Not run on EP32:
ctx64k-c16 and ctx128k-c16 (K3-XN-01's pass list); K3-INF-02 ran them.

On 4 nodes each, not the same ones (EP32 on `trig[0109-0112]`, K3-INF-02 on `trig[0139-0142]`), EP32 gives more than
TP8 x PP4 at conc256 (1.15x), conc1024 (1.16x) and ctx32k-c16 (1.08x), about the same at ctx8k-c16 (0.98x), and less
at the other passes: one request decodes at 50.5 against 8.9 ms per token
(conc1 0.18x), conc64 0.92x, conc512 0.95x (one late request, below), conc128 0.67x (two pauses, below). Against EP16
on 2 nodes (`trig[0109-0110]`), with twice the GPUs, it gives less at every pass but conc256 (1.01x) and ctx32k-c16
(1.02x); conc1 0.63x (TPOT 50.5 against 31.8 ms). One run each. Output and latency only: the power of these runs is the
main node's x 4 (extrapolated) and the other nodes are not sampled, so no energy per token is compared.

**The autotune** (first EP32 run with it): no gain seen, one run each. The only run without it at the same layout and
cap is 1272263 (`--max-num-seqs 128`): conc64 TPOT p50 58.05 against 53.93 ms (+7.7%), ITL p50 55.12 against 53.77
(+2.5%); conc1024 TPOT p50 94.2 against 93.3 (+1.0%), ITL p50 71.8 against 70.8 (+1.4%). 1271715 ran
`--max-num-seqs 512`: conc1 TPOT p50 50.53 against 51.65 ms (-2.2%), conc64 58.05 against 59.21 (-2.0%). conc1024 gives
20% more output than 1272263 (4,994 against 4,167 tok/s) with about the same TPOT and TTFT p50 (27.8 against 27.6 s),
its last requests starting sooner (TTFT p99 34.5 against 43.2 s); 1272263 ran conc1024 right after the warm-up, this
run after the eval and conc512, so the 20% is not put down to the autotune.

**conc512 and conc1024 at 128 sequences per engine:** SW-25 did not recur (1,024 and 2,048 done, 0 failed; one run).
The cap holds at most 512 sequences in the engines (4 x 128): conc512 fills it exactly, conc1024 queues the other
half. K3-INF-02's one engine holds all 1,024:

| | EP32 conc512 | K3-INF-02 conc512 | EP32 conc1024 | K3-INF-02 conc1024 |
|---|---|---|---|---|
| output tok/s | 3843 | 4038 | 4994 | 4292 |
| TTFT mean, p50 / p99 s | 4.4, 3.0 / 10.6 | 11.5, 11.3 / 22.2 | 23.0, 27.8 / 34.5 | 14.8, 7.8 / 44.2 |
| TPOT p50 ms | 91.7 | 81.9 | 94.2 | 210.7 |
| E2E latency p50 / p99 s | 28.5 / 33.9 | 32.2 / 32.9 | 51.8 / 58.5 | 60.8 / 98.4 |

At conc1024 EP32 comes out ahead in output (1.16x) and in end-to-end latency (p50 0.85x, p99 0.60x), with TPOT 94
against 211 ms and TTFT p50 27.8 against 7.8 s. Inferred, since the layouts differ (DP4 x TP8 with EP32 against TP8 x
PP4): the cap moves the wait from the decode to the queue. At conc512 the output is 0.95x because of one request: the
last (index 1023), dispatched at 04:35:34.5 (+34.9 s), waited 20.45 s for its first token (the others at most 10.8 s),
then decoded alone at 20 tok/s for 12.5 s (the other 1,023 were done at +55.7 s), 18% of the 68.2-s pass; up to the
1,023rd completion the pass ran at about 4.7k tok/s (derived, not measured). The DP coordinator's "Received stats for
out-of-order step" warnings run 04:35:27-34, up to its dispatch; nothing is logged during its wait. conc1024 has two
such requests: index 1531 (TTFT 47.0 s) and 1533 (46.1 s), dispatched at +32.7 and +33.6 s, the others at most 34.8 s;
they finish with the last batch (+104.5 of 105.0 s), so they do not lengthen the pass. Inferred, not shown: the API
server sent each to an engine already holding 128 sequences, where it waited for that engine's batch to finish, while
another engine had a free slot (the DP load balancer works from the engines' published counts). Power: as above, main
node x 4; not compared.

Anomalies, from the raw files:
- conc128, two pauses: from the per-request token times (`benchmark_results.db`), no request gets a token for 3.88 s
  (02:09:02.2-06.1, +2.9 to +6.8 s into the pass) and 5.08 s (02:09:25.3-30.4, +26.0 to +31.1 s), each at the end of a
  prefill wave with 128 in flight; 3 requests of the first wave and 32 of the second (indices 224-255) wait 6.75-7.04 s
  for their first token, the others 1.1-2.9 s; in between the decode runs at 1,826-1,905 tok/s. Hence TPOT p50 86.6 ms
  against ITL p50 67.9 (the pauses fall inside requests), and 1,355 tok/s. The log holds nothing in either pause; the
  only server lines near them are the DP coordinator's "Received stats for out-of-order step" warnings at 02:09:22-24,
  before the second (402 in the job). The longest such gap elsewhere: conc256 2.47 s (+26.1 to +28.6 s, its second
  wave), conc512 2.96 s (+5.6 to +8.6 s, its first wave), conc1024 1.04 s. Cause not logged.
- JIT during passes, logged by vLLM's `jit_monitor`: Triton `layer_norm_gated_fwd_kernel` at 02:02:36 (conc4, DP0);
  CuTeDSL `FlashPrepareScheduler` and `FlashAttentionForwardCombine` at the start of ctx8k-c16 (03:02:31-33; DP0, DP2,
  DP3) and of ctx32k-c16 (03:03:50-53, DP1): part of their TTFT p99 (ctx8k-c16's 5.8 s; ctx32k-c16's first 16 requests
  at 5.8-12 s), as on K3-INF.
- `JSONDecodeError`: during the run 2 lines came split (warm-up: one request's ITL and a `torchmem` sample; conc1024: a
  throughput-window sample and a `gpudata` sample) and were dropped. At the end `_parse_report` failed on 24 lines of
  `<entry>.0.server.nolog.dpworker.data` (the three other nodes' engines: FlashInfer's autotune progress bars, stored as
  Python repr `{'#unrepresentable': ...}`), as in K3-XN-01. Every pass `.data` file parses; no pass summary lost.
- Power: `power_w` is the main node's 8 GPUs x 4 (extrapolated), as K3-INF-02's; the other nodes are not sampled (no
  `benchmate.nodemon`, see K3-XN-02).

### Part 3: prefill and decode on separate nodes

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

Since then: `nixl` 1.5.2 in `venv/deepep` (`deepep_build.sh`). `kimi-vllm-4n-pd-ep16-bringup-notune`: `pd_serve.sh`
(`server.command`, run by milabench on the main node with the entry's vLLM flags; `num_machines: 1`) starts a prefill
instance on the job's nodes 0-1 and a decode instance on nodes 2-3, each EP16 (DP2 x TP8, all-gather all-to-all, so the
shared experts whole on every GPU: see below), with `--kv-transfer-config` NixlConnector (`kv_both`), `VLLM_SSM_CONV_STATE_LAYOUT=DS` and each
node's `VLLM_NIXL_SIDE_CHANNEL_HOST`, and `pd_proxy.py`: vLLM v0.32.0's NIXL toy proxy
(`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py`, downloaded to `$X/pd/`, unchanged) plus `/v1/models` for
milabench's readiness check. The proxy sends each request to the prefill instance with `max_tokens=1`, then to the decode
instance with the returned `kv_transfer_params`.

```bash
curl -sSfL -o $SCRATCH/trillium/new/k3-xn/pd/toy_proxy_server.py \
  https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py
JOB_PREFIX=nw-k3-xn NODELIST=trig[0103-0106] MILABENCH_BASE=$PWD src/scripts/trillium/kimi/serve.sh kimi-vllm-4n-pd-ep16-bringup-notune 4 1:00:00
```

**Bring-up, job 1271433**, `trig[0103-0106]` (prefill 0103-0104, decode 0105-0106), 2026-11-08 21:32:58-22:00:01 (27:03),
milabench exit 0, all four nodes "GPUs free". NIXL initialized on the four engines with the KDA state (no assertion), the
decode workers' "NIXL compatibility check passed" against the prefill's. KV memory 15.47 GiB per GPU, 588,134 tokens per
engine (4.42x) on both instances. Weights 55 s (prefill, warm), 392-412 s (decode, cold). Harmless: vLLM's probe of the
optional `deep_ep` module logs an `AssertionError` (DeepEP wants `CUDA_HOME`, unset here), and the bench client's
`/tokenize` gets a 404 from the proxy; `kv_role='kv_both'` is deprecated (vLLM: use `kv_producer` / `kv_consumer`).

| pass | done / failed | output tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | ITL p50 / p99 ms | colocated reference, 4 nodes (1271710) |
|---|---|---|---|---|---|---|
| warmup | 64 / 0 | 411.5 | | | | 646.1 |
| conc1 | 60 / 0 | 28.69 | 556 / 591 | 32.80 / 32.91 | 32.79 / 32.96 | 30.19, TTFT 95 / 123, TPOT 32.88 |
| conc64 | 128 / 0 | 1132.17 | 1581 / 7142 | 42.99 / 43.25 | 42.77 / 47.72 | 1444.25, TTFT 813 / 1281, TPOT 41.04 |

GSM8K's first 200: 197 / 200 (K3-INF-01: 197): the transferred KV and KDA state decode right. The KV transfer time per
request is not measured here: vLLM's stats log is off in these DP servers, but its Prometheus histograms are not, and
`pd_serve.sh` now samples `vllm:nixl_xfer_time_seconds` (count, sum) every 5 s (checked in 1271710, zero there): sum /
count is the mean transfer time per TP worker (each of a request's 8 workers records its own transfer handle, so sum per
request would be 8x too high), recorded on the decode side only (NIXL pulls), with no percentiles (sum and count only),
and it may include up to one engine step of polling. At
conc1 P/D's TTFT is 461 ms over the reference's, for the prefill instance's run, the KV and KDA transfer and the
proxy's second hop (derived, not measured).

**Its colocated reference, job 1271710** (`kimi-vllm-4n-pd-ep16-colocated-bringup-notune`, coordinator, 2026-11-08):
the same nodes `trig[0103-0106]`, instances, flags and proxy (`pd_proxy.py --colocated`: each request whole to one
instance, round-robin), 22:05:10-22:25:04 (19:54), milabench exit 0, all four nodes "GPUs free"; weights 52-55 s
(warm), KV 15.47 GiB per GPU, 588,134 tokens per engine as P/D's; GSM8K's first 200: 197 / 200.
At conc64 P/D gives 22% less output than its reference (1132 against 1444 tok/s), with a higher TTFT (p50 1.58 against
0.81 s) and about the same TPOT. Untested: P/D with `deepep_v2`, more than one instance per role.

### K3-XN-02: prefill and decode on separate nodes against colocated, 4 nodes

`kimi-vllm-4n-pd-ep16` and `kimi-vllm-4n-pd-ep16-colocated`, checkout `5b986c1b`, submitted by the experiment
coordinator at the same time as K3-XN-03 (other nodes): job 1272790 on `trig[0101-0104]` (prefill `trig0101-0102`, decode
`trig0103-0104`, proxy on `trig0101`), 2026-11-09 01:30:19-03:45:36 (2:15:17), and job 1272791 on `trig[0105-0108]`
(instance A `trig0105-0106`, B `trig0107-0108`), 01:30:19-03:44:50 (2:14:31); milabench exit 0 after 8,096 and 8,048
s, 18 passes and the eval each, no failed request; all eight nodes "GPUs free"; the `dmesg` watcher saw no CUDA error.
The bring-ups' venv (`venv/deepep`, NCCL 2.30.7), flags, instances and proxy (1271433, 1271710): two EP16 instances (DP2
x TP8, all-gather, shared experts whole: 127.13 GiB of weights per GPU; KV 15.47 GiB per GPU, 588,134 tokens per engine),
now with the autotune and both warm-ups. Weights 53-56 s on every node but colocated B's (249-272 s, cold); autotune
about 10 min; engines up 15.8 and 21.1 min after the start (B's cold load). Eval 42.7 and 39.7 min. P/D against its
colocated reference only (same venv; K3-XN-01's all-gather ran K3-INF's venv, NCCL 2.29.7).

| pass | P/D: out tok/s, TTFT p50 / p99 ms, TPOT p50 / p99 ms | colocated | P/D / colocated, output | TTFT mean, P/D - colocated (ms) | NIXL transfer, mean ms (requests) |
|---|---|---|---|---|---|
| warmup | 417.7, 1549 / 7426, 40.8 / 41.4 | 649.9, 2238 / 3897, 38.6 / 49.5 | 0.64 | +378 | 61.3 (64) |
| warmup-mixed | 1322.3, 1320 / 2497, 41.1 / 41.5 | 1441.3, 815 / 1361, 40.4 / 44.1 | 0.92 | +480 | 50.3 (128) |
| conc1 | 29.5, 541 / 541, 31.9 / 32.1 | 31.1, 98 / 102, 31.9 / 32.1 | 0.95 | +442 | 6.3 (60) |
| conc4 | 123.3, 293 / 748, 30.9 / 31.0 | 126.3, 182 / 470, 30.4 / 33.3 | 0.98 | +149 | 32.6 (60) |
| conc16 | 447.3, 808 / 1014, 32.5 / 32.6 | 428.1, 413 / 1018, 35.2 / 38.4 | 1.04 | +333 | 34.2 (64) |
| conc64 | 1343.1, 1212 / 2291, 41.4 / 41.4 | 1437.1, 843 / 1267, 39.6 / 42.6 | 0.93 | +443 | 57.4 (128) |
| conc128 | 2132.2, 1656 / 3810, 49.7 / 50.1 | 1991.6, 1219 / 1930, 49.5 / 64.3 | 1.07 | +860 | 59.7 (256) |
| conc256 | 3636.1, 1750 / 6419, 55.3 / 56.0 | 3282.7, 1812 / 10346, 59.9 / 92.0 | 1.11 | -514 | 64.4 (512) |
| ctx1k-c1 | 27.9, 541 / 553, 31.9 / 32.0 | 30.8, 97 / 99, 31.9 / 32.1 | 0.90 | +444 | 6.2 (60) |
| ctx8k-c1 | 27.5, 595 / 656, 32.0 / 32.0 | 29.1, 345 / 353, 32.0 / 32.1 | 0.95 | +254 | 12.7 (60) |
| ctx32k-c1 | 21.3, 1875 / 2814, 32.2 / 32.2 | 23.4, 1262 / 3447, 32.2 / 32.3 | 0.91 | +536 | 47.9 (60) |
| ctx64k-c1 | 17.0, 3413 / 3452, 32.4 / 32.4 | 18.8, 2677 / 2706, 32.5 / 32.5 | 0.90 | +734 | 90.7 (60) |
| ctx128k-c1 | 11.5, 6931 / 6954, 32.8 / 32.8 | 12.6, 5979 / 6000, 32.9 / 33.0 | 0.91 | +953 | 177.6 (60) |
| ctx1k-c16 | 411.9, 769 / 1013, 32.5 / 32.8 | 385.1, 388 / 952, 38.4 / 45.3 | 1.07 | +353 | 34.0 (64) |
| ctx8k-c16 | 338.0, 1293 / 3253, 33.4 / 33.5 | 294.6, 693 / 4753, 45.9 / 66.8 | 1.15 | +458 | 37.0 (64) |
| ctx32k-c16 | 155.0, 7686 / 12454, 33.7 / 33.7 | 151.2, 2291 / 9520, 86.3 / 125.1 | 1.02 | +4683 | 68.2 (64) |
| decodelong-c1 | 31.2, 540 / 541, 31.9 / 32.0 | 31.3, 103 / 120, 31.9 / 32.1 | 1.00 | +433 | 6.2 (4) |
| decodelong-c16 | 481.2, 739 / 1016, 33.0 / 33.2 | 492.5, 425 / 1001, 32.0 / 32.8 | 0.98 | +330 | 34.5 (32) |

Accuracy (the `eval` pass, GSM8K in full):

| | GSM8K | ARC-Challenge | GPQA diamond | MMLU-Pro (2002) |
|---|---|---|---|---|
| P/D | 97.8 | 98.3 | 81.8 | 86.8 |
| colocated | 97.4 | 98.2 | 85.4 | 86.7 |

GPQA: 162 against 169 of 198. Question by question, the colocated run is right where P/D is wrong 9 times, the reverse
2 times: exact McNemar p = 0.065, not significant at 5%, borderline; one run each. P/D's 162 is also the lowest of the
runs compared in this README (colocated 169, EP32 167, K3-XN-01 all-gather 173, K3-INF-02 166). The other three sets
are within 0.4 points.

**The NIXL transfer time** (`slurm/pd-nixl.1272790.txt`; per pass, the change of `vllm:nixl_xfer_time_seconds` sum
over count between the 5-s samples at the pass's start and end, both decode engines; 8 records per request, one per
decode TP worker): 61.0 ms over the whole run (6,509 requests), 63.1 ms in the eval (4,691). The post time, the decode
worker's call that starts the read, is 0.35-1.52 ms. One request at a time, it grows with the prompt: 6.2 ms at 1k
tokens, 12.7 at 8k, 47.9 at 32k, 90.7 at 64k, 177.6 at 128k, about 1.35 ms per 1k tokens beyond 32k. With 4 or more
1k prompts in flight it is 32.6-64.4 ms, of the order of one decode step (TPOT 31-55 ms) above conc1's 6.3: the decode side
checks for finished reads once per engine step (inferred; the histogram's "up to one engine step of polling"). The
prefill side records none (NIXL pulls), nor does the colocated run (zero throughout, as 1271710).

**Where P/D gains and loses** (4 nodes each, not the same ones: P/D on `trig[0101-0104]`, prefilling on 2 and decoding
on 2; colocated on `trig[0105-0108]`, both on all 4):
- It loses on TTFT at one request, every pass: +0.25 to +0.95 s (mean; conc1 0.54 against 0.10 s), so 0.90-0.95x
  the output (decodelong-c1 1.00x, its 4,096 tokens dilute it). The transfer is 6-178 ms of that gap; the rest (the
  prefill instance's request, the proxy's two hops, the decode side's scheduling of the pulled request) is not
  measured. It is not a fixed cost: from 1k to 8k tokens P/D's TTFT p50 grows 54 ms (541 to 595) and colocated's 248
  ms (97 to 345), so at short prompts P/D's TTFT has a floor near 0.54 s (inferred from these two points).
- It loses on TTFT with more in flight, up to conc128 (p50 1.4-2.0x), and most with long prompts at 16: ctx32k-c16
  TTFT p50 7.7 against 2.3 s, one 2-node instance prefilling everything against two.
- It gains on TPOT and ITL whenever prefills run during the decode, the decode nodes never prefilling: TPOT p50 32.5,
  33.4, 33.7 ms through ctx1k/8k/32k-c16 (colocated 38.4, 45.9, 86.3), ITL p99 34-39 ms (colocated 238, 331, 617);
  conc256 TPOT p99 56.0 against 92.0 ms (the ctx8k/32k-c16 figures include the colocated JIT stalls below). Output
  1.07x at conc128 and ctx1k-c16, 1.11x at conc256; mean end-to-end latency 0.85x at conc256.
- Most of the ctx8k-c16 gain (1.15x) is the colocated run's JIT: its pass starts at 02:44:33.8, and CuTeDSL
  `FlashPrepareScheduler` and `FlashAttentionForwardCombine` and Triton `layer_norm_gated_fwd_kernel` compile at
  02:44:35-37 on three of its four engines; from +1.88 to +4.48 s no request gets a token, and 13 requests stall
  2.83-2.88 s, on both instances. Without about 2.6-2.9 s the gain is about 1.03-1.04x (estimate, assuming the stall
  pushes the pass's end back by as much). ctx32k-c16 (1.02x) has a 3.4-s JIT stall on the fourth engine
  (02:45:41-02:46:26; 3 requests at 3.40-3.43 s). P/D logged no JIT in any pass (both runs:
  `FlashAttentionForwardSm100` on the first request, before the warm-up), as on K3-INF and K3-XN-03 for the colocated
  kernels. Inferred: P/D's engines never put a prefill and decodes in one batch, where these kernels are picked.
- conc128 and conc256 hold no JIT and no warm-up stall on either side (no `jit_monitor` line in their windows). They
  do hold colocated stalls of another kind, cause not logged: at conc128 two of 3.8-3.9 s, each on 63 requests at a
  wave's start (+1.9 to +5.8 s and +18.6 to +22.4 s into the pass) while the other requests keep getting tokens; at
  conc256, in its first burst, 5.2-5.3 s on 113 requests (+1.8 to +7.1 s, no token from any request for 5.18 s) and
  2.1-2.6 s on 241 (+7.8 to +10.4 s). Nothing is logged in those windows (the DP coordinators' out-of-order warnings
  fall just before or after them). P/D has no gap over 0.17 s between two tokens of a request in either. For scale,
  not a comparison (K3-INF's venv, NCCL 2.29.7): K3-XN-01's one EP16 instance at conc128, the colocated per-instance
  load of conc256, has no gap over 1 s, and 2.8 s at conc256. So P/D's decode is steadier there, measured; how much of
  the 1.07x and 1.11x the colocated stalls account for is not estimated. ctx1k-c16 (1.07x) holds no JIT and no gap
  over 0.27 s on either side.
- About even: conc4 0.98x, conc16 1.04x, decodelong-c16 0.98x; conc64 0.93x.

So P/D trades TTFT for a steady decode; it gives more output at conc128, conc256 and ctx1k-c16 (1.07x, 1.11x, 1.07x),
one run each and with the colocated stalls above, not a lower latency. Power is missing for 3 of
the 4 nodes in both runs (below), so these are output and latency only, no energy per token. The bring-ups' conc64
gap (-22%, no autotune, one warm-up) is -7% here: P/D's conc64 rose from 1,132 to 1,343 tok/s, the colocated's stayed
(1,444, 1,437).

Anomalies, from the raw files:
- Gaps with no token from any request, from the per-request token times (`benchmark_results.db`), leaving out a
  single request's own prefill at c1: P/D none over 1 s but the warm-up's (below); colocated conc256's 5.18 s and
  2.07 s and ctx8k-c16's 2.6 s (above). Gaps between two tokens of one request: P/D at most 0.17 s in every pass;
  colocated up to 3.9 s in conc128, 5.3 s in conc256, 2.9 s in ctx8k-c16 and 3.4 s in ctx32k-c16 (above).
- P/D warm-up: 17 of its 64 requests (from index 33) waited 7.2-7.4 s for the first token against a median of 1.55
  s, and no request got a token for 3.0 s (+4.19 to +7.19 s into the pass): the run's first requests through NIXL;
  nothing logged (vLLM logs no NIXL handshake at INFO). The bring-up's conc64 TTFT p99 (7.1 s, 1271433) is of the
  same size.
- `JSONDecodeError`: during the run 4 lines came split in each job (P/D: conc16, conc64, ctx128k-c1, decodelong-c1;
  colocated: conc64, conc128, ctx64k-c1, ctx32k-c16) and were dropped. Each held a voir telemetry sample (`gpudata` or
  `torchmem`) and one client record (a request's TTFT or end-to-end latency, a request-rate or a throughput-window
  sample). At the end `_parse_report` failed on 8 lines of `<entry>.0.server.nolog.data` in each (these entries have no
  `dpworker` file; `pd_serve.sh`'s output goes to the server's): FlashInfer's autotune progress bars stored as Python
  repr. Every pass `.data` file parses; no pass summary lost.
- At the end milabench's warden killed 6 (P/D) and 7 (colocated) worker processes of the main node's instance still on
  the GPUs, as after the bring-up 1271433; all nodes "GPUs free" after.

**Power: missing.** These entries have `num_machines: 1` (`pd_serve.sh` starts the other nodes), so `kimi_study.py`'s
`power_w` is the main node's 8 GPUs only, not extrapolated: in P/D a prefill node (`trig0101`), in the colocated run
instance A's first engine (`trig0105`). The other three nodes' power is not in the run: voir samples only the main
node, and `job.sh` runs no `benchmate.nodemon` (Node metrics, above), so there is no `nodemon/` folder for 1272790-1272792.
Whole-system power and energy per token, P/D against colocated, cannot be given from these runs.

Open: where P/D's short-prompt TTFT goes beyond the transfer (vLLM's `vllm:request_queue_time_seconds` and
`vllm:request_prefill_time_seconds` on both instances, sampled as the NIXL histograms, would split it; not done), EP32
conc128's two pauses and the colocated conc128/conc256 stalls (causes not logged), and `nodemon` in `job.sh` for the
power of every node.

## Stall investigation (K3-XN, 2026-11-09)

The engine-wide stalls above (intervals with requests in flight and no token at all) had no node metrics beside them.
`stall_gaps.py RUN_DIR [MIN_S]` lists them per pass from the client's per-request token times (`benchmark_results.db`),
with the wall-clock start as the server logs it. Over the runs above, gaps of 2.5 s or more (JIT-logged ones aside):

| layout | run | stalls (s), pass |
|---|---|---|
| TP8 x PP2, 2 nodes | K3-INF-01 1263799 | 4.60 conc128, 4.41 conc256, 8.64 ctx32k-c16 |
| TP8 x PP2, notune | 1263425 | 3.80 conc64 |
| TP8 x PP4, 4 nodes | K3-INF-02 1265338 | 14.18 conc64, 10.94 conc128, 8.51 + 8.46 conc256, 7.88 + 8.93 ctx128k-c16 |
| TP8 x PP4 | bring-up 1264956 | 17.82 conc64, 7.52 + 8.46 conc1024 |
| EP16 all-gather | K3-XN-01 1271952 | 3.80 warmup, 3.70 + 3.33 warmup-mixed, 2.77 + 2.78 conc256 |
| EP32 s128 | K3-XN-03 1272792 | 3.88 + 5.08 conc128, 2.96 conc512, 3.13 ctx8k-c16, 3.63 ctx32k-c16 |
| EP16 x 2 colocated | K3-XN-02 1272791 | 5.18 conc256 |

(DeepEP's 16.5-19.6 s stalls, K3-XN-01, are left out: inferred DeepEP JIT.) The 10-s throughput windows missed the
2-node ones: K3-INF-01 stalls too, about half as long as 4 nodes (PP2 4.4-4.6 s, PP4 8.5-17.8 s at conc64-256).
TP8 x PP4 stalls most reliably and longest: every pass of conc64 to conc256 in both runs (none in 1265338's conc512
and conc1024, after them).

`stall.yaml` (`kimi-vllm-4n-tp8pp4-stall`): K3-INF-02's server unchanged (autotune on), the warm-up, then conc64,
conc128 and conc256 twice each, so a cost paid once per batch shape shows in the first of each pair only. Logging only:
`--enable-logging-iteration-details` (one line per batch: context and generation requests and tokens, the engine's
wait for the batch, KV use), `VLLM_LOG_STATS_INTERVAL=1` (the throughput line every second), `TRITON_PRINT_AUTOTUNING=1`
(a line per Triton autotune). `job.sh` now runs `benchmate.nodemon` on every node for the whole run (`nodemon/<run>/`:
GPU power, SM clock, throttle reasons, utilization and GPM activity every 1 s, NVLink every 10 s, InfiniBand and host
memory every 1 s, CPU every 2 s) and saves each node's kernel log at the end (`<host>.dmesg.txt` there).

```bash
cd $SCRATCH/trillium/new/k3-xn && git -C src pull --ff-only
JOB_PREFIX=nw-k3-xn NODELIST=<4 nodes> MILABENCH_BASE=$PWD src/scripts/trillium/kimi/serve.sh kimi-vllm-4n-tp8pp4-stall 4 1:00:00 \
  CONFIG=$PWD/src/scripts/trillium/kimi/stall.yaml VLLM_LOG_STATS_INTERVAL=1 TRITON_PRINT_AUTOTUNING=1
python3 -I src/scripts/trillium/kimi/stall_gaps.py runs/kimi-vllm-4n-tp8pp4-stall.<job> 1
```

`stall_nodes.py NODEMON_DIR FROM TO` prints every node's metrics second by second over a window (GPU utilization, power,
lowest SM clock and clock event reasons, GPM SM activity, InfiniBand GB/s, CPU), to set beside a stall.

**Job 1278355**, `trig[0115-0118]` (PP0 `trig0115`, PP1 `trig0117`, PP2 `trig0118`, PP3 `trig0116`), 2026-11-09
18:00:53-18:50:10 (49:17), `1a92add0`, COMPLETED, milabench exit 0 after 2934 s, all four nodes "GPUs free", none
drained; nodemon closed cleanly on all four. Weights 488-888 s (cold), autotune as K3-INF-02, engine ready at 18:43:30.

| pass | duration s | stalls s (`stall_gaps.py`, over 1 s) | output tok/s (derived: requests x 256 / duration) |
|---|---|---|---|
| warmup | 22.5 | 18.77 | |
| conc64-1 | 23.3 | 7.94 | 1406 |
| conc64-2 | 14.8 | none | 2214 |
| conc128-1 | 22.2 | none | 2952 |
| conc128-2 | 22.7 | none | 2887 |
| conc256-1 | 52.2 | 8.49 + 8.51 | 2511 |
| conc256-2 | 37.1 | none | 3533 |

The stalls reproduce, the same size as K3-INF-02's (conc256 8.51 + 8.46 there), in the first pass of each pair only;
conc128 did not stall after two conc64 passes. Without the stall conc64 on 4 nodes gives 2214 tok/s against 2 nodes'
2003 (K3-INF-01): K3-INF-02's conc64 "halving" (1112) was its 14.2-s stall (one run each, other nodes).

What lines up with each stall:
- The engine (one line per batch): each stall is one batch, 7,947 ms (1 context request of 16 tokens + 11 decodes),
  8,486 and 8,509 ms (17 context requests, 16,352 and 16,336 tokens, + 32 and 48 decodes), 18,769 ms in the warm-up
  (12 + 1), against 0.66 s for every other batch of 16k tokens. A batch of the same composition ran in 0.66 s two
  batches before each conc256 stall (iterations 9866, 9870): the token and request counts alone do not set it off.
- The nodes (nodemon, 1 s): at each stall's start all 32 GPUs go idle (util 0, idle power, 2.1 kW per node) for 1.5-4
  s. Then the stages wake one after the other in pipeline order, PP0, PP1, PP2, PP3, 2 s apart (4-5 s in the warm-up):
  conc64-1 at 18:44:46, :48, :50, work again at :52; conc256-1 at 18:47:52, :54, :56, :58 and 18:48:02, :04, :06, :08. From
  its turn on, a stage's 8 GPUs show 100% utilization at idle power with 1-4% SM activity (99% on PP0 in conc256-1),
  waiting, until the stall ends; the last stage stays idle until then. The CPU of a stage goes from 0.2-0.4% to 5-10% at
  its turn (`trig0116`). No InfiniBand traffic in the stalls; SM clocks at 1965 MHz, no clock event reason but the power
  cap bit (0x4) that busy GPUs show; host memory flat, no OOM kill; no kernel log line during the passes (`dmesg`: only
  `knvlinkSendInbandData` at 18:05 and at the teardown).
- The logs: nothing in any stall window. No JIT-monitor line (its last, Triton `layer_norm_gated_fwd_kernel`, at
  18:44:36, 7 s before conc64-1's stall), no Triton autotune line (`TRITON_PRINT_AUTOTUNING=1`), no warning. vLLM does
  not torch.compile this model (compilation mode NONE).

Inferred, not shown: a host-side cost that each pipeline stage pays in turn, about 2 s of CPU work with its GPUs idle
before it runs the batch, the next stage starting only once it receives the batch, so the stall is about PP x 2 s (PP4
7.9-8.5 s, PP2 4.4-4.6 s in K3-INF-01), paid once and not again in the repeated passes. Ruled out for these stalls: GPU
compute, clocks and power (idle GPUs, no throttle), the fabric (no IB traffic, no link event), host memory, and the JITs
vLLM's monitor watches. The host work itself is not determined: no log names it. Candidates, not checked: a JIT or
lazily built object the monitor does not watch (kernel loading, CUDA graph or kernel-plan builds).
