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
