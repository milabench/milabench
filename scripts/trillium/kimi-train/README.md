# Kimi K3 pre-training (TorchTitan) on Trillium B200

Brief: `~/projects/tri-work/ops/K3_TRAINING_BRIEF.md`. Upstream TorchTitan's `kimi_k3` model as is (the commit the
torchtitan_native pack pins, 51b37c2): KDA (attn-gym kernels), gated MLA, LatentMoE (896 experts, top-16, 2 shared),
Attention Residuals, text only, random init. Entries: `config/torchtitan_native_kimi.yaml` (`--model kimi_k3`).

| step | command | what |
|---|---|---|
| 2 | `k3_train.sh --nodelist <node> k3-1n-l4` (`-bf16`) | 1 node, 4 layers (KDA dense, KDA-MoE x2, MLA-MoE), EP8: loss on FineWeb |
| 3 | `k3_train.sh --nodelist <2 nodes> k3-pp2-l4` (`-bf16`) | PP2 x EP8, 2 layers per stage, 16 micro-batches: memory per stage, pipeline hop |
| hop | `python hop_times.py $K3/runs/<run>/traces/iteration_20` | NCCL kernel times of the traced step, per rank and stream |

`k3_train.sh --dry-run ...` prints the sbatch command. Jobs are named `nw-k3trn-<bench>`, bring-up rules in
`~/projects/tri-work/ops/IMPL_BRIEF.md`.

## Install (login node: it downloads)

```bash
unset PYTHONPATH PIP_CONFIG_FILE; export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$SCRATCH/trillium/new/k3-trn/uv-cache
K3=$SCRATCH/trillium/new/k3-trn; mkdir -p $K3 && cd $K3
git clone -b trillium-new https://github.com/milabench/milabench.git
uv venv --python-preference only-managed --python 3.12 --seed env
VIRTUAL_ENV=$PWD/env uv pip install -e './milabench[cuda]'
export PATH=$PWD/env/bin:$PATH MILABENCH_BASE=$PWD/output MILABENCH_GPU_ARCH=cuda
milabench install --config milabench/config/torchtitan_native_kimi.yaml --select k3-1n-l4
milabench prepare --config milabench/config/torchtitan_native_kimi.yaml --select k3-1n-l4
```

The bench venv is the Qwen pack's (same requirements and torch nightly). `prepare` downloads FineWeb sample-10BT
(500,000 documents) and DeepSeek-V3.1's tokenizer: K3's own is tiktoken (`tiktoken.model` with remote code), which
TorchTitan's `HuggingFaceTokenizer` cannot read; DeepSeek's ids (< 129,280) index K3's 163,840-entry embedding.

## Measured

Feasibility, 1 node, random tokens (job 1260631, a disposable script, not these entries), seq 8192, EP8, full AC:

| layers | precision | peak GiB | tokens/s per GPU |
|---|---|---|---|
| 4 (KDA dense, KDA-MoE x2, MLA-MoE) | BF16 (no FP32 master) | 150.3 | 11,150 |
| 2 (KDA dense, KDA-MoE) | BF16 | 89.3 | 21,700 |
| 2 | FP32 master, BF16 moments | 96.0 | 21,500 |

## Untested

Everything above 1 node; the entries themselves (FineWeb, milabench); PP with the AttnRes block stack between nodes.
