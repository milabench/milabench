# RLHF

PPO-based reinforcement learning from human feedback benchmark using HuggingFace TRL's `PPOTrainer`.

## What it measures

GPU throughput on the full RLHF-PPO pipeline: generation rollouts from a policy LLM, reward scoring, value estimation, and PPO policy updates. Stresses autoregressive generation, multiple model forward/backward passes per step, and memory pressure from holding 4 models simultaneously.

## Workload

Trains a causal LM (EleutherAI/pythia-1b-deduped) using PPO with a separate reward model and value model. The policy generates completions, the reward model scores them, and the value model estimates advantages for PPO updates.

## Models (all pythia-1b-deduped, ~1B params each)

- **Policy**: `AutoModelForCausalLM` -- the model being trained
- **Reference policy**: frozen copy of the policy for KL penalty computation
- **Reward model**: `AutoModelForSequenceClassification` (1 label)
- **Value model**: `AutoModelForSequenceClassification` (1 label)

## Dataset

`trl-internal-testing/descriptiveness-sentiment-trl-style` (split: `descriptiveness`). Pre-tokenized during setup. Last 100 samples held out for eval.

## Framework / dependencies

- `trl` (HuggingFace TRL -- experimental PPO implementation)
- `transformers`, `accelerate`, `datasets`
- `torchcompat` (for accelerator abstraction)

## Execution

- `rlhf-single`: one process per GPU (`per_gpu`)
- `rlhf-gpus`: one job using all GPUs via `accelerate` (`njobs: 1`)
- Uses `AccelerateAllNodes` launch, not `voir` instrumentation directly. Precision: bfloat16.
