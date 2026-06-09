# RecursionGFN

GFlowNet benchmark for molecular fragment generation using the SEH (soluble epoxide hydrolase) task from the `gflownet` library.

## What it measures

GPU throughput on a generative flow network that constructs molecules fragment-by-fragment. The workload combines graph neural network forward passes (policy model) with a pretrained proxy reward model evaluation. Stresses both GNN scatter/gather and small dense MLP operations.

## Workload

Trains a GFlowNet policy to generate molecular graphs that maximize predicted binding affinity to SEH, as scored by a frozen pretrained proxy (`bengio2021flow`). Uses the fragment-based action space from `SEHFragTrainer`.

## Dataset

No fixed dataset -- the GFlowNet generates its own training data on-the-fly via online policy rollouts. The pretrained proxy model weights are downloaded during `prepare.py`.

## Framework / dependencies

- `gflownet` (cloned from `github.com/Delaunay/gflownet`, branch `milabench`)
- `torch-geometric`, `torch-scatter`, `torch-sparse`, `torch-cluster`
- `rdkit`, `pyro-ppl`, `botorch`, `gpytorch`
- Requires CUDA (`FORCE_CUDA=1`)

## Execution

Single-GPU only (`per_gpu`). Default: 100 training steps, batch size 128, 4-layer policy with width 128.
