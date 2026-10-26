# megatron

GLM-5 training on plain Megatron-LM (`pretrain`), launched by torchrun over srun: no Ray, no SGLang rollout, random
initialisation, random packed (THD) tokens. Same install as the `slime` pack (the GLM-5 layer spec, DSA indexer and
sparse MLA come from slime's `slime_plugins`; its benchfile is imported and subclassed). Metric: `rate` = global
tokens / step time (first rank), plus `step_time`.

## Frozen parameters (`frozen:`)
| key | meaning |
|---|---|
| `preset` | `none`, `experts` (regex `experts`, like slime's `--freeze-params-name-list experts`: it also matches `shared_experts`), `attention-only` |
| `freeze` / `only_train` | extra regexes on parameter names |
| `indexer` | also freeze the DSA indexer (`wq_b`, `wk`, `k_norm`, `weights_proj`) |

Frozen parameters are set before Megatron wraps the model, so they get no gradient buffer and no optimizer state.

## Placement (`placement:`)
- `nodes`: node rank = position in `system.nodes`. `as-is`, `sorted`, `topology` (nodes grouped per leaf switch, main
  node's switch first, fullest switches next) or an explicit list (must start with the main node).
- Switches come from `placement.topology {node: switch}`, else `node.switch`, else `scontrol show topology`.
- `order`: `tp-cp-ep-dp-pp` (Megatron default) or `tp-cp-ep-pp-dp` (adds `--use-tp-pp-dp-mapping`).
- `nccl`: env passed to the job (`NCCL_TOPO_FILE`, `NCCL_CROSS_NIC`, ...).
- `placement.py` simulates Megatron's rank groups; `main.py` logs, on rank 0, how many nodes / switches every group
  (tp, cp, dp, pp, ep, expert-dp) spans and how many pipeline hops cross a switch. Rule of thumb: keep EP (all-to-all)
  and TP under one switch; `nodes: topology` does that with the default order, `tp-cp-ep-pp-dp` makes the pipeline
  local instead and pushes the data-parallel reduce across switches.

## Status
Config only, **not run**: written against Megatron-LM `4b4acac` (the pinned slime commit) without a GPU. Unverified
points: that `slime_plugins.models.glm5.get_glm5_spec` works outside slime's training actor (it reads `args.hf_checkpoint`,
set by `main.py`), that the random THD batches satisfy `pretrain_gpt.forward_step`, and the 16-node layout memory
(PP8 x EP8, first/last stage 14/16 layers, optimizer CPU offload, as the slime frozen entry).
