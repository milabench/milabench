# GLM-5 runs: what to run, for the GLM owner

Rorqual, Megatron/slime. Companion to `EXPERIMENTS.md` (queue, roles, rules) and `FORMAT.md`
(output format). Values marked **to confirm** are open: decide them before the window and write
them here.

## Why these runs

The main GLM-5 question is whether using **every node** of Rorqual is worth it compared with the
largest set of nodes where no expert group crosses a switch. Using every node forces some expert
groups to span two switches, so their all-to-all crosses the spine. We hope that cost is small enough
that the bigger job still gives more total throughput. These runs feed:

| figure | question | runs |
|---|---|---|
| 11 Expert placement (tier 1) | every node vs. the largest aligned set; cost of crossing groups | G1-G4 |
| 12 All-to-all per EP group (tier 2) | the mechanism: crossing groups are slower | G1-G4, if per-group timers exist |
| 10 GLM-5 scaling, EP size, FP8 (tier 2) | EP size and FP8 at scale | pre-window small runs, G5, G6 |

GLM blocks come first in the window (complex, more likely to fail, more time to fix).

## Settings held fixed in every run

Every run below uses **the same model and training configuration**. Only the node list (and so the
data-parallel size in G4) changes. Otherwise the comparison is invalid.

| setting | value |
|---|---|
| model | GLM-5, **full depth** if it fits on 208 GPUs (**to confirm**). If not, one truncated depth for *all* runs |
| expert parallelism | **EP8**: one expert group = 8 GPUs = **2 nodes** (G6: EP16 = 4 nodes) |
| precision | BF16 (G5: FP8) |
| sequence length, microbatch, TP / PP / CP, optimizer, recompute | **to confirm**, then frozen |
| deviations from the real recipe (optimizer, MTP, data, ...) | listed in the summary's `deviations` |
| measurement | warm-up, then a fixed number of measured steps, as in Qwen (`FORMAT.md`) |

**Expert groups must be 2 consecutive hosts in the node list.** Megatron builds expert groups from
consecutive ranks, so with 4 GPUs per node, EP8 = hosts (1,2), (3,4), ... of the host list. This
holds only if the rank order puts EP innermost after TP/CP and expert-TP = 1. **Check it**: print
the hosts of each expert group at init on a small run and compare with the node list. The layouts
below are just node orders; if the mapping is different, they mean nothing.

## Node sets (Rorqual, 66 full-GPU nodes)

Leaves: 13 leaves of 3 nodes (ihpc134-136, ihpc317-326), ihpc217 (8), ihpc218 (7), ihpc316 (12).
MIG nodes (leaves ihpc125-133) are never used. Lists are computed **from the healthy set after the
window health check**, never by hand.

- **52-node set** (G1-G3, G5): an even number of nodes on every leaf. 2 of 3 on each 3-node leaf
  (13 x 2 = 26), plus 8 on ihpc217, 6 on ihpc218 and 12 on ihpc316. That is 52 nodes, 208 GPUs and
  26 expert groups.
- **Every node** (G4): every healthy node, an even count (66 if all healthy: 33 groups).

## Runs

| id | run | nodes | node order | groups crossing a switch | window blocks |
|---|---|---|---|---|---|
| G1 | **EP8 aligned** | 52-node set | leaf by leaf: both nodes of every group on one leaf | 0 / 26 | 1, 11 |
| G2 | **EP8 crossed** | same 52 | pairs taken from two leaves: (leaf A node, leaf B node), ... | 26 / 26 | 3, 13 |
| G3 | **EP8 mixed** | same 52 | aligned order, except a few pairs of 3-node leaves swap partners (A1 B1, A2 B2) | **6-7 / 26** (~1 in 4) | 5, 15 |
| G4 | **EP8 mixed, every node** | every healthy node | Megatron's default: leaf by leaf, all nodes | 8 / 33 if all 66 healthy (record it) | 7, 17 |
| G5 | FP8, otherwise = G1 | 52-node set | aligned | 0 | 22, 24 |
| G6 | EP16, otherwise = G1 | 52-node set | Megatron's default; record the crossing count | record | 35 (P3, if time remains) |

The G3 target (~1 in 4 crossing) matches the share in G4, so G3 isolates the crossing cost at equal
size and G4 adds the scale. G1-G4 run twice each (pass 1, pass 2). G5 also runs twice. Each run is one
block of ~15 min (assumed until measured).

During G1-G3 the 14 idle nodes (one per 3-node leaf, one on ihpc218) are free for the 1-node Qwen
smoke and 1-node health re-checks. G4 uses every node and runs alone.

## Before the window (checklist)

- [ ] Full model fits on 208 GPUs at EP8 (memory at the chosen microbatch). If not, choose the depth.
- [ ] Freeze the configuration in the table above and write it here.
- [ ] Expert-group-to-host mapping checked on a small run (see above).
- [ ] Node-order generator: from the healthy host list and `scontrol show topology`, output the four
      node lists (aligned, crossed, mixed, every node) and the crossing count of each. Check it on the
      login node (no GPU).
- [ ] Output in milabench format (`FORMAT.md` §1 and "MoE additions"), via a voirfile or a converter.
      Required: `rate` per step; `summary` with `placement.hosts` in rank order, `placement.ep_groups`,
      `moe` block, `params_trainable_fraction`, `deviations`. Wanted for Fig. 12: `per_ep_group`
      all-to-all time per step, if Megatron's MoE timers can give it.
- [ ] Small truncated runs (Fir or a Rorqual reservation): EP4 / EP8 / EP16, BF16 vs FP8, to test
      the pipeline end to end and give Fig. 10's small-scale points. EP4 fits only with part of the
      model frozen. Record `params_trainable_fraction` < 1: these runs are not compared with the
      others as the same model.
- [ ] Measure run length (startup + warm-up + measurement) on a few nodes. The queue assumes 15 min.
- [ ] One prepared command per block (G1-G6): node list from the generator, run tag, config.
- [ ] Check on Fir: `analyze.py all --no-synthetic` parses a GLM run with no manual step.

## If nodes are excluded

- An exclusion on ihpc217 or ihpc316, or 2 nodes on one 3-node leaf, shrinks the set to **50**
  (or 48). A single exclusion on a 3-node leaf or on ihpc218 changes nothing for the 52-node set.
- G1, G2 and G3 must run **on the same set**. If the set changes mid-comparison, rerun all three on
  the new set. Never mix sets.
- G4 always uses every healthy node; record the count and the crossing count.

## Quick check after each run

- `rate` events present and the summary parsed (`analyze.py quick`).
- Crossing count in the summary matches the table.
- Throughput: G1 and G2 differ in the expected direction. If G2 >= G1, check the group mapping first.
