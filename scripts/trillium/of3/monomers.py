"""Which long-monomer entries a run reads, from OpenFold3's sampler (OF3DistributedSampler, seed 42, epoch 0): the global
list draws each of epoch_len samples' dataset by weight; the in-order monomer set hands out entries 0, 1, 2, ... in the
order its samples appear, wraps back to 0 once (_get_ordered_subset) and raises IndexError past 2 N draws; rank r takes
positions r, r + RANKS, ... Each rank's DataLoader has up to 16 workers x 2 prefetched samples beyond the last step.
Prints the monomer draws a run of STEPS steps reaches, against N; with RUN, checks the prediction against the
monomers that run's steps-r*.jsonl logged, through the order of CACHE (the subset's long_monomer_cache.json).
Usage: python monomers.py RANKS STEPS N [RUN CACHE]"""
import glob
import json
import sys

import torch

ranks, steps, n = map(int, sys.argv[1:4])
weights = torch.tensor([0.5, 0.495, 0.05])  # weighted-pdb, long-monomer-distillation, RNA (initial_training.yml)
generator = torch.Generator().manual_seed(42)
datasets = torch.multinomial(weights, 128000, replacement=True, generator=generator).tolist()
draw, k = [None] * len(datasets), 0
for i, d in enumerate(datasets):
    if d == 1:
        draw[i], k = k, k + 1
prefetch = 16 * 2
cum = [0]
for d in datasets:
    cum.append(cum[-1] + (d == 1))


def draws(s):
    """Monomer draws reached by a run of s steps (epoch 0 only)."""
    return cum[min((s + prefetch) * ranks, len(datasets))]


used = draws(steps)
print(f"{ranks} ranks, {steps} steps (+{prefetch} prefetched per rank): {(steps + prefetch) * ranks} samples, {used}"
      f" monomer draws; N {n}: {'no repeat' if used <= n else 'repeats' if used <= 2 * n else 'IndexError'}")
last = len(datasets) // ranks - prefetch
norepeat = max((s for s in range(1, last + 1) if draws(s) <= n), default=0)
safe = max((s for s in range(1, last + 1) if draws(s) <= 2 * n), default=0)
print(f"  steps without a repeated monomer: up to {norepeat}; without IndexError: up to {safe}"
      f"{' (the whole epoch)' if safe == last else ''}")
if len(sys.argv) > 5:
    order = list(json.load(open(sys.argv[5]))["structure_data"])
    seen = bad = 0
    for f in glob.glob(f"{sys.argv[4]}/steps-r*.jsonl"):
        r = int(f.rsplit("-r", 1)[1].split(".")[0])
        for row in (json.loads(line) for line in open(f)):
            if "step" in row and row["pdb_id"][0].startswith("MGYP"):
                predicted = draw[(row["step"] - 1) * ranks + r]
                seen += 1
                bad += predicted is None or order[predicted % n] != row["pdb_id"][0]
    print(f"  check against {sys.argv[4]}: {seen} logged monomer samples, {bad} differ from the prediction")
