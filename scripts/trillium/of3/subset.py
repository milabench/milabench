"""OF3 training-data subset (OF3_BRIEF.md, step 1), from the published dataset caches in CACHES (data.sh fetches them):
- PDB: N_PDB entries of training_cache_with_templates.json, drawn without replacement with probability proportional to
  their summed datapoint weights (AF3 SI Algorithm 1 as WeightedPDBDataset computes them, default sample_weights);
- long monomer distillation: the first N_MONO entries of its cache.json (the dataset reads it in order,
  sample_in_order: true); CACHES/long_head.json is the start of that 15 GB file, cut anywhere;
- RNA distillation: N_RNA entries drawn uniformly.
Writes OUT/{pdb,long_monomer,rna}_cache.json (indented: OpenFold3 reads the cache type on its second line) and OUT/keys.txt, the S3 keys these entries read (structure, MSA arrays,
template caches, template structures, reference molecules).
Usage: python subset.py CACHES OUT N_PDB N_MONO N_RNA [SEED]"""
import json
import random
import sys
from pathlib import Path

caches, out = Path(sys.argv[1]), Path(sys.argv[2])
n_pdb, n_mono, n_rna = map(int, sys.argv[3:6])
rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
out.mkdir(parents=True, exist_ok=True)
P = "pdb_training_set"
TSA = f"{P}/templates/template_structure_arrays"
keys = set()


def templates(chain):
    for tid in chain.get("template_ids") or []:
        keys.add(f"{TSA}/{tid.split('_')[0]}/{tid}.npz")


def weighted_sample(items, weights, n):
    """Without replacement, probability proportional to weight (Efraimidis-Spirakis)."""
    return [i for _, i in sorted(((rng.random() ** (1 / w), i) for i, w in zip(items, weights)), reverse=True)[:n]]


# PDB: weight of an entry = sum over its chains and interfaces of w_type / cluster_size * (3 n_prot + 3 n_nuc + n_lig)
pdb = json.load(open(caches / "pdb_cache.json"))
A = {"PROTEIN": 3.0, "DNA": 3.0, "RNA": 3.0, "LIGAND": 1.0}
ids, weights = [], []
for pid, e in pdb["structure_data"].items():
    w = sum(0.5 / int(c["cluster_size"]) * A[c["molecule_type"]] for c in e["chains"].values())
    w += sum(1.0 / int(i["cluster_size"]) * sum(A[e["chains"][c]["molecule_type"]] for c in iid.split("_"))
             for iid, i in e["interfaces"].items())
    ids.append(pid)
    weights.append(w)
chosen = weighted_sample(ids, weights, n_pdb)
pdb["structure_data"] = {pid: pdb["structure_data"][pid] for pid in chosen}
for pid, e in pdb["structure_data"].items():
    keys.add(f"{P}/preprocessed_pdb_data/standard/structure_files/{pid}/{pid}.npz")
    for c in e["chains"].values():
        rep = c.get("alignment_representative_id")
        if rep:
            keys.add(f"{P}/alignment_arrays/{rep}.npz")
            if c.get("template_ids"):
                keys.add(f"{P}/templates/train_template_cache/{rep}.npz")
        templates(c)
json.dump(pdb, open(out / "pdb_cache.json", "w"), indent=4)
points = sum(len(e["chains"]) + len(e["interfaces"]) for e in pdb["structure_data"].values())
print(f"pdb: {len(chosen)} of {len(ids)} entries, {points} chains and interfaces")

# Long monomers: the first N_MONO entries of structure_data, decoded one by one from the cut file
text = (caches / "long_head.json").read_text(errors="ignore")
dec = json.JSONDecoder()
pos = text.index('"structure_data"')
head = text[:pos].rstrip().rstrip(",") + "}"
mono = dec.decode(head)
pos = text.index("{", pos) + 1
entries = {}
while len(entries) < n_mono:
    while text[pos] in " \n\t,":
        pos += 1
    key, pos = dec.raw_decode(text, pos)
    pos = text.index(":", pos) + 1
    while text[pos] in " \n\t":
        pos += 1
    entries[key], pos = dec.raw_decode(text, pos)
mono["structure_data"] = entries
M = "monomer_distillation_sets_v2/long_monomers/preprocessed"
for mid, e in entries.items():
    keys.update(f"{M}/{mid}/{f}.npz" for f in ("structure", "alignment"))
    for c in e["chains"].values():
        if c.get("template_ids"):
            keys.add(f"{M}/{c['alignment_representative_id']}/template.npz")
        templates(c)
json.dump(mono, open(out / "long_monomer_cache.json", "w"), indent=4)
print(f"long monomers: first {len(entries)}")

# RNA: uniform
rna = json.load(open(caches / "rna_cache.json"))
chosen = rng.sample(sorted(rna["structure_data"]), n_rna)
print(f"rna: {len(chosen)} of {len(rna['structure_data'])}")
rna["structure_data"] = {r: rna["structure_data"][r] for r in chosen}
for r in chosen:
    keys.update(f"rna_distillation_set/rna_monomer_preprocessed_cache/{r}/{f}.npz" for f in ("structure", "alignment"))
json.dump(rna, open(out / "rna_cache.json", "w"), indent=4)

(out / "keys.txt").write_text("".join(f"{k}\n" for k in sorted(keys)))
print(f"keys: {len(keys)} (reference molecules not included)")
