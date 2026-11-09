"""How far the PDB part of the step-1 subset is from the full weighted-PDB training set, as OpenFold3's sampler draws
from each: datapoints (chains and interfaces) weighted by AF3 SI Algorithm 1 (WeightedPDBDataset, default weights),
within the full training cache and within the subset cache. Properties per datapoint: its entry's token count before
cropping and chain count (metadata.json), its molecule types, the template hits of its chains (capped at the 4 used)
and the size of their MSA arrays (alignment_arrays, from a listing of the bucket: KEY<TAB>BYTES lines). The full
caches are read from the bucket into memory, not written to disk (1.42 and 1.68 GB).
Usage: python mix.py SUBSET_CACHE ALIGNMENT_LISTING"""
import json
import sys
import urllib.request

B = "https://openfold3-data.s3.amazonaws.com/pdb_training_set/"
A = {"PROTEIN": 3.0, "DNA": 3.0, "RNA": 3.0, "LIGAND": 1.0}
aln = {}
for line in open(sys.argv[2]):
    key, size = line.split("\t")
    aln[key.rsplit("/", 1)[1][:-4]] = int(size)
meta = json.load(urllib.request.urlopen(B + "preprocessed_pdb_data/standard/metadata.json"))["structure_data"]
tokens = {k: e.get("token_count") for k, e in meta.items()}
del meta


def points(structures):
    """(weight, properties) per datapoint."""
    for pid, e in structures.items():
        chains = e["chains"]
        props = {"entry_tokens": tokens.get(pid) or 0, "entry_chains": len(chains)}

        def chain(c):
            return (min(len(c.get("template_ids") or []), 4),
                    aln.get(c.get("alignment_representative_id") or "", 0) / 1e6, c["molecule_type"])

        for c in chains.values():
            t, m, kind = chain(c)
            yield 0.5 / int(c["cluster_size"]) * A[kind], {**props, "templates": t, "msa_mb": m,
                                                           "nucleic": kind in ("DNA", "RNA"), "ligand": kind == "LIGAND"}
        for iid, i in e["interfaces"].items():
            both = [chain(chains[c]) for c in iid.split("_")]
            yield 1.0 / int(i["cluster_size"]) * sum(A[k] for _, _, k in both), {
                **props, "templates": sum(t for t, _, _ in both), "msa_mb": sum(m for _, m, _ in both),
                "nucleic": any(k in ("DNA", "RNA") for _, _, k in both), "ligand": any(k == "LIGAND" for _, _, k in both)}


def describe(name, structures):
    pts = list(points(structures))
    total = sum(w for w, _ in pts)

    def quantiles(key):
        ordered = sorted((p[key], w) for w, p in pts)
        out, acc, qs = [], 0.0, [0.1, 0.5, 0.9]
        for value, w in ordered:
            acc += w
            while qs and acc >= qs[0] * total:
                out.append(value)
                qs.pop(0)
        return "/".join(f"{v:.3g}" for v in out)

    def share(test):
        return sum(w for w, p in pts if test(p)) / total

    print(f"{name}: {len(structures)} entries, {len(pts)} datapoints; weighted p10/p50/p90 and shares:")
    print(f"  entry tokens {quantiles('entry_tokens')}; at least 384 (the crop is full): {share(lambda p: p['entry_tokens'] >= 384):.1%};"
          f" at least 640: {share(lambda p: p['entry_tokens'] >= 640):.1%}")
    print(f"  entry chains {quantiles('entry_chains')}; entries of 100 chains or more: {share(lambda p: p['entry_chains'] >= 100):.1%}")
    print(f"  template hits (up to 4 per chain) {quantiles('templates')}; MSA arrays of the datapoint's chains, MB {quantiles('msa_mb')}")
    print(f"  with a nucleic acid chain {share(lambda p: p['nucleic']):.1%}, with a ligand {share(lambda p: p['ligand']):.1%}")


describe("subset", json.load(open(sys.argv[1]))["structure_data"])
describe("full", json.load(urllib.request.urlopen(B + "dataset_caches/training_cache_with_templates.json"))["structure_data"])
