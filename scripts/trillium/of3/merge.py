"""Print the runner yaml of one run: BASE (initial_training.yml) with @S3@ and @SUBSET@ set under DATA, deep-merged with
the entries of VARIANTS named in NAMES (comma-separated, in order; @RUN@ set to OUTPUT_DIR and @DATA@ to DATA in them),
then output_dir, num_nodes and max_steps.
Usage: python merge.py BASE VARIANTS NAMES DATA OUTPUT_DIR NODES STEPS > runner.yml"""
import sys

import yaml

base, variants, names, data, out, nodes, steps = sys.argv[1:]
cfg = yaml.safe_load(open(base).read().replace("@S3@", f"{data}/s3").replace("@SUBSET@", f"{data}/subset"))


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


for name in names.split(","):
    merge(cfg, yaml.safe_load(open(variants).read().replace("@RUN@", out).replace("@DATA@", data))[name] or {})
cfg["experiment_settings"]["output_dir"] = out
cfg["pl_trainer_args"].update(num_nodes=int(nodes), max_steps=int(steps))
yaml.safe_dump(cfg, sys.stdout, sort_keys=False)
