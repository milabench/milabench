"""B200 sizer config: B200 observations where they fit a curve, else the H100 curve (+ B200 OOM markers).

Memory per sample does not depend on the GPU: a bench with a single B200 point (its default batch)
borrows the H100 curve and extrapolates it to the B200 memory. Rebuild after merging new B200 runs:
    python scripts/b200_h100_scaling.py config/scaling/B200-H100.yaml
"""
import sys, yaml
from milabench.sizer import measured, is_oom
# H100 curves that do not transfer: torchatari (memory not monotonic in the batch), recursiongfn (H100 stops at 144)
SKIP_H100 = {"torchatari", "recursiongfn"}
b200 = yaml.safe_load(open("config/scaling/B200.yaml"))
h100 = yaml.safe_load(open("config/scaling/H100.yaml"))
def sizes(v):
    return {o["batch_size"] for o in measured((v or {}).get("observations", []))}
out, src = {}, {}
for bench in sorted(set(b200) | set(h100)):
    if bench == "version": continue
    b, h = b200.get(bench) or {}, h100.get(bench) or {}
    if len(sizes(b)) >= 2:
        out[bench], src[bench] = b, "B200"
    elif len(sizes(h)) >= 2 and bench not in SKIP_H100:
        ooms = [o for o in b.get("observations", []) if is_oom(o)]
        out[bench] = {"observations": list(h["observations"]) + ooms}
        src[bench] = "H100" + (" +oom" if ooms else "")
    elif b:
        out[bench], src[bench] = b, "B200 (1 point)"
yaml.safe_dump(out, open(sys.argv[1], "w"), sort_keys=True, width=1000)
for bench, origin in sorted(src.items()):
    if origin != "B200":
        print(f"{bench:28} {origin}")
