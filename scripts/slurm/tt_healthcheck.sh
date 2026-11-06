#!/bin/bash
# Node health check for the topology study / window (run inside an allocation).
#
#   sbatch --nodes=N ... scripts/slurm/tt_healthcheck.sh            # standalone
#   bash scripts/slurm/tt_healthcheck.sh                             # inside a job
#
# Per node: 4 non-MIG H100, no uncorrected ECC errors, InfiniBand ports Active
# (same adapter count / rates as most nodes of the allocation),
# bf16 matmul TFLOPS per GPU, intra-node all-reduce bandwidth. Then one
# all-reduce over all nodes of the allocation. A node fails if any check fails
# or if its slowest GPU / intra-node bandwidth is > 5 % (matmul) / 10 %
# (all-reduce) below the allocation median.
#
# Output in $TT38/health/<jobid>/: nodes.tsv, healthy.txt, excluded.txt.
#
#SBATCH --job-name=tt-health
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output=%x-%j.out

set -euo pipefail

TT38=${TT38:-$SCRATCH/tt38}
unset PYTHONPATH PIP_CONFIG_FILE
PY=$TT38/output/venv/torchtitan_native/bin/python
TORCHRUN=$TT38/output/venv/torchtitan_native/bin/torchrun
HEALTH=$TT38/milabench/scripts/slurm/tt_health.py
OUT=$TT38/health/$SLURM_JOB_ID
mkdir -p "$OUT"
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES=$(echo "$NODES" | wc -l)
echo "[tt-health] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST start=$(date +%s)"

# 1. Static checks + matmul + intra-node all-reduce, every node in parallel.
srun --ntasks-per-node=1 --overlap bash -c "
    unset PYTHONPATH
    host=\$(hostname -s)
    gpus=\$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'H100 80GB')
    mig=\$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | grep -c Enabled || true)
    ecc=\$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits \
          | awk '{s += (\$1 ~ /^[0-9]+\$/) ? \$1 : 0} END {print s + 0}')
    ib=\$(for ca in /sys/class/infiniband/*; do
            [ \"\$(cat \$ca/ports/1/link_layer 2>/dev/null)\" = InfiniBand ] || continue
            echo \"\$(cat \$ca/ports/1/state | awk '{print \$2}')@\$(cat \$ca/ports/1/rate | awk '{print \$1}')\"
          done | paste -sd+)
    echo \"[tt-health] static host=\$host gpus=\$gpus mig=\$mig ecc=\$ecc ib=\${ib:-none}\"
    $PY $HEALTH matmul
    TT_HEALTH_SCOPE=node:\$host $TORCHRUN --standalone --nproc-per-node=4 $HEALTH allreduce 2>/dev/null
" | tee "$OUT/raw.log"

# 2. All-reduce over the whole allocation.
HEAD=$(echo "$NODES" | head -1)
TT_HEALTH_SCOPE=all TT_HEALTH_NODES=$NNODES srun --ntasks-per-node=1 --overlap \
    "$TORCHRUN" --nnodes="$NNODES" --nproc-per-node=4 --rdzv-backend=c10d \
    --rdzv-endpoint="$HEAD:29511" --rdzv-conf=timeout=300 "$HEALTH" allreduce 2>/dev/null \
    | tee -a "$OUT/raw.log" || echo "[tt-health] allreduce scope=all FAILED" | tee -a "$OUT/raw.log"

# 3. Verdict per node.
"$PY" - "$OUT" <<'EOF'
import re, statistics, sys
out = sys.argv[1]
nodes = {}
for line in open(f"{out}/raw.log"):
    kv = dict(re.findall(r"(\w+)=(\S+)", line))
    if "[tt-health] static" in line:
        nodes.setdefault(kv["host"], {}).update(kv)
    elif "[tt-health] matmul" in line:
        nodes.setdefault(kv["host"], {}).setdefault("tflops", []).append(float(kv["tflops"]))
    elif "[tt-health] allreduce" in line and kv.get("scope", "").startswith("node:"):
        nodes.setdefault(kv["scope"][5:], {})["intra_busbw"] = float(kv["busbw_gbs"])
    elif "allreduce scope=all" in line:
        print(line.strip())
mm = statistics.median(min(n["tflops"]) for n in nodes.values() if n.get("tflops"))
# InfiniBand: every port ACTIVE, and the same adapter count / rates as most nodes
# of the allocation (1x200 on Rorqual, 4 adapters on Fir and Tamia).
ib_common = statistics.mode(n.get("ib", "none") for n in nodes.values())
bw = statistics.median(n["intra_busbw"] for n in nodes.values() if "intra_busbw" in n)
healthy, excluded = [], []
with open(f"{out}/nodes.tsv", "w") as tsv:
    tsv.write("host\tgpus\tmig\tecc\tib\tmin_tflops\tintra_busbw\tverdict\n")
    for host, n in sorted(nodes.items()):
        reasons = []
        if n.get("gpus") != "4": reasons.append(f"gpus={n.get('gpus')}")
        if n.get("mig") not in (None, "0"): reasons.append("mig")
        if n.get("ecc") not in (None, "0"): reasons.append(f"ecc={n.get('ecc')}")
        ib = n.get("ib", "none")
        if ib == "none" or any(not p.startswith("ACTIVE@") for p in ib.split("+")):
            reasons.append(f"ib={ib}")
        elif ib != ib_common:
            reasons.append(f"ib={ib} differs from {ib_common}")
        t = min(n.get("tflops", [0]))
        if t < 0.95 * mm: reasons.append(f"matmul {t:.0f} < 95% of {mm:.0f}")
        b = n.get("intra_busbw", 0)
        if b < 0.90 * bw: reasons.append(f"intra-node allreduce {b:.0f} < 90% of {bw:.0f} GB/s")
        verdict = "ok" if not reasons else "; ".join(reasons)
        (healthy if not reasons else excluded).append((host, verdict))
        tsv.write(f"{host}\t{n.get('gpus')}\t{n.get('mig')}\t{n.get('ecc')}\t{n.get('ib')}\t{t:.0f}\t{b:.0f}\t{verdict}\n")
open(f"{out}/healthy.txt", "w").write("".join(f"{h}\n" for h, _ in healthy))
open(f"{out}/excluded.txt", "w").write("".join(f"{h}\t{r}\n" for h, r in excluded))
print(f"[tt-health] healthy={len(healthy)} excluded={len(excluded)} median_tflops={mm:.0f} median_intra_busbw={bw:.0f}")
for h, r in excluded:
    print(f"[tt-health] EXCLUDED {h}: {r}")
EOF
echo "[tt-health] results in $OUT end=$(date +%s)"
