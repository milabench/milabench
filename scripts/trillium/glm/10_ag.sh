#!/usr/bin/env bash
# A-G: GLM-5 layouts (DESIGN.md), one job per node count, its entries one after the other (20 steps each).
#   ./10_ag.sh 13|26|52|12|48|36|10|20|30|40|25|50 [-- extra sbatch args]
#     13: pp13-ep8 (256), small batch (64 per copy), selective recompute
#     26: pp13-ep16, pp26-ep8 (one copy each), 512
#     52: pp13-ep8 x 4 copies, sel; pp13-ep16 x 2, pp13-ep32 x 1, pp26-ep8 x 2; all at 1024
#     12 / 48: pp6-ep16, 1 / 4 copies, 256 / 1024
#     10 / 20 / 30 / 40: pp10-ep8, 1 to 4 copies, 200 to 800      25: pp25-ep8, 512
#     50: pp10-ep8 x 5 copies, 1000; pp25-ep8 x 2, 1024
#   Every layout at a node count takes that count's weak-scaling batch (256 x nodes / 13, benchmarks/megatron/dev.yaml)
# env: TIME (default per node count, estimates: ~15 min startup at scale + 20 steps per entry), REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
nodes=${1:?usage: $0 13|26|52|12|48|36|10|20|30|40|25|50 [-- sbatch args]}
shift
[ $# -gt 0 ] && [ "$1" = -- ] && shift
export TT_EXPERIMENT=A-G

e=megatron-glm5-tri
case $nodes in
    13) entries=($e-pp13-ep8-n13-gbs256 $e-pp13-ep8-n13-gbs64 $e-pp13-ep8-n13-gbs256-sel) time=02:00:00 ;;
    26) entries=($e-pp13-ep16-n26-gbs512 $e-pp26-ep8-n26-gbs512) time=01:30:00 ;;
    52) entries=($e-pp13-ep8-n52-gbs1024 $e-pp13-ep8-n52-gbs1024-sel
                 $e-pp13-ep16-n52-gbs1024 $e-pp13-ep32-n52-gbs1024 $e-pp26-ep8-n52-gbs1024) time=03:00:00 ;;
    12) entries=($e-pp6-ep16-n12-gbs256) time=00:45:00 ;;
    48) entries=($e-pp6-ep16-n48-gbs1024) time=00:45:00 ;;
    36) entries=($e-pp6-ep16-n36-gbs768) time=00:45:00 ;;  # a 47-node window, instead of 48
    10|20|30|40) entries=($e-pp10-ep8-n$nodes-gbs$((nodes * 20))) time=01:00:00 ;;
    25) entries=($e-pp25-ep8-n25-gbs512) time=01:00:00 ;;
    50) entries=($e-pp10-ep8-n50-gbs1000 $e-pp25-ep8-n50-gbs1024) time=01:30:00 ;;
    *) echo "no A-G entries on $nodes nodes (13, 26, 52, 12, 48, 10, 20, 30, 40, 25, 50; 36 in a 47-node window)" >&2; exit 2 ;;
esac
submit "glm5-ag-n$nodes-r$TT_REPETITION" "${TIME:-$time}" "$nodes" "${entries[@]}" -- "$@"
