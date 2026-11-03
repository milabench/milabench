#!/usr/bin/env bash
# A-G: GLM-5 layouts (DESIGN.md), one job per node count, its entries one after the other (20 steps each).
#   ./10_ag.sh 13|26|52|12|48 [-- extra sbatch args]
#     13: pp13-ep8 (256), small batch (64), micro-batch 2, selective recompute
#     26: pp13-ep16, pp26-ep8 (one copy each)
#     52: pp13-ep8 x 4 copies, mb2, sel; pp13-ep16 x 2, pp13-ep32 x 1, pp26-ep8 x 2
#     12 / 48: pp6-ep16, 1 / 4 copies
# env: TIME (default per node count, estimates: ~15 min startup at scale + 20 steps per entry), REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
nodes=${1:?usage: $0 13|26|52|12|48 [-- sbatch args]}
shift
[ $# -gt 0 ] && [ "$1" = -- ] && shift
export TT_EXPERIMENT=A-G

e=megatron-glm5-tri
case $nodes in
    13) entries=($e-pp13-ep8-n13-gbs256 $e-pp13-ep8-n13-gbs64 $e-pp13-ep8-n13-gbs256-mb2 $e-pp13-ep8-n13-gbs256-sel) time=02:00:00 ;;
    26) entries=($e-pp13-ep16-n26-gbs256 $e-pp26-ep8-n26-gbs256) time=01:30:00 ;;
    52) entries=($e-pp13-ep8-n52-gbs1024 $e-pp13-ep8-n52-gbs1024-mb2 $e-pp13-ep8-n52-gbs1024-sel
                 $e-pp13-ep16-n52-gbs512 $e-pp13-ep32-n52-gbs256 $e-pp26-ep8-n52-gbs512) time=03:00:00 ;;
    12) entries=($e-pp6-ep16-n12-gbs256) time=00:45:00 ;;
    48) entries=($e-pp6-ep16-n48-gbs1024) time=00:45:00 ;;
    *) echo "no A-G entries on $nodes nodes (13, 26, 52, 12, 48)" >&2; exit 2 ;;
esac
submit "glm5-ag-n$nodes-r$TT_REPETITION" "${TIME:-$time}" "$nodes" "${entries[@]}" -- "$@"
