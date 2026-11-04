#!/usr/bin/env bash
# B-G: GLM-5 pp13-ep8 scaling over model copies (13 nodes each). Weak: 32 micro-batches per data-parallel rank (256
# x copies, bubble 27 % throughout). Strong: 768 in total (96 / 48 / 32 / 24 micro-batches, bubble 11 % to 33 %). The weak
# runs at 13 and 52 nodes are A-G's; at 39 nodes weak and strong share one run (768, tagged B-G-weak).
#   ./20_bg.sh 13|26|39|52 [-- extra sbatch args]
# env: TIME (default 01:30:00, an estimate), REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
nodes=${1:?usage: $0 13|26|39|52 [-- sbatch args]}
shift
[ $# -gt 0 ] && [ "$1" = -- ] && shift
export TT_EXPERIMENT=B-G-weak   # the strong-scaling entries are tagged @B-G-strong

e=megatron-glm5-tri-pp13-ep8
case $nodes in
    13) entries=($e-n13-gbs768@B-G-strong) ;;
    26) entries=($e-n26-gbs512 $e-n26-gbs768@B-G-strong) ;;
    39) entries=($e-n39-gbs768) ;;
    52) entries=($e-n52-gbs768@B-G-strong) ;;
    *) echo "no B-G entries on $nodes nodes (13, 26, 39, 52)" >&2; exit 2 ;;
esac
submit "glm5-bg-n$nodes-r$TT_REPETITION" "${TIME:-01:30:00}" "$nodes" "${entries[@]}" -- "$@"
