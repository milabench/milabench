#!/usr/bin/env bash
# B-G: GLM-5 pp13-ep8 scaling over model copies (13 nodes each): weak (256 per copy) and strong (256 in total; 264 at 39
# nodes, where data parallel 24 does not divide 256). 13 and 52 nodes at 256 per copy are A-G's runs.
#   ./20_bg.sh 26|39|52 [-- extra sbatch args]
# env: TIME (default 01:30:00, an estimate), REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
nodes=${1:?usage: $0 26|39|52 [-- sbatch args]}
shift
[ $# -gt 0 ] && [ "$1" = -- ] && shift
export TT_EXPERIMENT=B-G

e=megatron-glm5-tri-pp13-ep8
case $nodes in
    13) entries=($e-n13-gbs256) ;;
    26) entries=($e-n26-gbs512 $e-n26-gbs256) ;;
    39) entries=($e-n39-gbs768 $e-n39-gbs264) ;;
    52) entries=($e-n52-gbs1024 $e-n52-gbs256) ;;
    *) echo "no B-G entries on $nodes nodes (13, 26, 39, 52)" >&2; exit 2 ;;
esac
submit "glm5-bg-n$nodes-r$TT_REPETITION" "${TIME:-01:30:00}" "$nodes" "${entries[@]}" -- "$@"
