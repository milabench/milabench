#!/usr/bin/env bash
# C: precision. TransformerEngine FP8 recipes against BF16 on pp13-ep8.
#   ./30_c.sh speed 13|52 [-- extra sbatch args]     C-speed: bf16, fp8-tensorwise, fp8-blockwise, mxfp8, 20 steps each
#   ./30_c.sh loss [RECIPES] [-- extra sbatch args]  C-loss: 500 steps from seed 1234 at 13 nodes, one job per recipe
#                                                    (default "bf16 fp8-tensorwise fp8-blockwise mxfp8")
# env: TIME (speed: 02:00:00 at 13 nodes, 02:30:00 at 52; loss: 06:00:00: estimates, the B200 step time is not
# measured yet), REPETITION, and _common.sh's
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_common.sh"
mode=${1:?usage: $0 speed 13|52 | loss [RECIPES]}
shift

case $mode in
    speed)
        nodes=${1:?usage: $0 speed 13|52}
        shift
        [ $# -gt 0 ] && [ "$1" = -- ] && shift
        export TT_EXPERIMENT=C-speed
        case $nodes in
            13) base=megatron-glm5-tri-pp13-ep8-n13-gbs256 time=02:00:00 ;;
            52) base=megatron-glm5-tri-pp13-ep8-n52-gbs1024 time=02:30:00 ;;
            *) echo "C-speed runs on 13 or 52 nodes" >&2; exit 2 ;;
        esac
        submit "glm5-c-speed-n$nodes-r$TT_REPETITION" "${TIME:-$time}" "$nodes" \
            "$base" "$base-fp8-tensorwise" "$base-fp8-blockwise" "$base-mxfp8" -- "$@"
        ;;
    loss)
        recipes="bf16 fp8-tensorwise fp8-blockwise mxfp8"
        if [ $# -gt 0 ] && [ "$1" != -- ]; then recipes=$1; shift; fi
        [ $# -gt 0 ] && [ "$1" = -- ] && shift
        export TT_EXPERIMENT=C-loss
        for recipe in $recipes; do
            submit "glm5-c-loss-$recipe-r$TT_REPETITION" "${TIME:-06:00:00}" 13 "megatron-glm5-tri-loss-$recipe-n13" -- "$@"
        done
        ;;
    *) echo "unknown mode $mode (speed | loss)" >&2; exit 2 ;;
esac
