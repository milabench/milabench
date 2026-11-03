"""Figure style: the Rorqual conventions (``tt_analysis.style``: minimal, serif, measured filled /
synthetic hollow), with a validated categorical palette (fixed order, colour follows the entity)
and a SYNTHETIC watermark inside every figure drawn wholly or partly from synthetic data.
"""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap
from tt_analysis import style as tt
from tt_analysis.style import TEXT_WIDTH, apply, finish, marker_kw, save  # noqa: F401  (re-exported)

# Categorical slots, in this order (adjacent pairs pass the colour-vision-deficiency checks).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE = {250: "#86b6ef", 450: "#2a78d6", 650: "#104281"}  # ordinal steps of the sequential hue
CRITICAL = "#d03b3b"  # status: outliers and thresholds only, never a series
INK, INK2, MUTED, LIGHT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "tri_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

QWEN, GLM = "qwen3_8-27B", "glm5"
WORKLOAD_COLOR = {QWEN: SERIES[0], GLM: SERIES[1]}
WORKLOAD_LABEL = {QWEN: "Qwen3.8-27B", GLM: "GLM-5"}
COLLECTIVE_COLOR = {"all_reduce": SERIES[0], "all_gather": SERIES[1], "reduce_scatter": SERIES[2],
                    "alltoall": SERIES[3]}
COLLECTIVE_MARKER = {"all_reduce": "o", "all_gather": "s", "reduce_scatter": "^", "alltoall": "D"}
RECIPE_COLOR = {"fp8-tensorwise": SERIES[0], "fp8-rowwise": SERIES[1], "fp8-blockwise": SERIES[2],
                "mxfp8": SERIES[3]}
TOKENS_COLOR = {12288: BLUE[250], 24576: BLUE[450], 49152: BLUE[650]}  # ordinal: light to dark
SCALE_COLOR = {"small": BLUE[250], "large": BLUE[650]}


def synthetic_mark(fig, any_synthetic: bool, text="includes synthetic data (hollow markers)") -> None:
    """The Rorqual tag (sets the flag behind \\syntheticnote{}) plus a watermark across the figure."""
    tt.synthetic_tag(fig, any_synthetic, text)
    if any_synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", rotation=18, fontsize=40,
                 fontweight="bold", color=CRITICAL, alpha=0.10, zorder=10)


def used_synthetic() -> bool:
    return tt.USED_SYNTHETIC


def reset_synthetic() -> None:
    tt.USED_SYNTHETIC = False
