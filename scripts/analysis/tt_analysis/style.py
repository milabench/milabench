"""Minimal (Tufte-like) seaborn style shared by every figure.

No grid, no top/right spines, axes trimmed to the data range, direct labels
instead of legends where possible. Colors: Okabe-Ito (colorblind-safe), and
every color is paired with a marker or line style so figures read in gray.

Conventions:
- measured: filled marker, solid line;
- estimated (extrapolated, model-based): dashed line or hatched bar, "est." in the label;
- synthetic (placeholder data, not a measurement): hollow marker, and the
  figure carries a "synthetic data" tag.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

TEXT_WIDTH = 5.5  # NeurIPS text width, inches

OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
    "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000",
}
CLUSTER_COLOR = {"rorqual": OKABE_ITO["blue"], "fir": OKABE_ITO["orange"],
                 "tamia": OKABE_ITO["green"], "lynx": OKABE_ITO["purple"]}
CLUSTER_MARKER = {"rorqual": "o", "fir": "s", "tamia": "^", "lynx": "D"}
FAMILY_COLOR = {"H": OKABE_ITO["vermillion"], "S": OKABE_ITO["blue"], "P": OKABE_ITO["green"],
                "ALL": OKABE_ITO["black"], "slurm": "#888888", "1node": "#888888"}
FAMILY_MARKER = {"H": "^", "S": "o", "P": "s", "ALL": "D", "slurm": "v", "1node": "P"}
FAMILY_LABEL = {"H": "H: 3 nodes per switch", "S": "S: one switch", "P": "P: fewest switches",
                "ALL": "ALL: every healthy node", "slurm": "placed by Slurm"}
GRAY = "#777777"
LIGHT = "#BBBBBB"


def apply() -> None:
    sns.set_theme(context="paper", style="ticks", font="serif", rc={
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "legend.frameon": False, "lines.linewidth": 1.0, "lines.markersize": 4,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def finish(ax, trim=True) -> None:
    sns.despine(ax=ax, offset=4, trim=trim)


# Set by synthetic_tag while a figure is drawn; read by figures.render_all for the report.
USED_SYNTHETIC = False


def synthetic_tag(fig, any_synthetic: bool, text="includes synthetic data (hollow markers)") -> None:
    global USED_SYNTHETIC
    if any_synthetic:
        USED_SYNTHETIC = True
        fig.text(0.995, 0.995, text, ha="right", va="top",
                 fontsize=6, style="italic", color=OKABE_ITO["vermillion"])


def marker_kw(color: str, synthetic: bool) -> dict:
    return {"markerfacecolor": "white" if synthetic else color, "markeredgecolor": color,
            "markeredgewidth": 0.8}


def save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


mpl.rcParams["axes.unicode_minus"] = False
