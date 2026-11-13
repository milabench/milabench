"""Analysis of the Trillium B200 campaign (see ../../README.md and ../../DESIGN.md).

The Rorqual modules (``scripts/analysis/tt_analysis``) are reused where the formats are shared
(run data, node telemetry, srygaard health checks, style): they are imported, not copied.
"""

import sys
from pathlib import Path

RORQUAL_ANALYSIS = Path(__file__).resolve().parents[3] / "analysis"
if str(RORQUAL_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(RORQUAL_ANALYSIS))
