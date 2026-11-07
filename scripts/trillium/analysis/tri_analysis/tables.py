"""LaTeX tables and macros for the report (``report/generated/tables/``).

Every number taken from synthetic rows is wrapped in ``\\TriSyn{...}`` (red, "syn" superscript),
so a placeholder can never read as a measurement in the text.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .machine import (GLM_SEQS_PER_COPY, GLM_TOP_COPIES, GLM_TOP_NODES, GPU_MEMORY_GIB, N_NODES, NET_D2_ALL_GPUS_JOB, NET_D2_JOBS, NET_D3_INSIDE_JOB, NET_D3_JOB)
from .select import completed as _completed
from .select import (NOISE_ASIDE, QWEN_BASE, REF_NODES, SLOW_NODES, bandwidth_group, checkpoints, drift, e1_changes, failed_runs, loss_gaps, pinned_blocked, qwen_base,
                     with_energy)
from .style import GLM, QWEN, WORKLOAD_LABEL

# Key numbers for the text (meaning in key_numbers); every one is defined, "??" without data.
MACROS = ["TriMatmulBF", "TriMatmulFP", "TriNodesChecked", "TriNodesFlagged", "TriPairBw", "TriPairBwMin",
          "TriPairLat", "TriAllReduceFull", "TriAlltoallFull", "TriBisection", "TriInterferencePct",
          "TriDriftPct", "TriQwenBestLayout", "TriQwenBestTokens", "TriQwenBestTokS", "TriQwenBestMFU",
          "TriQwenFsdpCommPct", "TriQwenWeakEff", "TriQwenWeakEffFromTwo", "TriQwenWeakTwoNodes", "TriGlmBestLayout", "TriGlmBestTokS", "TriGlmBestMFU",
          "TriGlmCopiesEff", "TriGlmCopiesEffTwo", "TriGlmCopiesEffFromTwo", "TriQwenFpBestRecipe", "TriQwenFpSpeedupSmall", "TriQwenFpSpeedupFull",
          "TriGlmFpBestRecipe", "TriGlmFpSpeedupSmall", "TriGlmFpSpeedupFull", "TriCkptQwenSave",
          "TriCkptGlmFullSave", "TriCkptGlmFullBlocked", "TriCkptGlmLoadPeak", "TriCkptWriteGBs", "TriStartupQwen", "TriStartupGlm",
          "TriEnergyBest",
          "TriQwenTop", "TriGlmTop", "TriRunsInReport", "TriGlmCopiesTop",
          # Scaling, GLM-5, operations
          "TriGlmWeakThreeTok", "TriGlmWeakThreeStep", "TriGlmWeakThreeMem", "TriGlmWeakThreeAtoa", "TriGlmStrongThreeRatio", "TriGlmStrongThreeFormula",
          "TriSlowRunNodes", "TriSlowRunTok", "TriSlowRunStep", "TriSlowRunPct", "TriSlowRunEff", "TriStragNodes", "TriStragLowLo", "TriStragLowHi", "TriStragLowRest", "TriStragUtilSlow", "TriStragUtilRest",
          "TriWeakTop","TriWeakFTop","TriWeakHTop","TriWeakFCross","TriWeakHCross","TriWeakFOne","TriWeakFTopTok","TriWeakFStep","TriWeakFStepOne","TriWeakFEffOne","TriWeakFEffTwo","TriWeakHOne","TriWeakHTopTok","TriWeakHStep","TriWeakHStepOne","TriWeakHEffOne","TriWeakHEffTwo",
          "TriStrongOne","TriStrongOneStep","TriStrongTopNodes","TriStrongTop","TriStrongTopStep","TriStrongEff",
          "TriGlmThirteenTok","TriGlmThirteenStep","TriGlmThirteenMem","TriGlmThirteenMfu","TriGlmThirteenAtoa","TriGlmThirteenAtoaPct","TriGlmSmallTok","TriGlmSmallStep","TriGlmSmallMem","TriGlmSmallMfu","TriGlmSmallAtoa","TriGlmSmallAtoaPct",
          "TriGlmSixTok","TriGlmSixStep","TriGlmSixMem","TriGlmSixMfu","TriGlmSixAtoa","TriGlmSixAtoaPct","TriGlmTenTok","TriGlmTenStep","TriGlmTenMem","TriGlmTenMfu","TriGlmTenAtoa","TriGlmTenAtoaPct","TriGlmEpRatio","TriGlmSmallRatio","TriGlmSmallFormula",
          "TriGlmStrongTok","TriGlmStrongStep","TriGlmStrongRatio","TriGlmStrongFormula","TriGlmWeakTwoTok","TriGlmWeakTwoStep","TriGlmWeakTwoMem","TriGlmMemFallPct","TriGlmStrongTwoTok","TriGlmStrongTwoStep","TriGlmStrongTwoRatio","TriGlmStrongTwoFormula",
          "TriStartQLo","TriStartQHi","TriStartQMed","TriStartGLo","TriStartGHi","TriStartGMed","TriEnergyFsdp","TriEnergyHsdp","TriEnergyPp","TriEnergyGlm","TriPowerFsdp","TriPowerHsdp","TriPowerPp","TriPowerGlm",
          # Speed
          "TriQwenFpOne", "TriQwenFpTopNodes", "TriQwenFpTop", "TriQwenMemBf", "TriQwenMemMx", "TriQwenFpOneSpread", "TriQwenFpOneMean", "TriQwenFpTopMean", "TriStackedGain", "TriQwenFpFull", "TriQwenFullMem", "TriGlmFpTensor", "TriGlmFpBlock", "TriGlmFpMx",
          "TriGlmMemTensor", "TriGlmMemBlock", "TriGlmMemMx", "TriGlmMemPctTensor", "TriGlmMemPctBlock", "TriGlmMemPctMx", "TriGlmMemBf", "TriGlmStepBf", "TriProbeRowsLo", "TriProbeRowsHi", "TriProbeTimeLo", "TriProbeTimeHi", "TriProbeRefTime", "TriLossGlmMxEarly", "TriLossGlmMxFirstEnd", "TriLossGlmSameEarly", "TriLossGlmBlockRep", "TriGlmLongFpSameEnd", "TriGlmLongFpBlock", "TriGlmLongFpTensor", "TriLossGlmTensorRep", "TriLossGlmSameLast", "TriLossGlmSameEnd", "TriLossGlmMxFirstLast", "TriLossGlmBlockLast", "TriLossGlmBlockEnd", "TriLossGlmTensorLast", "TriLossGlmTensorEnd", "TriGlmLongRepMid", "TriGlmLongRepEnd", "TriGlmLongRepStep", "TriGlmLongFpSame", "TriGlmLongBlockMid", "TriGlmLongBlockEnd", "TriGlmLongBlockStep", "TriGlmLongBlockMem", "TriGlmLongTensorMid", "TriGlmLongTensorEnd", "TriGlmLongTensorStep", "TriGlmLongTensorMem", "TriBoardNodes", "TriBoardMixed", "TriBoardSlopeLo", "TriBoardSlopeHi", "TriBoardRestSlopeLo", "TriBoardRestSlopeHi", "TriBoardStepLo", "TriBoardStepHi", "TriBoardLowOldLo", "TriBoardLowOldHi", "TriBoardLowCtlLo", "TriBoardLowCtlHi", "TriLossGlmOtherLast", "TriLossGlmOtherEnd", "TriGlmLongSeedMid", "TriGlmLongSeedEnd", "TriGlmLongBfMid", "TriGlmLongBfEnd", "TriGlmLongMxMid", "TriGlmLongMxEnd", "TriProbeSlowRatio", "TriProbeSlowTimeLo", "TriProbeSlowTimeHi", "TriProbeFixTimeLo", "TriProbeFixTimeHi", "TriLoopRatioLo", "TriLoopRatioHi", "TriProbeLowMs", "TriProbeHighMs", "TriDropRows", "TriDropR", "TriDropRMain", "TriDropRMainRows", "TriAllNodes", "TriAllStepLo", "TriAllStepHi", "TriAllLowLo", "TriAllLowHi", "TriAllWattLo", "TriAllWattHi", "TriAllTflopLo", "TriAllTflopHi", "TriAllNvlsLo", "TriAllNvlsHi", "TriMeterIdleLo", "TriMeterIdleHi", "TriMeterPeakLo", "TriMeterPeakHi", "TriMeterOddIdle", "TriMeterOddPeak", "TriMeterOddPct", "TriMeterOddIdlePct", "TriReturnStep", "TriReturnWattLo", "TriReturnWattHi", "TriOtherHealthNodes", "TriOtherHealthPending", "TriGlmTwentyEff", "TriGlmTwentyTok", "TriGlmTwentyStep", "TriGlmTwentyMem", "TriGlmTwentyMfu", "TriGlmTwentyAtoa", "TriGlmTwentyAtoaPct", "TriFixPctLo", "TriFixPctHi", "TriGlmPpStepPct", "TriGlmPpTwoFiveTok", "TriGlmPpTwoFiveStep", "TriGlmPpTwoFiveMem", "TriGlmPpTwoFiveMfu", "TriGlmPpTwoFiveAtoa", "TriGlmPpTwoFiveAtoaPct", "TriGlmEpTwoSixRatio", "TriGlmEpAtoaShare", "TriGlmPpRatio", "TriGlmEpEightTok", "TriGlmEpEightStep", "TriGlmEpEightMem", "TriGlmEpEightMfu", "TriGlmEpEightAtoa", "TriGlmEpEightAtoaPct", "TriGlmEpSixteenTok", "TriGlmEpSixteenStep", "TriGlmEpSixteenMem", "TriGlmEpSixteenMfu", "TriGlmEpSixteenAtoa", "TriGlmEpSixteenAtoaPct", "TriGlmPpTwoSixTok", "TriGlmPpTwoSixStep", "TriGlmPpTwoSixMem", "TriGlmPpTwoSixMfu", "TriGlmPpTwoSixAtoa", "TriGlmPpTwoSixAtoaPct", "TriBackNodes", "TriBackPctLo", "TriBackPctHi", "TriBackSlowLo", "TriBackSlowHi", "TriNetInsideTop", "TriInGather", "TriInScatter", "TriInReduce", "TriCrossGather", "TriCrossScatter", "TriCrossReduce", "TriInAlltoall", "TriInThirteenLo", "TriInThirteenHi", "TriGlmThirtyTok", "TriGlmThirtyStep", "TriGlmThirtyMem", "TriGlmThirtyMfu", "TriGlmThirtyAtoa", "TriGlmThirtyAtoaPct", "TriGlmThirtyEff", "TriGlmThirtySixTok", "TriGlmThirtySixStep", "TriGlmThirtySixMem", "TriGlmThirtySixMfu", "TriGlmThirtySixAtoa", "TriGlmThirtySixAtoaPct", "TriGlmThirtySixEff", "TriGlmEpRatioTop", "TriClearedNodes", "TriClearedPowerLo", "TriClearedPowerHi", "TriClearedTFlopsLo", "TriClearedTFlopsHi", "TriClearedCtlLo", "TriClearedCtlHi", "TriGlmFortyTok", "TriGlmFortyStep", "TriGlmFortyMem", "TriGlmFortyMfu", "TriGlmFortyAtoa", "TriGlmFortyAtoaPct", "TriGlmFortyEff", "TriCkGBigWSync", "TriCkGBigWSyncGBs", "TriCkGBigWBlocked", "TriCkGBigWTotal", "TriCkGBigWLoadLo", "TriCkGBigWLoadHi", "TriCkGBigWLoadPeakLo", "TriCkGBigWLoadPeakHi", "TriCkGBigWLoadGBsLo", "TriCkGBigWLoadGBsHi", "TriCkGBigFSync", "TriCkGBigFSyncGBs", "TriCkGBigFBlocked", "TriCkGBigFTotal", "TriCkGBigFLoadLo", "TriCkGBigFLoadHi", "TriCkGBigFLoadPeakLo", "TriCkGBigFLoadPeakHi", "TriCkGBigFLoadGBsLo", "TriCkGBigFLoadGBsHi", "TriCkGBigFSyncPeak", "TriCkGBigFAsyncPeak", "TriEnergyFpQ", "TriPowerFpQBf", "TriPowerFpQMx", "TriEnergyFpG", "TriPowerFpGBf", "TriPowerFpGMx", "TriGlmFpTopNodes", "TriGlmMemBfTop", "TriGlmFpTensorTop", "TriGlmFpTensorTopMean", "TriGlmMemTensorTop", "TriGlmMemPctTensorTop", "TriGlmFpBlockTop", "TriGlmFpBlockTopMean", "TriGlmMemBlockTop", "TriGlmMemPctBlockTop", "TriGlmFpMxTop", "TriGlmFpMxTopMean", "TriGlmMemMxTop", "TriGlmMemPctMxTop", "TriGlmLongStepBf", "TriGlmLongStepMx", "TriGlmLongFp", "TriGlmLongMemBf", "TriStartQNodesLo", "TriStartQNodesHi", "TriStartQTopNodes", "TriStartQTopLayout", "TriStartGNodesLo", "TriStartGNodesHi", "TriStartGTopNodes", "TriStartGTopLayout", "TriStartSpreadMax", "TriLossGlmSameMax", "TriLossGlmThirdFirstLast", "TriLossGlmThirdFirstEnd", "TriLossGlmThirdFirstEarly", "TriLossGlmThirdRepLast", "TriLossGlmThirdRepEnd", "TriLossGlmThirdRepEarly", "TriLossGlmMxThirdLast", "TriLossGlmMxThirdEnd", "TriGlmLongThirdMid", "TriGlmLongThirdEnd", "TriGlmLongThirdStep",
          # Checkpoints
          *[f"TriCkQ{n}{k}{e}" for n in ("One", "Thirteen", "Top", "Big") for k in ("Sync", "Load", "Async", "AsyncTotal", "PinTotal", "WSave", "WLoad") for e in ("Lo", "Hi")],
          "TriCkQBigSyncGBs", "TriCkQPinBlockedLo", "TriCkQPinBlockedHi", "TriCkQSizeFull", "TriCkQSizeW", "TriCkGWSyncLo", "TriCkGWSyncHi", "TriCkGWAsyncBlockedLo", "TriCkGWAsyncBlockedHi", "TriCkGWAsyncTotalLo", "TriCkGWAsyncTotalHi",
          "TriCkGWLoadLo", "TriCkGWLoadHi", "TriCkGWSize", "TriCkGWLoadPeak", "TriCkGFSize", "TriCkGFSync", "TriCkGFSyncGBs", "TriCkGFSyncPeak", "TriCkGFBlocked", "TriCkGFTotal", "TriCkGFAsyncPeak",
          # Loss
          "TriLossSeedOneLo", "TriLossSeedOneHi", "TriLossSeedOneEnd", "TriLossSeedTwoLo", "TriLossSeedTwoHi", "TriLossSeedTwoEnd", "TriLossSeedThreeLo", "TriLossSeedThreeHi", "TriLossSeedThreeEnd",
          "TriLossSameMax", "TriLossSameMin", "TriLossOtherMax", "TriLossOtherMxMax", "TriLossGlmLast", "TriLossGlmEnd",
          # Layouts
          "TriLevTokens", "TriLevTwoLarge", "TriLevFourLarge", "TriLevPpTwo", "TriLevPpFour", "TriLevHsdp", "TriLevHsdpLarge", "TriLevReduce", "TriLevFull", "TriLevFullH", "TriLevFortyEight", "TriLevPpBest",
          "TriMemFsdp", "TriMemHsdp", "TriMemReduce", "TriMemFull", "TriTokFortyEight", "TriTokFsdp", "TriTokReduce", "TriTokTpFour", "TriTokPpBest", "TriLayoutTopGap", "TriPaddingPct", "TriLayoutRepeatMedian", "TriLayoutRepeatMax", "TriLayoutTopRepeat", "TriLevReduceTopNodes", "TriLevReduceTop", "TriTokReduceTop", "TriTokFsdpTop", "TriMemReduceTop", "TriMemFsdpTop",
          # Machine
          "TriCapNodes", "TriCapTFlops", "TriCapControlLo", "TriCapControlHi", "TriCapPowerLo", "TriCapPowerHi", "TriCapControlPowerLo",
          "TriCapControlPowerHi", "TriCapFactor", "TriCpuBeforeMed", "TriCpuBeforeLo", "TriCpuBeforeHi", "TriCpuAfterMed", "TriCpuAfterLo",
          "TriCpuAfterHi", "TriCpuOddNode", "TriCpuNodesBefore", "TriCpuNodesAfter", "TriThrottlePct", "TriThrottleStep", "TriThrottleOddStep",
          "TriSlowBehindLo", "TriSlowBehindHi", "TriSlowThirteenWith", "TriSlowThirteenWithout", "TriSlowThirteenPct", "TriKernelMaxChange",
          "TriKernelReruns", "TriKernelRepeatLo", "TriKernelRepeatHi",
          # E1, R
          "TriInterferenceLo", "TriInterferenceHi", "TriPlacementPct", "TriPlacementGroup", "TriPlacementRandom", "TriBesidePPSlow", "TriBesidePPFast", "TriRefRuns", "TriRefSpreadPct", "TriRefAfter",
          # D
          "TriDcgmHi", "TriDcgmLo", "TriDcgmFailed", "TriNodesCheckedAfter", "TriIntra", "TriPairBus", "TriLoopback", "TriPairBusLo", "TriPairBusLoHi", "TriPairWithinLo",
          "TriPairWithinHi", "TriPairAcrossLo", "TriPairAcrossHi", "TriPairsMeasured", "TriPairAllGpus", "TriNetTop", "TriGatherLo", "TriGatherHi", "TriAllReduceLo",
          "TriAllReduceHi", "TriAlltoallTwo"]
FAILED_SHOWN = 8  # configurations named in the note; the rest are in the ledger


def failed_note(t) -> str:
    """Text for the draft box and the ledger: measured failures whose configuration has no successful run."""
    names = [f"{r.experiment} {WORKLOAD_LABEL.get(r.model, r.model)} {r.layout}, {int(r.nodes)} nodes"
             for r in failed_runs(t["runs"]).itertuples() if r.experiment != "F1"]
    f = t["failed_ops"]
    if not f.empty:
        names += [f"F1 {WORKLOAD_LABEL.get(r.model, r.model)} {r.content} {r.op}, {int(r.nodes)} nodes" for r in f.itertuples()]
    if not names:
        return ""
    names = sorted(set(names))
    text = "; ".join(names[:FAILED_SHOWN]) + ("; ..." if len(names) > FAILED_SHOWN else "")
    return (rf"{{\color{{red!70!black}}\textbf{{{len(names)} configurations failed, with no successful run}} "
            rf"(no point, or a red cross, in their figures; synthetic data does not stand in): {text}.}}")


def over_memory_note(t) -> str:
    """A synthetic peak above a B200's memory: the placeholder says what a measured run could not reach."""
    c = checkpoints(t["checkpoints"], t["runs"])
    if c.empty:
        return ""
    over = c[c.synthetic.astype(bool) & (c.peak_allocated_gib > GPU_MEMORY_GIB)]
    if over.empty:
        return ""
    return (rf"The synthetic peaks of {over.peak_allocated_gib.min():.0f}--{over.peak_allocated_gib.max():.0f}\,GiB per GPU are above "
            rf"180\,GB ({GPU_MEMORY_GIB:.0f}\,GiB): such a run would not fit, it would fail with an out-of-memory error.")


SYN_DEF = r"\providecommand{\TriSyn}[1]{{\color{red!70!black}#1\textsuperscript{syn}}}"


def _value(v, fmt, synthetic) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "??"  # never "nan" in the text
    text = format(v, fmt) if not isinstance(v, str) else v
    return rf"\TriSyn{{{text}}}" if synthetic else text


def _span(values) -> tuple[float, float]:
    """Lowest and highest of cells such as 980-986, 667,668 or 2.9."""
    cells = [float(x) for v in values for x in str(v).replace(",", "-").split("-") if x]
    return min(cells), max(cells)


def _drop_numbers(t: dict[str, pd.DataFrame], out: dict) -> None:
    """The hardware agent's 1-node tables: what the low-clock share means, and the load test on all nodes."""
    probe = t.get("probe_share", pd.DataFrame())
    if len(probe):
        r = probe.groupby("run").agg(node=("node", "first"), rows=("probe_pct_rows", "mean"), time=("probe_pct_time", "mean"),
                                     low=("loop_low_us", "mean"), high=("loop_high_us", "mean"), low_s=("probe_dt_low_ms", "mean"), high_s=("probe_dt_high_ms", "mean"))
        slow = r[r.rows > 40]
        main = r[(r.rows <= 40) & (r.node != "trig0114")]
        ref = main[main.node == "trig0151"]
        fixed = main[main.node.isin(SLOW_NODES)]
        out["TriProbeRowsLo"], out["TriProbeRowsHi"] = (main.rows.min(), ".0f", False), (main.rows.max(), ".0f", False)
        out["TriProbeTimeLo"], out["TriProbeTimeHi"] = (main.time.min(), ".0f", False), (main.time.max(), ".0f", False)
        out["TriProbeRefTime"] = (ref.time.mean(), ".0f", False)
        out["TriProbeSlowRatio"] = (slow.time.mean() / ref.time.mean(), ".1f", False)
        out["TriProbeSlowTimeLo"], out["TriProbeSlowTimeHi"] = (slow.time.min(), ".0f", False), (slow.time.max(), ".0f", False)
        out["TriProbeFixTimeLo"], out["TriProbeFixTimeHi"] = (fixed.time.min(), ".0f", False), (fixed.time.max(), ".0f", False)
        ratio = main.low / main.high
        out["TriLoopRatioLo"], out["TriLoopRatioHi"] = (ratio.min(), ".1f", False), (ratio.max(), ".1f", False)
        out["TriProbeLowMs"], out["TriProbeHighMs"] = (main.low_s.median(), ".0f", False), (main.high_s.median(), ".0f", False)
    d = t.get("drop_runs", pd.DataFrame())
    if len(d):
        latest = d.sort_values("start").groupby(["node", "psu"]).tail(1)
        with_drops = latest[latest.node != "trig0114"]
        main = with_drops[with_drops.pct_low < 40]
        out["TriDropRows"] = (len(with_drops), "d", False)
        out["TriDropR"] = (with_drops.pct_low.corr(with_drops.step_s), ".2f", False)
        out["TriDropRMain"] = (main.pct_low.corr(main.step_s), ".2f", False)
        out["TriDropRMainRows"] = (len(main), "d", False)
    a = t.get("all_nodes", pd.DataFrame())
    if len(a):
        main = a[~a.node.isin(["trig0114", "trig0135", "trig0151"])]
        main = main[main.job == main.q_run]  # trig0113's second run is the side-by-side repeat
        odd, ctl = a[a.node == "trig0114"], a[a.node == "trig0135"]
        out["TriAllNodes"] = (main.node.nunique(), "d", False)
        out["TriAllStepLo"], out["TriAllStepHi"] = (main.step_s.min(), ".2f", False), (main.step_s.max(), ".2f", False)
        for name, col, fmt in (("Low", "probe_pct_low", ".0f"), ("Watt", "load_w", ",.0f"), ("Tflop", "load_tflop", ",.0f"), ("Nvls", "nccl_on", ".0f")):
            lo, hi = _span(main[col])
            out[f"TriAll{name}Lo"], out[f"TriAll{name}Hi"] = (lo, fmt, False), (hi, fmt, False)
        out["TriMeterIdleLo"], out["TriMeterIdleHi"] = (main.meter_idle.min(), ".1f", False), (main.meter_idle.max(), ".1f", False)
        out["TriMeterPeakLo"], out["TriMeterPeakHi"] = (main.meter_peak.min(), ".1f", False), (main.meter_peak.max(), ".1f", False)
        out["TriMeterOddIdle"], out["TriMeterOddPeak"] = (odd.meter_idle.mean(), ".1f", False), (odd.meter_peak.mean(), ".1f", False)
        out["TriMeterOddPct"] = (100 * (1 - odd.meter_peak.mean() / main.meter_peak.median()), ".0f", False)
        out["TriMeterOddIdlePct"] = (100 * (1 - odd.meter_idle.mean() / main.meter_idle.median()), ".0f", False)
        if len(ctl):
            out["TriReturnStep"] = (ctl.step_s.iloc[0], ".2f", False)
            lo, hi = _span(ctl.load_w)
            out["TriReturnWattLo"], out["TriReturnWattHi"] = (lo, ",.0f", False), (hi, ",.0f", False)
    b = t.get("board_parts", pd.DataFrame())
    if len(a) and len(b):
        old = set(b[b.gpus_000 == 8].node)
        out["TriBoardNodes"] = (len(old), "d", False)
        out["TriBoardMixed"] = (int(b.gpus_000.between(1, 7).sum()), "d", False)
        rest = b[(b.gpus_000 == 0) & (b.node != "trig0114")]
        o = b[b.node.isin(old)]
        out["TriBoardSlopeLo"], out["TriBoardSlopeHi"] = (o.slope_train.min(), ".2f", False), (o.slope_train.max(), ".2f", False)
        out["TriBoardRestSlopeLo"], out["TriBoardRestSlopeHi"] = (rest.slope_train.min(), ".2f", False), (rest.slope_train.max(), ".2f", False)
        # The nodes with 3 rounds of the test: 4 of the old revision against 4 others run with them
        three = a[a.groupby("node").q_run.transform("nunique") == 3].sort_values(["node", "q_run"])
        three = three.assign(round=three.groupby("node").cumcount(), is_old=three.node.isin(old))
        if three.is_old.any() and (~three.is_old).any():
            m = three.groupby(["round", "is_old"]).step_s.mean().unstack()
            gap = 100 * (m[True] / m[False] - 1)
            out["TriBoardStepLo"], out["TriBoardStepHi"] = (gap.min(), ".1f", False), (gap.max(), ".1f", False)
            for name, g in (("Old", three[three.is_old]), ("Ctl", three[~three.is_old])):
                lo, hi = _span(g.probe_pct_low)
                out[f"TriBoardLow{name}Lo"], out[f"TriBoardLow{name}Hi"] = (lo, ".0f", False), (hi, ".0f", False)
    h = t.get("other_health", pd.DataFrame())
    if len(h):
        out["TriOtherHealthNodes"] = (int((h.label == "current").sum()), "d", False)
        out["TriOtherHealthPending"] = (int((h.label != "current").sum()), "d", False)


def _machine_numbers(t) -> dict[str, tuple]:
    """The machine block: from the health checks, the hardware reruns and hw/post-pstate's readings. All measured."""
    out = {}
    h, runs, steps = t["node_health"], t["runs"], t["steps"]
    c = h[h.metric == "cpu_mhz_full_power"] if len(h) else h
    for period, prefix in (("before", "TriCpuBefore"), ("after", "TriCpuAfter")):
        d = c[c.period == period] if len(c) else c
        d = d[d.host != "trig0114"].groupby("host").value.median() if len(d) else d
        if len(d):
            out[f"{prefix}Med"] = (d.median(), ",.0f", False)
            out[f"{prefix}Lo"] = (d.min(), ",.0f", False)
            out[f"{prefix}Hi"] = (d.max(), ",.0f", False)
            out[f"TriCpuNodes{period.title()}"] = (len(d), "d", False)
    if len(c) and (c.host == "trig0114").any():
        out["TriCpuOddNode"] = (c[c.host == "trig0114"].value.max(), ",.0f", False)
    load = t.get("cap_load", pd.DataFrame())
    if len(load):
        capped, control = load[load.limit_w == 200], load[(load.node == "trig0151")]
        out["TriCapTFlops"] = (capped.tflop_s.median(), ",.0f", False)
        out["TriCapControlLo"] = (control.tflop_s.min(), ",.0f", False)
        out["TriCapControlHi"] = (control.tflop_s.max(), ",.0f", False)
        out["TriCapPowerLo"] = (capped.power_w_median.min(), ",.0f", False)
        out["TriCapPowerHi"] = (capped.power_w_median.max(), ",.0f", False)
        out["TriCapControlPowerLo"] = (control.power_w_median.min(), ",.0f", False)
        out["TriCapControlPowerHi"] = (control.power_w_median.max(), ",.0f", False)
        out["TriCapFactor"] = (control.tflop_s.median() / capped.tflop_s.median(), ".1f", False)
        out["TriCapNodes"] = (capped.node.nunique(), "d", False)
    cleared = t.get("cap_cleared", pd.DataFrame())
    if len(cleared):
        d = cleared[~cleared.node.isin(["trig0151", "trig0146"])]  # trig0151 is the normal node run beside them; trig0146 was never capped
        out["TriClearedNodes"] = (d.node.nunique(), "d", False)
        out["TriClearedPowerLo"], out["TriClearedPowerHi"] = (d.power_w_median.min(), ",.0f", False), (d.power_w_median.max(), ",.0f", False)
        out["TriClearedTFlopsLo"], out["TriClearedTFlopsHi"] = (d.tflop_s.min(), ",.0f", False), (d.tflop_s.max(), ",.0f", False)
        ctl = cleared[cleared.node == "trig0151"]
        if len(ctl):
            out["TriClearedCtlLo"], out["TriClearedCtlHi"] = (ctl.tflop_s.min(), ",.0f", False), (ctl.tflop_s.max(), ",.0f", False)
    _drop_numbers(t, out)
    if len(steps) and len(runs) and "hosts" in runs:
        probe = runs[runs.campaign == "hw-probe"]
        mean = steps[steps.run_id.isin(probe.run_id)].groupby("run_id").step_s.mean()
        p = probe.assign(mean_s=probe.run_id.map(mean), round=probe.run_name.str.extract(r"(h\d)-")[0])
        if len(p):
            odd = p[p.hosts == "trig0114"].mean_s.mean()
            control = p[p.hosts == "trig0151"]
            out["TriThrottleStep"] = (control.mean_s.mean(), ".2f", False)
            out["TriThrottleOddStep"] = (odd, ".2f", False)
            out["TriThrottlePct"] = (100 * (1 - odd / control.mean_s.mean()), ".1f", False)
            # Side by side with trig0151, once the nodes capped or out until Nov 6 had returned.
            h6 = p[p.run_name.str.startswith("hw-h6-")]
            ctl6 = h6[h6.hosts == "trig0151"]
            if len(ctl6):
                pct = 100 * (h6.set_index("hosts").mean_s / ctl6.mean_s.iloc[0] - 1)
                back, held = pct[~pct.index.isin(SLOW_NODES | {"trig0151"})], pct[pct.index.isin(SLOW_NODES)]
                out["TriBackNodes"] = (len(back), "d", False)
                out["TriBackPctLo"], out["TriBackPctHi"] = (back.min(), "+.1f", False), (back.max(), "+.1f", False)
                out["TriBackSlowLo"], out["TriBackSlowHi"] = (held.min(), "+.1f", False), (held.max(), "+.1f", False)
            # The same test, twice, after the administrators' change on the slow nodes.
            fix = p[p.run_name.str.startswith("hw-slowfix-")]
            if len(fix):
                fix = fix.assign(rep=fix.run_name.str.extract(r"slowfix-([a-z])-")[0])
                ctl = fix[fix.hosts == "trig0151"].set_index("rep").mean_s
                pct = 100 * (fix[fix.hosts != "trig0151"].mean_s / fix[fix.hosts != "trig0151"].rep.map(ctl) - 1)
                out["TriFixPctLo"], out["TriFixPctHi"] = (pct.min(), "+.1f", False), (pct.max(), "+.1f", False)
            slow = p[p.hosts.isin(SLOW_NODES) & p["round"].notna() & ~p.run_name.str.startswith("hw-h6-")].merge(control[["round", "mean_s"]], on="round", suffixes=("", "_control"))
            behind = 100 * (slow.mean_s / slow.mean_s_control - 1)
            out["TriSlowBehindLo"] = (behind.min(), ".1f", False)
            out["TriSlowBehindHi"] = (behind.max(), ".1f", False)
        e = runs[runs.experiment == "E1"]
        if len(e):
            m = e.assign(mean_s=e.run_id.map(steps[steps.run_id.isin(e.run_id)].groupby("run_id").step_s.mean()))
            has = m.slow_nodes.notna() & (m.slow_nodes != "")
            w, wo = m[has].mean_s.mean(), m[~has].mean_s.mean()
            out["TriSlowThirteenWith"] = (w, ".2f", False)
            out["TriSlowThirteenWithout"] = (wo, ".2f", False)
            out["TriSlowThirteenPct"] = (100 * (w / wo - 1), ".1f", False)
    k = t.get("kernel_change", pd.DataFrame())
    if len(k):
        d = k.dropna(subset=["step_change_pct"])
        change = pd.to_numeric(d.step_change_pct, errors="coerce").dropna()
        repeat = pd.to_numeric(d.pre_change_repeat_spread_pct, errors="coerce").dropna()
        out["TriKernelMaxChange"] = (change.abs().max(), ".2f", False)
        out["TriKernelReruns"] = (len(change), "d", False)
        out["TriKernelRepeatLo"] = (repeat.min(), ".2f", False)
        out["TriKernelRepeatHi"] = (repeat.max(), ".2f", False)
    return out


def _health_numbers(t) -> dict[str, tuple]:
    out = {}
    h = t["node_health"]
    if h.empty or "period" not in h:
        return out
    b = h[h.period.isin(["before", "after"])]
    d = b[b.metric == "dcgm_tflops"]
    if len(d):
        out["TriDcgmHi"] = (d.value.max(), ",.0f", False)
        out["TriDcgmLo"] = (d.value.min(), ",.0f", False)
    tests = b[b.metric.str.startswith("dcgm_") & (b.metric != "dcgm_tflops")]
    if len(tests):
        out["TriDcgmFailed"] = (int((tests.value == 0).sum()), "d", False)
    out["TriNodesChecked"] = (h[h.period == "before"].host.nunique(), "d", False)
    out["TriNodesCheckedAfter"] = (h[h.period == "after"].host.nunique(), "d", False)
    for metric, name, fmt in (("intra_busbw_gbs", "TriIntra", ".0f"), ("pair_busbw_gbs", "TriPairBus", ".0f"), ("loopback_busbw_gbs", "TriLoopback", ".1f")):
        d = b[b.metric == metric]
        if len(d):
            out[name] = (d.value.median(), fmt, False)
    pair = b[(b.metric == "pair_busbw_gbs") & (b.host == "trig0122")]
    if len(pair):
        out["TriPairBusLo"] = (pair.value.min(), ".0f", False)
        out["TriPairBusLoHi"] = (pair.value.max(), ".0f", False)
    return out


def _network_numbers(t) -> dict[str, tuple]:
    out = {}
    p, c = t["pairs"], t["collectives"]
    if len(p) and not p.synthetic.any():
        d = p[p.job_id.isin(NET_D2_JOBS) & (p.status == "ok")]
        if len(d):
            same = d.a.map(bandwidth_group) == d.b.map(bandwidth_group)
            out["TriPairWithinLo"], out["TriPairWithinHi"] = (d[same].bw_gbs.min(), ".1f", False), (d[same].bw_gbs.max(), ".1f", False)
            out["TriPairAcrossLo"], out["TriPairAcrossHi"] = (d[~same].bw_gbs.min(), ".1f", False), (d[~same].bw_gbs.max(), ".1f", False)
            out["TriPairsMeasured"] = (len(d), "d", False)
        a = p[p.job_id == NET_D2_ALL_GPUS_JOB]
        if len(a):
            out["TriPairAllGpus"] = (a.bw_gbs.median(), ".1f", False)
    if len(c) and not c.synthetic.any():
        d = c[(c.job_id == NET_D3_JOB) & (c.experiment == "D3") & (c.size_bytes == 2**30) & (c.status == "ok") & (c.nodes >= 2)]
        if len(d):
            med = d.groupby(["collective", "nodes"]).busbw_gbs.median()
            out["TriNetTop"] = (int(d.nodes.max()), "d", False)
            gather = med[med.index.get_level_values(0).isin(["all_gather", "reduce_scatter"])]
            out["TriGatherLo"], out["TriGatherHi"] = (gather.min(), ".0f", False), (gather.max(), ".0f", False)
            reduce = med["all_reduce"]
            reduce = reduce[reduce.index >= 4]
            out["TriAllReduceLo"], out["TriAllReduceHi"] = (reduce.min(), ".0f", False), (reduce.max(), ".0f", False)
            out["TriAlltoallTwo"] = (med["alltoall"][2], ".1f", False)
            out["TriAlltoallFull"] = (med["alltoall"][d.nodes.max()], ".1f", False)
            i = c[(c.job_id == NET_D3_INSIDE_JOB) & (c.experiment == "D3") & (c.size_bytes == 2**30) & (c.status == "ok")]
            if len(i):
                inside = i.groupby(["collective", "nodes"]).busbw_gbs.median()
                top, big = int(i.nodes.max()), int(d.nodes.max())
                out["TriNetInsideTop"] = (top, "d", False)
                for coll, tag in (("all_gather", "Gather"), ("reduce_scatter", "Scatter"), ("all_reduce", "Reduce")):
                    out[f"TriIn{tag}"], out[f"TriCross{tag}"] = (inside[coll][top], ".0f", False), (med[coll][big], ".0f", False)
                out["TriInAlltoall"] = (inside["alltoall"][top], ".1f", False)
                gain = [100 * (inside[k][13] / med[k][13] - 1) for k in ("all_gather", "reduce_scatter", "all_reduce")]
                out["TriInThirteenLo"], out["TriInThirteenHi"] = (min(gain), ".1f", False), (max(gain), ".1f", False)
    return out


def kernel_change_table(t) -> str:
    """Every rerun with a step time: before and after the kernel change, and how much repeats moved before it."""
    k = t.get("kernel_change", pd.DataFrame())
    if k.empty:
        return ""
    d = k.dropna(subset=["step_change_pct"])
    lines = [r"\begin{tabular}{lllrl}", r"\toprule", r"Rerun (nodes as before) & Step before (s) & After (s) & Change (\%) & Repeats before (\%) \\", r"\midrule"]
    for r in d.itertuples():
        name = str(r.rerun).replace("_", r"\_").replace("hw ", "1 node, ").replace("tri-aq-", "").replace(" nodes", " nodes").replace("L1", "(13 nodes)")
        lines.append(f"{name} & {str(r.step_s_before).replace('/', ', ')} & {str(r.step_s_after).replace('/', ', ')} & {float(r.step_change_pct):+.2f} & {r.pre_change_repeat_spread_pct} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _layout_numbers(runs: pd.DataFrame, steps: pd.DataFrame) -> dict[str, tuple]:
    """Qwen layouts at 13 nodes: levers, memory, repeats."""
    out = {}
    r = _completed(runs, "A-Q")
    if r.empty or (r.synthetic.any()):
        return out
    # The BF16 reduction at the largest count with both: the one lever repeated at scale.
    f = r[(r.layout == "fsdp") & (r.tokens_per_gpu == 24576) & (r.ac == "selective") & (r.precision == "bf16") & (r.nodes > 13)]
    m = f.groupby(["nodes", "grad_reduce"]).agg(v=("tok_s_gpu", "median"), mem=("mem_alloc_gib", "max")).unstack()
    both = m.v.dropna().index if len(m) and "bf16" in m.v else []
    if len(both):
        n = both.max()
        out["TriLevReduceTopNodes"] = (int(n), "d", False)
        out["TriLevReduceTop"] = (100 * (m.v.bf16[n] / m.v.fp32[n] - 1), "+.1f", False)
        out["TriTokReduceTop"], out["TriTokFsdpTop"] = (m.v.bf16[n], ",.0f", False), (m.v.fp32[n], ",.0f", False)
        out["TriMemReduceTop"], out["TriMemFsdpTop"] = (m.mem.bf16[n], ".0f", False), (m.mem.fp32[n], ".0f", False)
    r = r[(r.nodes == 13) | ((r.nodes == 12) & (r.layout == "hsdp16"))]
    m = r.groupby(["layout", "tokens_per_gpu", "grad_reduce", "ac"]).agg(v=("tok_s_gpu", "median"), mem=("mem_alloc_gib", "max"), share=("valid_token_share", "mean"), step=("step_s", "median"))
    get = lambda layout="fsdp", tokens=12288, reduce="fp32", ac="selective": m.loc[(layout, tokens, reduce, ac)] if (layout, tokens, reduce, ac) in m.index else None  # noqa: E731
    names = {"TriLevTokens": (get(tokens=24576), get()), "TriLevTwoLarge": (get("tp2-fsdp"), get()), "TriLevFourLarge": (get("tp4-fsdp"), get()), "TriLevPpTwo": (get("pp2-fsdp"), get()),
             "TriLevPpFour": (get("pp4-fsdp"), get()), "TriLevHsdp": (get("hsdp8"), get()), "TriLevHsdpLarge": (get("hsdp8", 24576), get(tokens=24576)), "TriLevReduce": (get(tokens=24576, reduce="bf16"), get(tokens=24576)),
             "TriLevFull": (get(tokens=24576, ac="full"), get(tokens=24576)), "TriLevFullH": (get("hsdp8", 24576, ac="full"), get("hsdp8", 24576)), "TriLevFortyEight": (get(tokens=49152, ac="full"), get(tokens=24576, ac="full")),
             "TriLevPpBest": (get("pp2-fsdp", 49152), get(tokens=24576))}
    for name, (a, b) in names.items():
        if a is not None and b is not None:
            out[name] = (100 * (a.v / b.v - 1), "+.0f", False)
    for name, a, col, fmt in (("TriMemFsdp", get(tokens=24576), "mem", ".0f"), ("TriMemHsdp", get("hsdp8", 24576), "mem", ".0f"), ("TriMemReduce", get(tokens=24576, reduce="bf16"), "mem", ".0f"),
                              ("TriMemFull", get(tokens=24576, ac="full"), "mem", ".0f"), ("TriTokFortyEight", get(tokens=49152, ac="full"), "v", ",.0f"), ("TriTokFsdp", get(tokens=24576), "v", ",.0f"),
                              ("TriTokReduce", get(tokens=24576, reduce="bf16"), "v", ",.0f"), ("TriTokTpFour", get("tp4-fsdp"), "v", ",.0f"), ("TriTokPpBest", get("pp2-fsdp", 49152), "v", ",.0f")):
        if a is not None:
            out[name] = (a[col], fmt, False)
    top = m.v.sort_values(ascending=False).iloc[:4]
    out["TriLayoutTopGap"] = (100 * (top.iloc[0] / top.iloc[-1] - 1), ".1f", False)
    padded = m[m.share < 0.99]
    if len(padded):
        out["TriPaddingPct"] = (100 * (1 - padded.share.mean()), ".0f", False)
    if len(steps):
        means = steps[steps.run_id.isin(r.run_id)].groupby("run_id").step_s.mean()
        x = r.assign(mean_s=r.run_id.map(means))
        x = x[~x.run_name.str.contains("|".join(NOISE_ASIDE))]  # stalled twice: out of anything about noise
        spread = x.groupby(["layout", "tokens_per_gpu", "grad_reduce", "ac"]).mean_s.agg(lambda v: 100 * (v.max() - v.min()) / v.mean() if len(v) > 1 else float("nan")).dropna()
        out["TriLayoutRepeatMedian"] = (spread.median(), ".1f", False)
        out["TriLayoutRepeatMax"] = (spread.max(), ".1f", False)
        out["TriLayoutTopRepeat"] = (spread.reindex(top.index).max(), ".0f", False)
    return out


def _loss_numbers(t) -> dict[str, tuple]:
    out = {}
    g = loss_gaps(t["runs"], t["loss"])
    if g.empty or g.synthetic.any():
        return out
    q, m = g[g.model == QWEN], g[g.model == GLM]
    r = q[q.kind == "recipe"]
    for seed, name in ((1234, "One"), (4321, "Two"), (5678, "Three")):
        d = r[r.seed_a == seed]
        if len(d):
            out[f"TriLossSeed{name}Lo"], out[f"TriLossSeed{name}Hi"] = (d.last100.min(), "+.2f", False), (d.last100.max(), "+.2f", False)
            out[f"TriLossSeed{name}End"] = (d.step500.mean(), "+.2f", False)
    same = q[q.kind == "same seed"]
    if len(same):
        out["TriLossSameMax"] = (same.last100.abs().max(), ".2f", False)
        out["TriLossSameMin"] = (same.last100.abs().min(), ".3f", False)
    other = q[(q.kind == "other seed") & (q.recipe == "bf16")]
    if len(other):
        out["TriLossOtherMax"] = (other.last100.abs().max(), ".2f", False)
    other = q[(q.kind == "other seed") & (q.recipe != "bf16")]
    if len(other):
        out["TriLossOtherMxMax"] = (other.last100.abs().max(), ".2f", False)
    d = m[(m.kind == "recipe") & (m.recipe == "mxfp8")]
    if len(d):
        out["TriLossGlmLast"] = (d.last100.iloc[0], "+.2f", False)
        out["TriLossGlmEnd"] = (d.step500.iloc[0], "+.2f", False)
    d = m[(m.kind == "other seed") & (m.recipe == "bf16")]
    if len(d):
        out["TriLossGlmOtherLast"] = (d.last100.iloc[0], "+.2f", False)
        out["TriLossGlmOtherEnd"] = (d.step500.iloc[0], "+.2f", False)
    mx = m[(m.kind == "recipe") & (m.recipe == "mxfp8")]
    same = m[(m.kind == "same seed") & (m.recipe == "bf16")]
    if len(same):
        out["TriLossGlmSameLast"] = (same.last100.iloc[0], "+.2f", False)
        out["TriLossGlmSameEnd"] = (same.step500.iloc[0], "+.2f", False)
        if len(mx) and mx.same_nodes.iloc[0]:  # the recipe row now uses the BF16 repeat on MXFP8's nodes; this is MXFP8 against the first BF16, on other nodes
            out["TriLossGlmMxFirstLast"] = (mx.last100.iloc[0] + same.last100.iloc[0], "+.2f", False)
            out["TriLossGlmMxFirstEnd"] = (mx.step500.iloc[0] + same.step500.iloc[0], "+.2f", False)
            out["TriLossGlmMxEarly"] = (mx.early.iloc[0], "+.2f", False)
        out["TriLossGlmSameEarly"] = (same.early.iloc[0], "+.2f", False)
        out["TriLossGlmSameMax"] = (same.last100.abs().max(), ".2f", False)
        first_id, rep_id = same.run_b.iloc[0], same.run_a.iloc[0]
        later = same[same.run_a != rep_id]  # a third BF16 run at the seed: against the first, and against the repeat
        if later.run_a.nunique() == 1:
            for b_id, name in ((first_id, "First"), (rep_id, "Rep")):
                d = later[later.run_b == b_id]
                if len(d):
                    out[f"TriLossGlmThird{name}Last"] = (d.last100.iloc[0], "+.2f", False)
                    out[f"TriLossGlmThird{name}End"] = (d.step500.iloc[0], "+.2f", False)
                    out[f"TriLossGlmThird{name}Early"] = (d.early.iloc[0], "+.2f", False)
            d = mx[mx.run_b == later.run_a.iloc[0]]
            if len(d):
                out["TriLossGlmMxThirdLast"] = (d.last100.iloc[0], "+.2f", False)
                out["TriLossGlmMxThirdEnd"] = (d.step500.iloc[0], "+.2f", False)
    for rec, name in (("fp8-blockwise", "Block"), ("fp8-tensorwise", "Tensor")):
        d = m[(m.kind == "recipe") & (m.recipe == rec)]
        if len(d):
            out[f"TriLossGlm{name}Last"] = (d.last100.iloc[0], "+.2f", False)
            out[f"TriLossGlm{name}End"] = (d.step500.iloc[0], "+.2f", False)
            if len(same):  # against the BF16 repeat, on other nodes
                out[f"TriLossGlm{name}Rep"] = (d.last100.iloc[0] - same.last100.iloc[0], "+.2f", False)
    return out


def _speed_numbers(runs: pd.DataFrame, steps: pd.DataFrame) -> dict[str, tuple]:
    """C-speed medians and memory, the full-recompute comparison (tagged p3-acfull), and GLM-5 over 500 steps (C-loss)."""
    out = {}
    if runs.empty or runs.synthetic.any():
        return out
    cs = _completed(runs, "C-speed")
    q, g = cs[cs.model == QWEN], cs[cs.model == GLM]
    if len(q):
        m = q.groupby(["nodes", "precision"]).agg(v=("tok_s_gpu", "median"), mem=("mem_alloc_gib", "max")).unstack()
        one = m.loc[1]
        out["TriQwenFpOne"] = (one.v.mxfp8 / one.v.bf16, ".2f", False)
        top = int(m.v.dropna().index.max())  # the largest node count with both recipes
        out["TriQwenFpTopNodes"] = (top, "d", False)
        out["TriQwenFpTop"] = (m.loc[top].v.mxfp8 / m.loc[top].v.bf16, ".2f", False)
        out["TriQwenMemBf"], out["TriQwenMemMx"] = (one.mem.bf16, ".0f", False), (one.mem.mxfp8, ".0f", False)
        b = q[(q.nodes == 1) & (q.precision == "bf16")].tok_s_gpu
        out["TriQwenFpOneSpread"] = (100 * (b.max() - b.min()) / b.median(), ".1f", False)
        means = q.assign(mean_s=q.run_id.map(steps[steps.run_id.isin(q.run_id)].groupby("run_id").step_s.mean())).groupby(["nodes", "precision"]).mean_s.mean().unstack()
        out["TriQwenFpOneMean"] = (means.bf16[1] / means.mxfp8[1], ".2f", False)
        out["TriQwenFpTopMean"] = (means.bf16[top] / means.mxfp8[top], ".2f", False)
        # C-speed's BF16 uses the torchtitan stacked layout, which MXFP8 needs; the weak-scaling hsdp8 runs, otherwise the same, use rows.
        rows = _completed(runs, "B-Q-weak")
        rows = rows[rows.layout == "hsdp8"].groupby("nodes").tok_s_gpu.median() if len(rows) else rows
        gain = (m.v.bf16 / rows).dropna() if len(rows) else rows
        gain = gain[gain.index > 1] if len(gain) else gain
        if len(gain):
            out["TriStackedGain"] = (gain.median(), ".2f", False)
    full = runs[(runs.experiment == "p3-acfull") & (runs.status == "completed")]
    if len(full):
        m = full.groupby("precision").agg(v=("tok_s_gpu", "median"), mem=("mem_alloc_gib", "max"))
        out["TriQwenFpFull"] = (m.v.mxfp8 / m.v.bf16, ".2f", False)
        out["TriQwenFullMem"] = (m.mem.bf16, ".1f", False)
    if len(g):
        m = g[g.nodes == 13].set_index("precision")
        for rec, name in (("fp8-tensorwise", "Tensor"), ("fp8-blockwise", "Block"), ("mxfp8", "Mx")):
            if rec in m.index:
                out[f"TriGlmFp{name}"] = (m.tok_s_gpu[rec] / m.tok_s_gpu.bf16, ".2f", False)
                out[f"TriGlmMem{name}"] = (m.mem_alloc_gib[rec], ".0f", False)
                out[f"TriGlmMemPct{name}"] = (100 * (m.mem_alloc_gib[rec] / m.mem_alloc_gib.bf16 - 1), ".0f", False)
        out["TriGlmMemBf"] = (m.mem_alloc_gib.bf16, ".0f", False)
        out["TriGlmStepBf"] = (m.step_s.bf16, ".1f", False)
        top = int(g.nodes.max())
        if top > 13:
            b = g[g.nodes == top].set_index("precision")
            # Means without the first measured step, slow at this size; the medians are not affected.
            b = b.assign(mean_s=b.run_id.map(steps[steps.run_id.isin(b.run_id)].sort_values("step").groupby("run_id").step_s.apply(lambda s: s.iloc[1:].mean())))
            out["TriGlmFpTopNodes"] = (top, "d", False)
            out["TriGlmMemBfTop"] = (b.mem_alloc_gib.bf16, ".0f", False)
            for rec, name in (("fp8-tensorwise", "Tensor"), ("fp8-blockwise", "Block"), ("mxfp8", "Mx")):
                if rec in b.index:
                    out[f"TriGlmFp{name}Top"] = (b.tok_s_gpu[rec] / b.tok_s_gpu.bf16, ".2f", False)
                    out[f"TriGlmFp{name}TopMean"] = (b.mean_s.bf16 / b.mean_s[rec], ".2f", False)
                    out[f"TriGlmMem{name}Top"] = (b.mem_alloc_gib[rec], ".0f", False)
                    out[f"TriGlmMemPct{name}Top"] = (100 * (b.mem_alloc_gib[rec] / b.mem_alloc_gib.bf16 - 1), ".0f", False)
    cl = _completed(runs, "C-loss")
    bf = cl[(cl.model == GLM) & (cl.precision == "bf16")].sort_values("started") if len(cl) else cl
    bf = bf[bf.seed != bf.seed.iloc[0]] if len(bf) else bf  # BF16 at another seed: other nodes, another day, the same step pattern or not
    if len(bf):
        x = steps[steps.run_id == bf.run_id.iloc[0]]
        out["TriGlmLongSeedMid"] = (x[x.step.between(26, 300)].step_s.median(), ".1f", False)
        out["TriGlmLongSeedEnd"] = (x[x.step > 475].step_s.median(), ".1f", False)
    g = cl[cl.model == GLM].sort_values("started") if len(cl) else cl
    mx = g[g.precision == "mxfp8"] if len(g) else g
    rep = g[(g.precision == "bf16") & (g.hosts == mx.hosts.iloc[0])] if len(mx) else mx  # the BF16 repeat on MXFP8's nodes: the same-node speed comparison
    if len(rep):
        x = steps[steps.run_id == rep.run_id.iloc[0]]
        out["TriGlmLongRepMid"] = (x[x.step.between(26, 300)].step_s.median(), ".1f", False)
        out["TriGlmLongRepEnd"] = (x[x.step > 475].step_s.median(), ".1f", False)
        out["TriGlmLongRepStep"] = (rep.step_s.iloc[0], ".1f", False)
        out["TriGlmLongFpSame"] = (mx.tok_s_gpu.iloc[0] / rep.tok_s_gpu.iloc[0], ".2f", False)
        y = steps[steps.run_id == mx.run_id.iloc[0]]
        out["TriGlmLongFpSameEnd"] = (x[x.step > 475].step_s.median() / y[y.step > 475].step_s.median(), ".2f", False)
    if len(rep) > 1:  # the third BF16 run, on the same nodes as the repeat
        x = steps[steps.run_id == rep.run_id.iloc[1]]
        out["TriGlmLongThirdMid"] = (x[x.step.between(26, 300)].step_s.median(), ".1f", False)
        out["TriGlmLongThirdEnd"] = (x[x.step > 475].step_s.median(), ".1f", False)
        out["TriGlmLongThirdStep"] = (rep.step_s.iloc[1], ".1f", False)
    # The first run of each recipe: later repetitions and seeds run on other nodes, and answer another question.
    cl = cl[cl.model == GLM].sort_values("started").drop_duplicates("precision").set_index("precision") if len(cl) else cl
    if len(cl) and {"bf16", "mxfp8"} <= set(cl.index):
        for rec, name in (("bf16", "Bf"), ("mxfp8", "Mx")):
            x = steps[steps.run_id == cl.run_id[rec]]
            out[f"TriGlmLong{name}Mid"] = (x[x.step.between(26, 300)].step_s.median(), ".1f", False)
            out[f"TriGlmLong{name}End"] = (x[x.step > 475].step_s.median(), ".1f", False)
        out["TriGlmLongStepBf"], out["TriGlmLongStepMx"] = (cl.step_s.bf16, ".1f", False), (cl.step_s.mxfp8, ".1f", False)
        out["TriGlmLongFp"] = (cl.tok_s_gpu.mxfp8 / cl.tok_s_gpu.bf16, ".2f", False)
        out["TriGlmLongMemBf"] = (cl.mem_alloc_gib.bf16, ".0f", False)
    for rec, name in (("fp8-blockwise", "Block"), ("fp8-tensorwise", "Tensor")):
        if len(cl) and rec in cl.index:
            x = steps[steps.run_id == cl.run_id[rec]]
            out[f"TriGlmLong{name}Mid"] = (x[x.step.between(26, 300)].step_s.median(), ".1f", False)
            out[f"TriGlmLong{name}End"] = (x[x.step > 475].step_s.median(), ".1f", False)
            out[f"TriGlmLong{name}Step"] = (cl.step_s[rec], ".1f", False)
            out[f"TriGlmLongFp{name}"] = (cl.tok_s_gpu[rec] / cl.tok_s_gpu.bf16, ".2f", False)
            out[f"TriGlmLong{name}Mem"] = (cl.mem_alloc_gib[rec], ".0f", False)
    return out


def _checkpoint_numbers(t) -> dict[str, tuple]:
    out = {}
    c = checkpoints(t["checkpoints"], t["runs"])
    if c.empty or c.synthetic.any():
        return out

    def rng(name, d, col, fmt=".0f"):
        if len(d):
            out[f"{name}Lo"], out[f"{name}Hi"] = (d[col].min(), fmt, False), (d[col].max(), fmt, False)

    q, g = c[c.model == QWEN], c[(c.model == GLM) & (c.nodes == 13)]
    for n, tag in ((1, "One"), (13, "Thirteen"), (25, "Top"), (39, "Big")):
        f = q[(q.nodes == n) & (q.content == "full")]
        rng(f"TriCkQ{tag}Sync", f[(f.op == "save") & (f["mode"] == "sync")], "total_s")
        rng(f"TriCkQ{tag}Load", f[f.op == "load"], "total_s")
        rng(f"TriCkQ{tag}Async", f[(f.op == "save") & (f["mode"] == "async")], "blocked_s", ".1f")
        rng(f"TriCkQ{tag}AsyncTotal", f[(f.op == "save") & (f["mode"] == "async")], "total_s")
        rng(f"TriCkQ{tag}PinTotal", f[(f.op == "save") & (f["mode"] == "async_pinned")], "total_s")
        w = q[(q.nodes == n) & (q.content == "weights")]
        rng(f"TriCkQ{tag}WSave", w[w.op == "save"], "total_s", ".1f")
        rng(f"TriCkQ{tag}WLoad", w[w.op == "load"], "total_s", ".1f")
    big = q[(q.nodes == 39) & (q.content == "full") & (q.op == "save") & (q["mode"] == "sync")]
    if len(big):
        out["TriCkQBigSyncGBs"] = (big.gb_s.median(), ".0f", False)
    p = q[(q["mode"] == "async_pinned") & (q.repetition >= 2)]
    rng("TriCkQPinBlocked", p, "blocked_s", ".1f")
    size = q[q.content == "full"].bytes.median() / 1e12
    out["TriCkQSizeFull"] = (size, ".2f", False)
    out["TriCkQSizeW"] = (q[q.content == "weights"].bytes.median() / 1e9, ".0f", False)
    s = g[g.content == "weights"]
    rng("TriCkGWSync", s[(s.op == "save") & (s["mode"] == "sync")], "total_s")
    rng("TriCkGWAsyncBlocked", s[(s.op == "save") & (s["mode"] == "async")], "blocked_s")
    rng("TriCkGWAsyncTotal", s[(s.op == "save") & (s["mode"] == "async")], "total_s")
    rng("TriCkGWLoad", s[s.op == "load"], "total_s")
    if len(s):
        out["TriCkGWSize"] = (s.bytes.median() / 1e12, ".2f", False)
        out["TriCkGWLoadPeak"] = (s[s.op == "load"].peak_allocated_gib.max(), ".0f", False)
    f = g[g.content == "full"]
    if len(f):
        a, y = f[(f.op == "save") & (f["mode"] == "async")].iloc[0], f[(f.op == "save") & (f["mode"] == "sync")].iloc[0]
        out["TriCkGFSize"] = (f.bytes.median() / 1e12, ".2f", False)
        out["TriCkGFSync"], out["TriCkGFSyncGBs"], out["TriCkGFSyncPeak"] = (y.total_s, ".0f", False), (y.gb_s, ".0f", False), (y.peak_allocated_gib, ".0f", False)
        out["TriCkGFBlocked"], out["TriCkGFTotal"], out["TriCkGFAsyncPeak"] = (a.blocked_s, ".0f", False), (a.total_s, ".0f", False), (a.peak_allocated_gib, ".0f", False)
    big = c[(c.model == GLM) & (c.nodes == 39)]  # one run of each save, two loads
    if len(big):
        w, f = big[big.content == "weights"], big[big.content == "full"]
        for name, d in (("W", w), ("F", f)):
            sync, asy, load = d[(d.op == "save") & (d["mode"] == "sync")], d[(d.op == "save") & (d["mode"] == "async")], d[d.op == "load"]
            if len(sync):
                out[f"TriCkGBig{name}Sync"], out[f"TriCkGBig{name}SyncGBs"] = (sync.total_s.iloc[0], ".0f", False), (sync.gb_s.iloc[0], ".0f", False)
            if len(asy):
                out[f"TriCkGBig{name}Blocked"], out[f"TriCkGBig{name}Total"] = (asy.blocked_s.iloc[0], ".0f", False), (asy.total_s.iloc[0], ".0f", False)
            rng(f"TriCkGBig{name}Load", load, "total_s")
            rng(f"TriCkGBig{name}LoadPeak", load, "peak_allocated_gib")
            rng(f"TriCkGBig{name}LoadGBs", load, "gb_s")
        if len(f[f.op == "save"]):
            s = f[f.op == "save"].set_index("mode").peak_allocated_gib
            out["TriCkGBigFSyncPeak"], out["TriCkGBigFAsyncPeak"] = (s.sync, ".0f", False), (s["async"], ".0f", False)
    return out


def _scaling_numbers(runs: pd.DataFrame) -> dict[str, tuple]:
    out = {}
    if runs.empty or runs.synthetic.any():
        return out
    w = _completed(runs, "B-Q-weak")
    w = w[(w.slow_nodes.fillna("") == "")]
    for layout, tag in (("fsdp", "F"), ("hsdp8", "H")):
        m = w[w.layout == layout].groupby("nodes").agg(v=("tok_s_gpu", "median"), s=("step_s", "median"))
        if m.empty:
            continue
        top = int(m.index.max())
        out[f"TriWeak{tag}Top"] = (top, "d", False)
        out[f"TriWeak{tag}One"] = (m.v[1], ",.0f", False)
        out[f"TriWeak{tag}TopTok"] = (m.v[top], ",.0f", False)
        out[f"TriWeak{tag}Step"] = (m.s[top], ".2f", False)
        out[f"TriWeak{tag}StepOne"] = (m.s[1], ".2f", False)
        out[f"TriWeak{tag}EffOne"] = (100 * m.v[top] / m.v[1], ".0f", False)
        if 2 in m.index:
            out[f"TriWeak{tag}EffTwo"] = (100 * m.v[top] / m.v[2], ".0f", False)
        if top > 28 and 26 in m.index:  # 26 nodes: one bandwidth group; above 28: both
            out[f"TriWeak{tag}Cross"] = (100 * (m.s[top] / m.s[26] - 1), "+.1f", False)
    if len(w):
        out["TriWeakTop"] = (int(w[w.layout.isin(["fsdp", "hsdp8"])].nodes.max()), "d", False)
    slow = _completed(runs, "B-Q-weak")
    slow = slow[slow.slow_nodes.fillna("") != ""] if len(slow) else slow
    clean = w[w.layout == "fsdp"].groupby("nodes").agg(v=("tok_s_gpu", "median"), s=("step_s", "median"))
    if len(slow) and len(clean):
        x = slow.iloc[0]
        top = clean.index.max()
        out["TriSlowRunNodes"] = (int(x.nodes), "d", False)
        out["TriSlowRunTok"] = (x.tok_s_gpu, ",.0f", False)
        out["TriSlowRunStep"] = (x.step_s, ".2f", False)
        out["TriSlowRunPct"] = (100 * (x.step_s / clean.s[top] - 1), "+.1f", False)
        out["TriSlowRunEff"] = (100 * x.tok_s_gpu / clean.v[1], ".0f", False)
    s = _completed(runs, "B-Q-strong")
    s = s[s.layout == "fsdp"] if len(s) else s
    if len(s):
        m = s.groupby("nodes").tok_s_gpu.median()
        out["TriStrongOne"] = (m.iloc[0], ",.0f", False)
        out["TriStrongOneStep"] = (s[s.nodes == m.index[0]].step_s.median(), ".1f", False)
        if len(m) > 1:
            n = int(m.index[-1])
            out["TriStrongTopNodes"] = (n, "d", False)
            out["TriStrongTop"] = (m.iloc[-1], ",.0f", False)
            out["TriStrongTopStep"] = (s[s.nodes == n].step_s.median(), ".1f", False)
            out["TriStrongEff"] = (100 * m.iloc[-1] * n / (m.iloc[0] * m.index[0]) / (n / m.index[0]), ".0f", False)
    return out


def _glm_numbers(runs: pd.DataFrame, ep: pd.DataFrame) -> dict[str, tuple]:
    out = {}
    if runs.empty or runs.synthetic.any():
        return out
    ag = _completed(runs, "A-G")
    if ag.empty:
        return out
    ag = ag.assign(seqs=ag.global_batch_tokens / ag.seq_len)
    def one(layout, nodes, seqs, ac="full"):
        d = ag[(ag.layout == layout) & (ag.nodes == nodes) & (ag.seqs == seqs) & (ag.ac == ac)]
        return d if len(d) else None
    for name, layout, nodes, seqs in (("Thirteen", "pp13-ep8", 13, 256), ("Small", "pp13-ep8", 13, 64), ("Six", "pp6-ep16", 12, 256), ("Ten", "pp10-ep8", 10, 200), ("Twenty", "pp10-ep8", 20, 400), ("Thirty", "pp10-ep8", 30, 600), ("Forty", "pp10-ep8", 40, 800), ("ThirtySix", "pp6-ep16", 36, 768),
                                         ("EpEight", "pp13-ep8", 26, 512), ("EpSixteen", "pp13-ep16", 26, 512), ("PpTwoSix", "pp26-ep8", 26, 512), ("PpTwoFive", "pp25-ep8", 25, 512)):
        d = one(layout, nodes, seqs)
        if d is not None:
            out[f"TriGlm{name}Tok"] = (d.tok_s_gpu.median(), ",.0f", False)
            out[f"TriGlm{name}Step"] = (d.step_s.median(), ".1f", False)
            out[f"TriGlm{name}Mem"] = (d.mem_alloc_gib.max(), ".0f", False)
            out[f"TriGlm{name}Mfu"] = (d.mfu_pct.median(), ".0f", False)
            if len(ep):
                a2a = ep[ep.run_id.isin(d.run_id)].a2a_s
                if len(a2a):
                    out[f"TriGlm{name}Atoa"] = (a2a.median(), ".1f", False)
                    out[f"TriGlm{name}AtoaPct"] = (100 * a2a.median() / d.step_s.median(), ".0f", False)
    # 26 nodes, same batch: expert groups in a node or across two at the same 13 stages, and twice the stages.
    e8, e16, p26 = one("pp13-ep8", 26, 512), one("pp13-ep16", 26, 512), one("pp26-ep8", 26, 512)
    if e8 is not None and e16 is not None:
        out["TriGlmEpTwoSixRatio"] = (e8.tok_s_gpu.median() / e16.tok_s_gpu.median(), ".1f", False)
        if len(ep):
            a8, a16 = ep[ep.run_id.isin(e8.run_id)].a2a_s.median(), ep[ep.run_id.isin(e16.run_id)].a2a_s.median()
            out["TriGlmEpAtoaShare"] = (100 * (a16 - a8) / (e16.step_s.median() - e8.step_s.median()), ".0f", False)
    if e8 is not None and p26 is not None:
        out["TriGlmPpRatio"] = (e8.tok_s_gpu.median() / p26.tok_s_gpu.median(), ".2f", False)
    p25 = one("pp25-ep8", 25, 512)
    if p25 is not None and p26 is not None:
        out["TriGlmPpStepPct"] = (100 * (p25.step_s.median() / p26.step_s.median() - 1), ".0f", False)
    six, six36 = one("pp6-ep16", 12, 256), one("pp6-ep16", 36, 768)
    if six is not None and six36 is not None:
        out["TriGlmThirtySixEff"] = (100 * six36.tok_s_gpu.median() / six.tok_s_gpu.median(), ".1f", False)
    top = one("pp13-ep8", 39, 768)
    if top is not None and six36 is not None:
        out["TriGlmEpRatioTop"] = (top.tok_s_gpu.median() / six36.tok_s_gpu.median(), ".1f", False)
    ten, forty = one("pp10-ep8", 10, 200), one("pp10-ep8", 40, 800)
    if ten is not None and forty is not None:
        out["TriGlmFortyEff"] = (100 * forty.tok_s_gpu.median() / ten.tok_s_gpu.median(), ".0f", False)
    twenty = one("pp10-ep8", 20, 400)
    if ten is not None and twenty is not None:
        out["TriGlmTwentyEff"] = (100 * twenty.tok_s_gpu.median() / ten.tok_s_gpu.median(), ".0f", False)
    thirty = one("pp10-ep8", 30, 600)
    if ten is not None and thirty is not None:
        out["TriGlmThirtyEff"] = (100 * thirty.tok_s_gpu.median() / ten.tok_s_gpu.median(), ".0f", False)
    a, b = one("pp13-ep8", 13, 256), one("pp6-ep16", 12, 256)
    if a is not None and b is not None:
        out["TriGlmEpRatio"] = (a.tok_s_gpu.median() / b.tok_s_gpu.median(), ".1f", False)
    c = one("pp13-ep8", 13, 64)
    if a is not None and c is not None:
        out["TriGlmSmallRatio"] = (c.tok_s_gpu.median() / a.tok_s_gpu.median(), ".2f", False)
        out["TriGlmSmallFormula"] = ((1 - 12 / 20) / (1 - 12 / 44), ".2f", False)  # (P-1)/(m+P-1) with 13 stages: 8 and 32 micro-batches
    wk = _completed(runs, "B-G-weak")
    wk = wk[(wk.layout == "pp13-ep8") & (wk.ac == "full") & (wk.precision == "bf16") & (wk.nodes == 26)] if len(wk) else wk
    if len(wk) and a is not None:
        out["TriGlmWeakTwoTok"] = (wk.tok_s_gpu.median(), ",.0f", False)
        out["TriGlmWeakTwoStep"] = (wk.step_s.median(), ".1f", False)
        out["TriGlmWeakTwoMem"] = (wk.mem_alloc_gib.max(), ".0f", False)
        out["TriGlmMemFallPct"] = (100 * (1 - wk.mem_alloc_gib.max() / a.mem_alloc_gib.max()), ".0f", False)
    wk3 = _completed(runs, "B-G-weak")
    wk3 = wk3[(wk3.layout == "pp13-ep8") & (wk3.ac == "full") & (wk3.precision == "bf16") & (wk3.nodes == 39)] if len(wk3) else wk3
    if len(wk3):
        out["TriGlmWeakThreeTok"] = (wk3.tok_s_gpu.median(), ",.0f", False)
        out["TriGlmWeakThreeStep"] = (wk3.step_s.median(), ".1f", False)
        out["TriGlmWeakThreeMem"] = (wk3.mem_alloc_gib.max(), ".0f", False)
        if len(ep):
            a2a = ep[ep.run_id.isin(wk3.run_id)].a2a_s
            if len(a2a):
                out["TriGlmWeakThreeAtoa"] = (a2a.median(), ".1f", False)
    st2 = _completed(runs, "B-G-strong")
    st2 = st2[st2.layout == "pp13-ep8"] if len(st2) else st2
    if len(st2) and {13, 39} <= set(st2.nodes):
        out["TriGlmStrongThreeRatio"] = (st2[st2.nodes == 39].tok_s_gpu.median() / st2[st2.nodes == 13].tok_s_gpu.median(), ".2f", False)
        out["TriGlmStrongThreeFormula"] = ((1 - 12 / 44) / (1 - 12 / 108), ".2f", False)  # 32 and 96 micro-batches, 13 stages
    if len(st2) and {13, 26} <= set(st2.nodes):
        one, two = st2[st2.nodes == 13], st2[st2.nodes == 26]
        out["TriGlmStrongTwoTok"] = (two.tok_s_gpu.median(), ",.0f", False)
        out["TriGlmStrongTwoStep"] = (two.step_s.median(), ".1f", False)
        out["TriGlmStrongTwoRatio"] = (two.tok_s_gpu.median() / one.tok_s_gpu.median(), ".2f", False)
        out["TriGlmStrongTwoFormula"] = ((1 - 12 / 60) / (1 - 12 / 108), ".2f", False)  # 48 and 96 micro-batches, 13 stages
    st = _completed(runs, "B-G-strong")
    st = st[(st.nodes == 13)] if len(st) else st
    if len(st) and a is not None:
        out["TriGlmStrongTok"] = (st.tok_s_gpu.median(), ",.0f", False)
        out["TriGlmStrongStep"] = (st.step_s.median(), ".1f", False)
        out["TriGlmStrongRatio"] = (st.tok_s_gpu.median() / a.tok_s_gpu.median(), ".2f", False)
        out["TriGlmStrongFormula"] = ((1 - 12 / 108) / (1 - 12 / 44), ".2f", False)
    return out


def _operations_numbers(t) -> dict[str, tuple]:
    out = {}
    runs = t["runs"]
    if runs.empty or runs.synthetic.any():
        return out
    done = _completed(runs)
    for model, tag in ((QWEN, "Q"), (GLM, "G")):
        d = done[(done.model == model) & done.experiment.isin(["A-Q", "A-G", "B-Q-weak", "B-G-weak", "B-G-strong", "C-speed"])].dropna(subset=["first_step_total_s"])
        if len(d):
            m = d.first_step_total_s / 60
            out[f"TriStart{tag}Lo"], out[f"TriStart{tag}Hi"], out[f"TriStart{tag}Med"] = (m.min(), ".1f", False), (m.max(), ".1f", False), (m.median(), ".1f", False)
            out[f"TriStart{tag}NodesLo"], out[f"TriStart{tag}NodesHi"] = (int(d.nodes.min()), "d", False), (int(d.nodes.max()), "d", False)
            top = d.loc[d.first_step_total_s.idxmax()]
            out[f"TriStart{tag}TopNodes"] = (int(top.nodes), "d", False)
            out[f"TriStart{tag}TopLayout"] = (top.layout, "s", False)
    d = done[done.experiment.isin(["A-Q", "A-G", "B-Q-weak", "B-G-weak", "B-G-strong", "C-speed"])] if len(done) else done
    if "start_spread_s" in d and d.start_spread_s.notna().any():
        out["TriStartSpreadMax"] = (d.start_spread_s.max(), ".1f", False)
    tele = t.get("node_telemetry", pd.DataFrame())
    big = done[(done.model == QWEN) & (done.layout == "fsdp") & (done.tokens_per_gpu == 24576) & (done.ac == "selective") & (done.grad_reduce == "fp32") & done.run_id.isin(tele.run_id if len(tele) else [])]
    if len(big):
        run = big.sort_values(["nodes", "started"]).iloc[-1]
        n = tele[tele.run_id == run.run_id]
        slow, rest = n[n.host.isin(SLOW_NODES)], n[~n.host.isin(SLOW_NODES | {"trig0114"})]
        out["TriStragNodes"] = (int(run.nodes), "d", False)
        if len(slow):
            out["TriStragLowLo"], out["TriStragLowHi"] = (100 * slow.cpu_low_share.min(), ".0f", False), (100 * slow.cpu_low_share.max(), ".0f", False)
            out["TriStragLowRest"] = (100 * rest.cpu_low_share.mean(), ".0f", False)
            out["TriStragUtilSlow"] = (100 * slow.gpu_util.mean(), ".1f", False)
            out["TriStragUtilRest"] = (100 * rest.gpu_util.mean(), ".1f", False)
    e = with_energy(done, t["run_power"])
    if len(e):
        for model, tag in ((QWEN, "Q"), (GLM, "G")):  # MXFP8 against BF16 at the model's largest C-speed run
            c = e[(e.model == model) & (e.experiment == "C-speed")]
            c = c[c.nodes == c.nodes.max()].set_index("precision") if len(c) else c
            if len(c) and {"bf16", "mxfp8"} <= set(c.index):
                out[f"TriEnergyFp{tag}"] = (c.mtok_kwh.mxfp8 / c.mtok_kwh.bf16, ".2f", False)
                out[f"TriPowerFp{tag}Bf"], out[f"TriPowerFp{tag}Mx"] = (c.power_w_gpu.bf16, ".0f", False), (c.power_w_gpu.mxfp8, ".0f", False)
        q = e[(e.model == QWEN) & (e.nodes == 13) & (e.tokens_per_gpu == 24576) & (e.experiment == "A-Q") & (e.ac == "selective") & (e.grad_reduce == "fp32")]
        for layout, tag in (("fsdp", "Fsdp"), ("hsdp8", "Hsdp"), ("pp2-fsdp", "Pp")):
            d = q[q.layout == layout]
            if len(d):
                out[f"TriEnergy{tag}"] = (d.mtok_kwh.median(), ".0f", False)
                out[f"TriPower{tag}"] = (d.power_w_gpu.median(), ",.0f", False)
        g = e[(e.model == GLM) & (e.nodes == 13) & (e.layout == "pp13-ep8") & (e.ac == "full") & (e.precision == "bf16") & (e.experiment == "A-G") & (e.global_batch_tokens == 256 * 8192)]
        if len(g):
            out["TriEnergyGlm"] = (g.mtok_kwh.median(), ".0f", False)
            out["TriPowerGlm"] = (g.power_w_gpu.median(), ",.0f", False)
    return out


def loss_table(t) -> str:
    g = loss_gaps(t["runs"], t["loss"])
    if g.empty:
        return ""
    lab = {"recipe": "{} minus BF16", "same seed": "{}, two runs, same seed", "other seed": "{}, two seeds"}
    lines = [r"\begin{tabular}{lllrr}", r"\toprule", r"Workload & Comparison & Seeds, nodes & Steps 401--500 & Step 500 \\", r"\midrule"]
    for model in (QWEN, GLM):
        for kind in ("recipe", "same seed", "other seed"):
            for x in g[(g.model == model) & (g.kind == kind)].itertuples():
                seeds = str(x.seed_a) if x.seed_a == x.seed_b else f"{x.seed_a}, {x.seed_b}"
                lines.append(f"{WORKLOAD_LABEL[model]} & {lab[kind].format(x.recipe.upper())} & {seeds}, {'same nodes' if x.same_nodes else 'other nodes'} & "
                             f"{x.last100:+.3f} & {x.step500:+.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def key_numbers(t) -> dict[str, tuple]:
    """Macro name -> (value, format, synthetic); a name missing here is written as "??"."""
    out = {}
    runs = t["runs"]
    h = t["node_health"]
    if not h.empty:
        for metric, name in (("matmul_bf16_tflops", "TriMatmulBF"), ("matmul_fp8_tflops", "TriMatmulFP")):
            d = h[h.metric == metric]
            if len(d):
                out[name] = (d.value.median(), ",.0f", d.synthetic.any())
    out.update(_machine_numbers(t))
    if len(runs) and "in_report" in runs and not runs.synthetic.any():
        r = runs[(runs.status == "completed") & runs.in_report.astype(bool)]
        q = r[r.model == QWEN].nodes
        g = r[r.model == GLM].nodes
        out["TriQwenTop"] = (int(q.max()), "d", False)
        out["TriGlmTop"] = (int(g.max()), "d", False)
        out["TriRunsInReport"] = (len(r), "d", False)
    out.update(_health_numbers(t))
    out.update(_network_numbers(t))
    p = t["pairs"]
    if not p.empty:
        if not p.synthetic.any():
            p = p[p.job_id.isin(NET_D2_JOBS)]
        out["TriPairBw"] = (p.bw_gbs.median(), ".0f", p.synthetic.any())
        out["TriPairBwMin"] = (p.bw_gbs.min(), ".0f", p.synthetic.any())
        out["TriPairLat"] = (p.lat_us.median(), ".1f", p.synthetic.any())
    c = t["collectives"]
    if not c.empty:
        if not c.synthetic.any():
            c = c[c.job_id == NET_D3_JOB]
        d3 = c[(c.experiment == "D3") & (c.size_bytes == 2**30)]
        for coll, name in (("all_reduce", "TriAllReduceFull"), ("alltoall", "TriAlltoallFull")):
            d = d3[(d3.collective == coll) & (d3.nodes == d3.nodes.max())]
            if len(d):
                out.setdefault(name, (d.busbw_gbs.median(), ".0f", d.synthetic.any()))
        d4 = c[c.experiment == "D4"]
        if len(d4):
            out["TriBisection"] = (d4.busbw_gbs.median(), ".0f", d4.synthetic.any())
    e1 = e1_changes(runs, t["steps"])
    if len(e1):
        syn = e1.synthetic.any()
        out["TriInterferencePct"] = (e1.change_pct.mean(), "+.2f", syn)
        out["TriInterferenceLo"] = (e1.change_pct.min(), "+.2f", syn)
        out["TriInterferenceHi"] = (e1.change_pct.max(), "+.2f", syn)
        clean = e1[e1.slow_nodes == ""]
        steps = (clean.alone_s + clean.together_s) / 2
        group, random = steps[~clean.bw_groups.str.contains(r"\+")], steps[clean.bw_groups.str.contains(r"\+")]
        if len(group) and len(random):
            out["TriPlacementGroup"] = (group.mean(), ".3f", syn)
            out["TriPlacementRandom"] = (random.mean(), ".3f", syn)
            out["TriPlacementPct"] = (100 * (random.mean() / group.mean() - 1), "+.1f", syn)
    steps = t["steps"]
    if len(steps) and "job" in runs:
        for job, name in (("1241064", "TriBesidePPSlow"), ("1241200", "TriBesidePPFast")):
            ids = runs[(runs.job == job) | (runs.job == float(job))].run_id
            d = steps[steps.run_id.isin(ids)].step_s
            if len(d):
                out[name] = (d.mean(), ".3f", False)
    rr = _completed(runs, "R")
    if len(rr) > 1:
        out["TriDriftPct"] = (drift(rr)[1], "+.2f", rr.synthetic.any())  # linear trend over the window
        fixed = rr[rr.hosts == REF_NODES] if not rr.synthetic.any() else rr
        out["TriRefRuns"] = (len(fixed), "d", rr.synthetic.any())
        if "pstate" in fixed:
            out["TriRefAfter"] = (int((fixed.pstate == "after").sum()), "d", rr.synthetic.any())
        out["TriRefSpreadPct"] = (100 * (fixed.tok_s_gpu.max() - fixed.tok_s_gpu.min()) / fixed.tok_s_gpu.median(), ".2f", rr.synthetic.any())
    aq = _completed(runs, "A-Q")
    base = qwen_base(runs)
    base = base[base.nodes == 13] if len(base) else base
    if len(base):
        m = base.groupby(["layout", "tokens_per_gpu"]).agg(v=("tok_s_gpu", "median"), mfu=("mfu_pct", "median"),
                                                           syn=("synthetic", "any"))
        (layout, tokens), best = m.v.idxmax(), m.loc[m.v.idxmax()]
        out["TriQwenBestLayout"] = (layout, "", best.syn)
        out["TriQwenBestTokens"] = (f"{int(tokens) // 1024}k", "", best.syn)
        out["TriQwenBestTokS"] = (best.v, ",.0f", best.syn)
        out["TriQwenBestMFU"] = (best.mfu, ".0f", best.syn)
        one = aq[(aq.nodes == 1) & (aq.layout == "fsdp") & (aq.tokens_per_gpu == 24576)
                 & (aq.ac == QWEN_BASE["ac"]) & (aq.grad_reduce == QWEN_BASE["grad_reduce"])]
        f13 = base[(base.layout == "fsdp") & (base.tokens_per_gpu == 24576)]
        if len(one) and len(f13):
            share = 100 * (1 - one.step_s.median() / f13.step_s.median())
            out["TriQwenFsdpCommPct"] = (share, ".0f", one.synthetic.any() or f13.synthetic.any())
    out.update(_layout_numbers(runs, t["steps"]))
    out.update(_loss_numbers(t))
    out.update(_speed_numbers(runs, t["steps"]))
    out.update(_checkpoint_numbers(t))
    out.update(_scaling_numbers(runs))
    out.update(_glm_numbers(runs, t["ep_groups"]))
    out.update(_operations_numbers(t))
    w = _completed(runs, "B-Q-weak")
    if len(w):
        g = w[w.layout != "fsdp"] if (w.layout != "fsdp").any() else w
        med = g.groupby("nodes").tok_s_gpu.median()
        out["TriQwenWeakEff"] = (100 * med.iloc[-1] / med.iloc[0], ".0f", g.synthetic.any())
        multi = med[med.index > 1]  # the first run with cross-node traffic (2 nodes, or 4 with four PP stages)
        if len(multi):
            out["TriQwenWeakEffFromTwo"] = (100 * med.iloc[-1] / multi.iloc[0], ".0f", g.synthetic.any())
            out["TriQwenWeakTwoNodes"] = (int(multi.index[0]), "d", False)
    ag = _completed(runs, "A-G")
    if len(ag):
        per_copy = ag.global_batch_tokens / ag.copies / ag.seq_len
        big = ag[(ag.nodes >= 0.9 * ag.nodes.max()) & (ag.ac == "full") & (per_copy >= GLM_SEQS_PER_COPY / 2)]
        m = big.groupby("layout").agg(v=("tok_s_gpu", "median"), mfu=("mfu_pct", "median"), syn=("synthetic", "any"))
        if len(m):
            best = m.loc[m.v.idxmax()]
            out["TriGlmBestLayout"] = (m.v.idxmax(), "", best.syn)
            out["TriGlmBestTokS"] = (best.v, ",.0f", best.syn)
            out["TriGlmBestMFU"] = (best.mfu, ".0f", best.syn)
    bg = _completed(runs, "B-G-weak")
    if len(bg) > 1:
        bg = bg[(bg.layout == "pp13-ep8") & (bg.ac == "full") & (bg.precision == "bf16")]
        med = bg.groupby("copies").tok_s_gpu.median()
        top = med.index.max() if len(med) else None
        if 1 in med.index and top and top > 1:  # the most copies run so far
            out["TriGlmCopiesEff"] = (100 * med[top] / med[1], ".0f", bg.synthetic.any())
            out["TriGlmCopiesTop"] = (int(top), "d", bg.synthetic.any())
        if 1 in med.index and 2 in med.index:
            out["TriGlmCopiesEffTwo"] = (100 * med[2] / med[1], ".0f", bg.synthetic.any())
        if 2 in med.index and top and top > 2:
            out["TriGlmCopiesEffFromTwo"] = (100 * med[top] / med[2], ".0f", bg.synthetic.any())
    cs = _completed(runs, "C-speed")
    for model, prefix in ((QWEN, "TriQwen"), (GLM, "TriGlm")):
        d = cs[cs.model == model] if len(cs) else cs
        if d.empty:
            continue
        m = d.groupby(["nodes", "precision"]).tok_s_gpu.median().unstack()
        if "bf16" not in m:
            continue
        speed = m.drop(columns="bf16").div(m.bf16, axis=0).dropna(how="all")  # a node count with both recipes
        if speed.empty:
            continue
        recipe = speed.iloc[-1].idxmax()
        out[f"{prefix}FpBestRecipe"] = (recipe, "", d.synthetic.any())
        out[f"{prefix}FpSpeedupSmall"] = (speed[recipe].iloc[0], ".2f", d.synthetic.any())
        out[f"{prefix}FpSpeedupFull"] = (speed[recipe].iloc[-1], ".2f", d.synthetic.any())
    ck = checkpoints(t["checkpoints"], runs)
    if not ck.empty:
        p = ck[(ck.model == GLM) & (ck.content == "full") & (ck.op == "load")].dropna(subset=["peak_allocated_gib"])
        if len(p):
            p = p[p.nodes == p.nodes.min()]
            out["TriCkptGlmLoadPeak"] = (p.peak_allocated_gib.median(), ".0f", p.synthetic.any())
        for model, content, col, mode, name in (
                (QWEN, "full", "total_s", "sync", "TriCkptQwenSave"),
                (GLM, "full", "total_s", "sync", "TriCkptGlmFullSave"),
                (GLM, "full", "blocked_s", "async", "TriCkptGlmFullBlocked")):
            d = ck[(ck.model == model) & (ck.content == content) & (ck.op == "save") & (ck["mode"] == mode)]
            d = d[d.nodes == d.nodes.max()]
            if len(d):
                out[name] = (d[col].median(), ".0f", d.synthetic.any())
        s = ck[(ck.op == "save") & (ck["mode"] == "sync")]
        if len(s):
            out["TriCkptWriteGBs"] = (s.groupby(["model", "content", "nodes"]).gb_s.median().max(), ".0f",
                                      s.synthetic.any())
    for model, name in ((QWEN, "TriStartupQwen"), (GLM, "TriStartupGlm")):
        d = _completed(runs)
        d = d[(d.model == model)].dropna(subset=["first_step_total_s"]) if len(d) else d
        if len(d):
            d = d[d.nodes == d.nodes.max()]
            out[name] = (d.first_step_total_s.median() / 60, ".0f", d.synthetic.any())
    pw = t["run_power"]
    if not pw.empty and len(base):
        e = with_energy(base, pw)
        if len(e):
            out["TriEnergyBest"] = (e.mtok_kwh.max(), ".0f", e.synthetic.any())
    return out


def macros(t) -> str:
    runs = t["runs"]
    real = runs[~runs.synthetic] if len(runs) else runs
    n_syn = int(runs.synthetic.sum()) if len(runs) else 0
    lines = [
        SYN_DEF,
        rf"\newcommand{{\DataGenerated}}{{{datetime.now():%Y-%m-%d %H:%M}}}",
        rf"\newcommand{{\NRunsReal}}{{{len(real)}}}",
        rf"\newcommand{{\NRunsFailed}}{{{int((real.status != 'completed').sum()) if len(real) else 0}}}",
        rf"\newcommand{{\NRunsSynthetic}}{{{n_syn}}}",
        rf"\newcommand{{\HasSynthetic}}{{{'1' if n_syn else '0'}}}",
        rf"\newcommand{{\TriNNodes}}{{{N_NODES}}}",
        rf"\newcommand{{\TriNGpus}}{{{N_NODES * 8}}}",
        rf"\newcommand{{\TriGlmTopNodes}}{{{GLM_TOP_NODES}}}",
        rf"\newcommand{{\TriGlmTopCopies}}{{{GLM_TOP_COPIES}}}",
        rf"\newcommand{{\TriPairsPerRound}}{{{N_NODES // 2}}}",
        rf"\newcommand{{\TriPairRounds}}{{{N_NODES - 1 + N_NODES % 2}}}",
        rf"\newcommand{{\TriPairsAll}}{{{N_NODES * (N_NODES - 1) // 2:,}}}",
    ]
    lines.append(rf"\newcommand{{\TriFailedNote}}{{{failed_note(t)}}}")
    lines.append(rf"\newcommand{{\TriCkptOverMemoryNote}}{{{over_memory_note(t)}}}")
    numbers = key_numbers(t)
    for name in MACROS:  # always defined: "??" until the data exists
        v, fmt, syn = numbers.get(name, ("??", "", False))
        lines.append(rf"\newcommand{{\{name}}}{{{_value(v, fmt, bool(syn))}}}")
    return "\n".join(lines) + "\n"


def qwen_layouts_table(runs: pd.DataFrame) -> str:
    """Qwen at 13 nodes, selective recompute: tokens/s per GPU (MFU %) by layout x tokens per GPU; synthetic in \\TriSyn."""
    r = qwen_base(runs)
    if r.empty:
        return ""
    r = r[(r.nodes == 13) | ((r.nodes == 12) & (r.layout == "hsdp16"))]
    m = r.groupby(["layout", "tokens_per_gpu"]).agg(v=("tok_s_gpu", "median"), mfu=("mfu_pct", "median"), mem=("mem_alloc_gib", "max"),
                                                    syn=("synthetic", "any")).reset_index()
    failed = failed_runs(runs, "A-Q")
    failed = failed[(failed.ac == "selective") & (failed.grad_reduce == "fp32")] if len(failed) else failed
    why = {("tp2-fsdp", 24576): "time limit", ("tp4-fsdp", 24576): "out of memory"}
    cols = sorted(m.tokens_per_gpu.unique())
    rows = []
    for layout in ("fsdp", "hsdp8", "hsdp16", "tp2-fsdp", "tp4-fsdp", "pp2-fsdp", "pp4-fsdp"):
        cells = m[m.layout == layout].set_index("tokens_per_gpu")
        row = [layout + (" (12 nodes)" if layout == "hsdp16" else "")]
        for c in cols:
            if c in cells.index:
                row.append(_value(f"{cells.v[c]:,.0f} ({cells.mfu[c]:.0f}), {cells.mem[c]:.0f}", "", cells.syn[c]))
            elif len(failed) and ((failed.layout == layout) & (failed.tokens_per_gpu == c)).any():
                row.append(why.get((layout, c), "failed"))
            else:
                row.append("")
        rows.append(row)
    df = pd.DataFrame(rows, columns=["layout"] + [f"{int(c) // 1024}k" for c in cols])
    # Not tt_analysis.tables.tabular: it escapes backslashes, and the cells hold \TriSyn{}.
    lines = [rf"\begin{{tabular}}{{l{'r' * len(cols)}}}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(str(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def checkpoint_table(t) -> str:
    """Checkpoint times, one row per workload, content and node count: ranges over repetitions, seconds. Pinned blocked: from the second save on."""
    c = checkpoints(t["checkpoints"], t["runs"])
    if c.empty:
        return ""

    def cell(d, col, fmt):
        if d.empty:
            return ""
        lo, hi = d[col].min(), d[col].max()
        return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}}--{hi:{fmt}}"

    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Workload & Content & Nodes & Sync save & GB/s & Async blocked & Async total & Pinned blocked & Pinned total & Load \\", r"\midrule"]
    failed = t.get("failed_ops", pd.DataFrame())
    for model in (QWEN, GLM):
        for content in ("weights", "full"):
            d = c[(c.model == model) & (c.content == content)]
            for n in sorted(d.nodes.unique()):
                g = d[d.nodes == n]
                save, load = g[g.op == "save"], g[g.op == "load"]
                sync, asy, pin = save[save["mode"] == "sync"], save[save["mode"] == "async"], save[save["mode"] == "async_pinned"]
                fmt = ".1f" if model == QWEN and content == "weights" else ".0f"
                loaded = cell(load, "total_s", fmt)
                if len(failed) and not ((failed.model == model) & (failed.content == content) & (failed.nodes == n)).empty:
                    if ((failed.model == model) & (failed.content == content) & (failed.nodes == n) & (failed.op == "load")).any():
                        loaded = "out of memory"
                lines.append(f"{WORKLOAD_LABEL[model]} & {content} ({d.bytes.median() / 1e12:.2f}\\,TB) & {int(n)} & {cell(sync, 'total_s', fmt)} & {cell(sync, 'gb_s', '.0f')} & {cell(asy, 'blocked_s', '.1f')} & "
                             f"{cell(asy, 'total_s', '.0f')} & {cell(pinned_blocked(pin), 'blocked_s', '.1f')} & {cell(pin, 'total_s', '.0f')} & {loaded} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _esc(v) -> str:
    return str(v).replace("_", r"\_").replace("%", r"\%")


def ledger(runs: pd.DataFrame) -> str:
    """Every attempt, on as many pages as it takes: start, entry, job, nodes, experiment, status, and why it is not in the figures."""
    r = runs[~runs.synthetic.astype(bool)].sort_values("started")
    lines = [r"\scriptsize", r"\begin{longtable}{lllrlll}",
             r"\caption{Run ledger: every attempt on Trillium, including failures. Use: report, in the figures; set aside, left out of them for the reason given; machine, the hardware reruns; test, bring-up and tests of one thing. "
             r"Why: a run kept out in spite of its tags.}\label{tab:ledger}\\",
             r"\toprule", r"Started & Entry & Job & Nodes & Experiment & Status & Use, why \\", r"\midrule", r"\endhead"]
    for x in r.itertuples():
        job = str(x.job).removesuffix(".0") if pd.notna(x.job) else ""
        nodes = f"{int(x.nodes)}" if pd.notna(x.nodes) else ""
        use = "set aside" if isinstance(x.set_aside, str) and x.set_aside else "report" if x.in_report else x.role
        why = f", {x.set_aside}" if isinstance(x.set_aside, str) and x.set_aside else ""
        entry = x.bench if isinstance(x.bench, str) else x.run_name
        lines.append(f"{str(x.started).replace('T', ' ')[5:16]} & {_esc(entry.removeprefix('megatron-glm5-'))} & {job} & {nodes} & {_esc(x.experiment) if pd.notna(x.experiment) else ''} & {x.status} & "
                     + r"\parbox[t]{5cm}{" + _esc(use + why) + r"} \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines) + "\n"


def progress_table(t) -> str:
    """What ran against what's planned, for the appendix."""
    from . import progress  # here, not at the top: progress imports select, and only this table needs it

    s = progress.summary(progress.table(t), t, Path("/nonexistent"))
    lines = [r"\begin{tabular}{lllll}", r"\toprule", r"Group & Done / planned, under 50 nodes & 50+ nodes & Compromise & What's left \\", r"\midrule"]
    for _, x in s.iterrows():
        lines.append(f"{_esc(x['Group'])} & {_esc(x['Done / planned'])} & {x['50+ nodes']} & {x['Compromise']} & " + r"\parbox[t]{6cm}{" + _esc(x["What's left"]) + "} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_all(tables: dict[str, pd.DataFrame], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs = tables["runs"]
    (out / "macros.tex").write_text(macros(tables))
    (out / "qwen_layouts.tex").write_text(qwen_layouts_table(runs))
    (out / "kernel_change.tex").write_text(kernel_change_table(tables))
    (out / "loss_gaps.tex").write_text(loss_table(tables))
    (out / "checkpoints.tex").write_text(checkpoint_table(tables))
    if len(runs):
        if "in_report" in runs:
            (out / "ledger.tex").write_text(ledger(runs))
            (out / "progress.tex").write_text(progress_table(tables))
