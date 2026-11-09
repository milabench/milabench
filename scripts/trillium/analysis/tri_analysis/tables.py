"""LaTeX tables and macros for the report (``report/generated/tables/``).

Every number taken from synthetic rows is wrapped in ``\\TriSyn{...}`` (red, "syn" superscript),
so a placeholder can never read as a measurement in the text.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .machine import (GLM_SEQS_PER_COPY, GLM_TOP_COPIES, GLM_TOP_NODES, GPU_MEMORY_GIB, N_NODES, NET_D2_AGAIN_JOB, NET_D2_ALL_GPUS_JOB, NET_D2_JOBS, NET_D3_INSIDE_JOB, NET_D3_JOB, NET_FULL_JOB)
from . import glm_fp4
from .kimi import LOG_INTERVAL_S
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
          "TriGlmMemTensor", "TriGlmMemBlock", "TriGlmMemMx", "TriGlmMemPctTensor", "TriGlmMemPctBlock", "TriGlmMemPctMx", "TriGlmMemBf", "TriGlmStepBf", "TriProbeRowsLo", "TriProbeRowsHi", "TriProbeTimeLo", "TriProbeTimeHi", "TriProbeRefTime", "TriLossGlmMxEarly", "TriLossGlmMxFirstEnd", "TriLossGlmSameEarly", "TriLossGlmBlockRep", "TriGlmLongFpSameEnd", "TriGlmLongFpBlock", "TriGlmLongFpTensor", "TriLossGlmTensorRep", "TriLossGlmSameLast", "TriLossGlmSameEnd", "TriLossGlmMxFirstLast", "TriLossGlmBlockLast", "TriLossGlmBlockEnd", "TriLossGlmTensorLast", "TriLossGlmTensorEnd", "TriGlmLongRepMid", "TriGlmLongRepEnd", "TriGlmLongRepStep", "TriGlmLongFpSame", "TriGlmLongBlockMid", "TriGlmLongBlockEnd", "TriGlmLongBlockStep", "TriGlmLongBlockMem", "TriGlmLongTensorMid", "TriGlmLongTensorEnd", "TriGlmLongTensorStep", "TriGlmLongTensorMem", "TriBoardNodes", "TriBoardMixed", "TriBoardSlopeLo", "TriBoardSlopeHi", "TriBoardRestSlopeLo", "TriBoardRestSlopeHi", "TriBoardStepLo", "TriBoardStepHi", "TriBoardLowOldLo", "TriBoardLowOldHi", "TriBoardLowCtlLo", "TriBoardLowCtlHi", "TriLossGlmOtherLast", "TriLossGlmOtherEnd", "TriGlmLongSeedMid", "TriGlmLongSeedEnd", "TriGlmLongBfMid", "TriGlmLongBfEnd", "TriGlmLongMxMid", "TriGlmLongMxEnd", "TriProbeSlowRatio", "TriProbeSlowTimeLo", "TriProbeSlowTimeHi", "TriProbeFixTimeLo", "TriProbeFixTimeHi", "TriLoopRatioLo", "TriLoopRatioHi", "TriProbeLowMs", "TriProbeHighMs", "TriDropRows", "TriDropR", "TriDropRMain", "TriDropRMainRows", "TriAllNodes", "TriAllStepLo", "TriAllStepHi", "TriAllLowLo", "TriAllLowHi", "TriAllWattLo", "TriAllWattHi", "TriAllTflopLo", "TriAllTflopHi", "TriAllNvlsLo", "TriAllNvlsHi", "TriMeterIdleLo", "TriMeterIdleHi", "TriMeterPeakLo", "TriMeterPeakHi", "TriMeterOddIdle", "TriMeterOddPeak", "TriMeterOddPct", "TriMeterOddIdlePct", "TriReturnStep", "TriReturnWattLo", "TriReturnWattHi", "TriOtherHealthNodes", "TriOtherHealthPending", "TriGlmTwentyEff", "TriGlmTwentyTok", "TriGlmTwentyStep", "TriGlmTwentyMem", "TriGlmTwentyMfu", "TriGlmTwentyAtoa", "TriGlmTwentyAtoaPct", "TriFixPctLo", "TriFixPctHi", "TriGlmPpStepPct", "TriGlmPpTwoFiveTok", "TriGlmPpTwoFiveStep", "TriGlmPpTwoFiveMem", "TriGlmPpTwoFiveMfu", "TriGlmPpTwoFiveAtoa", "TriGlmPpTwoFiveAtoaPct", "TriGlmEpTwoSixRatio", "TriGlmEpAtoaShare", "TriGlmPpRatio", "TriGlmEpEightTok", "TriGlmEpEightStep", "TriGlmEpEightMem", "TriGlmEpEightMfu", "TriGlmEpEightAtoa", "TriGlmEpEightAtoaPct", "TriGlmEpSixteenTok", "TriGlmEpSixteenStep", "TriGlmEpSixteenMem", "TriGlmEpSixteenMfu", "TriGlmEpSixteenAtoa", "TriGlmEpSixteenAtoaPct", "TriGlmPpTwoSixTok", "TriGlmPpTwoSixStep", "TriGlmPpTwoSixMem", "TriGlmPpTwoSixMfu", "TriGlmPpTwoSixAtoa", "TriGlmPpTwoSixAtoaPct", "TriBackNodes", "TriBackPctLo", "TriBackPctHi", "TriBackSlowLo", "TriBackSlowHi", "TriNetInsideTop", "TriInGather", "TriInScatter", "TriInReduce", "TriCrossGather", "TriCrossScatter", "TriCrossReduce", "TriInAlltoall", "TriInThirteenLo", "TriInThirteenHi", "TriGlmThirtyTok", "TriGlmThirtyStep", "TriGlmThirtyMem", "TriGlmThirtyMfu", "TriGlmThirtyAtoa", "TriGlmThirtyAtoaPct", "TriGlmThirtyEff", "TriGlmThirtySixTok", "TriGlmThirtySixStep", "TriGlmThirtySixMem", "TriGlmThirtySixMfu", "TriGlmThirtySixAtoa", "TriGlmThirtySixAtoaPct", "TriGlmThirtySixEff", "TriGlmEpRatioTop", "TriClearedNodes", "TriClearedPowerLo", "TriClearedPowerHi", "TriClearedTFlopsLo", "TriClearedTFlopsHi", "TriClearedCtlLo", "TriClearedCtlHi", "TriGlmFortyTok", "TriGlmFortyStep", "TriGlmFortyMem", "TriGlmFortyMfu", "TriGlmFortyAtoa", "TriGlmFortyAtoaPct", "TriGlmFortyEff", "TriCkGBigWSync", "TriCkGBigWSyncGBs", "TriCkGBigWBlocked", "TriCkGBigWTotal", "TriCkGBigWLoadLo", "TriCkGBigWLoadHi", "TriCkGBigWLoadPeakLo", "TriCkGBigWLoadPeakHi", "TriCkGBigWLoadGBsLo", "TriCkGBigWLoadGBsHi", "TriCkGBigFSync", "TriCkGBigFSyncGBs", "TriCkGBigFBlocked", "TriCkGBigFTotal", "TriCkGBigFLoadLo", "TriCkGBigFLoadHi", "TriCkGBigFLoadPeakLo", "TriCkGBigFLoadPeakHi", "TriCkGBigFLoadGBsLo", "TriCkGBigFLoadGBsHi", "TriCkGBigFSyncPeak", "TriCkGBigFAsyncPeak", "TriEnergyFpQ", "TriPowerFpQBf", "TriPowerFpQMx", "TriEnergyFpG", "TriPowerFpGBf", "TriPowerFpGMx", "TriGlmFpTopNodes", "TriGlmMemBfTop", "TriGlmFpTensorTop", "TriGlmFpTensorTopMean", "TriGlmMemTensorTop", "TriGlmMemPctTensorTop", "TriGlmFpBlockTop", "TriGlmFpBlockTopMean", "TriGlmMemBlockTop", "TriGlmMemPctBlockTop", "TriGlmFpMxTop", "TriGlmFpMxTopMean", "TriGlmMemMxTop", "TriGlmMemPctMxTop", "TriGlmLongStepBf", "TriGlmLongStepMx", "TriGlmLongFp", "TriGlmLongMemBf", "TriStragLowAllLo", "TriStragLowAllHi", "TriStragUtilAllLo", "TriStragUtilAllHi", "TriStragKwLo", "TriStragKwHi", "TriLongQNodes", "TriLongQSteps", "TriLongQMin", "TriLongQStep", "TriLongQTok", "TriLongQEarly", "TriLongQLate", "TriLongQDriftPct", "TriLongQMax", "TriLongQRefNodes", "TriLongQRefStep", "TriLongQVsRefPct", "TriLongQClockLo", "TriLongQClockHi", "TriLongQClockMed", "TriLongQWattLo", "TriLongQWattHi", "TriLongQWattMed", "TriLongQClockFirst", "TriLongQClockLast", "TriLongQClockDriftPct", "TriLongQWattFirst", "TriLongQWattLast", "TriLongQWattDriftPct", "TriAgainPairs", "TriAgainNodes", "TriAgainWithin", "TriAgainWithinLo", "TriAgainWithinHi", "TriAgainAcross", "TriAgainAcrossLo", "TriAgainAcrossHi", "TriAgainMax", "TriAgainPassPct", "TriAgainCommon", "TriAgainRatioLo", "TriAgainRatioHi", "TriAgainRatioMed", "TriLongQRefSteps", "TriLongQHead", "TriLongGNodes", "TriLongGSteps", "TriLongGMin", "TriLongGStep", "TriLongGTok", "TriLongGEarly", "TriLongGLate", "TriLongGDriftPct", "TriLongGMax", "TriLongGRefNodes", "TriLongGRefStep", "TriLongGVsRefPct", "TriLongGRefSteps", "TriLongGHead", "TriLongGClockLo", "TriLongGClockHi", "TriLongGClockMed", "TriLongGClockFirst", "TriLongGClockLast", "TriLongGClockDriftPct", "TriLongGWattLo", "TriLongGWattHi", "TriLongGWattMed", "TriLongGWattFirst", "TriLongGWattLast", "TriLongGWattDriftPct", "TriFullPairs", "TriFullNodes", "TriFullFast", "TriFullSlow", "TriFullFastLo", "TriFullFastHi", "TriFullSlowLo", "TriFullSlowHi", "TriFullPartnersLo", "TriFullPartnersHi", "TriFullLatLo", "TriFullLatHi", "TriFullWithin", "TriFullAcross", "TriFullSlowPct", "TriFullEightPct", "TriFullEightPairs", "TriFullCommon", "TriFullRatioLo", "TriFullRatioHi", "TriFullRatioMed", "TriBisectionCuts", "TriBisectionMin", "TriBisectionSlowLo", "TriBisectionSlowHi", "TriBisectionPairs", "TriSevenThirteenLo", "TriSevenThirteenHi", "TriSevenGather", "TriSevenScatter", "TriSevenReduce", "TriGlmFortyEightTok", "TriGlmFortyEightStep", "TriGlmFortyEightMem", "TriGlmFortyEightMfu", "TriGlmFortyEightAtoa", "TriGlmFortyEightAtoaPct", "TriGlmFortyEightEff", "TriCkQFiftySyncLo", "TriCkQFiftySyncHi", "TriCkQFiftyLoadLo", "TriCkQFiftyLoadHi", "TriCkQFiftyAsyncLo", "TriCkQFiftyAsyncHi", "TriCkQFiftyAsyncTotalLo", "TriCkQFiftyAsyncTotalHi", "TriCkQFiftyPinTotalLo", "TriCkQFiftyPinTotalHi", "TriCkQFiftyWSaveLo", "TriCkQFiftyWSaveHi", "TriCkQFiftyWLoadLo", "TriCkQFiftyWLoadHi", "TriCkQFiftySyncGBs", "TriWeakFCrossTop", "TriWeakHCrossTop", "TriSlowRunRefNodes", "TriSlowRunVsTopPct", "TriStragBeforeNodes", "TriStragBeforeLowLo", "TriStragBeforeLowHi", "TriStragBeforeLowRest", "TriStragBeforeUtilSlow", "TriStragBeforeUtilRest", "TriGlmFiftyTok", "TriGlmFiftyStep", "TriGlmFiftyMem", "TriGlmFiftyMfu", "TriGlmFiftyAtoa", "TriGlmFiftyAtoaPct", "TriGlmFiftyEff", "TriGlmPpTwoFiveFiftyTok", "TriGlmPpTwoFiveFiftyStep", "TriGlmPpTwoFiveFiftyMem", "TriGlmPpTwoFiveFiftyMfu", "TriGlmPpTwoFiveFiftyAtoa", "TriGlmPpTwoFiveFiftyAtoaPct", "TriGlmPpTwoFiveFiftyEff", "TriStartQNodesLo", "TriStartQNodesHi", "TriStartQTopNodes", "TriStartQTopLayout", "TriStartGNodesLo", "TriStartGNodesHi", "TriStartGTopNodes", "TriStartGTopLayout", "TriStartSpreadMax", "TriLossGlmSameMax", "TriLossGlmThirdFirstLast", "TriLossGlmThirdFirstEnd", "TriLossGlmThirdFirstEarly", "TriLossGlmThirdRepLast", "TriLossGlmThirdRepEnd", "TriLossGlmThirdRepEarly", "TriLossGlmMxThirdLast", "TriLossGlmMxThirdEnd", "TriGlmLongThirdMid", "TriGlmLongThirdEnd", "TriGlmLongThirdStep",
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
          "TriAllReduceHi", "TriAlltoallTwo",
          # New workloads: Kimi K3 inference (_kimi_numbers)
          "TriKimiNodes", "TriKimiGpus", "TriKimiLayout", "TriKimiVllm", "TriKimiMoeBackend", "TriKimiMaxSeqs", "TriKimiMaxLen", "TriKimiChunk",
          "TriKimiKvTokens", "TriKimiKvUsePct", "TriKimiKvFillSeqs", "TriKimiPasses", "TriKimiRequests", "TriKimiFailed",
          "TriKimiOneTok", "TriKimiOneTpot", "TriKimiOneTtft", "TriKimiSixtyFourTok", "TriKimiSixtyFourGpu", "TriKimiSixtyFourTpot", "TriKimiSixtyFourTtft",
          "TriKimiPeakConc", "TriKimiPeakTok", "TriKimiPeakGpu", "TriKimiPeakNode", "TriKimiPeakTpot", "TriKimiTopConc", "TriKimiTopTok", "TriKimiTopTtft", "TriKimiTopPct",
          "TriKimiCtxTopK", "TriKimiCtxOneTtft", "TriKimiCtxTtft", "TriKimiCtxTok", "TriKimiCtxOneTpot", "TriKimiCtxTpot", "TriKimiCtxTpotPct", "TriKimiCtxSixteenTtft",
          "TriKimiPowerOne", "TriKimiJOne", "TriKimiPowerSixtyFour", "TriKimiJSixtyFour", "TriKimiPowerPeak", "TriKimiJPeak", "TriKimiMainGpuW", "TriKimiMainHost",
          "TriKimiGsm", "TriKimiGsmErr", "TriKimiArc", "TriKimiArcErr", "TriKimiGpqa", "TriKimiGpqaErr", "TriKimiMmlu", "TriKimiMmluErr", "TriKimiMmluN",
          "TriKimiCtxSixteenTok", "TriKimiCtxSixteenTokThirtyTwo", "TriKimiCtxSixteenTokSixtyFour", "TriKimiTopMainW", "TriKimiStalls", "TriKimiMmluUnparsed",
          # ... and the 4-node server (_kimi_four_numbers), with its gains over the 2-node one
          "TriKimiFourNodes", "TriKimiFourGpus", "TriKimiFourLayout", "TriKimiFourMaxSeqs", "TriKimiFourMainHost", "TriKimiFourKvTokens", "TriKimiFourKvUsePct",
          "TriKimiFourPasses", "TriKimiFourRequests", "TriKimiFourFailed", "TriKimiFourOneTok", "TriKimiFourOneTpot",
          "TriKimiFourSixtyFourTok", "TriKimiFourSixtyFourPct", "TriKimiFourSixtyFourNoStall", "TriKimiFourSixtyFourNoStallPct",
          "TriKimiFourFiveTwelveTok", "TriKimiFourFiveTwelvePct", "TriKimiFourTopTok", "TriKimiFourTopGpu", "TriKimiFourTopNode", "TriKimiFourTopTpot",
          "TriKimiFourTopPct", "TriKimiFourPowerTop", "TriKimiFourJTop", "TriKimiFourTopMainW",
          "TriKimiFourCtxTtft", "TriKimiFourCtxSixteenTok", "TriKimiFourCtxSixteenTokThirtyTwo", "TriKimiFourCtxSixteenTokSixtyFour", "TriKimiFourCtxSixteenPct",
          "TriKimiFourStalls", "TriKimiFourGsm", "TriKimiFourGsmErr", "TriKimiFourArc", "TriKimiFourArcErr", "TriKimiFourGpqa", "TriKimiFourGpqaErr",
          "TriKimiFourMmlu", "TriKimiFourMmluErr", "TriKimiFourMmluN", "TriKimiFourMmluUnparsed", "TriKimiMmluOnlyTwo", "TriKimiMmluOnlyFour", "TriKimiMmluPairZ",
          # ... and with the experts over 2 nodes (_kimi_xn_numbers): EP16 with all-gather (Ag) and DeepEP (Dp) all-to-all, against TP8 x PP2 (Pp)
          "TriKxnNodes", "TriKxnGpus", "TriKxnDp", "TriKxnTp", "TriKxnEp", "TriKxnLayout", "TriKxnMaxSeqs", "TriKxnPasses", "TriKxnRequests",
          "TriKxnFailed", "TriKxnWarmFailed", "TriKxnNotRun", "TriKxnFasterAg", "TriKxnFasterDp", "TriKxnDpLeadN", "TriKxnDpLeadLo", "TriKxnDpLeadHi",
          "TriKxnTwoFiftySixTokPp", "TriKxnCtxEightSixteenTokPp", "TriKxnGsmN",
          *[f"TriKxn{m}{b}" for b in ("Ag", "Dp") for m in (
              "Weights", "KvGib", "KvTok", "Up", "OneTok", "OneRatio", "OneTpot", "FourTok", "FourRatio", "SixteenTok", "SixteenRatio",
              "SixtyFourTok", "SixtyFourRatio", "SixtyFourTpot", "OneTwentyEightTok", "OneTwentyEightRatio", "TwoFiftySixTok", "TwoFiftySixRatio",
              "CtxTok", "CtxRatio", "CtxEightSixteenTok", "CtxEightSixteenRatio", "CtxThirtyTwoSixteenTok", "CtxThirtyTwoSixteenRatio",
              "Gsm", "GsmErr", "Arc", "ArcErr", "Gpqa", "GpqaErr", "Mmlu", "MmluErr")],
          "TriKxnDpAgFour", "TriKxnDpAgTwoFiftySix",
          # ... and on 4 nodes: prefill and decode on separate nodes against colocated (_kimi_pd_numbers, colocated: Co), and the experts
          # over all 4 (_kimi_xn_four_numbers) against TP8 x PP4 (Pp) and EP16 with all-gather (Ep)
          "TriKpdNodes", "TriKpdGpus", "TriKpdInstNodes", "TriKpdLayout", "TriKpdMaxSeqs", "TriKpdWeights", "TriKpdKvGib", "TriKpdKvTok", "TriKpdUp",
          "TriKpdUpCo", "TriKpdPasses", "TriKpdRequests", "TriKpdFailed", "TriKpdWarmFailed", "TriKpdOneRatio", "TriKpdFourRatio",
          "TriKpdSixteenRatio", "TriKpdSixtyFourRatio", "TriKpdOneTwentyEightRatio", "TriKpdTwoFiftySixRatio", "TriKpdCtxOneSixteenRatio",
          "TriKpdCtxEightSixteenRatio", "TriKpdCtxThirtyTwoSixteenRatio", "TriKpdDecodeOneRatio", "TriKpdDecodeSixteenRatio", "TriKpdOneTok",
          "TriKpdOneTokCo", "TriKpdSixtyFourTok", "TriKpdSixtyFourTokCo", "TriKpdTwoFiftySixTok", "TriKpdTwoFiftySixTokCo", "TriKpdLeadN",
          "TriKpdOneTtft", "TriKpdOneTtftCo", "TriKpdOneTtftGapLo", "TriKpdOneTtftGapHi", "TriKpdOneRatioLo", "TriKpdOneRatioHi",
          "TriKpdCtxOneSixteenTpot", "TriKpdCtxOneSixteenTpotCo", "TriKpdCtxEightSixteenTpot", "TriKpdCtxEightSixteenTpotCo",
          "TriKpdCtxThirtyTwoSixteenTpot", "TriKpdCtxThirtyTwoSixteenTpotCo", "TriKpdCtxThirtyTwoSixteenTtft", "TriKpdCtxThirtyTwoSixteenTtftCo",
          "TriKpdNotRun", "TriKpdGsm", "TriKpdGsmErr", "TriKpdArc", "TriKpdArcErr", "TriKpdGpqa", "TriKpdGpqaErr", "TriKpdMmlu", "TriKpdMmluErr",
          "TriKpdGsmCo", "TriKpdGsmErrCo", "TriKpdArcCo", "TriKpdArcErrCo", "TriKpdGpqaCo", "TriKpdGpqaErrCo", "TriKpdMmluCo", "TriKpdMmluErrCo",
          "TriKpdGpqaN", "TriKpdGpqaRight", "TriKpdGpqaRightCo", "TriKpdGpqaOnlyCo", "TriKpdGpqaOnlyPd", "TriKpdGpqaP",
          "TriKxnFourNodes", "TriKxnFourGpus", "TriKxnFourLayout", "TriKxnFourMaxSeqs", "TriKxnFourHeld", "TriKxnFourWeights", "TriKxnFourKvGib",
          "TriKxnFourKvTok", "TriKxnFourUp", "TriKxnFourPasses", "TriKxnFourRequests", "TriKxnFourFailed", "TriKxnFourWarmFailed",
          "TriKxnFourOneTok", "TriKxnFourFourTok", "TriKxnFourSixteenTok", "TriKxnFourSixtyFourTok", "TriKxnFourOneTwentyEightTok",
          "TriKxnFourTwoFiftySixTok", "TriKxnFourFiveTwelveTok", "TriKxnFourTopTok", "TriKxnFourCtxTok", "TriKxnFourCtxEightSixteenTok",
          "TriKxnFourCtxThirtyTwoSixteenTok", "TriKxnFourOneTpot", "TriKxnFourTopTpot", "TriKxnFourOneRatio", "TriKxnFourFourRatio",
          "TriKxnFourSixteenRatio", "TriKxnFourSixtyFourRatio", "TriKxnFourOneTwentyEightRatio", "TriKxnFourTwoFiftySixRatio",
          "TriKxnFourFiveTwelveRatio", "TriKxnFourTopRatio", "TriKxnFourCtxRatio", "TriKxnFourCtxEightSixteenRatio",
          "TriKxnFourCtxThirtyTwoSixteenRatio", "TriKxnFourLeadN", "TriKxnFourCommon", "TriKxnFourNotRun", "TriKxnFourTwoFiftySixTokPp",
          "TriKxnFourTopTtft", "TriKxnFourTopTtftPp", "TriKxnFourEpOneRatio", "TriKxnFourEpTwoFiftySixRatio", "TriKxnFourEpCtxThirtyTwoSixteenRatio",
          "TriKxnFourEpLeadN", "TriKxnFourEpCommon", "TriKxnFourGsm", "TriKxnFourGsmErr", "TriKxnFourArc", "TriKxnFourArcErr", "TriKxnFourGpqa",
          "TriKxnFourGpqaErr", "TriKxnFourMmlu", "TriKxnFourMmluErr", "TriKxnFourAccZMax",
          # New workloads: Kimi K3 pretraining (_kimi_train_numbers): the full-size run, then the 49-layer one (Small)
          "TriKtRuns", "TriKtFailed", "TriKtLayers", "TriKtNodes", "TriKtGpus", "TriKtPp", "TriKtEp", "TriKtMicro", "TriKtSeq", "TriKtTokens",
          "TriKtParams", "TriKtActive", "TriKtStepOne", "TriKtStepTwo", "TriKtSweepFirst", "TriKtSweepSplit", "TriKtSweepHi", "TriKtSweepLo",
          "TriKtSweepTotal", "TriKtMeasured", "TriKtMeasuredFrom", "TriKtStep", "TriKtStepPninety", "TriKtStepMax", "TriKtStepFirst", "TriKtStepLast",
          "TriKtTok", "TriKtTokLo", "TriKtTokHi", "TriKtMfu", "TriKtMfuSixN", "TriKtFlops", "TriKtFetch", "TriKtLowLo", "TriKtLowHi", "TriKtBubblePct",
          "TriKtMemWorst", "TriKtMemWorstAlloc", "TriKtMemWorstStage", "TriKtMemWorstHost", "TriKtMemLast", "TriKtMemLastAlloc", "TriKtMemZero",
          "TriKtMemZeroAlloc", "TriKtMemEdgeLo", "TriKtMemEdgeHi", "TriKtMemMidLo", "TriKtMemMidHi", "TriKtMemLimit", "TriKtHeadroom",
          "TriKtLossFirst", "TriKtLossLast", "TriKtSmallLayers", "TriKtSmallStep", "TriKtSmallStepLo", "TriKtSmallStepHi", "TriKtSmallTok",
          "TriKtSmallMfu", "TriKtSmallMemLast", "TriKtSmallMemZero", "TriKtSmallStepTwo", "TriKtSmallSweep", "TriKtSmallLossLast",
          # New workloads: GLM-5.3 GRPO with slime (_glm_rl_numbers): setup, the warm-up step 0, then steps 1 and 2 (One, Two)
          "TriGrNodes", "TriGrGpus", "TriGrLayers", "TriGrPp", "TriGrEp", "TriGrDp", "TriGrFirstLayers", "TriGrMidLayers", "TriGrLastLayers",
          "TriGrMeanLayers", "TriGrLastRatio", "TriGrEngines", "TriGrEngineGpus", "TriGrEngineEp", "TriGrPrompts", "TriGrSamplesPer", "TriGrSamples",
          "TriGrMaxResp", "TriGrMemFraction", "TriGrSteps", "TriGrStartup", "TriGrEnginesUp",
          "TriGrWarmRollout", "TriGrWarmTrain", "TriGrWarmStep", "TriGrWarmStepMin", "TriGrWarmTrainTok", "TriGrCompilesAfter",
          "TriGrRolloutOne", "TriGrRolloutTwo", "TriGrRolloutMinOne", "TriGrRolloutMinTwo", "TriGrTrainOne", "TriGrTrainTwo", "TriGrSyncOne",
          "TriGrSyncTwo", "TriGrSyncFinal", "TriGrRestOne", "TriGrRestTwo", "TriGrStepOne", "TriGrStepTwo", "TriGrStepMinOne", "TriGrStepMinTwo",
          "TriGrRolloutPctOne", "TriGrRolloutPctTwo", "TriGrTrainPctOne", "TriGrTrainPctTwo",
          "TriGrTrainTokOne", "TriGrTrainTokTwo", "TriGrTrainTokensOne", "TriGrTrainTokensTwo", "TriGrMicroOne", "TriGrMicroTwo", "TriGrBubbleOne",
          "TriGrBubbleTwo", "TriGrRollTokOne", "TriGrRollTokTwo", "TriGrRespMeanOne", "TriGrRespMeanTwo", "TriGrRespMedOne", "TriGrRespMedTwo",
          "TriGrTruncOne", "TriGrTruncTwo", "TriGrRewardWarm", "TriGrRewardOne", "TriGrRewardTwo", "TriGrLatMedOne", "TriGrLatMedTwo",
          "TriGrLatMaxOne", "TriGrLatMaxTwo", "TriGrDecodeReqOne", "TriGrDecodeReqTwo", "TriGrDecodeFast", "TriGrDecodeSlow", "TriGrSlowFirstOne",
          "TriGrSlowFirstTwo", "TriGrGpuRollout", "TriGrGpuCap", "TriGrHostLastLo", "TriGrHostLastHi", "TriGrHostLastTb", "TriGrHostOtherHi",
          "TriGrHostTotal", "TriGrHostFirst", "TriGrSlowEnginesOne", "TriGrSlowEnginesTwo",
          # GLM-RL-03 (without routing replay) for the comparison, then GLM-RL-04 with it (TriGrRp*), steps 1 and 2
          "TriGrTisClipLow", "TriGrTisClipHigh", "TriGrTisAbsOne", "TriGrTisAbsTwo", "TriGrTisClipOne", "TriGrTisClipTwo", "TriGrPrepOne",
          "TriGrPrepTwo", "TriGrStepMean", "TriGrRolloutSpread", "TriGrHeadHost", "TriGrGradOthersLo", "TriGrGradOthersHi",
          "TriGrRpRolloutOne", "TriGrRpRolloutTwo", "TriGrRpTrainOne", "TriGrRpTrainTwo", "TriGrRpSyncOne", "TriGrRpSyncTwo", "TriGrRpStepOne",
          "TriGrRpStepTwo", "TriGrRpStepDiffOne", "TriGrRpStepDiffTwo", "TriGrRpStepMean", "TriGrRpRolloutSpread", "TriGrRpPrepOne", "TriGrRpPrepTwo",
          "TriGrRpTisAbsOne", "TriGrRpTisAbsTwo", "TriGrRpTisClipOne", "TriGrRpTisClipTwo", "TriGrRpTisRatioOne", "TriGrRpTisRatioTwo",
          "TriGrRpClipRatioOne", "TriGrRpClipRatioTwo", "TriGrRpRewardOne", "TriGrRpRewardTwo", "TriGrRpTruncOne", "TriGrRpTruncTwo", "TriGrRpGradTwo",
          "TriGrRpDecodeReqOne", "TriGrRpDecodeReqTwo", "TriGrRpDecodeDropOne", "TriGrRpDecodeDropTwo", "TriGrRpHostLo", "TriGrRpHostHi",
          "TriGrRpHostNodes", "TriGrRpHostLastHi", "TriGrRpHeadHost", "TriGrRpGpuRollout",
          # GLM-RL-05, disaggregated (TriGrDg*): the layout, step 0, steps 1 and 2, the top-k log-prob retries, memory
          "TriGrDgTrainerNodes", "TriGrDgEngineNodes", "TriGrDgTrainerGpus", "TriGrDgEngineGpus", "TriGrDgEngines", "TriGrDgEp", "TriGrDgDp",
          "TriGrDgStartup", "TriGrDgWarmRollout", "TriGrDgWarmTrain", "TriGrDgWarmStep", "TriGrDgSyncFirst",
          "TriGrDgRolloutOne", "TriGrDgRolloutTwo", "TriGrDgRolloutMinOne", "TriGrDgTrainOne", "TriGrDgTrainTwo", "TriGrDgSyncOne",
          "TriGrDgSyncTwo", "TriGrDgRestOne", "TriGrDgRestTwo", "TriGrDgStepOne", "TriGrDgStepTwo", "TriGrDgStepMinOne", "TriGrDgStepMinTwo",
          "TriGrDgTrainTokOne", "TriGrDgTrainTokTwo", "TriGrDgTrainPerGpu", "TriGrDgMicroOne", "TriGrDgMicroTwo", "TriGrDgBubbleOne",
          "TriGrDgBubbleTwo", "TriGrDgRollTokOne", "TriGrDgRollTokTwo", "TriGrDgDecodeReqOne", "TriGrDgDecodeReqTwo", "TriGrDgTisAbsOne",
          "TriGrDgTisAbsTwo", "TriGrDgTisClipOne", "TriGrDgTisClipTwo", "TriGrDgRewardOne", "TriGrDgRewardTwo", "TriGrDgTruncOne",
          "TriGrDgTruncTwo", "TriGrDgTopkWarm", "TriGrDgTopkOne", "TriGrDgTopkTwo", "TriGrDgTopkAll", "TriGrDgRetryMinOne",
          "TriGrDgLastGroupMinOne", "TriGrDgLatMaxOne", "TriGrDgHostLastLo", "TriGrDgHostLastHi", "TriGrDgHostLastTb", "TriGrDgHostFirstLo",
          "TriGrDgHostFirstHi", "TriGrDgHostMidLo", "TriGrDgHostMidHi", "TriGrDgHostMidNodes", "TriGrDgHostMidAll", "TriGrDgHostEngine",
          "TriGrDgGpuRollout", "TriGrDgGpuTrainer",
          # GLM-RL-06, colocated on B200's default rollout kernels with max running requests capped (TriGrKn*), against GLM-RL-03 (Ref)
          "TriGrKnCap", "TriGrKnCapRank", "TriGrKnRunningMax", "TriGrKnEnginesUp", "TriGrKnStartup", "TriGrKnGraphLo", "TriGrKnGraphHi",
          "TriGrKnGraphRefLo", "TriGrKnGraphRefHi", "TriGrKnRolloutOne", "TriGrKnRolloutTwo", "TriGrKnTrainOne", "TriGrKnTrainTwo",
          "TriGrKnSyncOne", "TriGrKnSyncTwo", "TriGrKnStepOne", "TriGrKnStepTwo", "TriGrKnDecodeReqOne", "TriGrKnDecodeReqTwo",
          "TriGrKnDecodeGainOne", "TriGrKnDecodeGainTwo", "TriGrKnDecodeFast", "TriGrKnDecodeSlow", "TriGrKnLatMaxOne", "TriGrKnLatMaxTwo",
          "TriGrKnSwitchOne", "TriGrKnSwitchTwo", "TriGrKnSwitchRefOne", "TriGrKnSwitchRefTwo", "TriGrKnTisAbsOne", "TriGrKnTisAbsTwo",
          "TriGrKnTisClipOne", "TriGrKnTisClipTwo", "TriGrKnTisRatioOne", "TriGrKnTisRatioTwo", "TriGrKnClipRatioOne", "TriGrKnClipRatioTwo",
          "TriGrKnRewardOne", "TriGrKnRewardTwo", "TriGrKnGpuRollout",
          # New workloads: GLM-5 C-speed with NVFP4 (_glm_fp4_numbers, GLM-FP4-01): setup, then per recipe (Bf, Mx, Nv), ratios, power cap,
          # and the campaign's 13-node run (Camp*) on the same measure
          "TriFpFourNodes", "TriFpFourGpus", "TriFpFourSeqs", "TriFpFourSeqLen", "TriFpFourLayers", "TriFpFourBfLast", "TriFpFourSteps",
          "TriFpFourFrom", "TriFpFourMeasured", "TriFpFourLastHost", "TriFpFourLr",
          "TriFpFourStepBf", "TriFpFourMedBf", "TriFpFourSdBf", "TriFpFourMemBf", "TriFpFourLossBf", "TriFpFourTokBf", "TriFpFourStartBf",
          "TriFpFourEndBf", "TriFpFourCapBf",
          "TriFpFourStepMx", "TriFpFourMedMx", "TriFpFourSdMx", "TriFpFourMemMx", "TriFpFourLossMx", "TriFpFourTokMx", "TriFpFourStartMx",
          "TriFpFourEndMx", "TriFpFourCapMx",
          "TriFpFourStepNv", "TriFpFourMedNv", "TriFpFourSdNv", "TriFpFourMemNv", "TriFpFourLossNv", "TriFpFourTokNv", "TriFpFourStartNv",
          "TriFpFourEndNv", "TriFpFourCapNv",
          "TriFpFourNvBf", "TriFpFourNvMx", "TriFpFourMxBf", "TriFpFourNvBfPct", "TriFpFourMxBfPct", "TriFpFourNvMxDiff", "TriFpFourNvMxDiffMed",
          "TriFpFourMemMxNv", "TriFpFourMemNvBf", "TriFpFourCapOtherBfLo", "TriFpFourCapOtherBfHi", "TriFpFourCapOtherMx", "TriFpFourCapOtherNv",
          "TriFpFourMxBfSpeed", "TriFpFourCampBf", "TriFpFourCampMx", "TriFpFourCampMxBf",
          # New workloads: Qwen3 32B dense with NVFP4 (_qw_fp4_numbers): C-speed at 13 nodes (GLM-FP4-03): setup, per recipe (Bf, Mx, Nv),
          # speed-ups, power; C-loss on 1 node, 3 runs per recipe (GLM-FP4-02, -04; TriQwFpFourL*, Win*, Sep*): setup, per recipe the step time
          # and the window means (mean of the runs, Lo/Hi: lowest and highest run), where the quantized runs sit above all BF16 runs, chance
          "TriQwFpFourNodes", "TriQwFpFourGpus", "TriQwFpFourTp", "TriQwFpFourDp", "TriQwFpFourSeqs", "TriQwFpFourSeqLen", "TriQwFpFourLayers",
          "TriQwFpFourBfLast", "TriQwFpFourSteps", "TriQwFpFourFrom", "TriQwFpFourMeasured",
          *[f"TriQwFpFour{m}{r}" for r in ("Bf", "Mx", "Nv") for m in ("Step", "Med", "Sd", "Tok", "Mem", "CapMed", "CapLo", "CapHi", "Power", "Sm")],
          "TriQwFpFourSpNvBf", "TriQwFpFourSpNvMx", "TriQwFpFourSpMxBf", "TriQwFpFourSmBfNvPct", "TriQwFpFourXid",
          "TriQwFpFourLNodes", "TriQwFpFourLTp", "TriQwFpFourLPp", "TriQwFpFourLDp", "TriQwFpFourLSeqs", "TriQwFpFourLSteps", "TriQwFpFourLLr",
          "TriQwFpFourLMinLr", "TriQwFpFourLWarm", "TriQwFpFourLSeed", "TriQwFpFourLRuns", "TriQwFpFourLJobs",
          *[f"TriQwFpFour{m}{r}" for r in ("Bf", "Mx", "Nv") for m in ("LStep", "LStepLo", "LStepHi", "LMem", "WinEarly", "WinEarlyLo", "WinEarlyHi",
                                                                      "WinEnd", "WinEndLo", "WinEndHi")],
          "TriQwFpFourLSpMxBf", "TriQwFpFourLSpNvBf", "TriQwFpFourLSpNvMx",
          *[f"TriQwFpFourWin{w}{e}" for w in ("Early", "Mid", "Late", "End") for e in ("From", "To")],
          "TriQwFpFourEarlyMargin", "TriQwFpFourSepWinBoth", "TriQwFpFourSepWinMx", "TriQwFpFourCut", "TriQwFpFourSepSteps", "TriQwFpFourSepLast",
          "TriQwFpFourSepRunFrom", "TriQwFpFourSepRunTo", "TriQwFpFourSepMxSteps", "TriQwFpFourSepMxIn", "TriQwFpFourSepMxLast",
          "TriQwFpFourSepNvSteps", "TriQwFpFourSepNvChance", "TriQwFpFourOddsOne", "TriQwFpFourOddsAll", "TriQwFpFourNewBf", "TriQwFpFourNewQLo",
          "TriQwFpFourNewOdds", "TriQwFpFourLossDiffFrom", "TriQwFpFourGnDiffFrom", "TriQwFpFourLMaxLo", "TriQwFpFourLMaxHi", "TriQwFpFourLNonFinite",
          "TriQwFpFourLEndLo", "TriQwFpFourLEndHi",
          # New workloads: Qwen3 32B at 32k and 128k with context parallelism (_qw_cp_numbers; QW-CP-01 and its 8k reference, GLM-FP4-03's BF16
          # entry): setup, per entry (Ref: 8k, ThirtyTwo: 32k CP4, CpFour: 128k CP4, CpEight: 128k CP8 on 12 nodes) the step (mean, median, SD),
          # tokens/s per GPU and against 8k, MFU, peak memory, bytes across nodes, power; CP8 against CP4; ranges over the three CP entries
          "TriQwCpTp", "TriQwCpLayers", "TriQwCpSteps", "TriQwCpFrom", "TriQwCpMeasured", "TriQwCpLr", "TriQwCpTokGpu",
          *[f"TriQwCp{m}{e}" for e in ("Ref", "ThirtyTwo", "CpFour", "CpEight", "CpFourSel") for m in ("Step", "Tok", "Mfu", "Mem", "IbLo", "IbHi")],
          *[f"TriQwCp{m}{e}" for e in ("ThirtyTwo", "CpFour", "CpEight", "CpFourSel") for m in ("Nodes", "Split", "Dp", "Seqs", "Med", "Sd", "Ratio")],
          "TriQwCpCapLoRef", "TriQwCpCapHiRef", "TriQwCpEightShorter", "TriQwCpCapLo", "TriQwCpCapHi", "TriQwCpPowerLo", "TriQwCpPowerHi",
          "TriQwCpSmLo", "TriQwCpSmHi", "TriQwCpPortZero", "TriQwCpPortOne", "TriQwCpPortFour", "TriQwCpPortNine", "TriQwCpPortBusy",
          # QW-CP-02 (CpFourSel: 128k CP4 with selective recompute) against CP8 and CP4 with full recompute, its power, stragglers and memory drops
          "TriQwCpEightShorterMed", "TriQwCpSelShorter", "TriQwCpSelShorterMed", "TriQwCpSelEight", "TriQwCpSelEightMed",
          "TriQwCpCapLoCpFourSel", "TriQwCpCapHiCpFourSel", "TriQwCpCapLoCpFour", "TriQwCpCapHiCpFour",
          "TriQwCpSlowSteps", "TriQwCpSlowMulti", "TriQwCpSlowNodes", "TriQwCpSlowNodeList", "TriQwCpFastNodes", "TriQwCpBehindLo", "TriQwCpBehindHi",
          "TriQwCpDropLo", "TriQwCpDropHi", "TriQwCpDropOther", "TriQwCpDropOne", "TriQwCpBehindFull", "TriQwCpBehindEight", "TriQwCpCleanSteps",
          "TriQwCpCleanLo", "TriQwCpCleanHi", "TriQwCpCleanBelowLo", "TriQwCpCleanBelowHi",
          # New workloads: OpenFold3 training (_of3_numbers; OF3 step 2 at 1, 2 and 4 nodes, OF3-01 at 13): setup; per variant (Pub: published,
          # Ds: DeepSpeed attention, Cueq: cuEquivariance + TF32) and node count the wall-clock step, its median, samples/s per GPU and in all, the
          # loader's wait, against 1 node and against published; at the top node count, ranges over the variants; telemetry, memory, InfiniBand
          "TriOfMeasured", "TriOfFrom", "TriOfSteps", "TriOfBatch", "TriOfWorkers", "TriOfCrop", "TriOfParams", "TriOfBuffer",
          *[f"TriOf{m}{v}{n}" for v, n in (("Pub", "One"), ("Pub", "Two"), ("Pub", "Four"), ("Pub", "Thirteen"), ("Ds", "One"), ("Ds", "Thirteen"),
                                          ("Cueq", "One"), ("Cueq", "Two"), ("Cueq", "Four"), ("Cueq", "Thirteen"))
            for m in ("Sps", "Step", "Med", "Total", "Gap")],
          *[f"TriOfEff{v}{n}" for v, n in (("Pub", "Two"), ("Pub", "Four"), ("Pub", "Thirteen"), ("Ds", "Thirteen"), ("Cueq", "Two"), ("Cueq", "Four"),
                                          ("Cueq", "Thirteen"))],
          *[f"TriOfRatio{v}{n}" for v, n in (("Ds", "One"), ("Ds", "Thirteen"), ("Cueq", "One"), ("Cueq", "Two"), ("Cueq", "Four"), ("Cueq", "Thirteen"))],
          "TriOfTopNodes", "TriOfTopGpus", "TriOfEffLo", "TriOfEffHi", "TriOfLossPub", "TriOfLossLo", "TriOfLossHi",
          *[f"TriOf{m}{e}" for m in ("Gap", "GapWorst", "Power", "Util", "CpuMin", "IbBuffers", "GapSmall") for e in ("Lo", "Hi")],
          *[f"TriOf{m}{v}" for m in ("Power", "Util") for v in ("Pub", "Ds", "Cueq")],
          "TriOfSm", "TriOfSmMin", "TriOfCap", "TriOfCapPlain", "TriOfSmMinPlain", "TriOfCapCueq", "TriOfXid", "TriOfCpuMed", "TriOfMemAlloc", "TriOfMemRes", "TriOfNonFinite", "TriOfIbLo", "TriOfIbHi",
          "TriOfIbTree", "TriOfIbExcess", "TriOfFirstDs", "TriOfFirstLo", "TriOfFirstHi"]
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
    if len(p) and not p.synthetic.any():
        f = p[(p.job_id == NET_FULL_JOB) & (p.status == "ok")]
        if len(f):
            cut = 0.75 * f.bw_gbs.max()  # two classes, about 189 and 96 GB/s, nothing in between
            slow, fast = f[f.bw_gbs < cut], f[f.bw_gbs >= cut]
            hosts = pd.concat([f.a, f.b]).unique()
            partners = pd.concat([slow.a, slow.b]).value_counts().reindex(hosts, fill_value=0)
            same = f.a.map(bandwidth_group) == f.b.map(bandwidth_group)
            eight = (f.a.str[4:].astype(int) - f.b.str[4:].astype(int)).abs() % 8 == 0
            out.update({"TriFullPairs": (len(f), ",d", False), "TriFullNodes": (len(hosts), "d", False),
                        "TriFullFast": (len(fast), "d", False), "TriFullSlow": (len(slow), "d", False),
                        "TriFullFastLo": (fast.bw_gbs.min(), ".1f", False), "TriFullFastHi": (fast.bw_gbs.max(), ".1f", False),
                        "TriFullSlowLo": (slow.bw_gbs.min(), ".1f", False), "TriFullSlowHi": (slow.bw_gbs.max(), ".1f", False),
                        "TriFullPartnersLo": (int(partners.min()), "d", False), "TriFullPartnersHi": (int(partners.max()), "d", False),
                        "TriFullLatLo": (f.lat_us.min(), ".1f", False), "TriFullLatHi": (f.lat_us.max(), ".1f", False),
                        "TriFullWithin": (f[same].bw_gbs.median(), ".1f", False), "TriFullAcross": (f[~same].bw_gbs.median(), ".1f", False),
                        "TriFullSlowPct": (100 * len(slow) / len(f), ".0f", False),
                        "TriFullEightPct": (100 * (f[eight].bw_gbs < cut).mean(), ".0f", False), "TriFullEightPairs": (int(eight.sum()), "d", False)})
            old = p[p.job_id.isin(NET_D2_JOBS) & (p.status == "ok")]
            new = {tuple(sorted(k)): v for k, v in zip(zip(f.a, f.b), f.bw_gbs)}
            ratio = pd.Series([new[tuple(sorted(k))] / v for k, v in zip(zip(old.a, old.b), old.bw_gbs) if tuple(sorted(k)) in new])
            if len(ratio):
                out.update({"TriFullCommon": (len(ratio), "d", False), "TriFullRatioLo": (ratio.min(), ".1f", False),
                            "TriFullRatioHi": (ratio.max(), ".1f", False), "TriFullRatioMed": (ratio.median(), ".1f", False)})
        a = p[(p.job_id == NET_D2_AGAIN_JOB) & (p.status == "ok")]  # Nov 7 15:08, 48 nodes, two passes
        if len(a):
            key = lambda d: [tuple(sorted(k)) for k in zip(d.a, d.b)]
            a = a.assign(key=key(a))
            per = a.groupby("key").bw_gbs.mean()
            same = a.a.map(bandwidth_group) == a.b.map(bandwidth_group)
            w, x = a[same].bw_gbs, a[~same].bw_gbs
            passes = a.groupby("key").bw_gbs.agg(["min", "max"])
            out.update({"TriAgainPairs": (len(per), ",d", False), "TriAgainNodes": (len(set(a.a) | set(a.b)), "d", False),
                        "TriAgainWithin": (w.median(), ".1f", False), "TriAgainWithinLo": (w.min(), ".1f", False), "TriAgainWithinHi": (w.max(), ".1f", False),
                        "TriAgainAcross": (x.median(), ".1f", False), "TriAgainAcrossLo": (x.min(), ".1f", False), "TriAgainAcrossHi": (x.max(), ".1f", False),
                        "TriAgainMax": (a.bw_gbs.max(), ".1f", False), "TriAgainPassPct": (100 * (passes["max"] / passes["min"] - 1).max(), ".1f", False)})
            old = p[p.job_id.isin(NET_D2_JOBS) & (p.status == "ok")]
            o = old.assign(key=key(old)).groupby("key").bw_gbs.mean()
            common = o.index.intersection(per.index)
            if len(common):
                r = per[common] / o[common]
                out.update({"TriAgainCommon": (len(common), "d", False), "TriAgainRatioLo": (r.min(), ".2f", False),
                            "TriAgainRatioHi": (r.max(), ".2f", False), "TriAgainRatioMed": (r.median(), ".3f", False)})
    if len(c) and not c.synthetic.any():
        d4 = c[(c.job_id == NET_FULL_JOB) & (c.experiment == "D4") & (c.status == "ok")]
        if len(d4):
            out.update({"TriBisectionCuts": (len(d4), "d", False), "TriBisectionMin": (d4.min_pair_gbs.min(), ".1f", False),
                        "TriBisectionSlowLo": (int(d4.slow_pairs.min()), "d", False), "TriBisectionSlowHi": (int(d4.slow_pairs.max()), "d", False),
                        "TriBisectionPairs": (int(d4.n_pairs.max()), "d", False)})
        s = c[(c.job_id == NET_FULL_JOB) & (c.experiment == "D3") & (c.size_bytes == 2**30) & (c.status == "ok")]
        s = s[s.collective.isin(["all_gather", "reduce_scatter", "all_reduce"])]
        if len(s):
            t13, t26 = s[s.nodes == 13], s[s.nodes == 26].groupby("collective").busbw_gbs.median()
            if len(t13):
                out["TriSevenThirteenLo"], out["TriSevenThirteenHi"] = (t13.busbw_gbs.min(), ".0f", False), (t13.busbw_gbs.max(), ".0f", False)
            for coll, tag in (("all_gather", "Gather"), ("reduce_scatter", "Scatter"), ("all_reduce", "Reduce")):
                if coll in t26:
                    out[f"TriSeven{tag}"] = (t26[coll], ".1f", False)
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
    for n, tag in ((1, "One"), (13, "Thirteen"), (25, "Top"), (39, "Big"), (50, "Fifty")):
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
    top = q[(q.nodes == 50) & (q.content == "full") & (q.op == "save") & (q["mode"] == "sync")]
    if len(top):
        out["TriCkQFiftySyncGBs"] = (top.gb_s.median(), ".1f", False)
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
        if 39 in m.index and 26 in m.index:  # 26 nodes: one bandwidth group; 39: the first runs across both
            out[f"TriWeak{tag}Cross"] = (100 * (m.s[39] / m.s[26] - 1), "+.1f", False)
            if top > 39:
                out[f"TriWeak{tag}CrossTop"] = (100 * (m.s[top] / m.s[26] - 1), "+.1f", False)
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
        ref = max([n for n in clean.index if n < x.nodes], default=top)  # the largest clean run with fewer nodes
        out["TriSlowRunRefNodes"] = (int(ref), "d", False)
        out["TriSlowRunPct"] = (100 * (x.step_s / clean.s[ref] - 1), "+.1f", False)
        if top > x.nodes:  # a larger clean run, after the fix: more nodes and a shorter step, or not
            out["TriSlowRunVsTopPct"] = (100 * (x.step_s / clean.s[top] - 1), "+.1f", False)
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
    for name, layout, nodes, seqs in (("Thirteen", "pp13-ep8", 13, 256), ("Small", "pp13-ep8", 13, 64), ("Six", "pp6-ep16", 12, 256), ("Ten", "pp10-ep8", 10, 200), ("Twenty", "pp10-ep8", 20, 400), ("Thirty", "pp10-ep8", 30, 600), ("Forty", "pp10-ep8", 40, 800), ("Fifty", "pp10-ep8", 50, 1000), ("ThirtySix", "pp6-ep16", 36, 768), ("FortyEight", "pp6-ep16", 48, 1024),
                                         ("EpEight", "pp13-ep8", 26, 512), ("EpSixteen", "pp13-ep16", 26, 512), ("PpTwoSix", "pp26-ep8", 26, 512), ("PpTwoFive", "pp25-ep8", 25, 512), ("PpTwoFiveFifty", "pp25-ep8", 50, 1024)):
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
    six48 = one("pp6-ep16", 48, 1024)
    if six is not None and six48 is not None:
        out["TriGlmFortyEightEff"] = (100 * six48.tok_s_gpu.median() / six.tok_s_gpu.median(), ".1f", False)
    top = one("pp13-ep8", 39, 768)
    if top is not None and six36 is not None:
        out["TriGlmEpRatioTop"] = (top.tok_s_gpu.median() / six36.tok_s_gpu.median(), ".1f", False)
    ten, forty = one("pp10-ep8", 10, 200), one("pp10-ep8", 40, 800)
    if ten is not None and forty is not None:
        out["TriGlmFortyEff"] = (100 * forty.tok_s_gpu.median() / ten.tok_s_gpu.median(), ".0f", False)
    fifty = one("pp10-ep8", 50, 1000)
    if ten is not None and fifty is not None:
        out["TriGlmFiftyEff"] = (100 * fifty.tok_s_gpu.median() / ten.tok_s_gpu.median(), ".0f", False)
    p25b = one("pp25-ep8", 50, 1024)
    if p25 is not None and p25b is not None:
        out["TriGlmPpTwoFiveFiftyEff"] = (100 * p25b.tok_s_gpu.median() / p25.tok_s_gpu.median(), ".0f", False)
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


def _long_run_numbers(t, done) -> dict[str, tuple]:
    """The long runs (F2), TriLongQ* and TriLongG*: steps over the hour against the same layout's short run, and the GPUs'
    clocks and power."""
    out = {}
    if not len(done):
        return out
    for model, tag, short in ((QWEN, "Q", ("B-Q-weak",)), (GLM, "G", ("A-G", "B-G-weak"))):
        f2 = done[(done.model == model) & (done.experiment == "F2")]
        if not len(f2):
            continue
        run = f2.sort_values(["nodes", "measured_steps"]).iloc[-1]
        s = t["steps"][t["steps"].run_id == run.run_id].sort_values("t_s")
        out[f"TriLong{tag}Nodes"] = (int(run.nodes), "d", False)
        out[f"TriLong{tag}Steps"] = (int(run.measured_steps), "d", False)
        out[f"TriLong{tag}Min"] = ((run.measure_end - run.measure_start) / 60, ".0f", False)
        out[f"TriLong{tag}Step"] = (run.step_s, ".2f", False)
        out[f"TriLong{tag}Tok"] = (run.tok_s_gpu, ".0f", False)
        if len(s):
            early, late = s[s.t_s <= s.t_s.min() + 600], s[s.t_s >= s.t_s.max() - 600]  # first and last 10 minutes
            out[f"TriLong{tag}Early"], out[f"TriLong{tag}Late"] = (early.step_s.median(), ".2f", False), (late.step_s.median(), ".2f", False)
            out[f"TriLong{tag}DriftPct"] = (100 * (late.step_s.median() / early.step_s.median() - 1), ".1f", False)
            out[f"TriLong{tag}Max"] = (s.step_s.max(), ".1f", False)
        ref = done[(done.model == model) & done.experiment.isin(short) & (done.layout == run.layout) & (done.tokens_per_gpu == run.tokens_per_gpu)]
        if len(ref):  # the same node count if it ran, else the most nodes
            same = ref[ref.nodes == run.nodes]
            r0 = (same if len(same) else ref).sort_values(["nodes", "started"]).iloc[-1]
            out[f"TriLong{tag}RefNodes"], out[f"TriLong{tag}RefStep"] = (int(r0.nodes), "d", False), (r0.step_s, ".2f", False)
            out[f"TriLong{tag}VsRefPct"] = (100 * (run.step_s / r0.step_s - 1), ".1f", False)
            if len(s) >= r0.measured_steps:  # the long run over as many steps as the short run measured
                head = s.step_s.iloc[:int(r0.measured_steps)].median()
                out[f"TriLong{tag}RefSteps"], out[f"TriLong{tag}Head"] = (int(r0.measured_steps), "d", False), (head, ".2f", False)
        mon = t.get("gpu_monitor", pd.DataFrame())
        if len(mon):
            m = mon[(mon.source == run.run_name) & mon.time.between(run.measure_start, run.measure_end)]
            for metric, mtag, fmt in (("sm_clock_mhz", "Clock", ".0f"), ("power_w", "Watt", ".0f")):
                g = m[m.metric == metric].groupby(["host", "gpu"]).value.mean()  # each GPU's mean over the measurement
                if len(g):
                    out[f"TriLong{tag}{mtag}Lo"], out[f"TriLong{tag}{mtag}Hi"] = (g.min(), fmt, False), (g.max(), fmt, False)
                    out[f"TriLong{tag}{mtag}Med"] = (g.median(), fmt, False)
                    v = m[m.metric == metric]  # mean over all GPUs, first and last 10 minutes (as the step times above)
                    first, last = v[v.time <= run.measure_start + 600].value.mean(), v[v.time >= run.measure_end - 600].value.mean()
                    out[f"TriLong{tag}{mtag}First"], out[f"TriLong{tag}{mtag}Last"] = (first, ",.0f", False), (last, ",.0f", False)
                    out[f"TriLong{tag}{mtag}DriftPct"] = (100 * (last / first - 1), ".1f", False)
    return out


def _kimi_numbers(t) -> dict[str, tuple]:
    """Kimi K3 inference, TriKimi*: the study's passes on the 2-node server (kimi_passes, kimi_runs, kimi_accuracy); TriKimiFour* for
    the 4-node one (_kimi_four_numbers). Power and J per token are estimates: the main node's GPUs, times the node count."""
    p, runs, acc = (t.get(k, pd.DataFrame()) for k in ("kimi_passes", "kimi_runs", "kimi_accuracy"))
    out = {}
    if p.empty or runs.empty or not (runs.nodes == 2).any():
        return out
    run = runs[runs.nodes == 2].sort_values("job").iloc[-1]
    p = p[(p.job.astype(str) == str(run.job)) & ~p.throwaway.astype(bool)].set_index("pass")
    out.update({"TriKimiNodes": (int(run.nodes), "d", False), "TriKimiGpus": (int(run.gpus), "d", False),
                "TriKimiLayout": (rf"TP{int(run.tp)}\,$\times$\,PP{int(run.pp)}", "", False), "TriKimiVllm": (run.vllm_version, "", False),
                "TriKimiMoeBackend": (str(run.moe_backend).replace("_", r"\_"), "", False), "TriKimiMaxSeqs": (int(run.max_num_seqs), "d", False),
                "TriKimiMaxLen": (int(run.max_model_len), ",d", False), "TriKimiChunk": (int(run.max_batched_tokens), ",d", False),
                "TriKimiKvTokens": (run.kv_cache_tokens / 1e6, ".1f", False), "TriKimiKvUsePct": (run.kv_use_pct_full, ".1f", False),
                # inferred: KV use in proportion to the sequences running
                "TriKimiKvFillSeqs": (round(run.max_num_seqs * 100 / run.kv_use_pct_full, -1), ",.0f", False),
                # measured on the main node (the one voir samples): its GPUs' mean power, highest pass
                "TriKimiMainGpuW": (p.power_w_est.max() / run.gpus, ",.0f", False), "TriKimiMainHost": (str(run.hosts).split(";")[0], "", False),
                "TriKimiPasses": (len(p), "d", False), "TriKimiRequests": (int(p.completed.sum()), ",d", False), "TriKimiFailed": (int(p.failed.sum()), "d", False)})
    for name, tag in (("conc1", "One"), ("conc64", "SixtyFour")):
        if name in p.index:
            r = p.loc[name]
            out.update({f"TriKimi{tag}Tok": (r.output_throughput, ",.1f" if tag == "One" else ",.0f", False), f"TriKimi{tag}Tpot": (r.tpot_ms_p50, ".2f" if tag == "One" else ".1f", False),
                        f"TriKimi{tag}Ttft": (r.ttft_ms_p50, ",.0f", False), f"TriKimiPower{tag}": (r.power_w_est, ",.0f", False),
                        f"TriKimiJ{tag}": (r.j_per_output_tok_est, ".1f" if tag == "One" else ".2f", False)})
            if tag == "SixtyFour":
                out["TriKimiSixtyFourGpu"] = (r.output_throughput_gpu, ".1f", False)
    conc = p[p.index.str.fullmatch(r"conc\d+")].sort_values("concurrency")
    if len(conc):
        peak, top = conc.loc[conc.output_throughput.idxmax()], conc.iloc[-1]
        out.update({"TriKimiPeakConc": (int(peak.concurrency), "d", False), "TriKimiPeakTok": (peak.output_throughput, ",.0f", False),
                    "TriKimiPeakGpu": (peak.output_throughput_gpu, ".1f", False), "TriKimiPeakNode": (peak.output_throughput_node, ",.0f", False),
                    "TriKimiPeakTpot": (peak.tpot_ms_p50, ".0f", False), "TriKimiPowerPeak": (peak.power_w_est, ",.0f", False),
                    "TriKimiJPeak": (peak.j_per_output_tok_est, ".1f", False), "TriKimiTopConc": (int(top.concurrency), ",d", False),
                    "TriKimiTopTok": (top.output_throughput, ",.0f", False), "TriKimiTopTtft": (top.ttft_ms_p50 / 1000, ".0f", False),
                    "TriKimiTopPct": (100 * (1 - top.output_throughput / peak.output_throughput), ".1f", False)})
    ctx = p[p.index.str.fullmatch(r"ctx\d+k-c\d+")]
    one, sixteen = ctx[ctx.concurrency == 1].sort_values("input_len"), ctx[ctx.concurrency == 16].sort_values("input_len")
    if len(one):
        lo, hi = one.iloc[0], one.iloc[-1]
        out.update({"TriKimiCtxTopK": (int(hi.input_len) // 1024, "d", False), "TriKimiCtxOneTtft": (lo.ttft_ms_p50, ",.0f", False),
                    "TriKimiCtxTtft": (hi.ttft_ms_p50, ",.0f", False), "TriKimiCtxTok": (hi.output_throughput, ".1f", False),
                    "TriKimiCtxOneTpot": (lo.tpot_ms_p50, ".2f", False), "TriKimiCtxTpot": (hi.tpot_ms_p50, ".2f", False),
                    "TriKimiCtxTpotPct": (100 * (hi.tpot_ms_p50 / lo.tpot_ms_p50 - 1), ".0f", False)})
    if len(sixteen):
        out["TriKimiCtxSixteenTtft"] = (sixteen.iloc[-1].ttft_ms_p50 / 1000, ".1f", False)
    if not acc.empty:
        a = acc[acc.job.astype(str) == str(run.job)].set_index("task")
        for task, tag in (("gsm8k", "Gsm"), ("arc_challenge", "Arc"), ("gpqa_diamond", "Gpqa"), ("mmlu_pro", "Mmlu")):
            if task in a.index:
                out[f"TriKimi{tag}"], out[f"TriKimi{tag}Err"] = (100 * a.value[task], ".1f", False), (100 * a.stderr[task], ".1f", False)
        if "mmlu_pro" in a.index:
            out["TriKimiMmluN"] = (int(a.n["mmlu_pro"]), ",d", False)
    out.update(_kimi_more(p, run, t.get("kimi_questions", pd.DataFrame()), "TriKimi"))
    out.update(_kimi_four_numbers(t, run, p))
    return out


def _kimi_more(p, run, questions, prefix) -> dict[str, tuple]:
    """Both servers: output tok/s at 16 concurrent requests of 32k, 64k and 128k tokens, the main node's mean GPU power at the top
    concurrency (per GPU, measured), zero windows over the passes, unparsed MMLU-Pro answers."""
    out = {}
    sixteen = p[p.index.str.fullmatch(r"ctx\d+k-c16")].set_index("input_len")
    for k, tag in ((131072, ""), (32768, "ThirtyTwo"), (65536, "SixtyFour")):
        if k in sixteen.index:
            out[f"{prefix}CtxSixteenTok{tag}"] = (sixteen.output_throughput[k], ".1f", False)
    conc = p[p.index.str.fullmatch(r"conc\d+")].sort_values("concurrency")
    if len(conc):
        out[f"{prefix}TopMainW"] = (conc.iloc[-1].power_w_est / run.gpus, ",.0f", False)
    out[f"{prefix}Stalls"] = (int(p.zero_windows.sum()), "d", False)
    if not questions.empty:
        q = questions[(questions.job.astype(str) == str(run.job)) & (questions.task == "mmlu_pro")]
        if len(q):
            out[f"{prefix}MmluUnparsed"] = (int((~q.parsed.astype(bool)).sum()), "d", False)
    return out


def _kimi_four_numbers(t, two, p2) -> dict[str, tuple]:
    """Kimi K3 inference on the 4-node server, TriKimiFour*, and its change against the 2-node one (two, p2: its run and passes),
    pass for pass. Without the zero windows (NoStall): output tokens over the pass's duration less 10 s per window, a derived bound."""
    p, runs, acc = (t.get(k, pd.DataFrame()) for k in ("kimi_passes", "kimi_runs", "kimi_accuracy"))
    out = {}
    if not (runs.nodes == 4).any():
        return out
    run = runs[runs.nodes == 4].sort_values("job").iloc[-1]
    p = p[(p.job.astype(str) == str(run.job)) & ~p.throwaway.astype(bool)].set_index("pass")
    if p.empty:
        return out

    def gain(name):
        return 100 * (p.output_throughput[name] / p2.output_throughput[name] - 1)

    out.update({"TriKimiFourNodes": (int(run.nodes), "d", False), "TriKimiFourGpus": (int(run.gpus), "d", False),
                "TriKimiFourLayout": (rf"TP{int(run.tp)}\,$\times$\,PP{int(run.pp)}", "", False), "TriKimiFourMaxSeqs": (int(run.max_num_seqs), ",d", False),
                "TriKimiFourMainHost": (str(run.hosts).split(";")[0], "", False), "TriKimiFourKvTokens": (run.kv_cache_tokens / 1e6, ".1f", False),
                "TriKimiFourKvUsePct": (run.kv_use_pct_full, ".1f", False), "TriKimiFourPasses": (len(p), "d", False),
                "TriKimiFourRequests": (int(p.completed.sum()), ",d", False), "TriKimiFourFailed": (int(p.failed.sum()), "d", False)})
    if "conc1" in p.index:
        out.update({"TriKimiFourOneTok": (p.output_throughput["conc1"], ",.1f", False), "TriKimiFourOneTpot": (p.tpot_ms_p50["conc1"], ".2f", False)})
    if "conc64" in p.index and "conc64" in p2.index:
        r = p.loc["conc64"]
        nostall = r.output_throughput * r.duration_s / (r.duration_s - LOG_INTERVAL_S * r.zero_windows)
        out.update({"TriKimiFourSixtyFourTok": (r.output_throughput, ",.0f", False), "TriKimiFourSixtyFourPct": (-gain("conc64"), ".0f", False),
                    "TriKimiFourSixtyFourNoStall": (nostall, ",.0f", False),
                    "TriKimiFourSixtyFourNoStallPct": (100 * (1 - nostall / p2.output_throughput["conc64"]), ".0f", False)})
    if "conc512" in p.index and "conc512" in p2.index:
        out.update({"TriKimiFourFiveTwelveTok": (p.output_throughput["conc512"], ",.0f", False), "TriKimiFourFiveTwelvePct": (gain("conc512"), ".0f", False)})
    conc = p[p.index.str.fullmatch(r"conc\d+")].sort_values("concurrency")
    if len(conc) and conc.index[-1] in p2.index:
        top = conc.iloc[-1]
        out.update({"TriKimiFourTopTok": (top.output_throughput, ",.0f", False), "TriKimiFourTopGpu": (top.output_throughput_gpu, ".1f", False),
                    "TriKimiFourTopNode": (top.output_throughput_node, ",.0f", False), "TriKimiFourTopTpot": (top.tpot_ms_p50, ".0f", False),
                    "TriKimiFourTopPct": (gain(conc.index[-1]), ".0f", False), "TriKimiFourPowerTop": (top.power_w_est, ",.0f", False),
                    "TriKimiFourJTop": (top.j_per_output_tok_est, ".1f", False)})
    one = p[p.index.str.fullmatch(r"ctx\d+k-c1")].sort_values("input_len")
    if len(one):
        out["TriKimiFourCtxTtft"] = (one.iloc[-1].ttft_ms_p50, ",.0f", False)
    sixteen = p[p.index.str.fullmatch(r"ctx\d+k-c16")].sort_values("input_len")
    if len(sixteen) and sixteen.index[-1] in p2.index:
        out["TriKimiFourCtxSixteenPct"] = (gain(sixteen.index[-1]), ".0f", False)
    out.update(_kimi_more(p, run, t.get("kimi_questions", pd.DataFrame()), "TriKimiFour"))
    if not acc.empty:
        a = acc[acc.job.astype(str) == str(run.job)].set_index("task")
        for task, tag in (("gsm8k", "Gsm"), ("arc_challenge", "Arc"), ("gpqa_diamond", "Gpqa"), ("mmlu_pro", "Mmlu")):
            if task in a.index:
                out[f"TriKimiFour{tag}"], out[f"TriKimiFour{tag}Err"] = (100 * a.value[task], ".1f", False), (100 * a.stderr[task], ".1f", False)
        if "mmlu_pro" in a.index:
            out["TriKimiFourMmluN"] = (int(a.n["mmlu_pro"]), ",d", False)
    q = t.get("kimi_questions", pd.DataFrame())
    if not q.empty:  # MMLU-Pro paired on the same questions: right on one server only; z of McNemar's test, (b - c) / sqrt(b + c)
        q = q[q.task == "mmlu_pro"].assign(job=q.job.astype(str))
        m = q[q.job == str(two.job)].merge(q[q.job == str(run.job)], on="doc_hash", suffixes=("_two", "_four"))
        if len(m):
            b, c = int(((m.correct_two == 1) & (m.correct_four == 0)).sum()), int(((m.correct_two == 0) & (m.correct_four == 1)).sum())
            out.update({"TriKimiMmluOnlyTwo": (b, "d", False), "TriKimiMmluOnlyFour": (c, "d", False),
                        "TriKimiMmluPairZ": ((b - c) / (b + c) ** 0.5 if b + c else float("nan"), ".1f", False)})
    return out


XN_BACKENDS = (("Ag", "allgather_reducescatter", "all-gather"), ("Dp", "deepep_v2", "DeepEP"))
XN_PASSES = (("conc1", "One"), ("conc4", "Four"), ("conc16", "Sixteen"), ("conc64", "SixtyFour"), ("conc128", "OneTwentyEight"),
             ("conc256", "TwoFiftySix"), ("ctx128k-c1", "Ctx"), ("ctx8k-c16", "CtxEightSixteen"), ("ctx32k-c16", "CtxThirtyTwoSixteen"))


def _kimi_two_node_passes(t):
    """The 2-node TP8 x PP2 server's measured passes, by pass (the run TriKimi* reads), or an empty frame."""
    p, runs = t.get("kimi_passes", pd.DataFrame()), t.get("kimi_runs", pd.DataFrame())
    if p.empty or runs.empty or not (runs.nodes == 2).any():
        return pd.DataFrame()
    run = runs[runs.nodes == 2].sort_values("job").iloc[-1]
    return p[(p.job.astype(str) == str(run.job)) & ~p.throwaway.astype(bool)].set_index("pass")


def _kimi_xn_servers(t) -> dict:
    """EP16 on 2 nodes (K3-XN-01), {tag: (run, measured passes by pass)} for each all-to-all backend (Ag, Dp) that ran."""
    p, runs = t.get("kimi_xn_passes", pd.DataFrame()), t.get("kimi_xn_runs", pd.DataFrame())
    out = {}
    if p.empty or runs.empty:
        return out
    for tag, backend, _ in XN_BACKENDS:
        r = runs[runs.all2all == backend]
        if len(r):
            run = r.sort_values("job").iloc[-1]
            out[tag] = (run, p[(p.job.astype(str) == str(run.job)) & ~p.throwaway.astype(bool)].set_index("pass"))
    return out


def _and(items) -> str:
    items = list(items)
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + (items[-1] if items else "")


def _kimi_xn_numbers(t) -> dict[str, tuple]:
    """Kimi K3 inference with the experts over 2 nodes (K3-XN-01), TriKxn*: EP16 (DP2 x TP8, expert parallelism) with each all-to-all
    backend, Ag (vLLM's all-gather / reduce-scatter) and Dp (DeepEP V2), one run each, against the 2-node TP8 x PP2 server (TriKimi*, Pp),
    pass for pass. Ratios: EP16's output tokens/s over TP8 x PP2's, derived. Weights, KV and engine-up times from the server logs."""
    servers, base = _kimi_xn_servers(t), _kimi_two_node_passes(t)
    out = {}
    if not servers:
        return out
    run0 = next(iter(servers.values()))[0]
    dp, tp = int(run0.dp), int(run0.tp)
    out.update({"TriKxnNodes": (int(run0.nodes), "d", False), "TriKxnGpus": (int(run0.gpus), "d", False), "TriKxnDp": (dp, "d", False),
                "TriKxnTp": (tp, "d", False), "TriKxnEp": (dp * tp, "d", False),
                "TriKxnLayout": (rf"DP{dp}\,$\times$\,TP{tp}, EP{dp * tp}", "", False),
                "TriKxnMaxSeqs": (int(run0.max_num_seqs), "d", False),
                "TriKxnPasses": (min(len(ps) for _, ps in servers.values()), "d", False),
                "TriKxnRequests": (int(sum(ps.completed.sum() for _, ps in servers.values())), ",d", False),
                "TriKxnFailed": (int(sum(ps.failed.sum() for _, ps in servers.values())), "d", False)})
    allp = t.get("kimi_xn_passes", pd.DataFrame())
    out["TriKxnWarmFailed"] = (int(allp[allp.throwaway.astype(bool)].failed.sum()), "d", False)
    acc = t.get("kimi_xn_accuracy", pd.DataFrame())
    for tag, (run, ps) in servers.items():
        out.update({f"TriKxnWeights{tag}": (run.weights_gib, ".1f", False), f"TriKxnKvGib{tag}": (run.kv_gib, ".1f", False),
                    f"TriKxnKvTok{tag}": (int(run.kv_cache_tokens), ",d", False), f"TriKxnUp{tag}": (run.engine_up_min, ".0f", False)})
        for name, ptag in XN_PASSES:
            if name in ps.index:
                v = ps.output_throughput[name]
                out[f"TriKxn{ptag}Tok{tag}"] = (v, ".1f" if v < 1000 else ",.0f", False)
                if name in base.index:
                    out[f"TriKxn{ptag}Ratio{tag}"] = (v / base.output_throughput[name], ".2f", False)
        for name, ptag in (("conc1", "One"), ("conc64", "SixtyFour")):
            if name in ps.index:
                out[f"TriKxn{ptag}Tpot{tag}"] = (ps.tpot_ms_p50[name], ".1f", False)
        if len(base):  # passes where EP16 gives more output than TP8 x PP2
            common = ps.index.intersection(base.index)
            out[f"TriKxnFaster{tag}"] = (int((ps.output_throughput[common] > base.output_throughput[common]).sum()), "d", False)
        if not acc.empty:
            a = acc[acc.job.astype(str) == str(run.job)].set_index("task")
            for task, atag in (("gsm8k", "Gsm"), ("arc_challenge", "Arc"), ("gpqa_diamond", "Gpqa"), ("mmlu_pro", "Mmlu")):
                if task in a.index:
                    out[f"TriKxn{atag}{tag}"] = (100 * a.value[task], ".1f", False)
                    out[f"TriKxn{atag}Err{tag}"] = (100 * a.stderr[task], ".1f", False)
            if "gsm8k" in a.index:
                out["TriKxnGsmN"] = (int(a.n["gsm8k"]), ",d", False)
    if len(base):
        for name, mtag in (("conc256", "TwoFiftySixTokPp"), ("ctx8k-c16", "CtxEightSixteenTokPp")):
            if name in base.index:
                v = base.output_throughput[name]
                out[f"TriKxn{mtag}"] = (v, ".1f" if v < 1000 else ",.0f", False)
        ran = set().union(*(set(ps.index) for _, ps in servers.values()))
        out["TriKxnNotRun"] = (_and(n for n in base.sort_values("order").index if n not in ran), "", False)
    if {"Ag", "Dp"} <= set(servers):
        ag, dpp = servers["Ag"][1], servers["Dp"][1]
        common = ag.index.intersection(dpp.index)
        r = dpp.output_throughput[common] / ag.output_throughput[common]
        lead = 100 * (r[r > 1] - 1)
        out["TriKxnDpLeadN"] = (len(lead), "d", False)
        if len(lead):
            out.update({"TriKxnDpLeadLo": (lead.min(), ".1f", False), "TriKxnDpLeadHi": (lead.max(), ".1f", False)})
        for name, ptag in (("conc4", "Four"), ("conc256", "TwoFiftySix")):
            if name in r.index:
                out[f"TriKxnDpAg{ptag}"] = (r[name], ".2f", False)
    return out


def kimi_xn_passes_table(t) -> str:
    """Kimi K3 inference on 2 nodes, pass for pass: TP8 x PP2 (K3-INF-01) against EP16 with each all-to-all backend (K3-XN-01); output
    tokens/s, TTFT and TPOT medians, and the ratios of output tokens/s (derived). A pass EP16 did not run reads "not run"."""
    servers, base = _kimi_xn_servers(t), _kimi_two_node_passes(t)
    if not servers or base.empty:
        return ""
    ag, dp = (servers[tag][1] if tag in servers else pd.DataFrame() for tag, _, _ in XN_BACKENDS)
    cols = [base, ag, dp]
    lines = [r"\begin{tabular}{lrr" + "r" * 12 + "}", r"\toprule",
             r"Pass & Conc. & In / out & \multicolumn{3}{c}{Output tok/s} & \multicolumn{3}{c}{TTFT p50, ms} & \multicolumn{3}{c}{TPOT p50, ms} & "
             r"\multicolumn{3}{c}{Output ratio (derived)} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}\cmidrule(lr){13-15}",
             r" & & tokens & PP2 & AG & DeepEP & PP2 & AG & DeepEP & PP2 & AG & DeepEP & AG / PP2 & DeepEP / PP2 & DeepEP / AG \\", r"\midrule"]

    def cell(ps, name, col, fmt):
        return format(ps[col][name], fmt) if len(ps) and name in ps.index else "--"

    def ratio(a, b, name):
        if not (len(a) and len(b) and name in a.index and name in b.index):
            return "--"
        return f"{a.output_throughput[name] / b.output_throughput[name]:.2f}"

    group = None
    for name in base.sort_values("order").index:
        x = base.loc[name]
        g = (re.sub(r"\d+k?$", "", name.split("-")[0]), x.concurrency if name.startswith("ctx") else None)
        if group is not None and g != group:
            lines.append(r"\midrule")
        group = g
        head = f"{name} & {int(x.concurrency)} & {int(x.input_len):,} / {int(x.output_len):,} & "
        if not any(len(ps) and name in ps.index for ps in (ag, dp)):
            lines.append(head + f"{x.output_throughput:,.1f} & " + r"\multicolumn{2}{c}{not run} & " + f"{x.ttft_ms_p50:,.0f} & & & "
                         + f"{x.tpot_ms_p50:.1f} & & & & & " + r"\\")
            continue
        lines.append(head + " & ".join([*(cell(ps, name, "output_throughput", ",.1f") for ps in cols),
                                        *(cell(ps, name, "ttft_ms_p50", ",.0f") for ps in cols),
                                        *(cell(ps, name, "tpot_ms_p50", ".1f") for ps in cols),
                                        ratio(ag, base, name), ratio(dp, base, name), ratio(dp, ag, name)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# Kimi K3 on 4 nodes (kimi_xn4_*): P/D against colocated (K3-XN-02, TriKpd*), EP32 against the 4-node pipeline (K3-XN-03, TriKxnFour*)
XN4_PD_PASSES = (("conc1", "One"), ("conc4", "Four"), ("conc16", "Sixteen"), ("conc64", "SixtyFour"), ("conc128", "OneTwentyEight"),
                 ("conc256", "TwoFiftySix"), ("ctx1k-c16", "CtxOneSixteen"), ("ctx8k-c16", "CtxEightSixteen"),
                 ("ctx32k-c16", "CtxThirtyTwoSixteen"), ("decodelong-c1", "DecodeOne"), ("decodelong-c16", "DecodeSixteen"))
XN4_EP_PASSES = (("conc1", "One"), ("conc4", "Four"), ("conc16", "Sixteen"), ("conc64", "SixtyFour"), ("conc128", "OneTwentyEight"),
                 ("conc256", "TwoFiftySix"), ("conc512", "FiveTwelve"), ("conc1024", "Top"), ("ctx128k-c1", "Ctx"),
                 ("ctx8k-c16", "CtxEightSixteen"), ("ctx32k-c16", "CtxThirtyTwoSixteen"))
ACC_TASKS = (("gsm8k", "Gsm"), ("arc_challenge", "Arc"), ("gpqa_diamond", "Gpqa"), ("mmlu_pro", "Mmlu"))


def _kimi_xn4_servers(t) -> dict:
    """The 4-node K3-XN runs, {"pd" | "colocated" | "ep32": (run, measured passes by pass)}: P/D and its colocated reference by role,
    EP32 by layout (one engine per node)."""
    p, runs = t.get("kimi_xn4_passes", pd.DataFrame()), t.get("kimi_xn4_runs", pd.DataFrame())
    out = {}
    if p.empty or runs.empty:
        return out
    roles = runs.role if "role" in runs else pd.Series(index=runs.index, dtype=object)
    for key, sel in (("pd", roles == "pd"), ("colocated", roles == "colocated"), ("ep32", roles.isna() & runs.layout.str.startswith("dp"))):
        r = runs[sel]
        if len(r):
            run = r.sort_values("job").iloc[-1]
            out[key] = (run, p[(p.job.astype(str) == str(run.job)) & ~p.throwaway.astype(bool)].set_index("pass"))
    return out


def _kimi_four_node_passes(t):
    """The 4-node TP8 x PP4 server's measured passes, by pass (the run TriKimiFour* reads), or an empty frame."""
    p, runs = t.get("kimi_passes", pd.DataFrame()), t.get("kimi_runs", pd.DataFrame())
    if p.empty or runs.empty or not (runs.nodes == 4).any():
        return pd.DataFrame()
    run = runs[runs.nodes == 4].sort_values("job").iloc[-1]
    return p[(p.job.astype(str) == str(run.job)) & ~p.throwaway.astype(bool)].set_index("pass")


def _tok(v) -> tuple:
    return (v, ".1f" if v < 1000 else ",.0f", False)


def _accuracy(acc, job, prefix, suffix="") -> dict[str, tuple]:
    out = {}
    if acc.empty:
        return out
    a = acc[acc.job.astype(str) == str(job)].set_index("task")
    for task, tag in ACC_TASKS:
        if task in a.index:
            out[f"{prefix}{tag}{suffix}"] = (100 * a.value[task], ".1f", False)
            out[f"{prefix}{tag}Err{suffix}"] = (100 * a.stderr[task], ".1f", False)
    return out


def _kimi_pd_numbers(t) -> dict[str, tuple]:
    """Prefill and decode on separate nodes (K3-XN-02), TriKpd*: P/D against its colocated reference (Co), one run each, pass for pass.
    Ratios: P/D's output tokens/s over colocated's, derived. One-request passes: conc1 and ctx*-c1 (128 or 256 out); decodelong-c1 apart.
    TTFT gap: P/D's mean less colocated's, derived. GPQA paired on doc_hash, exact McNemar p (two-sided binomial on the discordant pairs)."""
    s = _kimi_xn4_servers(t)
    out = {}
    if not {"pd", "colocated"} <= set(s):
        return out
    (run, ps), (co_run, co) = s["pd"], s["colocated"]
    dp, tp = int(run.dp), int(run.tp)
    out.update({"TriKpdNodes": (int(run.nodes), "d", False), "TriKpdGpus": (int(run.gpus), "d", False),
                "TriKpdInstNodes": (int(run.nodes) // int(run.instances), "d", False),
                "TriKpdLayout": (rf"DP{dp}\,$\times$\,TP{tp}, EP{dp * tp}", "", False), "TriKpdMaxSeqs": (int(run.max_num_seqs), "d", False),
                "TriKpdWeights": (run.weights_gib, ".1f", False), "TriKpdKvGib": (run.kv_gib, ".1f", False),
                "TriKpdKvTok": (int(run.kv_cache_tokens), ",d", False), "TriKpdUp": (run.engine_up_min, ".0f", False),
                "TriKpdUpCo": (co_run.engine_up_min, ".0f", False), "TriKpdPasses": (min(len(ps), len(co)), "d", False),
                "TriKpdRequests": (int(ps.completed.sum() + co.completed.sum()), ",d", False),
                "TriKpdFailed": (int(ps.failed.sum() + co.failed.sum()), "d", False)})
    allp = t.get("kimi_xn4_passes", pd.DataFrame())
    warm = allp[allp.throwaway.astype(bool) & allp.job.astype(str).isin([str(run.job), str(co_run.job)])]
    out["TriKpdWarmFailed"] = (int(warm.failed.sum()), "d", False)
    common = ps.index.intersection(co.index)
    ratio = ps.output_throughput[common] / co.output_throughput[common]
    for name, tag in XN4_PD_PASSES:
        if name in common:
            out[f"TriKpd{tag}Ratio"] = (ratio[name], ".2f", False)
    for name, tag in (("conc1", "One"), ("conc64", "SixtyFour"), ("conc256", "TwoFiftySix")):
        if name in common:
            out[f"TriKpd{tag}Tok"], out[f"TriKpd{tag}TokCo"] = _tok(ps.output_throughput[name]), _tok(co.output_throughput[name])
    out["TriKpdLeadN"] = (int((ratio > 1).sum()), "d", False)
    if "conc1" in common:
        out.update({"TriKpdOneTtft": (ps.ttft_ms_p50["conc1"], ",.0f", False), "TriKpdOneTtftCo": (co.ttft_ms_p50["conc1"], ",.0f", False)})
    one = [n for n in common if re.fullmatch(r"conc1|ctx\d+k-c1|decodelong-c1", n)]
    if one:  # TTFT over every one-request pass; output over those with 128 or 256 out (decodelong-c1's 4,096 out dilute the TTFT)
        gap = (ps.ttft_ms_mean[one] - co.ttft_ms_mean[one]) / 1000
        short = [n for n in one if n != "decodelong-c1"]
        out.update({"TriKpdOneTtftGapLo": (gap.min(), ".2f", False), "TriKpdOneTtftGapHi": (gap.max(), ".2f", False),
                    "TriKpdOneRatioLo": (ratio[short].min(), ".2f", False), "TriKpdOneRatioHi": (ratio[short].max(), ".2f", False)})
    for name, tag in (("ctx1k-c16", "CtxOneSixteen"), ("ctx8k-c16", "CtxEightSixteen"), ("ctx32k-c16", "CtxThirtyTwoSixteen")):
        if name in common:
            out.update({f"TriKpd{tag}Tpot": (ps.tpot_ms_p50[name], ".1f", False), f"TriKpd{tag}TpotCo": (co.tpot_ms_p50[name], ".1f", False)})
    if "ctx32k-c16" in common:
        out.update({"TriKpdCtxThirtyTwoSixteenTtft": (ps.ttft_ms_p50["ctx32k-c16"] / 1000, ".1f", False),
                    "TriKpdCtxThirtyTwoSixteenTtftCo": (co.ttft_ms_p50["ctx32k-c16"] / 1000, ".1f", False)})
    base = _kimi_two_node_passes(t)
    if len(base):
        ran = set(ps.index) | set(co.index)
        out["TriKpdNotRun"] = (_and(n for n in base.sort_values("order").index if n not in ran), "", False)
    acc = t.get("kimi_xn4_accuracy", pd.DataFrame())
    out.update(_accuracy(acc, run.job, "TriKpd"))
    out.update(_accuracy(acc, co_run.job, "TriKpd", "Co"))
    q = t.get("kimi_xn4_questions", pd.DataFrame())
    if not q.empty:
        q = q[q.task == "gpqa_diamond"].assign(job=q.job.astype(str))
        m = q[q.job == str(run.job)].merge(q[q.job == str(co_run.job)], on="doc_hash", suffixes=("_pd", "_co"))
        if len(m):
            b, c = int(((m.correct_co == 1) & (m.correct_pd == 0)).sum()), int(((m.correct_pd == 1) & (m.correct_co == 0)).sum())
            n, k = b + c, min(b, c)
            p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else float("nan")
            out.update({"TriKpdGpqaN": (len(m), "d", False), "TriKpdGpqaRight": (int(m.correct_pd.sum()), "d", False),
                        "TriKpdGpqaRightCo": (int(m.correct_co.sum()), "d", False), "TriKpdGpqaOnlyCo": (b, "d", False),
                        "TriKpdGpqaOnlyPd": (c, "d", False), "TriKpdGpqaP": (p, ".3f", False)})
    return out


def _kimi_xn_four_numbers(t) -> dict[str, tuple]:
    """The experts over 4 nodes (K3-XN-03), TriKxnFour*: EP32 (DP4 x TP8, expert parallelism, a sequence cap per engine) against the 4-node
    TP8 x PP4 server (TriKimiFour*, Pp) and, where useful, EP16 on 2 nodes with all-gather (K3-XN-01, Ep), one run each, pass for pass.
    Ratios: EP32's output tokens/s over the other's, derived. AccZMax: the largest |difference| / standard error of the difference over
    the four tasks, against TP8 x PP4, derived."""
    s, base = _kimi_xn4_servers(t), _kimi_four_node_passes(t)
    out = {}
    if "ep32" not in s:
        return out
    run, ps = s["ep32"]
    dp, tp = int(run.dp), int(run.tp)
    out.update({"TriKxnFourNodes": (int(run.nodes), "d", False), "TriKxnFourGpus": (int(run.gpus), "d", False),
                "TriKxnFourLayout": (rf"DP{dp}\,$\times$\,TP{tp}, EP{dp * tp}", "", False), "TriKxnFourMaxSeqs": (int(run.max_num_seqs), "d", False),
                "TriKxnFourHeld": (dp * int(run.max_num_seqs), "d", False), "TriKxnFourWeights": (run.weights_gib, ".1f", False),
                "TriKxnFourKvGib": (run.kv_gib, ".1f", False), "TriKxnFourKvTok": (int(run.kv_cache_tokens), ",d", False),
                "TriKxnFourUp": (run.engine_up_min, ".0f", False), "TriKxnFourPasses": (len(ps), "d", False),
                "TriKxnFourRequests": (int(ps.completed.sum()), ",d", False), "TriKxnFourFailed": (int(ps.failed.sum()), "d", False)})
    allp = t.get("kimi_xn4_passes", pd.DataFrame())
    warm = allp[allp.throwaway.astype(bool) & (allp.job.astype(str) == str(run.job))]
    out["TriKxnFourWarmFailed"] = (int(warm.failed.sum()), "d", False)
    for name, tag in XN4_EP_PASSES:
        if name in ps.index:
            out[f"TriKxnFour{tag}Tok"] = _tok(ps.output_throughput[name])
    for name, tag in (("conc1", "One"), ("conc1024", "Top")):
        if name in ps.index:
            out[f"TriKxnFour{tag}Tpot"] = (ps.tpot_ms_p50[name], ".1f" if tag == "One" else ".0f", False)
    if len(base):
        common = ps.index.intersection(base.index)
        ratio = ps.output_throughput[common] / base.output_throughput[common]
        for name, tag in XN4_EP_PASSES:
            if name in common:
                out[f"TriKxnFour{tag}Ratio"] = (ratio[name], ".2f", False)
        out.update({"TriKxnFourLeadN": (int((ratio > 1).sum()), "d", False), "TriKxnFourCommon": (len(common), "d", False),
                    "TriKxnFourNotRun": (_and(n for n in base.sort_values("order").index if n not in ps.index), "", False)})
        if "conc256" in base.index:
            out["TriKxnFourTwoFiftySixTokPp"] = _tok(base.output_throughput["conc256"])
        if "conc1024" in common:
            out.update({"TriKxnFourTopTtft": (ps.ttft_ms_p50["conc1024"] / 1000, ".1f", False),
                        "TriKxnFourTopTtftPp": (base.ttft_ms_p50["conc1024"] / 1000, ".1f", False)})
    ep16 = _kimi_xn_servers(t).get("Ag")
    if ep16 is not None:
        e = ep16[1]
        common = ps.index.intersection(e.index)
        ratio = ps.output_throughput[common] / e.output_throughput[common]
        for name, tag in (("conc1", "One"), ("conc256", "TwoFiftySix"), ("ctx32k-c16", "CtxThirtyTwoSixteen")):
            if name in common:
                out[f"TriKxnFourEp{tag}Ratio"] = (ratio[name], ".2f", False)
        out.update({"TriKxnFourEpLeadN": (int((ratio > 1).sum()), "d", False), "TriKxnFourEpCommon": (len(common), "d", False)})
    acc = t.get("kimi_xn4_accuracy", pd.DataFrame())
    out.update(_accuracy(acc, run.job, "TriKxnFour"))
    runs = t.get("kimi_runs", pd.DataFrame())
    if not acc.empty and len(runs) and (runs.nodes == 4).any():
        four = runs[runs.nodes == 4].sort_values("job").iloc[-1]
        a = acc[acc.job.astype(str) == str(run.job)].set_index("task")
        b = t.get("kimi_accuracy", pd.DataFrame())
        b = b[b.job.astype(str) == str(four.job)].set_index("task") if len(b) else b
        z = [abs(a.value[k] - b.value[k]) / math.hypot(a.stderr[k], b.stderr[k]) for k, _ in ACC_TASKS if k in a.index and k in b.index]
        if z:
            out["TriKxnFourAccZMax"] = (max(z), ".2f", False)
    return out


def kimi_pd_passes_table(t) -> str:
    """Kimi K3 inference on 4 nodes, pass for pass: prefill and decode on separate nodes (P/D) against colocated (K3-XN-02), one run
    each; output tokens/s and its ratio (derived), TTFT medians and the difference of the means (derived), TPOT medians."""
    s = _kimi_xn4_servers(t)
    if not {"pd", "colocated"} <= set(s):
        return ""
    ps, co = s["pd"][1], s["colocated"][1]
    lines = [r"\begin{tabular}{lrr" + "r" * 8 + "}", r"\toprule",
             r"Pass & Conc. & In / out & \multicolumn{2}{c}{Output tok/s} & Ratio & \multicolumn{2}{c}{TTFT p50, ms} & TTFT mean, ms & "
             r"\multicolumn{2}{c}{TPOT p50, ms} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){7-8}\cmidrule(lr){10-11}",
             r" & & tokens & P/D & coloc. & (derived) & P/D & coloc. & P/D $-$ coloc. & P/D & coloc. \\", r"\midrule"]
    group = None
    for name in ps.sort_values("order").index:
        if name not in co.index:
            continue
        x, y = ps.loc[name], co.loc[name]
        g = (re.sub(r"\d+k?$", "", name.split("-")[0]), x.concurrency if name.startswith("ctx") else None)
        if group is not None and g != group:
            lines.append(r"\midrule")
        group = g
        lines.append(f"{name} & {int(x.concurrency)} & {int(x.input_len):,} / {int(x.output_len):,} & {x.output_throughput:,.1f} & "
                     f"{y.output_throughput:,.1f} & {x.output_throughput / y.output_throughput:.2f} & {x.ttft_ms_p50:,.0f} & {y.ttft_ms_p50:,.0f} & "
                     f"{x.ttft_ms_mean - y.ttft_ms_mean:+,.0f} & {x.tpot_ms_p50:.1f} & {y.tpot_ms_p50:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def kimi_xn_four_passes_table(t) -> str:
    """Kimi K3 inference on 4 nodes, pass for pass: TP8 x PP4 (K3-INF-02) against EP32 (K3-XN-03), one run each; output tokens/s, TTFT
    and TPOT medians, the ratio of output tokens/s (derived), and EP32's against EP16 on 2 nodes with all-gather (K3-XN-01, derived)."""
    s, base = _kimi_xn4_servers(t), _kimi_four_node_passes(t)
    if "ep32" not in s or base.empty:
        return ""
    ps = s["ep32"][1]
    ep16 = _kimi_xn_servers(t).get("Ag", (None, pd.DataFrame()))[1]
    lines = [r"\begin{tabular}{lrr" + "r" * 8 + "}", r"\toprule",
             r"Pass & Conc. & In / out & \multicolumn{2}{c}{Output tok/s} & \multicolumn{2}{c}{TTFT p50, ms} & \multicolumn{2}{c}{TPOT p50, ms} & "
             r"\multicolumn{2}{c}{Output ratio (derived)} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
             r" & & tokens & PP4 & EP32 & PP4 & EP32 & PP4 & EP32 & EP32 / PP4 & EP32 / EP16 \\", r"\midrule"]
    group = None
    for name in base.sort_values("order").index:
        x = base.loc[name]
        g = (re.sub(r"\d+k?$", "", name.split("-")[0]), x.concurrency if name.startswith("ctx") else None)
        if group is not None and g != group:
            lines.append(r"\midrule")
        group = g
        head = f"{name} & {int(x.concurrency)} & {int(x.input_len):,} / {int(x.output_len):,} & {x.output_throughput:,.1f} & "
        if name not in ps.index:
            lines.append(head + r"not run & " + f"{x.ttft_ms_p50:,.0f} & & {x.tpot_ms_p50:.1f} & & & " + r"\\")
            continue
        y = ps.loc[name]
        vs16 = f"{y.output_throughput / ep16.output_throughput[name]:.2f}" if len(ep16) and name in ep16.index else "--"
        lines.append(head + f"{y.output_throughput:,.1f} & {x.ttft_ms_p50:,.0f} & {y.ttft_ms_p50:,.0f} & {x.tpot_ms_p50:.1f} & {y.tpot_ms_p50:.1f} & "
                     f"{y.output_throughput / x.output_throughput:.2f} & {vs16} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def kimi_passes_table(t, nodes: int) -> str:
    """Kimi K3 inference: the study's passes on the server of that many nodes (throwaway warm-up left out), one row each; power
    and J per token are estimates."""
    p = t.get("kimi_passes", pd.DataFrame())
    if p.empty or not (p.nodes == nodes).any():
        return ""
    p = p[(p.nodes == nodes) & ~p.throwaway.astype(bool)].rename(columns={"pass": "name"})  # "pass" is no attribute name
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"Pass & Conc. & In / out & Output tok/s (/GPU, /node) & Total tok/s (/GPU, /node) & TTFT p50 / p99 & TPOT p50 / p99 & ITL p50 / p99 & Requests ok / failed & GPU W, J / output tok \\",
             rf" & & tokens & & & ms & ms & ms & & est.: main node $\times$\,{int(p.nodes.max())} \\", r"\midrule"]

    def ms(v):
        return f"{v:.2f}" if v < 100 else f"{v:,.0f}"

    group = None
    for x in p.itertuples():
        # a rule between kinds: conc, ctx at each concurrency, decodelong
        g = (re.sub(r"\d+k?$", "", x.name.split("-")[0]), x.concurrency if x.name.startswith("ctx") else None)
        if group is not None and g != group:
            lines.append(r"\midrule")
        group = g
        lines.append(f"{x.name} & {int(x.concurrency)} & {int(x.input_len):,} / {int(x.output_len):,} & "
                     f"{x.output_throughput:,.1f} ({x.output_throughput_gpu:.1f}, {x.output_throughput_node:,.0f}) & "
                     f"{x.total_throughput:,.0f} ({x.total_throughput_gpu:,.0f}, {x.total_throughput_node:,.0f}) & "
                     f"{x.ttft_ms_p50:,.0f} / {x.ttft_ms_p99:,.0f} & {ms(x.tpot_ms_p50)} / {ms(x.tpot_ms_p99)} & {ms(x.itl_ms_p50)} / {ms(x.itl_ms_p99)} & "
                     f"{int(x.completed):,} / {int(x.failed)} & {x.power_w_est:,.0f}, {x.j_per_output_tok_est:.1f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


KIMI_TASKS = (("gsm8k", "GSM8K"), ("arc_challenge", "ARC-Challenge"), ("gpqa_diamond", "GPQA diamond"), ("mmlu_pro", "MMLU-Pro"))


def kimi_mi355x_table(t) -> str:
    """Kimi K3 inference against MI355X, pass for pass: a template until the MI355X study's numbers are in. B200 is the 2-node
    TP8 x PP2 server (the run TriKimi* reads); the MI355X and ratio cells are placeholders, not data."""
    p = _kimi_two_node_passes(t)
    if p.empty:
        return ""
    todo = r"\textit{to come}"
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Pass & Conc. & In / out & B200, ours & MI355X & MI355X / B200 \\",
             r" & & tokens & \multicolumn{2}{c}{output tok/s per GPU} & \\", r"\midrule"]
    group = None
    for name, x in p.sort_values("order").iterrows():
        g = (re.sub(r"\d+k?$", "", name.split("-")[0]), x.concurrency if name.startswith("ctx") else None)
        if group is not None and g != group:
            lines.append(r"\midrule")
        group = g
        lines.append(f"{name} & {int(x.concurrency)} & {int(x.input_len):,} / {int(x.output_len):,} & {x.output_throughput_gpu:,.1f} & {todo} & {todo} " + r"\\")
    acc = t.get("kimi_accuracy", pd.DataFrame())
    if not acc.empty:
        a = acc[acc.job.astype(str) == str(p.job.iloc[0])].set_index("task")
        rows = [f"{label} ({int(a.n[task]):,}) & & & {100 * a.value[task]:.1f} $\\pm$ {100 * a.stderr[task]:.1f} & {todo} & {todo} " + r"\\"
                for task, label in KIMI_TASKS if task in a.index]
        if rows:
            lines += [r"\midrule", r"Accuracy, \% (questions) & & & B200, ours & MI355X & MI355X $-$ B200, points \\", r"\midrule", *rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def kimi_runs_table(t) -> str:
    """Kimi K3 inference runs in the report: entry, job, nodes, layout, the server's sequence cap, start and end, Slurm state."""
    r = t.get("kimi_runs", pd.DataFrame())
    if r.empty:
        return ""
    # EP16 and EP32 (K3-XN-01, -03): one engine per node, so their sequence cap is per engine; P/D and colocated (K3-XN-02): two EP16
    # instances of 2 nodes each
    xn = [x for x in (t.get("kimi_xn_runs", pd.DataFrame()), t.get("kimi_xn4_runs", pd.DataFrame())) if len(x)]
    r = pd.concat([r, *xn], ignore_index=True)
    labels = {backend: label for _, backend, label in XN_BACKENDS}
    roles = {"pd": "P/D", "colocated": "colocated"}
    lines = [r"\begin{tabular}{llllrlll}", r"\toprule", r"Entry & Job & Nodes & Layout & Max seqs & Started & Ended & State \\", r"\midrule"]
    for x in r.sort_values(["start", "job"]).itertuples():
        hosts = str(x.hosts).replace(";", ", ")
        dp = int(x.dp) if "dp" in r and pd.notna(x.dp) else 1
        layout = (f"DP{dp}\\,$\\times$\\,TP{int(x.tp)}, EP{dp * int(x.tp)}, {labels.get(x.all2all, _esc(x.all2all))}" if dp > 1
                  else f"TP{int(x.tp)}\\,$\\times$\\,PP{int(x.pp)}")
        if "role" in r and pd.notna(x.role):
            layout = f"{int(x.instances)}\\,$\\times$\\,({layout}), {roles.get(x.role, _esc(x.role))}"
        lines.append(f"\\texttt{{{_esc(x.entry)}}} & {x.job} & {int(x.nodes)} ({hosts}) & {layout} & {int(x.max_num_seqs):,}{' per engine' if dp > 1 else ''} & "
                     f"{str(x.start).replace('T', ' ')[:16]} & "
                     f"{str(x.end).replace('T', ' ')[:16]} & {x.job_state} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _kimi_train_numbers(t) -> dict[str, tuple]:
    """Kimi K3 pretraining at 48 nodes, TriKt*: the completed run with the most layers (full size), and TriKtSmall* for a completed run
    with fewer. Step times on milabench's timer (measured steps); memory max over each stage's ranks; MFU with TorchTitan's FLOPs per token."""
    r, st, sg = (t.get(k, pd.DataFrame()) for k in ("kimi_train_runs", "kimi_train_steps", "kimi_train_stages"))
    out = {}
    if r.empty:
        return out
    r = r.assign(job=r.job.astype(str))
    out.update({"TriKtRuns": (len(r), "d", False), "TriKtFailed": (int((r.job_state != "COMPLETED").sum()), "d", False)})
    done = r[(r.job_state == "COMPLETED") & r.step_median.notna()].sort_values("layers") if "step_median" in r else r.iloc[:0]
    if done.empty:
        return out
    x = done.iloc[-1]
    steps = st[st.job.astype(str) == x.job].sort_values("step") if len(st) else st
    meas = steps[steps.measured.astype(bool)] if len(steps) else steps
    out.update({"TriKtLayers": (int(x.layers), "d", False), "TriKtNodes": (int(x.nodes), "d", False), "TriKtGpus": (int(x.gpus), "d", False),
                "TriKtPp": (int(x.pp), "d", False), "TriKtEp": (int(x.ep), "d", False), "TriKtMicro": (int(x.microbatches), "d", False),
                "TriKtSeq": (int(x.seq_len), ",d", False), "TriKtTokens": (int(x.global_tokens), ",d", False),
                "TriKtParams": (x.params_total / 1e12, ".2f", False), "TriKtActive": (x.params_active / 1e9, ".0f", False),
                "TriKtStepOne": (x.step1_min, ".1f", False), "TriKtStepTwo": (x.step2_s, ",.0f", False),
                "TriKtMeasured": (int(x.steps_measured), "d", False), "TriKtMeasuredFrom": (int(x.measured_from), "d", False),
                "TriKtStep": (x.step_median, ".2f", False), "TriKtStepPninety": (x.step_p90, ".2f", False), "TriKtStepMax": (x.step_max, ".2f", False),
                "TriKtTok": (x.tok_s_gpu, ".1f", False), "TriKtTokLo": (x.tok_s_gpu_lo, ".0f", False), "TriKtTokHi": (x.tok_s_gpu_hi, ".0f", False),
                "TriKtMfu": (x.mfu_pct, ".2f", False), "TriKtMfuSixN": (x.mfu_6n_pct, ".2f", False),
                "TriKtFlops": (rf"${x.flops_per_token / 1e11:.2f}\times10^{{11}}$", "", False), "TriKtFetch": (x.data_wait_s, ".1f", False),
                # 1F1B's idle share with p stages and m micro-batches, (p - 1) / (m + p - 1): a formula, not a measurement
                "TriKtBubblePct": (100 * (x.pp - 1) / (x.microbatches + x.pp - 1), ".0f", False),
                "TriKtMemLimit": (x.capacity_gib, ".1f", False), "TriKtLossFirst": (x.loss_first, ".2f", False), "TriKtLossLast": (x.loss_last, ".2f", False)})
    if len(meas):
        out["TriKtStepFirst"], out["TriKtStepLast"] = (meas.step_s.iloc[0], ".1f", False), (meas.step_s.iloc[-1], ".1f", False)
    g = sg[sg.job.astype(str) == x.job].set_index("stage") if len(sg) else sg
    if len(g):
        worst = g.reserved_gib.idxmax()
        last, zero = g.index.max(), g.index.min()
        out.update({"TriKtMemWorst": (g.reserved_gib[worst], ".1f", False), "TriKtMemWorstAlloc": (g.allocated_gib[worst], ".1f", False),
                    "TriKtMemWorstStage": (int(worst), "d", False), "TriKtMemWorstHost": (g.host[worst], "", False),
                    "TriKtMemLast": (g.reserved_gib[last], ".1f", False), "TriKtMemLastAlloc": (g.allocated_gib[last], ".1f", False),
                    "TriKtMemZero": (g.reserved_gib[zero], ".1f", False), "TriKtMemZeroAlloc": (g.allocated_gib[zero], ".1f", False),
                    "TriKtHeadroom": (x.capacity_gib - g.reserved_gib[worst], ".1f", False)})
        # Stages that start an Attention Residuals block (12 layers, 2 per stage: every 6th stage, while stages hold 2 layers) against
        # the other 2-layer stages after stage 0. Which layers a stage holds is inferred from the split (93 = 45 x 2 + 3 x 1).
        two = g[(g.index > zero) & (g.index < x.layers - x.pp)]
        edge = two[two.index % 6 == 0]
        mid = two[two.index % 6 != 0]
        if len(edge) and len(mid):
            out.update({"TriKtMemEdgeLo": (edge.reserved_gib.min(), ".1f", False), "TriKtMemEdgeHi": (edge.reserved_gib.max(), ".1f", False),
                        "TriKtMemMidLo": (mid.reserved_gib.min(), ".1f", False), "TriKtMemMidHi": (mid.reserved_gib.max(), ".1f", False)})
        if "low_power_pct" in g and g.low_power_pct.notna().any():
            out["TriKtLowLo"], out["TriKtLowHi"] = (g.low_power_pct.min(), ".0f", False), (g.low_power_pct.max(), ".0f", False)
        sw = g.sweep_s.dropna() if "sweep_s" in g else pd.Series(dtype=float)
        split = 13  # kimi_train.SWEEP_SPLIT: slower above it
        if len(sw) and zero in sw and split in sw:
            first = sw.idxmin()  # the first stage step 2's backward reaches
            out.update({"TriKtSweepFirst": (int(first), "d", False), "TriKtSweepSplit": (split, "d", False),
                        "TriKtSweepHi": ((sw[split] - sw[first]) / (first - split), ".1f", False), "TriKtSweepLo": ((sw[zero] - sw[split]) / split, ".1f", False),
                        "TriKtSweepTotal": (sw[zero] - sw[first], ",.0f", False)})
    small = done[done.layers < x.layers]
    if len(small):
        y = small.iloc[-1]
        ys = st[(st.job.astype(str) == y.job) & st.measured.astype(bool)] if len(st) else st
        out.update({"TriKtSmallLayers": (int(y.layers), "d", False), "TriKtSmallStep": (y.step_median, ".2f", False),
                    "TriKtSmallTok": (y.tok_s_gpu, ".0f", False), "TriKtSmallMfu": (y.mfu_pct, ".1f", False),
                    "TriKtSmallStepTwo": (y.step2_s, ",.0f", False), "TriKtSmallLossLast": (y.loss_last, ".2f", False)})
        if len(ys):
            out["TriKtSmallStepLo"], out["TriKtSmallStepHi"] = (ys.step_s.min(), ".0f", False), (ys.step_s.max(), ".0f", False)
        h = sg[sg.job.astype(str) == y.job].set_index("stage") if len(sg) else sg
        if len(h):
            out["TriKtSmallMemLast"], out["TriKtSmallMemZero"] = (h.reserved_gib[h.index.max()], ".1f", False), (h.reserved_gib[h.index.min()], ".1f", False)
            sw = h.sweep_s.dropna() if "sweep_s" in h else pd.Series(dtype=float)
            if len(sw) and h.index.min() in sw:
                # 1-layer stages pause about 3 s each, too short for 1 s telemetry stage by stage: only the two ends are used
                first = sw.idxmin()
                out["TriKtSmallSweep"] = ((sw[h.index.min()] - sw[first]) / (first - h.index.min()), ".1f", False)
    return out


def kimi_train_runs_table(t) -> str:
    """Kimi K3 pretraining at 48 nodes: every job, its state, and why it stopped (RUNS.md) or what sets it apart."""
    r = t.get("kimi_train_runs", pd.DataFrame())
    if r.empty:
        return ""

    def num(v, fmt):
        return format(v, fmt) if pd.notna(v) else "--"

    lines = [r"\begin{tabular}{llrlllrrrrl}", r"\toprule",
             r"Request & Job & Layers & Grad.\ red. & Detach & State & Step 1 & Step median / p90 & Tok/s & MFU & Note \\",
             r" & & & & losses & & min & s & per GPU & \% & \\", r"\midrule"]
    for x in r.sort_values("start").itertuples():
        step = f"{x.step_median:.2f} / {x.step_p90:.2f}" if pd.notna(getattr(x, "step_median", None)) else "--"
        lines.append(f"{x.request} & {x.job} & {int(x.layers)} & {str(x.grad_reduce).upper()} & "
                     f"{'yes' if x.detach_losses else 'no'} & {x.job_state} & {num(getattr(x, 'step1_min', None), '.1f')} & {step} & "
                     f"{num(getattr(x, 'tok_s_gpu', None), '.1f')} & {num(getattr(x, 'mfu_pct', None), '.2f')} & "
                     + r"\parbox[t]{5cm}{\raggedright " + _esc(x.note) + r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _glm_rl_numbers(t) -> dict[str, tuple]:
    """GLM-5.3 GRPO with slime, TriGr*: the completed run without routing replay (GLM-RL-03); TriGrRp*: the one with it (GLM-RL-04),
    against it. Per-step numbers are slime's own timers and metrics (glm_rl.py); steps 1 and 2 are the measured steps (One, Two), step 0
    the warm-up. The bubble is computed, (p - 1) / (m + p - 1), not measured."""
    r, st, h, dc = (t.get(k, pd.DataFrame()) for k in ("glm_rl_runs", "glm_rl_steps", "glm_rl_hosts", "glm_rl_decode"))
    out = {}
    done = r[r.job_state == "COMPLETED"] if len(r) else r
    coloc = done[done.colocate.astype(bool)] if len(done) else done
    fallback = ~coloc.default_kernels.astype(bool) if "default_kernels" in coloc else True
    base = coloc[~coloc.routing_replay.astype(bool) & fallback] if len(coloc) else coloc
    if base.empty or st.empty:
        return out
    x = base.iloc[-1]
    job = str(x.job)
    s = st[st.job.astype(str) == job].set_index("step")
    out.update({"TriGrNodes": (int(x.nodes), "d", False), "TriGrGpus": (int(x.gpus), "d", False), "TriGrLayers": (int(x.layers), "d", False),
                "TriGrPp": (int(x.pp), "d", False), "TriGrEp": (int(x.ep), "d", False), "TriGrDp": (int(x.dp), "d", False),
                "TriGrFirstLayers": (int(x.first_stage_layers), "d", False), "TriGrMidLayers": (x.middle_stage_layers, "g", False),
                "TriGrLastLayers": (int(x.last_stage_layers), "d", False), "TriGrMeanLayers": (x.layers / x.pp, "g", False),
                "TriGrLastRatio": (x.last_stage_layers / (x.layers / x.pp), ".1f", False),
                "TriGrEngines": (int(x.engines), "d", False), "TriGrEngineGpus": (int(x.engine_gpus), "d", False), "TriGrEngineEp": (int(x.engine_ep), "d", False),
                "TriGrPrompts": (int(x.prompts), "d", False), "TriGrSamplesPer": (int(x.samples_per_prompt), "d", False), "TriGrSamples": (int(x.samples), ",d", False),
                "TriGrMaxResp": (int(x.max_response), ",d", False), "TriGrMemFraction": (float(x.mem_fraction), "g", False),
                "TriGrSteps": (int(x.steps_logged), "d", False), "TriGrStartup": (x.startup_min, ".1f", False), "TriGrEnginesUp": (x.engines_up_min, ".1f", False),
                "TriGrSyncFinal": (x.get("sync_after_last_s"), ".1f", False)})
    if 0 in s.index:
        w = s.loc[0]
        out.update({"TriGrWarmRollout": (w.rollout_s, ",.1f", False), "TriGrWarmTrain": (w.train_s, ",.1f", False), "TriGrWarmStep": (w.step_s, ",.1f", False),
                    "TriGrWarmStepMin": (w.step_s / 60, ".1f", False), "TriGrWarmTrainTok": (w.train_tok_s, ",.0f", False),
                    "TriGrRewardWarm": (w.raw_reward, ".3f", False),
                    "TriGrCompilesAfter": (int(s.loc[s.index > 0, "compiles"].sum()), "d", False)})
    for k, n in ((1, "One"), (2, "Two")):
        if k not in s.index:
            continue
        y = s.loc[k]
        out.update({f"TriGrRollout{n}": (y.rollout_s, ",.1f", False), f"TriGrRolloutMin{n}": (y.rollout_s / 60, ".1f", False),
                    f"TriGrTrain{n}": (y.train_s, ".1f", False), f"TriGrSync{n}": (y.sync_s, ".1f", False), f"TriGrRest{n}": (y.rest_s, ".1f", False),
                    f"TriGrStep{n}": (y.step_s, ",.1f", False), f"TriGrStepMin{n}": (y.step_s / 60, ".1f", False),
                    f"TriGrRolloutPct{n}": (y.rollout_pct, ".1f", False), f"TriGrTrainPct{n}": (100 * y.train_s / y.step_s, ".0f", False),
                    f"TriGrTrainTok{n}": (y.train_tok_s, ",.0f", False), f"TriGrTrainTokens{n}": (y.train_tokens / 1e6, ".2f", False),
                    f"TriGrMicro{n}": (int(y.microbatches), "d", False), f"TriGrBubble{n}": (y.bubble_pct, ".0f", False),
                    f"TriGrRollTok{n}": (y.rollout_tok_s_gpu, ".2f", False), f"TriGrRespMean{n}": (y.resp_mean, ",.0f", False),
                    f"TriGrRespMed{n}": (y.resp_median, ",.0f", False), f"TriGrTrunc{n}": (y.truncated_pct, ".1f", False),
                    f"TriGrReward{n}": (y.raw_reward, ".3f", False), f"TriGrLatMed{n}": (y.latency_median_s, ".0f", False),
                    f"TriGrLatMax{n}": (y.latency_max_s, ",.0f", False), f"TriGrDecodeReq{n}": (y.decode_tok_s_median, ".2f", False)})
    out["TriGrGpuRollout"] = (s.gpu_rollout_gib.max(), ".1f", False)
    if len(dc):
        d = dc[(dc.job.astype(str) == job) & (dc.step > 0) & ~dc.idle.astype(bool)]
        one = d[d.running == 1]  # one sequence per DP-attention rank: the line's throughput is that sequence's speed
        out.update({"TriGrDecodeFast": (one[~one.slow.astype(bool)].tok_s_seq.median(), ".1f", False),
                    "TriGrDecodeSlow": (one[one.slow.astype(bool)].tok_s_seq.median(), ".1f", False)})
        first = d[d.slow.astype(bool)].groupby("step").minute.min()
        engines = d[d.slow.astype(bool)].groupby("step").engine.nunique()  # engines with any slower line (slime's placement, glm_rl.py)
        for k, n in ((1, "One"), (2, "Two")):
            if k in first:
                out[f"TriGrSlowFirst{n}"] = (first[k], ".1f", False)
                out[f"TriGrSlowEngines{n}"] = (int(engines[k]), "d", False)
    hh = h[h.job.astype(str) == job] if len(h) else h
    if len(hh):
        head = str(x.hosts).split(";")[0]  # the job's first node: Ray's head, slime's driver and rollout manager
        last = hh[hh.stage == x.pp - 1]
        other = hh[(hh.stage != x.pp - 1) & (hh.host != head)]
        if (hh.host == head).any():
            out["TriGrHostFirst"] = (hh[hh.host == head].host_used_gib.max(), ".0f", False)
        out.update({"TriGrHostLastLo": (last.host_used_gib.min(), ",.1f", False), "TriGrHostLastHi": (last.host_used_gib.max(), ",.1f", False),
                    "TriGrHostLastTb": (last.host_used_gib.max() * 2**30 / 1e12, ".2f", False),
                    "TriGrHostOtherHi": (other.host_used_gib.max(), ".0f", False), "TriGrHostTotal": (hh.host_total_gib.max(), ",.0f", False),
                    "TriGrGpuCap": (hh.gpu_total_gib.max(), ".1f", False)})
    replay = coloc[coloc.routing_replay.astype(bool)]
    kernels = coloc[coloc.default_kernels.astype(bool)] if "default_kernels" in coloc else coloc.iloc[:0]
    if len(kernels):
        out.update(_glm_rl_kernel_numbers(x, kernels.iloc[-1], st, dc))
    out.update(_glm_rl_replay_numbers(x, replay.iloc[-1] if len(replay) else None, st, h, t.get("glm_rl_memory", pd.DataFrame())))
    dis = done[~done.colocate.astype(bool)]
    if len(dis):
        out.update(_glm_rl_disagg_numbers(x, dis.iloc[-1], st, t.get("glm_rl_memory", pd.DataFrame())))
    return out


def _glm_rl_kernel_numbers(x, z, st, dc) -> dict[str, tuple]:
    """GLM-RL-06 (z), colocated on B200's default SGLang rollout kernels with max running requests capped, against GLM-RL-03 (x, the
    fallback kernels), steps 1 and 2. Decode modes: the engines' lines with one sequence per rank (a Ray-deduplicated sample), as for
    GLM-RL-03. The last request's engine: the engine of the rollout's last decode line; its switch: its first slower line (glm_rl.SLOW_DECODE,
    as for GLM-RL-03), in minutes into the rollout; a line caught mid-switch, above the threshold, does not count."""
    s = st[st.job.astype(str) == str(z.job)].set_index("step")
    b = st[st.job.astype(str) == str(x.job)].set_index("step")
    out = {"TriGrKnCap": (int(z.max_running_requests), ",d", False), "TriGrKnCapRank": (int(z.max_running_requests // z.engine_dp), "d", False),
           "TriGrKnEnginesUp": (z.engines_up_min, ".1f", False), "TriGrKnStartup": (z.startup_min, ".1f", False),
           "TriGrKnGraphLo": (z.graph_s_lo, ".1f", False), "TriGrKnGraphHi": (z.graph_s_hi, ".1f", False),
           "TriGrKnGraphRefLo": (x.graph_s_lo, ".1f", False), "TriGrKnGraphRefHi": (x.graph_s_hi, ".1f", False),
           "TriGrKnGpuRollout": (s.gpu_rollout_gib.max(), ".1f", False)}
    for k, n in ((1, "One"), (2, "Two")):
        if k not in s.index or k not in b.index:
            continue
        y, w = s.loc[k], b.loc[k]
        out.update({f"TriGrKnRollout{n}": (y.rollout_s, ",.1f", False), f"TriGrKnTrain{n}": (y.train_s, ".1f", False),
                    f"TriGrKnSync{n}": (y.sync_s, ".1f", False), f"TriGrKnStep{n}": (y.step_s, ",.1f", False),
                    f"TriGrKnDecodeReq{n}": (y.decode_tok_s_median, ".2f", False),
                    f"TriGrKnDecodeGain{n}": (100 * (y.decode_tok_s_median / w.decode_tok_s_median - 1), ".0f", False),
                    f"TriGrKnLatMax{n}": (y.latency_max_s, ",.0f", False), f"TriGrKnTisAbs{n}": (y.tis_abs, ".4f", False),
                    f"TriGrKnTisClip{n}": (y.tis_clipfrac, ".4f", False), f"TriGrKnTisRatio{n}": (y.tis_abs / w.tis_abs, ".1f", False),
                    f"TriGrKnClipRatio{n}": (y.tis_clipfrac / w.tis_clipfrac, ".1f", False), f"TriGrKnReward{n}": (y.raw_reward, ".3f", False)})
    if len(dc):
        d = dc[(dc.job.astype(str) == str(z.job)) & ~dc.idle.astype(bool)]
        if len(d):
            out["TriGrKnRunningMax"] = (int(d.running.max()), "d", False)
            one = d[(d.step > 0) & (d.running == 1)]
            out.update({"TriGrKnDecodeFast": (one[~one.slow.astype(bool)].tok_s_seq.median(), ".1f", False),
                        "TriGrKnDecodeSlow": (one[one.slow.astype(bool)].tok_s_seq.median(), ".1f", False)})
        for job, ref in ((str(z.job), ""), (str(x.job), "Ref")):
            g = dc[(dc.job.astype(str) == job) & ~dc.idle.astype(bool) & (dc.step > 0)]
            for k, n in ((1, "One"), (2, "Two")):
                r = g[g.step == k]
                if r.empty:
                    continue
                last = r.loc[r.time.idxmax(), "engine"]
                switch = r[(r.engine == last) & r.slow.astype(bool)].minute
                if len(switch):
                    out[f"TriGrKnSwitch{ref}{n}"] = (switch.min(), ".1f", False)
    return out


def _glm_rl_disagg_numbers(x, z, st, mem) -> dict[str, tuple]:
    """GLM-RL-05 (z), disaggregated: the trainer on its own nodes, the engines on the others, against GLM-RL-03 (x, colocated, without
    routing replay). Training per GPU: the means of steps 1 and 2 of slime's tokens/s over each run's trainer GPUs. Host memory: slime's
    snapshots (not peaks) from step 0's training on, each node's highest; the first node is an engine node here, and its host
    figure is milabench's continuous sample."""
    s = st[st.job.astype(str) == str(z.job)].set_index("step")
    b = st[st.job.astype(str) == str(x.job)].set_index("step")
    out = {"TriGrDgTrainerNodes": (int(z.trainer_nodes), "d", False), "TriGrDgEngineNodes": (int(z.nodes - z.trainer_nodes), "d", False),
           "TriGrDgTrainerGpus": (int(z.gpus), "d", False), "TriGrDgEngineGpus": (int(z.rollout_gpus), "d", False),
           "TriGrDgEngines": (int(z.engines), "d", False), "TriGrDgEp": (int(z.ep), "d", False), "TriGrDgDp": (int(z.dp), "d", False),
           "TriGrDgStartup": (z.startup_min, ".1f", False), "TriGrDgHostEngine": (z.head_host_gib, ".0f", False),
           "TriGrDgGpuRollout": (s.gpu_rollout_gib.max(), ".1f", False),
           "TriGrDgTopkAll": (int(s.topk_errors.sum()), "d", False)}
    if 0 in s.index:
        w = s.loc[0]
        out.update({"TriGrDgWarmRollout": (w.rollout_s, ",.1f", False), "TriGrDgWarmTrain": (w.train_s, ",.1f", False),
                    "TriGrDgWarmStep": (w.step_s, ",.1f", False), "TriGrDgSyncFirst": (w.sync_s, ".1f", False),
                    "TriGrDgTopkWarm": (int(w.topk_errors), "d", False)})
    for k, n in ((1, "One"), (2, "Two")):
        if k not in s.index:
            continue
        y = s.loc[k]
        out.update({f"TriGrDgRollout{n}": (y.rollout_s, ",.1f", False), f"TriGrDgTrain{n}": (y.train_s, ".1f", False),
                    f"TriGrDgSync{n}": (y.sync_s, ".1f", False), f"TriGrDgRest{n}": (y.rest_s, ".1f", False),
                    f"TriGrDgStep{n}": (y.step_s, ",.1f", False), f"TriGrDgStepMin{n}": (y.step_s / 60, ".1f", False),
                    f"TriGrDgTrainTok{n}": (y.train_tok_s, ",.0f", False), f"TriGrDgMicro{n}": (int(y.microbatches), "d", False),
                    f"TriGrDgBubble{n}": (y.bubble_pct, ".0f", False), f"TriGrDgRollTok{n}": (y.rollout_tok_s_gpu, ".2f", False),
                    f"TriGrDgDecodeReq{n}": (y.decode_tok_s_median, ".2f", False), f"TriGrDgTisAbs{n}": (y.tis_abs, ".4f", False),
                    f"TriGrDgTisClip{n}": (y.tis_clipfrac, ".4f", False), f"TriGrDgReward{n}": (y.raw_reward, ".3f", False),
                    f"TriGrDgTrunc{n}": (y.truncated_pct, ".1f", False), f"TriGrDgTopk{n}": (int(y.topk_errors), "d", False)})
    if 1 in s.index:
        y = s.loc[1]
        out.update({"TriGrDgRolloutMinOne": (y.rollout_s / 60, ".1f", False), "TriGrDgLatMaxOne": (y.latency_max_s, ",.0f", False)})
        if pd.notna(y.get("topk_last_min")):
            out["TriGrDgRetryMinOne"] = (y.topk_last_min, ".1f", False)
        if pd.notna(y.get("last_group_min")):
            out["TriGrDgLastGroupMinOne"] = (y.last_group_min, ".1f", False)
    both = [k for k in (1, 2) if k in s.index and k in b.index]
    if both:
        out["TriGrDgTrainPerGpu"] = ((s.loc[both, "train_tok_s"].mean() / z.gpus) / (b.loc[both, "train_tok_s"].mean() / x.gpus), ".2f", False)
    m = mem[(mem.job.astype(str) == str(z.job)) & mem.after_train0.astype(bool)] if len(mem) else mem
    if len(m):
        node = m.groupby(["host", "stage"]).host_used_gib.max().reset_index()  # each node's highest snapshot
        last, first = node[node.stage == z.pp - 1], node[node.stage == 0]
        mid = node[(node.stage > 0) & (node.stage < z.pp - 1)]
        out.update({"TriGrDgHostLastLo": (last.host_used_gib.min(), ",.1f", False), "TriGrDgHostLastHi": (last.host_used_gib.max(), ",.1f", False),
                    "TriGrDgHostLastTb": (last.host_used_gib.max() * 2**30 / 1e12, ".2f", False),
                    "TriGrDgHostFirstLo": (first.host_used_gib.min(), ",.0f", False), "TriGrDgHostFirstHi": (first.host_used_gib.max(), ",.0f", False),
                    "TriGrDgHostMidLo": (mid.host_used_gib.min(), ".0f", False), "TriGrDgHostMidHi": (mid.host_used_gib.max(), ".0f", False),
                    "TriGrDgHostMidNodes": (len(mid), "d", False),
                    "TriGrDgHostMidAll": (int(z.trainer_nodes * (z.pp - 2) / z.pp), "d", False)})
        trainer = mem[mem.job.astype(str) == str(z.job)]
        if "gpu_used_gib" in trainer:
            out["TriGrDgGpuTrainer"] = (trainer.gpu_used_gib.max(), ".1f", False)
    return out


def _glm_rl_replay_numbers(x, y, st, h, mem) -> dict[str, tuple]:
    """GLM-RL-03's TIS metrics, data preprocessing and spreads (x), and GLM-RL-04's (y, with routing replay) against them, steps 1 and 2.
    Host memory: slime's snapshots are Ray-deduplicated, not peaks, so a node is compared only where both runs logged it in the same
    phase of the same step, from step 0's training on (glm_rl_memory), outside the first node (Ray's head and slime's driver)."""
    out = {"TriGrTisClipLow": (float(x.tis_clip_low), "g", False), "TriGrTisClipHigh": (float(x.tis_clip), "g", False),
           "TriGrHeadHost": (x.head_host_gib, ".1f", False)}
    s = st[st.job.astype(str) == str(x.job)].set_index("step")
    two = s.loc[[k for k in (1, 2) if k in s.index]]
    for k, n in ((1, "One"), (2, "Two")):
        if k in s.index:
            out.update({f"TriGrTisAbs{n}": (s.loc[k, "tis_abs"], ".4f", False), f"TriGrTisClip{n}": (s.loc[k, "tis_clipfrac"], ".4f", False),
                        f"TriGrPrep{n}": (s.loc[k, "data_preprocess_s"], ".2f", False)})
    if len(two) == 2:
        out.update({"TriGrStepMean": (two.step_s.mean(), ",.1f", False), "TriGrRolloutSpread": (two.rollout_s.max() - two.rollout_s.min(), ".1f", False)})
    if y is None:
        return out
    q = st[st.job.astype(str) == str(y.job)].set_index("step")
    for k, n in ((1, "One"), (2, "Two")):
        if k not in q.index or k not in s.index:
            continue
        a, b = q.loc[k], s.loc[k]
        diff = f"{a.step_s - b.step_s:+,.1f}".replace("-", "$-$")
        out.update({f"TriGrRpRollout{n}": (a.rollout_s, ",.1f", False), f"TriGrRpTrain{n}": (a.train_s, ".1f", False),
                    f"TriGrRpSync{n}": (a.sync_s, ".1f", False), f"TriGrRpStep{n}": (a.step_s, ",.1f", False), f"TriGrRpStepDiff{n}": (diff, "", False),
                    f"TriGrRpPrep{n}": (a.data_preprocess_s, ".2f", False), f"TriGrRpTisAbs{n}": (a.tis_abs, ".4f", False),
                    f"TriGrRpTisClip{n}": (a.tis_clipfrac, ".5f", False), f"TriGrRpTisRatio{n}": (b.tis_abs / a.tis_abs, ".1f", False),
                    f"TriGrRpClipRatio{n}": (b.tis_clipfrac / a.tis_clipfrac, ".0f", False), f"TriGrRpReward{n}": (a.raw_reward, ".3f", False),
                    f"TriGrRpTrunc{n}": (a.truncated_pct, ".1f", False), f"TriGrRpDecodeReq{n}": (a.decode_tok_s_median, ".2f", False),
                    f"TriGrRpDecodeDrop{n}": (100 * (1 - a.decode_tok_s_median / b.decode_tok_s_median), ".1f", False)})
    qq = q.loc[[k for k in (1, 2) if k in q.index]]
    if len(qq) == 2:
        out.update({"TriGrRpStepMean": (qq.step_s.mean(), ",.1f", False), "TriGrRpRolloutSpread": (qq.rollout_s.max() - qq.rollout_s.min(), ".1f", False)})
    if 2 in q.index:
        both = pd.concat([s, q.drop(index=2)])
        out.update({"TriGrRpGradTwo": (q.loc[2, "grad_norm"], ".3f", False), "TriGrGradOthersLo": (both.grad_norm.min(), ".3f", False),
                    "TriGrGradOthersHi": (both.grad_norm.max(), ".3f", False)})
    out.update({"TriGrRpHeadHost": (y.head_host_gib, ".1f", False), "TriGrRpGpuRollout": (q.gpu_rollout_gib.max(), ".1f", False)})
    if len(mem):
        head = str(x.hosts).split(";")[0]
        m = mem[mem.after_train0.astype(bool) & (mem.host != head) & mem.job.astype(str).isin([str(x.job), str(y.job)])]
        m = m.assign(job=m.job.astype(str))
        same = m.groupby(["host", "phase", "step", "job"]).host_used_gib.max().unstack("job").dropna()
        if str(x.job) in same and str(y.job) in same and len(same):
            d = same[str(y.job)] - same[str(x.job)]
            out.update({"TriGrRpHostLo": (d.min(), ".0f", False), "TriGrRpHostHi": (d.max(), ".0f", False),
                        "TriGrRpHostNodes": (int(d.index.get_level_values("host").nunique()), "d", False)})
    last = h[(h.job.astype(str) == str(y.job)) & (h.stage == y.pp - 1)] if len(h) else h
    if len(last):
        out["TriGrRpHostLastHi"] = (last.host_used_gib.max(), ",.1f", False)
    return out


FP4_RECIPES = (("bf16", "Bf", "BF16"), ("mxfp8", "Mx", "MXFP8"), ("nvfp4", "Nv", "NVFP4"))


def _glm_fp4_numbers(t) -> dict[str, tuple]:
    """GLM-5 C-speed with NVFP4 (GLM-FP4-01, glm_fp4.py), TriFpFour*: step times over steps 3-20 on milabench's timer, against the job's own
    BF16 entry; the campaign's 13-node C-speed run (Camp*) on the same measure, from the campaign's runs and steps tables."""
    r = t.get("glm_fp4_runs", pd.DataFrame())
    out = {}
    done = r[r.job_state == "COMPLETED"] if len(r) else r
    if done.empty or not {"bf16", "mxfp8", "nvfp4"} <= set(done.precision):
        return out
    m = done.set_index("precision")
    nv = m.loc["nvfp4"]
    for key, fmt in (("nodes", "d"), ("gpus", "d"), ("seqs", "d"), ("seq_len", ",d"), ("n_layers", "d"), ("measured", "d")):
        out[{"nodes": "TriFpFourNodes", "gpus": "TriFpFourGpus", "seqs": "TriFpFourSeqs", "seq_len": "TriFpFourSeqLen",
             "n_layers": "TriFpFourLayers", "measured": "TriFpFourMeasured"}[key]] = (int(nv[key]), fmt, False)
    out["TriFpFourBfLast"] = (int(nv.bf16_layers_last), "d", False)
    out["TriFpFourSteps"] = (int(nv.steps_logged), "d", False)
    out["TriFpFourFrom"] = (glm_fp4.FIRST_STEP, "d", False)
    out["TriFpFourLastHost"] = (nv.last_host, "", False)
    if pd.notna(nv.lr) and set(done.lr_schedule) == {"constant"} and done.lr.nunique() == 1:  # the same constant rate in every entry
        e = math.log10(nv.lr)
        out["TriFpFourLr"] = (f"$10^{{{int(round(e))}}}$" if abs(e - round(e)) < 1e-9 else f"{nv.lr:g}", "", False)
    for rec, name, _ in FP4_RECIPES:
        x = m.loc[rec]
        for col, macro, fmt in (("step_mean", "Step", ".2f"), ("step_median", "Med", ".2f"), ("step_sd", "Sd", ".2f"), ("mem_alloc_gib", "Mem", ".0f"),
                                ("loss_last", "Loss", ".3f"), ("tok_s_gpu", "Tok", ".0f"), ("step_start3", "Start", ".1f"), ("step_end3", "End", ".1f"),
                                ("cap_last_pct", "Cap", ".0f")):
            out[f"TriFpFour{macro}{name}"] = (x[col], fmt, False)
    s = m.step_mean
    out["TriFpFourNvBf"] = (s.nvfp4 / s.bf16, ".3f", False)
    out["TriFpFourNvMx"] = (s.nvfp4 / s.mxfp8, ".3f", False)
    out["TriFpFourMxBf"] = (s.mxfp8 / s.bf16, ".3f", False)
    out["TriFpFourNvBfPct"] = (100 * (1 - s.nvfp4 / s.bf16), ".1f", False)
    out["TriFpFourMxBfPct"] = (100 * (1 - s.mxfp8 / s.bf16), ".1f", False)
    out["TriFpFourNvMxDiff"] = (s.nvfp4 - s.mxfp8, "+.2f", False)
    out["TriFpFourNvMxDiffMed"] = (m.step_median.nvfp4 - m.step_median.mxfp8, "+.2f", False)
    out["TriFpFourMemMxNv"] = (m.mem_alloc_gib.mxfp8 - m.mem_alloc_gib.nvfp4, ".0f", False)
    out["TriFpFourMemNvBf"] = (m.mem_alloc_gib.nvfp4 - m.mem_alloc_gib.bf16, ".0f", False)
    out["TriFpFourCapOtherBfLo"] = (m.cap_others_lo.bf16, ".0f", False)
    out["TriFpFourCapOtherBfHi"] = (m.cap_others_hi.bf16, ".0f", False)
    out["TriFpFourCapOtherMx"] = (m.cap_others_hi.mxfp8, ".0f", False)
    out["TriFpFourCapOtherNv"] = (m.cap_others_hi.nvfp4, ".0f", False)
    out["TriFpFourMxBfSpeed"] = (m.tok_s_gpu_measured.mxfp8 / m.tok_s_gpu_measured.bf16, ".2f", False)  # as TriGlmFpMx
    runs, steps = t["runs"], t["steps"]
    if len(runs) and not runs.synthetic.any():
        cs = _completed(runs, "C-speed")
        g = cs[(cs.model == GLM) & (cs.nodes == 13) & cs.precision.isin(["bf16", "mxfp8"])].set_index("precision")
        if {"bf16", "mxfp8"} <= set(g.index):
            # Megatron's steps table counts from the first measured step (step 2); steps 3-20 leave it out, as TriGlmFp*TopMean
            mean = {rec: steps[steps.run_id == g.run_id[rec]].sort_values("step").step_s.iloc[1:].mean() for rec in ("bf16", "mxfp8")}
            out["TriFpFourCampBf"] = (mean["bf16"], ".2f", False)
            out["TriFpFourCampMx"] = (mean["mxfp8"], ".2f", False)
            out["TriFpFourCampMxBf"] = (mean["mxfp8"] / mean["bf16"], ".3f", False)
    return out


# Qwen3 32B C-loss windows, mean loss over the steps: 401-500 and 491-500 are GLM-FP4-02's; 91-100 was chosen after GLM-FP4-02's four runs
# showed an early gap, 241-250 after all nine runs. None was set before the runs it is used on.
QW_WINDOWS = (("Early", 91, 100), ("Mid", 241, 250), ("Late", 401, 500), ("End", 491, 500))
# Where the quantized runs' window means sit above all BF16 runs': windows read from the data after all nine runs (post hoc)
QW_SEP_WINDOWS = ((1, 50), (51, 110), (91, 100), (101, 110))


def _power10(x: float) -> str:
    """3e-4 -> $3\\times10^{-4}$, 1e-6 -> $10^{-6}$."""
    e = math.floor(math.log10(x))
    m = x / 10 ** e
    m = round(m, 6)
    return f"$10^{{{e}}}$" if m == 1 else f"${m:g}\\times10^{{{e}}}$"


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _qw_fp4(t) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Qwen3 32B with NVFP4 (glm_fp4.py, qw_fp4_*): the completed C-speed entries by precision, the completed C-loss runs by job (the key: the
    runs' repetition numbers do not tell them apart; ``recipe`` is the precision), and the loss per step, a column per job."""
    r = t.get("qw_fp4_runs", pd.DataFrame())
    if r.empty:
        return r, r, r
    done = r[r.job_state == "COMPLETED"].assign(job=lambda d: d.job.astype(str))
    speed, loss = done[done.experiment == "C-speed"], done[done.experiment == "C-loss"].sort_values("job")
    if speed.precision.duplicated().any() or loss.job.duplicated().any():
        raise SystemExit("[report] qw_fp4_runs: two completed C-speed entries with the same recipe, or two C-loss rows for one job")
    l = t.get("qw_fp4_loss", pd.DataFrame())
    curves = (l.assign(job=l.job.astype(str)).pipe(lambda d: d[d.job.isin(set(loss.job))]).pivot(index="step", columns="job", values="loss")
              if len(l) and len(loss) else pd.DataFrame())
    return speed.set_index("precision"), loss.set_index("job"), curves


def _qw_windows(lo: pd.DataFrame, curves: pd.DataFrame) -> dict[tuple[int, int], dict[str, pd.Series]]:
    """Per window (first, last step), per recipe: each run's mean loss over the window, indexed by job."""
    out = {}
    for a, b in {(a, b) for _, a, b in QW_WINDOWS} | set(QW_SEP_WINDOWS):
        means = curves.loc[a:b].mean()
        out[(a, b)] = {rec: means[lo.index[lo.recipe == rec]] for rec, _, _ in FP4_RECIPES}
    return out


def _qw_fp4_numbers(t) -> dict[str, tuple]:
    """Qwen3 32B with NVFP4, TriQwFpFour*: C-speed at 13 nodes (GLM-FP4-03) on milabench's timer, steps 3-20, the GPU telemetry over the same
    steps; C-loss on 1 node, 3 runs per recipe (GLM-FP4-02, -04): window means per recipe (mean, lowest and highest run), where the quantized
    runs sit above all BF16 runs (window means, and step by step), the step times, and where runs of one recipe first differ."""
    m, lo, curves = _qw_fp4(t)
    out = {}
    if {"bf16", "mxfp8", "nvfp4"} <= set(m.index):
        nv = m.loc["nvfp4"]
        for key, macro, fmt in (("nodes", "Nodes", "d"), ("gpus", "Gpus", "d"), ("tp", "Tp", "d"), ("dp", "Dp", "d"), ("seqs", "Seqs", "d"),
                                ("seq_len", "SeqLen", ",d"), ("n_layers", "Layers", "d"), ("bf16_layers_last", "BfLast", "d"),
                                ("steps_logged", "Steps", "d"), ("measured", "Measured", "d")):
            out[f"TriQwFpFour{macro}"] = (int(nv[key]), fmt, False)
        out["TriQwFpFourFrom"] = (glm_fp4.FIRST_STEP, "d", False)
        for rec, name, _ in FP4_RECIPES:
            x = m.loc[rec]
            for col, macro, fmt in (("step_mean", "Step", ".2f"), ("step_median", "Med", ".2f"), ("step_sd", "Sd", ".3f"), ("tok_s_gpu", "Tok", ",.0f"),
                                    ("mem_alloc_gib", "Mem", ".1f"), ("cap_med", "CapMed", ".1f"), ("cap_lo", "CapLo", ".1f"), ("cap_hi", "CapHi", ".1f"),
                                    ("power_w", "Power", ".0f"), ("sm_mhz", "Sm", ",.0f")):
                out[f"TriQwFpFour{macro}{name}"] = (x[col], fmt, False)
        s = m.step_mean
        out["TriQwFpFourSpNvBf"] = (s.bf16 / s.nvfp4, ".2f", False)
        out["TriQwFpFourSpNvMx"] = (s.mxfp8 / s.nvfp4, ".2f", False)
        out["TriQwFpFourSpMxBf"] = (s.bf16 / s.mxfp8, ".2f", False)
        out["TriQwFpFourSmBfNvPct"] = (100 * (1 - m.sm_mhz.bf16 / m.sm_mhz.nvfp4), ".0f", False)
        out["TriQwFpFourXid"] = (int(m.xid_rows.sum()), "d", False)
    if len(curves) and {"bf16", "mxfp8", "nvfp4"} <= set(lo.recipe):
        for key, macro, fmt in (("nodes", "LNodes", "d"), ("tp", "LTp", "d"), ("pp", "LPp", "d"), ("dp", "LDp", "d"), ("seqs", "LSeqs", "d"),
                                ("steps_logged", "LSteps", "d"), ("lr_warmup", "LWarm", "d"), ("seed", "LSeed", "d")):
            if lo[key].nunique() == 1:  # the same setup in every run
                out[f"TriQwFpFour{macro}"] = (int(lo[key].iloc[0]), fmt, False)
        if lo.lr.nunique() == 1 and lo.min_lr.nunique() == 1 and set(lo.lr_schedule) == {"cosine"}:
            out["TriQwFpFourLLr"] = (_power10(lo.lr.iloc[0]), "", False)
            out["TriQwFpFourLMinLr"] = (_power10(lo.min_lr.iloc[0]), "", False)
        counts = lo.recipe.value_counts()
        if counts.nunique() == 1:
            out["TriQwFpFourLRuns"] = (int(counts.iloc[0]), "d", False)
        out["TriQwFpFourLJobs"] = (len(lo), "d", False)
        step = {}
        for rec, name, _ in FP4_RECIPES:
            x = lo[lo.recipe == rec]
            step[rec] = x.step_mean.mean()
            out[f"TriQwFpFourLStep{name}"] = (step[rec], ".2f", False)
            out[f"TriQwFpFourLStepLo{name}"] = (x.step_mean.min(), ".2f", False)
            out[f"TriQwFpFourLStepHi{name}"] = (x.step_mean.max(), ".2f", False)
            out[f"TriQwFpFourLMem{name}"] = (x.mem_alloc_gib.max(), ".1f", False)
        out["TriQwFpFourLSpMxBf"] = (step["bf16"] / step["mxfp8"], ".2f", False)
        out["TriQwFpFourLSpNvBf"] = (step["bf16"] / step["nvfp4"], ".2f", False)
        out["TriQwFpFourLSpNvMx"] = (step["mxfp8"] / step["nvfp4"], ".2f", False)
        win = _qw_windows(lo, curves)
        for w, a, b in QW_WINDOWS:
            out[f"TriQwFpFourWin{w}From"], out[f"TriQwFpFourWin{w}To"] = (a, "d", False), (b, "d", False)
            if w not in ("Early", "End"):
                continue
            for rec, name, _ in FP4_RECIPES:
                v = win[(a, b)][rec]
                out[f"TriQwFpFourWin{w}{name}"] = (v.mean(), ".3f", False)
                out[f"TriQwFpFourWin{w}Lo{name}"] = (v.min(), ".3f", False)
                out[f"TriQwFpFourWin{w}Hi{name}"] = (v.max(), ".3f", False)
        _, a, b = QW_WINDOWS[0]
        e = win[(a, b)]
        out["TriQwFpFourEarlyMargin"] = (min(e["mxfp8"].min(), e["nvfp4"].min()) - e["bf16"].max(), ".3f", False)
        # Window means: where every run of a recipe sits above every BF16 run (post hoc windows)
        above = {rec: [(a, b) for a, b in QW_SEP_WINDOWS if win[(a, b)][rec].min() > win[(a, b)]["bf16"].max()] for rec in ("mxfp8", "nvfp4")}
        both = [w for w in QW_SEP_WINDOWS if w in above["mxfp8"] and w in above["nvfp4"]]
        mx_only = [w for w in above["mxfp8"] if w not in both]
        if both:
            out["TriQwFpFourSepWinBoth"] = (_and_list([f"{a}-{b}" for a, b in both]), "", False)
            out["TriQwFpFourCut"] = (max(b for _, b in both), "d", False)
        if mx_only:
            out["TriQwFpFourSepWinMx"] = (_and_list([f"{a}-{b}" for a, b in mx_only]), "", False)
        # Step by step: every quantized run above every BF16 run; each recipe alone; what chance gives under exchangeable runs
        jobs = {rec: list(lo.index[lo.recipe == rec]) for rec, _, _ in FP4_RECIPES}
        bf_max = curves[jobs["bf16"]].max(axis=1)
        sep = {"all": curves[jobs["mxfp8"] + jobs["nvfp4"]].min(axis=1) > bf_max,
               "Mx": curves[jobs["mxfp8"]].min(axis=1) > bf_max, "Nv": curves[jobs["nvfp4"]].min(axis=1) > bf_max}
        steps_all = sep["all"][sep["all"]].index
        out["TriQwFpFourSepSteps"] = (len(steps_all), "d", False)
        if len(steps_all):
            out["TriQwFpFourSepLast"] = (int(steps_all.max()), "d", False)
            run_id = (sep["all"] != sep["all"].shift()).cumsum()[sep["all"]]  # consecutive stretches
            longest = run_id.value_counts().idxmax()
            stretch = run_id[run_id == longest].index
            out["TriQwFpFourSepRunFrom"], out["TriQwFpFourSepRunTo"] = (int(stretch.min()), "d", False), (int(stretch.max()), "d", False)
        s = sep["Mx"][sep["Mx"]].index
        out["TriQwFpFourSepMxSteps"] = (len(s), "d", False)
        if len(s):
            out["TriQwFpFourSepMxLast"] = (int(s.max()), "d", False)
            if both:
                out["TriQwFpFourSepMxIn"] = (int((s <= max(b for _, b in both)).sum()), "d", False)  # of them, up to the cutoff
        out["TriQwFpFourSepNvSteps"] = (int(sep["Nv"].sum()), "d", False)
        n_bf, n_q = len(jobs["bf16"]), len(jobs["mxfp8"])
        out["TriQwFpFourOddsOne"] = (math.comb(n_bf + n_q, n_bf), "d", False)  # one recipe's runs all above BF16's: 1 in this many orderings
        out["TriQwFpFourOddsAll"] = (math.comb(len(lo), n_bf), "d", False)  # every quantized run above every BF16 run
        out["TriQwFpFourSepNvChance"] = (len(curves) / math.comb(n_bf + len(jobs["nvfp4"]), n_bf), ".0f", False)
        # The five GLM-FP4-04 runs alone, over the early window: is the new BF16 run below every new quantized run?
        new = lo[lo.request == "GLM-FP4-04"]
        nb, nq = e["bf16"][e["bf16"].index.isin(new.index)], pd.concat([e["mxfp8"], e["nvfp4"]])
        nq = nq[nq.index.isin(new.index)]
        if len(nb) == 1 and len(nq):
            out["TriQwFpFourNewBf"] = (nb.iloc[0], ".3f", False)
            out["TriQwFpFourNewQLo"] = (nq.min(), ".3f", False)
            out["TriQwFpFourNewOdds"] = (len(nb) + len(nq), "d", False)  # one BF16 run lowest of the five: 1 in 5
        # Runs of one recipe, from one seed: first step where their loss, and their gradient norm, differ (the earliest over the recipes)
        lt = t["qw_fp4_loss"].assign(job=lambda d: d.job.astype(str))
        gn = lt[lt.job.isin(set(lo.index))].pivot(index="step", columns="job", values="grad_norm")
        for frame, macro in ((curves, "LossDiffFrom"), (gn, "GnDiffFrom")):
            firsts = []
            for js in jobs.values():
                d = frame.index[frame[js].nunique(axis=1) > 1]
                if len(d):
                    firsts.append(int(d.min()))
            if firsts:
                out[f"TriQwFpFour{macro}"] = (min(firsts), "d", False)
        last = int(curves.index.max())
        out["TriQwFpFourLMaxLo"] = (lo.loss_max_after50.min(), ".2f", False)
        out["TriQwFpFourLMaxHi"] = (lo.loss_max_after50.max(), ".2f", False)
        out["TriQwFpFourLNonFinite"] = (int(lo.loss_nonfinite.sum()), "d", False)
        out["TriQwFpFourLEndLo"] = (curves.loc[last].min(), ".2f", False)
        out["TriQwFpFourLEndHi"] = (curves.loc[last].max(), ".2f", False)
    return out


QW_CP_ENTRIES = (("8k", "Ref"), ("32k-cp4-none", "ThirtyTwo"), ("128k-cp4-full", "CpFour"), ("128k-cp8-selective", "CpEight"),
                 ("128k-cp4-selective", "CpFourSel"))


def _qw_cp(t) -> pd.DataFrame:
    """Qwen3 32B with context parallelism (glm_fp4.build_cp, qw_cp_runs): the completed entries by run label (8k, 32k-cp4, 128k-cp4, 128k-cp8)."""
    r = t.get("qw_cp_runs", pd.DataFrame())
    if r.empty:
        return r
    done = r[r.job_state == "COMPLETED"]
    if done.run.duplicated().any():
        raise SystemExit("[report] qw_cp_runs: two completed entries with the same label")
    return done.set_index("run")


def _range(lo, hi, fmt) -> str:
    a, b = format(lo, fmt), format(hi, fmt)
    return a if a == b else f"{a}-{b}"


def _qw_cp_numbers(t) -> dict[str, tuple]:
    """Qwen3 32B at 32k and 128k with context parallelism, TriQwCp*: QW-CP-01's three entries and the 8k reference (GLM-FP4-03's BF16 entry, the
    same run as TriQwFpFour*Bf), means over steps 3-20 on milabench's timer, tokens/s per GPU at the mean step, MFU with the summary's FLOPs per
    token; bytes across nodes per node and step; GPU telemetry over the same steps (per node: cap share, mean power, median SM clock)."""
    m = _qw_cp(t)
    out = {}
    if m.empty or "8k" not in m.index:
        return out
    fp4, _, _ = _qw_fp4(t)
    if "bf16" in fp4.index and abs(fp4.step_mean.bf16 - m.step_mean["8k"]) > 1e-9:
        raise SystemExit("[report] qw_cp_runs: the 8k reference's mean step differs from the Qwen3 NVFP4 block's BF16 one")
    ref = m.loc["8k"]
    for key, macro, fmt in (("tp", "Tp", "d"), ("n_layers", "Layers", "d"), ("steps_logged", "Steps", "d"), ("first", "From", "d"),
                            ("measured", "Measured", "d"), ("tokens_per_gpu", "TokGpu", ",d")):
        out[f"TriQwCp{macro}"] = (int(ref[key]), fmt, False)
    if m.lr.nunique() == 1 and set(m.lr_schedule) == {"constant"}:
        out["TriQwCpLr"] = (_power10(m.lr.iloc[0]), "", False)
    for run, name in QW_CP_ENTRIES:
        if run not in m.index:
            continue
        x = m.loc[run]
        for col, macro, fmt in (("nodes", "Nodes", "d"), ("gpus", "Gpus", "d"), ("cp", "Split", "d"), ("dp", "Dp", "d"), ("seqs", "Seqs", "d")):
            out[f"TriQwCp{macro}{name}"] = (int(x[col]), fmt, False)
        for col, macro, fmt in (("step_mean", "Step", ".2f"), ("step_median", "Med", ".3f"), ("step_sd", "Sd", ".3f"), ("tok_s_gpu", "Tok", ",.0f"),
                                ("mfu_pct", "Mfu", ".1f"), ("mem_alloc_gib", "Mem", ".1f"), ("step1_s", "First", ".0f"), ("ib_lo", "IbLo", ".1f"),
                                ("ib_hi", "IbHi", ".1f"), ("cap_med", "CapMed", ".1f"), ("cap_lo", "CapLo", ".1f"), ("cap_hi", "CapHi", ".1f"),
                                ("power_lo", "PowerLo", ".0f"), ("power_hi", "PowerHi", ".0f"), ("sm_med_lo", "SmLo", ",.0f"), ("sm_med_hi", "SmHi", ",.0f")):
            out[f"TriQwCp{macro}{name}"] = (x[col], fmt, False)
        if run != "8k":
            out[f"TriQwCpRatio{name}"] = (x.tok_s_gpu / ref.tok_s_gpu, ".3f", False)
    full, eight, sel = "128k-cp4-full", "128k-cp8-selective", "128k-cp4-selective"
    if {full, eight} <= set(m.index):
        out["TriQwCpEightShorter"] = (100 * (1 - m.step_mean[eight] / m.step_mean[full]), ".1f", False)
        out["TriQwCpEightShorterMed"] = (100 * (1 - m.step_median[eight] / m.step_median[full]), ".1f", False)
        out["TriQwCpBehindFull"] = (m.behind_max[full], ".2f", False)
        out["TriQwCpBehindEight"] = (m.behind_max[eight], ".2f", False)
        out["TriQwCpDropOne"] = (max(m.drop_other_max[full], m.drop_other_max[eight]), ".1f", False)
        out["TriQwCpCapLoCpFour"], out["TriQwCpCapHiCpFour"] = (m.cap_lo[full], ".1f", False), (m.cap_hi[full], ".1f", False)
    if {full, eight, sel} <= set(m.index):
        x = m.loc[sel]
        out["TriQwCpSelShorter"] = (100 * (1 - x.step_mean / m.step_mean[full]), ".1f", False)
        out["TriQwCpSelShorterMed"] = (100 * (1 - x.step_median / m.step_median[full]), ".1f", False)
        out["TriQwCpSelEight"] = (100 * abs(x.step_mean / m.step_mean[eight] - 1), ".1f", False)
        out["TriQwCpSelEightMed"] = (100 * abs(x.step_median / m.step_median[eight] - 1), ".1f", False)
        for col, macro, fmt in (("cap_lo", "CapLo", ".1f"), ("cap_hi", "CapHi", ".1f"), ("power_lo", "PowerLo", ".0f"), ("power_hi", "PowerHi", ".0f"),
                                ("sm_med_lo", "SmLo", ",.0f"), ("sm_med_hi", "SmHi", ",.0f")):
            out[f"TriQwCp{macro}CpFourSel"] = (x[col], fmt, False)
        nodes = x.slow_nodes.split(";") if isinstance(x.slow_nodes, str) and x.slow_nodes else []
        out["TriQwCpSlowSteps"], out["TriQwCpSlowMulti"] = (int(x.slow_steps), "d", False), (int(x.slow_steps_multi), "d", False)
        out["TriQwCpSlowNodes"], out["TriQwCpFastNodes"] = (len(nodes), "d", False), (int(x.cap_nodes) - len(nodes), "d", False)
        out["TriQwCpSlowNodeList"] = (_and_list([rf"\texttt{{{h}}}" for h in nodes]) if nodes else None, "", False)
        for col, macro, fmt in (("slow_behind_lo", "BehindLo", ".1f"), ("slow_behind_hi", "BehindHi", ".1f"), ("drop_slow_lo", "DropLo", ".1f"),
                                ("drop_slow_hi", "DropHi", ".1f"), ("drop_other_max", "DropOther", ".1f"), ("clean_lo", "CleanLo", ".2f"),
                                ("clean_hi", "CleanHi", ".2f")):
            out[f"TriQwCp{macro}"] = (x[col], fmt, False)
        out["TriQwCpCleanSteps"] = (int(x.clean_steps), "d", False)
        out["TriQwCpCleanBelowLo"] = (100 * (1 - x.clean_hi / m.step_mean[eight]), ".1f", False)
        out["TriQwCpCleanBelowHi"] = (100 * (1 - x.clean_lo / m.step_mean[eight]), ".1f", False)
    if eight in m.index:
        e = m.loc[eight]
        for port, name in (("mlx5_0", "Zero"), ("mlx5_1", "One"), ("mlx5_4", "Four"), ("mlx5_9", "Nine")):
            out[f"TriQwCpPort{name}"] = (_range(e[f"ib_{port}_min"], e[f"ib_{port}_max"], ".0f"), "", False)
        out["TriQwCpPortBusy"] = (e.ib_port_max_gbs, ".1f", False)
    cp = m.drop(index="8k")
    if len(cp):
        for col, agg, macro, fmt in (("cap_lo", "min", "CapLo", ".0f"), ("cap_hi", "max", "CapHi", ".0f"), ("power_lo", "min", "PowerLo", ".0f"),
                                     ("power_hi", "max", "PowerHi", ".0f"), ("sm_med_lo", "min", "SmLo", ",.0f"), ("sm_med_hi", "max", "SmHi", ",.0f")):
            out[f"TriQwCp{macro}"] = (getattr(cp[col], agg)(), fmt, False)
    out["TriQwCpXid"] = (int(m.xid_rows.sum()), "d", False)
    return out


def qw_cp_speed_table(t) -> str:
    """Qwen3 32B at 8k (the reference), 32k and 128k tokens, BF16, one row per entry: layout, means over steps 3-20 on milabench's timer, tokens/s
    per GPU and against 8k, MFU, peak allocated memory, bytes across nodes per node and step (range over the nodes, and the steps), power cap."""
    m = _qw_cp(t)
    if m.empty or "8k" not in m.index:
        return ""
    ref = m.loc["8k"]
    lines = [r"\begin{tabular}{lrllrrrrrrrr}", r"\toprule",
             r"Context & Nodes & Layout & Recompute & Mean step & SD & Tokens/s & vs 8k & MFU & Peak alloc. & Across nodes, GiB & At power cap, \% \\",
             r" & & & & s & s & per GPU & & \% & GiB & per node and step & median node (range) \\", r"\midrule"]
    for run, _ in QW_CP_ENTRIES:
        if run not in m.index:
            continue
        x = m.loc[run]
        context = f"{int(x.seq_len) // 1024}k" + (", slots" if x.doc_masking else "")
        layout = f"TP{int(x.tp)}" + (rf" $\times$ CP{int(x.cp)}" if x.cp > 1 else "") + rf" $\times$ DP{int(x.dp)}"
        recompute = x.recompute + (f" ({x.recompute_modules.upper()})" if isinstance(x.recompute_modules, str) else "") + (", graphs" if x.cuda_graphs else "")
        ib = f"{x.ib_lo:.1f}--{x.ib_hi:.1f} ({int(x.first)}--{int(x.ib_last_lo)})" if pd.notna(x.ib_lo) else ""
        lines.append(f"{context} & {int(x.nodes)} & {layout} & {recompute} & {x.step_mean:.2f} & {x.step_sd:.2f} & {x.tok_s_gpu:,.0f} & {x.tok_s_gpu / ref.tok_s_gpu:.3f} & "
                     f"{x.mfu_pct:.1f} & {x.mem_alloc_gib:.1f} & {ib} & {x.cap_med:.1f} ({x.cap_lo:.1f}--{x.cap_hi:.1f}) " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# OpenFold3 training (of3.build): the variants and node counts measured, macro suffixes
OF3_VARIANTS = (("published", "Pub"), ("ds", "Ds"), ("cueq+tf32", "Cueq"))
OF3_NODES = {1: "One", 2: "Two", 4: "Four", 13: "Thirteen"}
OF3_POINTS = (("Pub", "One"), ("Pub", "Two"), ("Pub", "Four"), ("Pub", "Thirteen"), ("Ds", "One"), ("Ds", "Thirteen"),
              ("Cueq", "One"), ("Cueq", "Two"), ("Cueq", "Four"), ("Cueq", "Thirteen"))


def _of3(t) -> pd.DataFrame:
    """OpenFold3's completed runs (of3_runs), indexed by (variant, nodes), with the per-GPU throughput against the same variant on 1 node."""
    r = t.get("of3_runs", pd.DataFrame())
    if r.empty:
        return r
    done = r[r.job_state == "COMPLETED"]
    if done.duplicated(["run", "nodes"]).any():
        raise SystemExit("[report] of3_runs: two completed runs of one variant at one node count")
    one = done[done.nodes == 1].set_index("run").sps_gpu
    return done.assign(eff_pct=100 * done.sps_gpu / done.run.map(one)).set_index(["run", "nodes"]).sort_index()


def _of3_numbers(t) -> dict[str, tuple]:
    """OpenFold3 training, TriOf*: per variant and node count, the wall-clock step period over steps 7-40 (rank 0's step ends), samples/s per GPU
    and in all, against 1 node; the loader's wait, GPU and CPU telemetry, peak memory, InfiniBand bytes against one gradient buffer."""
    m = _of3(t)
    out = {}
    if m.empty:
        return out
    x = m.iloc[0]
    for col, macro, fmt in (("measured", "Measured", "d"), ("first", "From", "d"), ("last", "Steps", "d"), ("batch_per_gpu", "Batch", "d"),
                            ("workers", "Workers", "d"), ("crop_tokens", "Crop", "d")):
        out[f"TriOf{macro}"] = (int(x[col]), fmt, False)
    out["TriOfParams"] = (x.params / 1e6, ".1f", False)
    out["TriOfBuffer"] = (x.buffer_bytes / 1e9, ".3f", False)
    names = dict(OF3_VARIANTS)
    for (run, nodes), y in m.iterrows():
        v, n = names.get(run), OF3_NODES.get(int(nodes))
        if (v, n) not in OF3_POINTS:
            continue
        for col, macro, fmt in (("sps_gpu", "Sps", ".4f"), ("step_wall", "Step", ".3f"), ("step_median", "Med", ".3f"), ("sps_total", "Total", ".2f"),
                                ("gap_pct", "Gap", ".1f")):
            out[f"TriOf{macro}{v}{n}"] = (y[col], fmt, False)
        if nodes > 1:
            out[f"TriOfEff{v}{n}"] = (y.eff_pct, ".1f", False)
        if run != "published" and ("published", nodes) in m.index:
            out[f"TriOfRatio{v}{n}"] = (y.step_wall / m.step_wall["published", nodes], ".3f", False)
    top_nodes = int(m.index.get_level_values("nodes").max())
    top = m.xs(top_nodes, level="nodes")
    out["TriOfTopNodes"], out["TriOfTopGpus"] = (top_nodes, "d", False), (int(top.gpus.iloc[0]), "d", False)
    out["TriOfEffLo"], out["TriOfEffHi"] = (top.eff_pct.min(), ".1f", False), (top.eff_pct.max(), ".1f", False)
    loss = 100 - top.eff_pct
    if "published" in top.index:
        out["TriOfLossPub"] = (loss["published"], ".1f", False)
        others = loss.drop(index="published")
        if len(others):
            out["TriOfLossLo"], out["TriOfLossHi"] = (others.min(), ".1f", False), (others.max(), ".1f", False)
    for col, macro, fmt in (("gap_pct", "Gap", ".1f"), ("gap_worst_pct", "GapWorst", ".1f"), ("power_w", "Power", ".0f"), ("util_pct", "Util", ".0f"),
                            ("cpu_busy_min", "CpuMin", ",.0f"), ("ib_buffers", "IbBuffers", ".2f")):
        out[f"TriOf{macro}Lo"], out[f"TriOf{macro}Hi"] = (top[col].min(), fmt, False), (top[col].max(), fmt, False)
    for run, v in OF3_VARIANTS:
        if run in top.index:
            out[f"TriOfPower{v}"] = (top.power_w[run], ".0f", False)
            out[f"TriOfUtil{v}"] = (top.util_pct[run], ".0f", False)
    small = m[m.index.get_level_values("nodes") < top_nodes]
    out["TriOfGapSmallLo"], out["TriOfGapSmallHi"] = (small.gap_pct.min(), ".1f", False), (small.gap_pct.max(), ".1f", False)
    out["TriOfSm"] = (m.sm_med.median(), ",.0f", False) if m.sm_med.nunique() == 1 else (None, "", False)
    out["TriOfSmMin"] = (m.sm_min.min(), ",.0f", False)
    out["TriOfCap"] = (m.cap_pct.max(), ".1f", False)  # the most of any run's GPU samples with the software power cap
    plain = m[m.index.get_level_values("run") != "cueq+tf32"]
    out["TriOfCapPlain"] = (plain.cap_pct.max(), ".1f", False)  # published and DeepSpeed's attention
    out["TriOfSmMinPlain"] = (plain.sm_min.min(), ",.0f", False)
    if "cueq+tf32" in top.index:
        out["TriOfCapCueq"] = (top.cap_pct["cueq+tf32"], ".1f", False)
    out["TriOfXid"] = (int(m.xid_rows.sum()), "d", False)
    out["TriOfCpuMed"] = (m.cpu_busy_med.median(), ",.0f", False) if m.cpu_busy_med.nunique() == 1 else (None, "", False)
    out["TriOfMemAlloc"], out["TriOfMemRes"] = (m.mem_alloc_gib.max(), ".1f", False), (m.mem_reserved_gib.max(), ".1f", False)
    out["TriOfNonFinite"] = (int(m.loss_nonfinite.sum()), "d", False)
    pub = top.loc["published"] if "published" in top.index else None
    if pub is not None and pd.notna(pub.ib_lo_gb):
        out["TriOfIbLo"], out["TriOfIbHi"] = (pub.ib_lo_gb, ".3f", False), (pub.ib_hi_gb, ".3f", False)
        out["TriOfIbTree"] = (2 * (top_nodes - 1), "d", False)
        out["TriOfIbExcess"] = (100 * (pub.ib_buffers / (2 * (top_nodes - 1)) - 1), ".1f", False)
    if "ds" in top.index:
        out["TriOfFirstDs"] = (top.step1_s["ds"], ".0f", False)
        rest = m.drop(index=("ds", top_nodes)).step1_s
        out["TriOfFirstLo"], out["TriOfFirstHi"] = (rest.min(), ".0f", False), (rest.max(), ".0f", False)
    return out


def of3_speed_table(t) -> str:
    """OpenFold3 training by variant and node count: wall-clock step period over steps 7-40 and its median, samples/s per GPU and in all, against
    the same variant on 1 node, the loader's wait, mean GPU power."""
    m = _of3(t)
    if m.empty:
        return ""
    names = {"published": "Published", "ds": "DeepSpeed attention", "cueq+tf32": "cuEquivariance + TF32"}
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"Variant & Nodes & GPUs & Step & Median & Samples/s & Samples/s & vs 1 node & Loader & Mean GPU \\",
             r" & & & s & s & per GPU & in all & \% & wait, \% & power, W \\", r"\midrule"]
    for i, (run, _) in enumerate(OF3_VARIANTS):
        if run not in m.index.get_level_values("run"):
            continue
        if i:
            lines.append(r"\addlinespace")
        for nodes, y in m.xs(run, level="run").iterrows():
            lines.append(f"{names[run]} & {int(nodes)} & {int(y.gpus)} & {y.step_wall:.3f} & {y.step_median:.3f} & {y.sps_gpu:.4f} & {y.sps_total:.2f} & "
                         f"{y.eff_pct:.1f} & {y.gap_pct:.1f} & {format(y.power_w, '.0f') if pd.notna(y.power_w) else ''} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def qw_fp4_speed_table(t) -> str:
    """Qwen3 32B C-speed at 13 nodes by recipe, one job (GLM-FP4-03): steps 3-20 on milabench's timer, speed-ups, peak allocated memory, and the
    GPU telemetry over the same steps (all nodes)."""
    m, _, _ = _qw_fp4(t)
    if not {"bf16", "mxfp8"} <= set(m.index):
        return ""
    lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"Recipe & Mean step & Median & SD & Speed-up & Speed-up & Tokens/s & Peak alloc. & At power cap, \% & Mean GPU & Mean SM \\",
             r" & s & s & s & vs BF16 & vs MXFP8 & per GPU & GiB & median node (range) & power, W & clock, MHz \\", r"\midrule"]
    for rec, _, label in FP4_RECIPES:
        if rec not in m.index:
            continue
        x = m.loc[rec]
        if x.bf16_layers_last:
            label += f" (last {int(x.bf16_layers_last)} layers BF16)"
        lines.append(f"{label} & {x.step_mean:.2f} & {x.step_median:.2f} & {x.step_sd:.3f} & {m.step_mean.bf16 / x.step_mean:.2f} & "
                     f"{m.step_mean.mxfp8 / x.step_mean:.2f} & {x.tok_s_gpu:,.0f} & {x.mem_alloc_gib:.1f} & {x.cap_med:.1f} ({x.cap_lo:.1f}--{x.cap_hi:.1f}) & "
                     f"{x.power_w:.0f} & {x.sm_mhz:,.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def qw_fp4_loss_table(t) -> str:
    """Qwen3 32B C-loss on 1 node, one job per run, 3 runs per recipe (GLM-FP4-02, -04), a column per recipe: the jobs and their nodes, mean step
    (steps 3-500, milabench's timer), peak allocated memory, and the mean loss over each window, as the mean of the runs [lowest--highest run]."""
    _, lo, curves = _qw_fp4(t)
    if curves.empty or not {"bf16", "mxfp8", "nvfp4"} <= set(lo.recipe):
        return ""
    win = _qw_windows(lo, curves)
    cols = []
    for rec, _, label in FP4_RECIPES:
        x = lo[lo.recipe == rec]
        if x.bf16_layers_last.max():
            label += f" (last {int(x.bf16_layers_last.max())} layers BF16)"
        mem = x.mem_alloc_gib
        cols.append({"label": label, "jobs": ", ".join(x.index), "nodes": ", ".join(rf"\texttt{{{h}}}" for h in x.hosts),
                     "step": f"{x.step_mean.mean():.3f} [{x.step_mean.min():.3f}--{x.step_mean.max():.3f}]",
                     "mem": f"{mem.max():.1f}" if mem.nunique() == 1 else f"{mem.min():.1f}--{mem.max():.1f}",
                     **{w: f"{v.mean():.3f} [{v.min():.3f}--{v.max():.3f}]" for (w, a, b) in QW_WINDOWS for v in [win[(a, b)][rec]]}})
    row = lambda name, key: f"{name} & " + " & ".join(c[key] for c in cols) + r" \\"
    lines = [r"\begin{tabular}{lccc}", r"\toprule", " & " + " & ".join(c["label"] for c in cols) + r" \\", r"\midrule",
             row("Jobs", "jobs"), row("Nodes", "nodes"), row("Mean step, s", "step"), row("Peak alloc., GiB", "mem"), r"\midrule",
             r"\multicolumn{4}{l}{Mean loss over steps} \\", *[row(f"\\quad {a}--{b}", w) for w, a, b in QW_WINDOWS],
             r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def glm_fp4_runs_table(t) -> str:
    """GLM-5 C-speed at 13 nodes by recipe, one job (GLM-FP4-01): steps 3-20 on milabench's timer, peak allocated memory, loss at step 20,
    and the last stage's node's share of GPU samples at the software power cap."""
    r = t.get("glm_fp4_runs", pd.DataFrame())
    if r.empty:
        return ""
    m = r.set_index("precision")
    if not {"bf16", "mxfp8"} <= set(m.index):
        return ""
    lines = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
             r"Recipe & Mean step & Median & SD & Step vs & Step vs & Tokens/s & Peak alloc. & Loss at & Last node at \\",
             r" & s & s & s & BF16 & MXFP8 & per GPU & GiB & step 20 & power cap, \% \\", r"\midrule"]
    for rec, _, label in FP4_RECIPES:
        if rec not in m.index:
            continue
        x = m.loc[rec]
        if x.bf16_layers_last:
            label += f" (last {int(x.bf16_layers_last)} layers BF16)"
        lines.append(f"{label} & {x.step_mean:.2f} & {x.step_median:.2f} & {x.step_sd:.2f} & {x.step_mean / m.step_mean.bf16:.3f} & "
                     f"{x.step_mean / m.step_mean.mxfp8:.3f} & {x.tok_s_gpu:,.0f} & {x.mem_alloc_gib:.1f} & {x.loss_last:.3f} & {x.cap_last_pct:.0f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def glm_rl_runs_table(t) -> str:
    """GLM-5.3 GRPO runs in the report: request, job, nodes, the trainer's and the engines' layout, the batch, steps, start and end, state."""
    r = t.get("glm_rl_runs", pd.DataFrame())
    if r.empty:
        return ""
    lines = [r"\begin{tabular}{llllllllrrlll}", r"\toprule",
             r"Request & Job & Layout & Routing & Nodes & Trainer & Rollout engines & Rollout & Samples & Steps & Started & Ended & State \\",
             r" & & & replay & & & & kernels & per step & & & & \\", r"\midrule"]
    for x in r.sort_values("start").itertuples():
        trainer = f"TP{int(x.tp)} PP{int(x.pp)} CP{int(x.cp)} EP{int(x.ep)} (DP{int(x.dp)})"
        engines = f"{int(x.engines)} $\\times$ {int(x.engine_gpus)} GPUs (EP{int(x.engine_ep)})"
        replay = "with" if bool(x.routing_replay) else "without"
        colocate = bool(getattr(x, "colocate", True))
        layout = "colocated" if colocate else "disaggregated"
        nodes = f"{int(x.nodes)}" if colocate else f"{int(x.nodes)}: {int(x.trainer_nodes)} + {int(x.nodes - x.trainer_nodes)}"
        kernels = (f"B200 default, {int(x.max_running_requests // x.engine_dp)} requests per rank" if bool(getattr(x, "default_kernels", False))
                   else "fallback")
        lines.append(f"{x.request} & {x.job} & {layout} & {replay} & {nodes} & {trainer} & {engines} & {kernels} & {int(x.samples):,} & {int(x.steps_logged)} & "
                     f"{str(x.start).replace('T', ' ')[:16]} & {str(x.end).replace('T', ' ')[:16]} & {x.job_state} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def glm_rl_steps_table(t) -> str:
    """GLM-5.3 GRPO per step (slime's timers and metrics): where the time goes, training and rollout speed, responses, reward."""
    st, order = _glm_rl_by_run(t)
    if st.empty:
        return ""
    lines = [r"\begin{tabular}{llrrrrrrrrrrrrr}", r"\toprule",
             r"Run & Step & Rollout & Training & Sync & Rest & Step & Rollout & Train & Micro- & Bubble & Rollout & Response & Trunc. & Reward \\",
             r" & & s & s & s & s & s & \% of step & tok/s & batches & \% (comp.) & tok/s/GPU & mean / median & \% & \\", r"\midrule"]
    for i, (job, replay) in enumerate(order):
        if i:
            lines.append(r"\midrule")
        for x in st[st.job.astype(str) == job].sort_values("step").itertuples():
            step = f"{x.step} (warm-up)" if x.warmup else str(x.step)
            lines.append(f"{replay} & {step} & {x.rollout_s:,.1f} & {x.train_s:,.1f} & {x.sync_s:.1f} & {x.rest_s:.1f} & {x.step_s:,.1f} & "
                         f"{x.rollout_pct:.1f} & {x.train_tok_s:,.0f} & {int(x.microbatches)} & {x.bubble_pct:.0f} & {x.rollout_tok_s_gpu:.2f} & "
                         f"{x.resp_mean:,.0f} / {x.resp_median:,.0f} & {x.truncated_pct:.1f} & {x.raw_reward:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _glm_rl_by_run(t) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """The GLM-RL steps and the runs in start order, as (job, label): colocated, colocated with routing replay, or disaggregated."""
    r, st = t.get("glm_rl_runs", pd.DataFrame()), t.get("glm_rl_steps", pd.DataFrame())
    if r.empty or st.empty:
        return pd.DataFrame(), []
    def label(x):
        layout = "colocated" if bool(getattr(x, "colocate", True)) else "disaggregated"
        return layout + (" + replay" if bool(x.routing_replay) else "") + (", default kernels" if bool(getattr(x, "default_kernels", False)) else "")
    return st, [(str(x.job), label(x)) for x in r.sort_values("start").itertuples()]


def glm_rl_replay_table(t) -> str:
    """GLM-5.3 GRPO per run and step (colocated without and with routing replay, disaggregated): data preprocessing, the TIS metrics,
    the gradient norm, the longest request, the median decode speed per request (slime's metrics) and the requests slime re-sent
    after a top-k log-prob error."""
    st, order = _glm_rl_by_run(t)
    if st.empty:
        return ""
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Run & Step & Data & TIS mean & TIS clip & Grad & Longest & Median decode & Top-k \\",
             r" & & preprocess s & $|w-1|$ & fraction & norm & request s & tok/s per request & retries \\", r"\midrule"]
    for i, (job, replay) in enumerate(order):
        if i:
            lines.append(r"\midrule")
        for x in st[st.job.astype(str) == job].sort_values("step").itertuples():
            step = f"{x.step} (warm-up)" if x.warmup else str(x.step)
            lines.append(f"{replay} & {step} & {x.data_preprocess_s:.2f} & {x.tis_abs:.4f} & {x.tis_clipfrac:.5f} & "
                         f"{x.grad_norm:.4f} & {x.latency_max_s:,.1f} & {x.decode_tok_s_median:.2f} & {int(getattr(x, 'topk_errors', 0) or 0)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


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
        before = big[big.slow_nodes.fillna("") != ""] if "slow_nodes" in big else big.iloc[0:0]
        if len(before):  # the largest run with the slow nodes before their fix, for the same numbers
            b = before.sort_values(["nodes", "started"]).iloc[-1]
            n = tele[tele.run_id == b.run_id]
            slow, rest = n[n.host.isin(SLOW_NODES)], n[~n.host.isin(SLOW_NODES | {"trig0114"})]
            out["TriStragBeforeNodes"] = (int(b.nodes), "d", False)
            if len(slow):
                out["TriStragBeforeLowLo"], out["TriStragBeforeLowHi"] = (100 * slow.cpu_low_share.min(), ".0f", False), (100 * slow.cpu_low_share.max(), ".0f", False)
                out["TriStragBeforeLowRest"] = (100 * rest.cpu_low_share.mean(), ".0f", False)
                out["TriStragBeforeUtilSlow"] = (100 * slow.gpu_util.mean(), ".1f", False)
                out["TriStragBeforeUtilRest"] = (100 * rest.gpu_util.mean(), ".1f", False)
        long = done[(done.model == QWEN) & (done.experiment == "F2") & done.run_id.isin(tele.run_id)]
        run = (long if len(long) else big).sort_values(["nodes", "started"]).iloc[-1]  # the long run once it has run (DESIGN.md: stragglers, F2)
        n = tele[tele.run_id == run.run_id]
        slow, rest = n[n.host.isin(SLOW_NODES)], n[~n.host.isin(SLOW_NODES | {"trig0114"})]
        out["TriStragNodes"] = (int(run.nodes), "d", False)
        if len(slow):
            out["TriStragLowLo"], out["TriStragLowHi"] = (100 * slow.cpu_low_share.min(), ".0f", False), (100 * slow.cpu_low_share.max(), ".0f", False)
            out["TriStragLowRest"] = (100 * rest.cpu_low_share.mean(), ".0f", False)
            out["TriStragUtilSlow"] = (100 * slow.gpu_util.mean(), ".1f", False)
            out["TriStragUtilRest"] = (100 * rest.gpu_util.mean(), ".1f", False)
        if len(n):
            out["TriStragLowAllLo"], out["TriStragLowAllHi"] = (100 * n.cpu_low_share.min(), ".0f", False), (100 * n.cpu_low_share.max(), ".0f", False)
            out["TriStragUtilAllLo"], out["TriStragUtilAllHi"] = (100 * n.gpu_util.min(), ".1f", False), (100 * n.gpu_util.max(), ".1f", False)
            out["TriStragKwLo"], out["TriStragKwHi"] = (n.node_gpu_kw_mean.min(), ".2f", False), (n.node_gpu_kw_mean.max(), ".2f", False)
    out.update(_long_run_numbers(t, done))
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
        d4 = t["collectives"]
        d4 = d4[(d4.experiment == "D4") & ((d4.job_id == NET_FULL_JOB) | d4.synthetic)]
        if len(d4):
            out["TriBisection"] = (d4.busbw_gbs.median(), ".1f", d4.synthetic.any())
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
    out.update(_kimi_numbers(t))
    out.update(_kimi_xn_numbers(t))
    out.update(_kimi_pd_numbers(t))
    out.update(_kimi_xn_four_numbers(t))
    out.update(_kimi_train_numbers(t))
    out.update(_glm_rl_numbers(t))
    out.update(_glm_fp4_numbers(t))
    out.update(_qw_fp4_numbers(t))
    out.update(_qw_cp_numbers(t))
    out.update(_of3_numbers(t))
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
        rf"\newcommand{{\NRunsFailed}}{{{int((real.status == 'failed').sum()) if len(real) else 0}}}",
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
    (out / "kimi_passes.tex").write_text(kimi_passes_table(tables, 2))
    (out / "kimi_passes_four.tex").write_text(kimi_passes_table(tables, 4))
    (out / "kimi_mi355x.tex").write_text(kimi_mi355x_table(tables))
    (out / "kimi_runs.tex").write_text(kimi_runs_table(tables))
    (out / "kimi_xn_passes.tex").write_text(kimi_xn_passes_table(tables))
    (out / "kimi_pd_passes.tex").write_text(kimi_pd_passes_table(tables))
    (out / "kimi_xn_four_passes.tex").write_text(kimi_xn_four_passes_table(tables))
    (out / "kimi_train_runs.tex").write_text(kimi_train_runs_table(tables))
    (out / "glm_rl_runs.tex").write_text(glm_rl_runs_table(tables))
    (out / "glm_fp4_runs.tex").write_text(glm_fp4_runs_table(tables))
    (out / "qw_fp4_speed.tex").write_text(qw_fp4_speed_table(tables))
    (out / "qw_fp4_loss.tex").write_text(qw_fp4_loss_table(tables))
    (out / "qw_cp_speed.tex").write_text(qw_cp_speed_table(tables))
    (out / "of3_speed.tex").write_text(of3_speed_table(tables))
    (out / "glm_rl_steps.tex").write_text(glm_rl_steps_table(tables))
    (out / "glm_rl_replay.tex").write_text(glm_rl_replay_table(tables))
    if len(runs):
        if "in_report" in runs:
            (out / "ledger.tex").write_text(ledger(runs))
            (out / "progress.tex").write_text(progress_table(tables))
