"""A pinned, small RULER slice for the long-context benches (config/homelab/longctx.yaml), as lm-eval tasks.

lm-eval's own RULER tasks (lm_eval/tasks/ruler) generate 500 samples per length and only at 4k unless
``--metadata`` is given, which the serving client's ``eval`` pass does not pass. These tasks reuse lm-eval's
RULER generators unchanged (same templates, same seeds: sample i here is sample i of lm-eval's task), but take
the lengths and the number of samples from the benchmark's config (``MILABENCH_CONFIG``, key ``longctx``):

    longctx:
      ruler_lengths: [8192, 32768, 65536, 131072]   # the server's --max-model-len must exceed the largest by ~1k
      ruler_samples: 8                               # per task and length

Each task reports one score per length (string match, as RULER), and -1 for a length it did not run.
Run from the serving pack's client (cwd benchmarks/serving) as ``../longctx/eval_tasks/<task>.yaml``.

Needs ``wonderwords`` and ``nltk`` in the client's venv (lm-eval's ``ruler`` extra). Data: Paul Graham essays
(HF dataset baber/paul_graham_essays, HF datasets cache) for the NIAH haystacks, NLTK punkt_tab, HotpotQA dev
(lm-eval downloads it into memory at every run from a pinned HF mirror: the QA task needs the network).
``python longctx_ruler.py --prepare`` fetches the cached ones. Generating the samples is CPU work before the
first request: ~1.5 min per task at 128k (Qwen tokenizer), a few minutes per model.
"""

from __future__ import annotations

import itertools
import json
import os

LENGTHS = [8192, 32768, 65536, 131072]
SAMPLES = 8
# every length a yaml lists in metric_list (a run reports -1 for the ones it skips)
ALL_LENGTHS = [8192, 16384, 32768, 65536, 131072, 262144]


def _settings(kwargs: dict) -> tuple[list[int], int]:
    try:
        cfg = json.loads(os.environ.get("MILABENCH_CONFIG", "{}")).get("longctx") or {}
    except json.JSONDecodeError:
        cfg = {}
    lengths = kwargs.pop("max_seq_lengths", None) or cfg.get("ruler_lengths") or LENGTHS
    samples = int(kwargs.pop("num_samples", None) or cfg.get("ruler_samples") or SAMPLES)
    unknown = sorted(set(map(int, lengths)) - set(ALL_LENGTHS))
    if unknown:
        raise ValueError(f"ruler_lengths {unknown} not in the tasks' metric_list {ALL_LENGTHS}")
    return [int(n) for n in lengths], samples


def _tokenizer(kwargs: dict):
    from lm_eval.tasks.ruler.common_utils import get_tokenizer, resolve_tokenizer_name

    return get_tokenizer(resolve_tokenizer_name(kwargs))


def _dataset(rows) -> dict:
    import datasets

    return {"test": datasets.Dataset.from_list(list(itertools.chain.from_iterable(rows)), split=datasets.Split.TEST)}


def _niah(kwargs: dict, **needles) -> dict:
    from lm_eval.tasks.ruler.niah_utils import TEMPLATE
    from lm_eval.tasks.ruler.prepare_niah import generate_samples, get_haystack

    lengths, samples = _settings(kwargs)
    tokenizer = _tokenizer(kwargs)
    return _dataset(
        generate_samples(get_haystack(type_haystack="essay"), max_seq_length=seq, template=TEMPLATE,
                         type_haystack="essay", type_needle_k="words", type_needle_v="numbers",
                         num_samples=samples, TOKENIZER=tokenizer, **needles)
        for seq in lengths
    )


# the parameters of lm-eval's niah_multikey_1 / niah_multivalue / niah_multiquery
def niah_multikey(**kwargs):
    """4 keys in the essay, one asked: retrieval among distractors."""
    return _niah(kwargs, num_needle_k=4)


def niah_multivalue(**kwargs):
    """4 values under one key: all must be found."""
    return _niah(kwargs, num_needle_v=4)


def niah_multiquery(**kwargs):
    """4 keys asked at once."""
    return _niah(kwargs, num_needle_q=4)


def variable_tracking(**kwargs):
    """Multi-hop: follow a chain of variable assignments (X1 = 12345 ... X2 = X1 ...) through the noise."""
    from lm_eval.tasks.ruler import vt_utils

    lengths, samples = _settings(kwargs)
    tokenizer = _tokenizer(kwargs)
    icl = vt_utils.sys_vartrack_w_noise_random(tokenizer=tokenizer, num_samples=1, max_seq_length=500, incremental=5)[0]
    return _dataset(
        vt_utils.sys_vartrack_w_noise_random(tokenizer=tokenizer, num_samples=samples, max_seq_length=seq, icl_example=icl)
        for seq in lengths
    )


def qa_hotpot(**kwargs):
    """Multi-document QA: the HotpotQA question's documents among distractor documents (chat-with-a-PDF shape)."""
    from lm_eval.tasks.ruler import qa_utils

    lengths, samples = _settings(kwargs)
    tokenizer = _tokenizer(kwargs)
    qas, docs = qa_utils.read_hotpotqa()
    return _dataset(
        qa_utils.generate_samples(tokenizer=tokenizer, docs=docs, qas=qas, num_samples=samples,
                                  tokens_to_generate=32, max_seq_length=seq)
        for seq in lengths
    )


# --- scoring: RULER's string match, one metric per length -----------------------------------------------

def _score(doc: dict, results: list[str], part: bool) -> dict:
    from lm_eval.tasks.ruler.common_utils import postprocess_pred, string_match_all, string_match_part

    metrics = {str(n): -1.0 for n in ALL_LENGTHS}
    pred = postprocess_pred(results)
    match = string_match_part if part else string_match_all
    metrics[str(doc["max_length"])] = match(pred, [doc["outputs"]])
    return metrics


def process_results(doc: dict, results: list[str]) -> dict:
    """Share of the expected values found in the reply (NIAH, variable tracking)."""
    return _score(doc, results, part=False)


def process_results_part(doc: dict, results: list[str]) -> dict:
    """Any of the reference answers found in the reply (QA)."""
    return _score(doc, results, part=True)


def aggregate(scores: list[float]) -> float:
    kept = [s for s in scores if s != -1]
    return sum(kept) / len(kept) if kept else -1.0


def prepare():
    """Fetch what the generators read (the runs may be offline): NLTK punkt_tab, the essays, HotpotQA."""
    from lm_eval.tasks.ruler import prepare_niah, qa_utils  # noqa: F401  (prepare_niah downloads punkt_tab on import)

    prepare_niah.get_haystack(type_haystack="essay")
    qa_utils.read_hotpotqa()
    print("RULER data ready")


if __name__ == "__main__":
    import sys

    if "--prepare" in sys.argv:
        prepare()
