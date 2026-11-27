"""Helpers of the milabench lm-eval tasks (referenced as ``!function utils.<name>`` from the task YAMLs)."""

import random
import re

# MMLU-Pro: questions per category in the stratified sample (14 categories -> ~2000 questions)
MMLU_PRO_PER_CATEGORY = 143

LETTERS = "ABCDEFGHIJ"
# case-insensitive: the task's filter lowercases the responses
ANSWER_INT = re.compile(r"(?:answer:\s*\**\s*|\\boxed\{)\s*(\d+)", re.IGNORECASE)


def mmlu_pro_sample(dataset):
    """Same number of questions from every category, picked with a fixed seed (the test split is sorted)."""
    by_category = {}
    for i, category in enumerate(dataset["category"]):
        by_category.setdefault(category, []).append(i)
    rng = random.Random(0)
    keep = []
    for category in sorted(by_category):
        rows = by_category[category]
        keep += rng.sample(rows, min(MMLU_PRO_PER_CATEGORY, len(rows)))
    return dataset.select(sorted(keep))


def mmlu_pro_text(doc):
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(doc["options"]))
    return (f"{doc['question'].strip()}\n\n{options}\n\n"
            "Think step by step, then end your response with a final line of the form \"Answer: <letter>\".")


def aime_avg(doc, results):
    """Accuracy averaged over the task's ``repeats`` samples (avg@k): the last integer answer of each response."""
    responses = [r for group in results for r in (group if isinstance(group, list) else [group])]
    target = int(str(doc["answer"]).strip())
    correct = []
    for response in responses:
        found = ANSWER_INT.findall(response or "")
        correct.append(bool(found) and int(found[-1]) == target)
    return {"exact_match": sum(correct) / len(correct) if correct else 0.0}
