"""Real-text prompts for the serving client, from OpenHermes-2.5 (ShareGPT format).

``prepare.py`` extracts a sample of (first user turn, assistant reply) pairs once into
``$MILABENCH_DIR_DATA/serving/openhermes_pairs.json``; the 1.9 GB source is too slow to load in every pass.

* chat passes: vLLM's own ``--dataset-name sharegpt`` on that file (real prompt and reply lengths)
* ``--real-text`` passes: :class:`RealTextDataset`, real text cut to exactly ``--random-input-len`` tokens, so a
  pass differs from its random-token twin by the content only
"""

from __future__ import annotations

import json
import os
import random

import numpy as np

SOURCE_REPO = "teknium/OpenHermes-2.5"
SOURCE_FILE = "openhermes2_5.json"
NUM_PAIRS = 20000


def corpus_path() -> str:
    return os.path.join(os.environ["MILABENCH_DIR_DATA"], "serving", "openhermes_pairs.json")


def _pair(conversation: list[dict]) -> tuple[str, str] | None:
    """First user turn and the assistant reply that follows it (system turns skipped)."""
    turns = [t for t in conversation if t.get("from") != "system"]
    if len(turns) >= 2 and turns[0].get("from") == "human" and turns[1].get("from") == "gpt":
        return turns[0]["value"], turns[1]["value"]
    return None


def build_corpus(path: str | None = None, num_pairs: int = NUM_PAIRS, seed: int = 0) -> str:
    path = path or corpus_path()
    if os.path.exists(path):
        print(f"Text corpus already built: {path}")
        return path

    from huggingface_hub import hf_hub_download

    source = hf_hub_download(SOURCE_REPO, SOURCE_FILE, repo_type="dataset")
    print(f"Building text corpus from {source}")
    with open(source, encoding="utf-8") as f:
        data = json.load(f)
    pairs = [p for entry in data if (p := _pair(entry.get("conversations") or []))]
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:num_pairs]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([{"conversations": [{"from": "human", "value": p}, {"from": "gpt", "value": c}]} for p, c in pairs], f)
    os.replace(tmp, path)
    print(f"Text corpus: {len(pairs)} pairs -> {path}")
    return path


def make_real_text_dataset(datasets):
    """:class:`RealTextDataset` on ``vllm.benchmarks.datasets`` (imported by the caller, it is slow)."""

    class RealTextDataset(datasets.BenchmarkDataset):
        """Prompts of exactly ``input_len`` tokens cut from the corpus' conversations, joined into one stream.

        Each request starts at its own random offset in the stream.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            with open(self.dataset_path or corpus_path(), encoding="utf-8") as f:
                self.data = json.load(f)
            random.Random(self.random_seed).shuffle(self.data)

        def _stream(self, tokenizer, min_tokens: int) -> list[int]:
            ids = []
            for entry in self.data:
                for turn in entry["conversations"]:
                    ids += tokenizer(turn["value"] + "\n\n", add_special_tokens=False).input_ids
                if len(ids) >= min_tokens:
                    return ids
            raise ValueError(f"text corpus has {len(ids)} tokens, {min_tokens} needed")

        def sample(self, tokenizer, num_requests: int, input_len: int, output_len: int,
                   request_id_prefix: str = "", **kwargs) -> list:
            rng = np.random.default_rng(self.random_seed)
            stream = self._stream(tokenizer, max(4 * input_len, 262144))
            samples = []
            for i in range(num_requests):
                start = int(rng.integers(0, len(stream) - input_len))
                prompt, token_ids, _ = datasets.gen_prompt_decode_to_target_len(
                    tokenizer=tokenizer,
                    token_sequence=stream[start:start + input_len],
                    target_token_len=input_len,
                    add_special_tokens=False,
                    rng=rng,
                )
                samples.append(datasets.SampleRequest(
                    prompt=prompt,
                    prompt_len=len(token_ids),
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i),
                ))
            return samples

    return RealTextDataset
