"""Synthetic data for throughput runs: random token ids, no file I/O.

Every microbatch is fully packed (no padding), so each step processes exactly
``num_tokens_per_microbatch`` tokens per data-parallel rank.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TokenizedTrainingMicrobatch
from torchtitan.components.tokenizer import BaseTokenizer


class NullTokenizer(BaseTokenizer):
    """Tokenizer stand-in: random-token runs never encode or decode text.

    Avoids reading tokenizer files from the shared filesystem on every rank.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        vocab_size: int

    def __init__(self, config: Config, **kwargs: Any) -> None:
        del kwargs
        super().__init__()
        self._vocab_size = config.vocab_size

    def encode(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError("NullTokenizer cannot encode text")

    def decode(self, *args, **kwargs) -> str:
        raise NotImplementedError("NullTokenizer cannot decode tokens")

    def get_vocab_size(self) -> int:
        return self._vocab_size


class RandomTokenDataLoader(BaseDataLoader):
    """Cycles through a small pool of pre-generated random microbatches.

    Each microbatch holds ``num_tokens_per_microbatch // max_context_length``
    sequences of ``max_context_length`` tokens, with positions restarting at
    each sequence boundary (same layout as TorchTitan's packed text collator).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        vocab_size: int
        pool_size: int = 8
        """Distinct microbatches kept in pinned host memory and cycled."""
        seed: int = 1234

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs, dp_world_size
        if num_tokens_per_microbatch % max_context_length != 0:
            raise ValueError(
                f"num_tokens_per_microbatch ({num_tokens_per_microbatch}) must be a "
                f"multiple of max_context_length ({max_context_length})"
            )
        self.max_num_documents = config.max_num_documents
        num_sequences = num_tokens_per_microbatch // max_context_length
        positions = torch.arange(max_context_length, dtype=torch.int64).repeat(
            num_sequences
        )
        pin = torch.cuda.is_available()
        generator = torch.Generator().manual_seed(config.seed + dp_rank)

        self._pool = []
        for _ in range(config.pool_size):
            tokens = torch.randint(
                0,
                config.vocab_size,
                (num_tokens_per_microbatch + 1,),
                dtype=torch.int64,
                generator=generator,
            )
            microbatch = TokenizedTrainingMicrobatch(
                input=tokens[:-1].clone(),
                labels=tokens[1:].clone(),
                positions=positions.clone(),
                padding_mask=torch.zeros(num_tokens_per_microbatch, dtype=torch.bool),
                num_valid_tokens=num_tokens_per_microbatch,
            )
            if pin:
                for name in ("input", "labels", "positions", "padding_mask"):
                    setattr(microbatch, name, getattr(microbatch, name).pin_memory())
            self._pool.append(microbatch)

    def __iter__(self) -> Iterator[TokenizedTrainingMicrobatch]:
        while True:
            yield from self._pool

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass
