from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pa_superbpe.pretokenize import PretokenMode, SymbolSequence, flatten, pretokenize


Pair = tuple[str, str]


def apply_merge(sequence: SymbolSequence, pair: Pair) -> SymbolSequence:
    """Apply one BPE merge to a single symbol sequence."""
    if len(sequence) < 2:
        return sequence

    merged: list[str] = []
    index = 0
    left, right = pair
    replacement = left + right

    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
            merged.append(replacement)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return tuple(merged)


def apply_merges(sequences: Iterable[SymbolSequence], merges: Iterable[Pair]) -> list[SymbolSequence]:
    updated = list(sequences)
    for pair in merges:
        updated = [apply_merge(sequence, pair) for sequence in updated]
    return updated


@dataclass(frozen=True)
class BPETokenizer:
    name: str
    merges: list[Pair]
    pretokenizer: PretokenMode

    def encode(self, text: str) -> list[str]:
        sequences = pretokenize(text, self.pretokenizer)
        return flatten(apply_merges(sequences, self.merges))

    def count_tokens(self, texts: Iterable[str]) -> int:
        return sum(len(self.encode(text)) for text in texts)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pretokenizer": self.pretokenizer,
            "merges": [[left, right] for left, right in self.merges],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "BPETokenizer":
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(
            name=str(payload["name"]),
            pretokenizer=str(payload["pretokenizer"]),  # type: ignore[arg-type]
            merges=merges,  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

