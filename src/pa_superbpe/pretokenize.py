from __future__ import annotations

import unicodedata
from typing import Iterable, Literal


PretokenMode = Literal["stage1", "stage2"]
SymbolSequence = tuple[str, ...]


def _category_bucket(char: str) -> str:
    category = unicodedata.category(char)
    if char.isspace():
        return "space"
    if category[0] in {"P", "S"}:
        return "punct"
    if category[0] == "N":
        return "number"
    if category[0] == "M":
        return "mark"
    return "letter"


def stage1_pretokenize(text: str) -> list[SymbolSequence]:
    """Approximate the GPT-style stage-1 rule: no token crosses whitespace/category boundaries."""
    if not text:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []
    current_bucket: str | None = None

    for char in text:
        bucket = _category_bucket(char)
        if bucket == "space":
            if current:
                chunks.append(current)
                current = []
            chunks.append([char])
            current_bucket = None
            continue

        if current and bucket != current_bucket:
            chunks.append(current)
            current = []

        current.append(char)
        current_bucket = bucket

    if current:
        chunks.append(current)

    return [tuple(chunk) for chunk in chunks if chunk]


def stage2_pretokenize(text: str) -> list[SymbolSequence]:
    """SuperBPE-style whitespace lifting: allow the whole sentence to be merged across spaces."""
    return [tuple(text)] if text else []


def pretokenize(text: str, mode: PretokenMode) -> list[SymbolSequence]:
    if mode == "stage1":
        return stage1_pretokenize(text)
    if mode == "stage2":
        return stage2_pretokenize(text)
    raise ValueError(f"unknown pretokenizer mode: {mode}")


def flatten(sequences: Iterable[SymbolSequence]) -> list[str]:
    return [symbol for sequence in sequences for symbol in sequence]

