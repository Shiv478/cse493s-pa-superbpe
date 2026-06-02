from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict


TextsByLanguage = dict[str, list[str]]


def load_jsonl(path: str | Path) -> TextsByLanguage:
    """Load {"lang": ..., "text": ...} records into a language-indexed corpus."""
    grouped: DefaultDict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                lang = record["lang"]
                text = record["text"]
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} missing field {exc}") from exc
            if not isinstance(lang, str) or not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} lang and text must be strings")
            if text:
                grouped[lang].append(text)
    return dict(sorted(grouped.items()))


def split_train_eval(
    texts_by_language: TextsByLanguage,
    eval_every: int = 4,
) -> tuple[TextsByLanguage, TextsByLanguage]:
    """Create a deterministic tiny train/eval split for smoke experiments."""
    train: TextsByLanguage = {}
    eval_set: TextsByLanguage = {}
    for lang, texts in texts_by_language.items():
        if len(texts) < 2:
            train[lang] = list(texts)
            eval_set[lang] = list(texts)
            continue

        train_rows: list[str] = []
        eval_rows: list[str] = []
        for idx, text in enumerate(texts):
            if idx % eval_every == eval_every - 1:
                eval_rows.append(text)
            else:
                train_rows.append(text)

        train[lang] = train_rows or list(texts)
        eval_set[lang] = eval_rows or texts[-1:]
    return train, eval_set


def character_counts(texts_by_language: TextsByLanguage) -> dict[str, int]:
    return {
        lang: sum(len(text) for text in texts) or 1
        for lang, texts in texts_by_language.items()
    }

