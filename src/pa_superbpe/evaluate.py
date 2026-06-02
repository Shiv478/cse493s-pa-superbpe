from __future__ import annotations

from dataclasses import dataclass

from pa_superbpe.data import TextsByLanguage
from pa_superbpe.tokenizer import BPETokenizer


@dataclass(frozen=True)
class LanguageMetric:
    method: str
    language: str
    total_chars: int
    total_tokens: int
    tokens_per_char: float


@dataclass(frozen=True)
class SummaryMetric:
    method: str
    mean_tokens_per_char: float
    gini: float
    max_min_ratio: float
    english_tokens_per_char: float | None


def gini(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(value for value in values if value >= 0)
    total = sum(sorted_values)
    if total == 0:
        return 0.0
    n = len(sorted_values)
    weighted_sum = sum((2 * index - n - 1) * value for index, value in enumerate(sorted_values, 1))
    return weighted_sum / (n * total)


def max_min_ratio(values: list[float]) -> float:
    positive = [value for value in values if value > 0]
    if not positive:
        return 0.0
    return max(positive) / min(positive)


def evaluate_tokenizer(tokenizer: BPETokenizer, texts_by_language: TextsByLanguage) -> list[LanguageMetric]:
    rows: list[LanguageMetric] = []
    for lang, texts in texts_by_language.items():
        total_chars = sum(len(text) for text in texts)
        total_tokens = tokenizer.count_tokens(texts)
        tokens_per_char = total_tokens / total_chars if total_chars else 0.0
        rows.append(
            LanguageMetric(
                method=tokenizer.name,
                language=lang,
                total_chars=total_chars,
                total_tokens=total_tokens,
                tokens_per_char=tokens_per_char,
            )
        )
    return rows


def summarize(method: str, rows: list[LanguageMetric]) -> SummaryMetric:
    values = [row.tokens_per_char for row in rows]
    english = next((row.tokens_per_char for row in rows if row.language == "eng_Latn"), None)
    return SummaryMetric(
        method=method,
        mean_tokens_per_char=sum(values) / len(values) if values else 0.0,
        gini=gini(values),
        max_min_ratio=max_min_ratio(values),
        english_tokens_per_char=english,
    )

