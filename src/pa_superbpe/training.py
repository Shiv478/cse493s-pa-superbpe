from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal

from pa_superbpe.data import TextsByLanguage, character_counts
from pa_superbpe.pretokenize import PretokenMode, SymbolSequence, pretokenize
from pa_superbpe.tokenizer import BPETokenizer, Pair, apply_merge


SelectionMode = Literal["global", "parity"]
SequencesByLanguage = dict[str, Counter[SymbolSequence]]


@dataclass(frozen=True)
class TrainConfig:
    stage1_merges: int = 80
    stage2_merges: int = 80
    min_frequency: int = 2
    max_stage2_words: int = 4
    progress_interval: int = 0


@dataclass(frozen=True)
class MergeStep:
    stage: str
    pair: Pair
    count: int
    selected_language: str


@dataclass(frozen=True)
class TrainingResult:
    tokenizer: BPETokenizer
    trace: list[MergeStep]


def build_sequences(texts_by_language: TextsByLanguage, mode: PretokenMode) -> SequencesByLanguage:
    return {
        lang: Counter(sequence for text in texts for sequence in pretokenize(text, mode))
        for lang, texts in texts_by_language.items()
    }


def count_pairs(sequences: Counter[SymbolSequence]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for sequence, frequency in sequences.items():
        for index in range(len(sequence) - 1):
            counts[(sequence[index], sequence[index + 1])] += frequency
    return counts


def apply_merge_by_language(sequences: SequencesByLanguage, pair: Pair) -> SequencesByLanguage:
    updated: SequencesByLanguage = {}
    for lang, lang_sequences in sequences.items():
        merged_sequences: Counter[SymbolSequence] = Counter()
        for sequence, frequency in lang_sequences.items():
            merged_sequences[apply_merge(sequence, pair)] += frequency
        updated[lang] = merged_sequences
    return updated


def _token_rates(sequences: SequencesByLanguage, chars_by_language: dict[str, int]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for lang, lang_sequences in sequences.items():
        token_count = sum(len(sequence) * frequency for sequence, frequency in lang_sequences.items())
        rates[lang] = token_count / chars_by_language[lang]
    return rates


def _stage2_filter(pair: Pair, max_words: int) -> bool:
    merged = pair[0] + pair[1]
    words = [part for part in merged.strip().split() if part]
    return len(words) <= max_words


def _best_global_pair(
    sequences: SequencesByLanguage,
    min_frequency: int,
    pair_filter: Callable[[Pair], bool] | None = None,
) -> tuple[Pair, int, str] | None:
    counts: Counter[Pair] = Counter()
    for lang_sequences in sequences.values():
        counts.update(count_pairs(lang_sequences))

    for pair, count in counts.most_common():
        if count < min_frequency:
            return None
        if pair_filter is None or pair_filter(pair):
            return pair, count, "global"
    return None


def _best_parity_pair(
    sequences: SequencesByLanguage,
    chars_by_language: dict[str, int],
    min_frequency: int,
    pair_filter: Callable[[Pair], bool] | None = None,
) -> tuple[Pair, int, str] | None:
    rates = _token_rates(sequences, chars_by_language)
    for lang, _rate in sorted(rates.items(), key=lambda item: item[1], reverse=True):
        counts = count_pairs(sequences[lang])
        for pair, count in counts.most_common():
            if count < min_frequency:
                break
            if pair_filter is None or pair_filter(pair):
                return pair, count, lang
    return _best_global_pair(sequences, min_frequency, pair_filter)


def _train_merges(
    sequences: SequencesByLanguage,
    chars_by_language: dict[str, int],
    steps: int,
    stage: str,
    selection: SelectionMode,
    min_frequency: int,
    pair_filter: Callable[[Pair], bool] | None = None,
    progress_interval: int = 0,
) -> tuple[list[Pair], list[MergeStep], SequencesByLanguage]:
    merges: list[Pair] = []
    trace: list[MergeStep] = []
    current = sequences

    for _ in range(steps):
        if selection == "global":
            selected = _best_global_pair(current, min_frequency, pair_filter)
        else:
            selected = _best_parity_pair(current, chars_by_language, min_frequency, pair_filter)

        if selected is None:
            break

        pair, count, selected_language = selected
        merges.append(pair)
        trace.append(MergeStep(stage=stage, pair=pair, count=count, selected_language=selected_language))
        current = apply_merge_by_language(current, pair)
        if progress_interval and len(merges) % progress_interval == 0:
            print(f"  {stage}: learned {len(merges)}/{steps} merges", flush=True)

    return merges, trace, current


def train_single_stage(
    name: str,
    texts_by_language: TextsByLanguage,
    num_merges: int,
    selection: SelectionMode,
    min_frequency: int,
    progress_interval: int = 0,
) -> TrainingResult:
    chars = character_counts(texts_by_language)
    sequences = build_sequences(texts_by_language, "stage1")
    merges, trace, _current = _train_merges(
        sequences=sequences,
        chars_by_language=chars,
        steps=num_merges,
        stage="stage1",
        selection=selection,
        min_frequency=min_frequency,
        progress_interval=progress_interval,
    )
    return TrainingResult(tokenizer=BPETokenizer(name=name, merges=merges, pretokenizer="stage1"), trace=trace)


def train_two_stage(
    name: str,
    texts_by_language: TextsByLanguage,
    selection: SelectionMode,
    config: TrainConfig,
) -> TrainingResult:
    chars = character_counts(texts_by_language)

    stage1_sequences = build_sequences(texts_by_language, "stage1")
    stage1_merges, stage1_trace, _stage1_current = _train_merges(
        sequences=stage1_sequences,
        chars_by_language=chars,
        steps=config.stage1_merges,
        stage="stage1",
        selection=selection,
        min_frequency=config.min_frequency,
        progress_interval=config.progress_interval,
    )

    stage2_sequences = build_sequences(texts_by_language, "stage2")
    for pair in stage1_merges:
        stage2_sequences = apply_merge_by_language(stage2_sequences, pair)
    stage2_filter = lambda pair: _stage2_filter(pair, config.max_stage2_words)
    stage2_merges, stage2_trace, _stage2_current = _train_merges(
        sequences=stage2_sequences,
        chars_by_language=chars,
        steps=config.stage2_merges,
        stage="stage2",
        selection=selection,
        min_frequency=config.min_frequency,
        pair_filter=stage2_filter,
        progress_interval=config.progress_interval,
    )

    tokenizer = BPETokenizer(
        name=name,
        merges=stage1_merges + stage2_merges,
        pretokenizer="stage2",
    )
    return TrainingResult(tokenizer=tokenizer, trace=stage1_trace + stage2_trace)


def train_all_tokenizers(texts_by_language: TextsByLanguage, config: TrainConfig) -> dict[str, TrainingResult]:
    total_single_stage_merges = config.stage1_merges + config.stage2_merges
    return {
        "bpe": train_single_stage(
            name="bpe",
            texts_by_language=texts_by_language,
            num_merges=total_single_stage_merges,
            selection="global",
            min_frequency=config.min_frequency,
            progress_interval=config.progress_interval,
        ),
        "superbpe": train_two_stage(
            name="superbpe",
            texts_by_language=texts_by_language,
            selection="global",
            config=config,
        ),
        "parity_bpe": train_single_stage(
            name="parity_bpe",
            texts_by_language=texts_by_language,
            num_merges=total_single_stage_merges,
            selection="parity",
            min_frequency=config.min_frequency,
            progress_interval=config.progress_interval,
        ),
        "pa_superbpe": train_two_stage(
            name="pa_superbpe",
            texts_by_language=texts_by_language,
            selection="parity",
            config=config,
        ),
    }
