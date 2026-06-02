from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_superbpe.evaluate import evaluate_tokenizer, summarize
from pa_superbpe.tokenizer import BPETokenizer
from pa_superbpe.training import TrainConfig, train_all_tokenizers


TOY_CORPUS = {
    "eng_Latn": [
        "new york is big",
        "new york is busy",
        "new york is bright",
        "new york has trains",
        "new york has parks",
        "new york has lights",
    ],
    "zho_Hans": [
        "北京很大",
        "北京很忙",
        "北京明亮",
    ],
}


EXAMPLES = {
    "eng_Latn": "new york is bright",
    "zho_Hans": "北京很大",
}


def compact(tokens: list[str]) -> str:
    return " | ".join(token.replace(" ", "_") for token in tokens)


def print_merges(tokenizer: BPETokenizer, limit: int = 12) -> None:
    print(f"\n{tokenizer.name} first merges:")
    for left, right in tokenizer.merges[:limit]:
        merged = left + right
        print(f"  {left!r} + {right!r} -> {merged!r}")

    space_merges = [(left, right) for left, right in tokenizer.merges if " " in left + right]
    if space_merges:
        print(f"{tokenizer.name} merges that cross/include spaces:")
        for left, right in space_merges[:limit]:
            merged = left + right
            print(f"  {left!r} + {right!r} -> {merged!r}")
    else:
        print(f"{tokenizer.name} has no merges that cross/include spaces.")


def main() -> None:
    config = TrainConfig(stage1_merges=8, stage2_merges=6, min_frequency=1)
    results = train_all_tokenizers(TOY_CORPUS, config)

    print("Toy corpus:")
    print("  English appears 6 times and repeats the phrase 'new york'.")
    print("  Chinese appears 3 times and repeats '北京'.")
    print("  This makes it easy to see which method chases global frequency vs fairness.\n")

    rows = []
    for name, result in results.items():
        tokenizer = result.tokenizer
        metrics = evaluate_tokenizer(tokenizer, TOY_CORPUS)
        summary = summarize(name, metrics)
        rows.append(summary)
        print("=" * 80)
        print_merges(tokenizer)
        print(f"\n{name} tokenization examples:")
        for lang, text in EXAMPLES.items():
            tokens = tokenizer.encode(text)
            print(f"  {lang}: {text!r}")
            print(f"    tokens ({len(tokens)}): {compact(tokens)}")
        print(
            f"\n{name} summary: mean tokens/char={summary.mean_tokens_per_char:.3f}, "
            f"gini={summary.gini:.3f}, max/min={summary.max_min_ratio:.2f}"
        )

    print("\nOverall summary:")
    for row in sorted(rows, key=lambda item: item.gini):
        print(
            f"  {row.method:12s} gini={row.gini:.3f} "
            f"mean_tokens/char={row.mean_tokens_per_char:.3f}"
        )


if __name__ == "__main__":
    main()
