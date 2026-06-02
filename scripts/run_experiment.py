from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_superbpe.data import load_jsonl, split_train_eval
from pa_superbpe.evaluate import LanguageMetric, SummaryMetric, evaluate_tokenizer, summarize
from pa_superbpe.training import (
    TrainConfig,
    TrainingResult,
    train_single_stage,
    train_two_stage,
)


def write_language_csv(path: Path, rows: list[LanguageMetric]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "language", "total_chars", "total_tokens", "tokens_per_char"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "language": row.language,
                    "total_chars": row.total_chars,
                    "total_tokens": row.total_tokens,
                    "tokens_per_char": f"{row.tokens_per_char:.6f}",
                }
            )


def write_summary_csv(path: Path, rows: list[SummaryMetric]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "mean_tokens_per_char", "gini", "max_min_ratio", "eng_Latn_tokens_per_char"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "mean_tokens_per_char": f"{row.mean_tokens_per_char:.6f}",
                    "gini": f"{row.gini:.6f}",
                    "max_min_ratio": f"{row.max_min_ratio:.6f}",
                    "eng_Latn_tokens_per_char": ""
                    if row.english_tokens_per_char is None
                    else f"{row.english_tokens_per_char:.6f}",
                }
            )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    train_texts: dict[str, list[str]],
    eval_texts: dict[str, list[str]],
    results: dict[str, TrainingResult],
) -> None:
    manifest = {
        "train_data": args.data,
        "eval_data": args.eval_data or "held out from training data",
        "train_rows": sum(len(rows) for rows in train_texts.values()),
        "eval_rows": sum(len(rows) for rows in eval_texts.values()),
        "languages": sorted(train_texts),
        "requested": {
            "stage1_merges": args.stage1_merges,
            "stage2_merges": args.stage2_merges,
            "single_stage_total_merges": args.stage1_merges + args.stage2_merges,
            "min_frequency": args.min_frequency,
        },
        "actual_merges": {
            method: len(result.tokenizer.merges)
            for method, result in sorted(results.items())
        },
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def markdown_table(rows: list[SummaryMetric]) -> str:
    lines = [
        "| method | mean tokens/char | Gini | max/min | eng_Latn tokens/char |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        english = "n/a" if row.english_tokens_per_char is None else f"{row.english_tokens_per_char:.4f}"
        lines.append(
            f"| {row.method} | {row.mean_tokens_per_char:.4f} | {row.gini:.4f} | "
            f"{row.max_min_ratio:.2f} | {english} |"
        )
    return "\n".join(lines) + "\n"


def write_gini_svg(path: Path, rows: list[SummaryMetric]) -> None:
    width = 760
    height = 360
    margin_left = 140
    margin_bottom = 70
    chart_width = width - margin_left - 40
    chart_height = height - 70 - margin_bottom
    max_value = max((row.gini for row in rows), default=1.0) or 1.0
    bar_height = chart_height / max(len(rows), 1) * 0.62
    gap = chart_height / max(len(rows), 1) * 0.38

    palette = ["#2563eb", "#059669", "#d97706", "#7c3aed"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="36" font-family="Arial" font-size="22" font-weight="700">Cross-lingual compression disparity</text>',
        '<text x="24" y="60" font-family="Arial" font-size="13" fill="#555">Lower Gini is more equitable across languages.</text>',
    ]

    y = 86
    for index, row in enumerate(rows):
        bar_width = (row.gini / max_value) * chart_width
        color = palette[index % len(palette)]
        parts.append(
            f'<text x="24" y="{y + bar_height * 0.65:.1f}" font-family="Arial" font-size="14">{row.method}</text>'
        )
        parts.append(
            f'<rect x="{margin_left}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>'
        )
        parts.append(
            f'<text x="{margin_left + bar_width + 8:.1f}" y="{y + bar_height * 0.65:.1f}" '
            f'font-family="Arial" font-size="13">{row.gini:.4f}</text>'
        )
        y += bar_height + gap

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def train_methods(train_texts: dict[str, list[str]], config: TrainConfig) -> dict[str, TrainingResult]:
    total_single_stage_merges = config.stage1_merges + config.stage2_merges
    jobs = [
        (
            "bpe",
            lambda: train_single_stage(
                name="bpe",
                texts_by_language=train_texts,
                num_merges=total_single_stage_merges,
                selection="global",
                min_frequency=config.min_frequency,
                progress_interval=config.progress_interval,
            ),
        ),
        (
            "superbpe",
            lambda: train_two_stage(
                name="superbpe",
                texts_by_language=train_texts,
                selection="global",
                config=config,
            ),
        ),
        (
            "parity_bpe",
            lambda: train_single_stage(
                name="parity_bpe",
                texts_by_language=train_texts,
                num_merges=total_single_stage_merges,
                selection="parity",
                min_frequency=config.min_frequency,
                progress_interval=config.progress_interval,
            ),
        ),
        (
            "pa_superbpe",
            lambda: train_two_stage(
                name="pa_superbpe",
                texts_by_language=train_texts,
                selection="parity",
                config=config,
            ),
        ),
    ]

    results: dict[str, TrainingResult] = {}
    for method, job in jobs:
        print(f"Training {method}...", flush=True)
        results[method] = job()
        print(f"Finished {method} with {len(results[method].tokenizer.merges)} merges.", flush=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA-SuperBPE compression disparity experiments.")
    parser.add_argument("--data", default="data/sample_multilingual.jsonl", help="Training JSONL corpus path.")
    parser.add_argument("--eval-data", default="", help="Optional evaluation JSONL path. Defaults to held-out train rows.")
    parser.add_argument("--out", default="outputs/prototype_demos/smoke_test", help="Output directory.")
    parser.add_argument("--stage1-merges", type=int, default=80, help="Stage-1 merge budget.")
    parser.add_argument("--stage2-merges", type=int, default=80, help="Stage-2 merge budget.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum pair frequency for a merge.")
    parser.add_argument("--eval-every", type=int, default=4, help="Hold out every Nth row per language for eval.")
    parser.add_argument("--progress-interval", type=int, default=0, help="Print progress every N merges.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out)
    tokenizer_dir = output_dir / "tokenizers"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    train_source = load_jsonl(args.data)
    if args.eval_data:
        train_texts = train_source
        eval_texts = load_jsonl(args.eval_data)
    else:
        train_texts, eval_texts = split_train_eval(train_source, eval_every=args.eval_every)
    config = TrainConfig(
        stage1_merges=args.stage1_merges,
        stage2_merges=args.stage2_merges,
        min_frequency=args.min_frequency,
        progress_interval=args.progress_interval,
    )

    print(f"Training languages: {', '.join(sorted(train_texts))}", flush=True)
    print(f"Evaluation languages: {', '.join(sorted(eval_texts))}", flush=True)
    print(f"Train rows: {sum(len(rows) for rows in train_texts.values())}", flush=True)
    print(f"Eval rows: {sum(len(rows) for rows in eval_texts.values())}", flush=True)
    results = train_methods(train_texts, config)

    language_rows: list[LanguageMetric] = []
    summary_rows: list[SummaryMetric] = []
    for method, result in results.items():
        result.tokenizer.save(tokenizer_dir / f"{method}.json")
        rows = evaluate_tokenizer(result.tokenizer, eval_texts)
        language_rows.extend(rows)
        summary_rows.append(summarize(method, rows))

    summary_rows = sorted(summary_rows, key=lambda row: row.gini)
    write_language_csv(output_dir / "compression_by_language.csv", language_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_manifest(output_dir / "experiment_manifest.json", args, train_texts, eval_texts, results)
    (output_dir / "summary.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    write_gini_svg(output_dir / "gini_bar.svg", summary_rows)

    print(markdown_table(summary_rows))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
