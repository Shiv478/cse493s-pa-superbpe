from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_superbpe.data import load_jsonl
from pa_superbpe.evaluate import gini, max_min_ratio


def load_hf_tokenizer(tokenizer_path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "This evaluator needs the Hugging Face tokenizers package. "
            "Use the SuperBPE Python 3.12 environment for exact SuperBPE tokenizers."
        ) from exc

    if tokenizer_path.is_dir():
        tokenizer_json = tokenizer_path / "tokenizer.json"
    else:
        tokenizer_json = tokenizer_path
    return Tokenizer.from_file(str(tokenizer_json))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Hugging Face tokenizer on JSONL language data.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json or tokenizer directory.")
    parser.add_argument("--data", required=True, help="JSONL eval data with {'lang', 'text'} rows.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    args = parser.parse_args()

    tokenizer = load_hf_tokenizer(Path(args.tokenizer))
    data = load_jsonl(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lang, texts in sorted(data.items()):
        total_chars = sum(len(text) for text in texts)
        total_tokens = sum(len(tokenizer.encode(text).ids) for text in texts)
        rows.append(
            {
                "language": lang,
                "total_chars": total_chars,
                "total_tokens": total_tokens,
                "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
            }
        )

    values = [row["tokens_per_char"] for row in rows]
    summary = {
        "tokenizer": str(args.tokenizer),
        "data": args.data,
        "languages": len(rows),
        "mean_tokens_per_char": sum(values) / len(values) if values else 0.0,
        "gini": gini(values),
        "max_min_ratio": max_min_ratio(values),
    }

    with (out_dir / "compression_by_language.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["language", "total_chars", "total_tokens", "tokens_per_char"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "language": row["language"],
                    "total_chars": row["total_chars"],
                    "total_tokens": row["total_tokens"],
                    "tokens_per_char": f"{row['tokens_per_char']:.6f}",
                }
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
