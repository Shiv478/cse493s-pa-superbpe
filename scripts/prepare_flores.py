from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "deu_Latn",
    "rus_Cyrl",
    "arb_Arab",
    "hin_Deva",
    "zho_Hans",
    "jpn_Jpan",
    "tha_Thai",
    "swh_Latn",
]


def write_split(
    source_dir: Path,
    split: str,
    languages: list[str],
    output_path: Path,
    limit_per_language: int,
) -> int:
    split_dir = source_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing FLORES split directory: {split_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output_path.open("w", encoding="utf-8") as output:
        for lang in languages:
            path = split_dir / f"{lang}.{split}"
            if not path.exists():
                raise FileNotFoundError(f"Missing FLORES language file: {path}")
            count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    output.write(json.dumps({"lang": lang, "text": text}, ensure_ascii=False) + "\n")
                    rows += 1
                    count += 1
                    if limit_per_language and count >= limit_per_language:
                        break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 10-language FLORES train/test JSONL files.")
    parser.add_argument(
        "--source-dir",
        default="data/external/flores200_dataset",
        help="Extracted directory containing dev/ and devtest/.",
    )
    parser.add_argument("--train-out", default="data/processed/flores10_train.jsonl")
    parser.add_argument("--test-out", default="data/processed/flores10_test.jsonl")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="FLORES language tags to include.",
    )
    parser.add_argument(
        "--limit-per-language",
        type=int,
        default=0,
        help="Optional cap for faster local iteration. 0 means use full split.",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    train_rows = write_split(
        source_dir=source_dir,
        split="dev",
        languages=args.languages,
        output_path=Path(args.train_out),
        limit_per_language=args.limit_per_language,
    )
    test_rows = write_split(
        source_dir=source_dir,
        split="devtest",
        languages=args.languages,
        output_path=Path(args.test_out),
        limit_per_language=args.limit_per_language,
    )
    print(f"Wrote {train_rows} train rows to {args.train_out}")
    print(f"Wrote {test_rows} test rows to {args.test_out}")


if __name__ == "__main__":
    main()
