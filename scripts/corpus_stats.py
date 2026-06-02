from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_superbpe.data import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Print corpus stats for tokenizer training.")
    parser.add_argument("data", help="JSONL file with {'lang', 'text'} rows.")
    args = parser.parse_args()

    corpus = load_jsonl(args.data)
    all_text = "".join(text for rows in corpus.values() for text in rows)
    total_rows = sum(len(rows) for rows in corpus.values())
    total_chars = len(all_text)
    alphabet = set(all_text)

    print(f"file: {args.data}")
    print(f"languages: {len(corpus)}")
    print(f"rows: {total_rows}")
    print(f"characters: {total_chars}")
    print(f"unique characters: {len(alphabet)}")
    print()
    print("rows by language:")
    for lang, rows in sorted(corpus.items()):
        chars = sum(len(text) for text in rows)
        print(f"  {lang:9s} rows={len(rows):5d} chars={chars:8d}")
    print()
    print("rule of thumb:")
    print("  40k/200k tokenizers need much larger training corpora than FLORES alone.")
    print("  If requested merges stop early, check outputs/*/experiment_manifest.json.")


if __name__ == "__main__":
    main()
