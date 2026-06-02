from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_superbpe.data import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split JSONL corpus into one text file per language.")
    parser.add_argument("--data", required=True, help="Input JSONL with {'lang', 'text'} rows.")
    parser.add_argument("--out-dir", required=True, help="Output directory for <lang>.txt files.")
    args = parser.parse_args()

    corpus = load_jsonl(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang, texts in sorted(corpus.items()):
        path = out_dir / f"{lang}.txt"
        path.write_text("\n".join(texts) + "\n", encoding="utf-8")
        print(f"{lang}: {len(texts)} rows -> {path}")


if __name__ == "__main__":
    main()

