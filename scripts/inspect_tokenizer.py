from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pa_superbpe.tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect learned tokenizer merges.")
    parser.add_argument("tokenizer", help="Path to tokenizer JSON.")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    print(f"name: {tokenizer.name}")
    print(f"pretokenizer: {tokenizer.pretokenizer}")
    print(f"merges: {len(tokenizer.merges)}")
    print("\nfirst merges:")
    for pair in tokenizer.merges[: args.limit]:
        print(repr(pair[0]), "+", repr(pair[1]), "=>", repr(pair[0] + pair[1]))

    superword_merges = [pair for pair in tokenizer.merges if " " in pair[0] + pair[1]]
    if superword_merges:
        print("\nmerges containing spaces:")
        for pair in superword_merges[: args.limit]:
            print(repr(pair[0]), "+", repr(pair[1]), "=>", repr(pair[0] + pair[1]))


if __name__ == "__main__":
    main()

