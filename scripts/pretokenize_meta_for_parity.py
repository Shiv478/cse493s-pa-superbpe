from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pretokenize_for_parity import get_pretokenizer


def write_dict_from_meta(meta_path: Path, output_path: Path, mode: str) -> None:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_files = [Path(path) for path in meta["train_files"]]
    pretokenizer = get_pretokenizer(mode)
    counts: Counter[str] = Counter()

    for path in train_files:
        print(f"Reading {path}", flush=True)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                for pretoken, _span in pretokenizer.pre_tokenize_str(line):
                    if pretoken:
                        counts[pretoken] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for pretoken, count in counts.most_common():
            if " " in pretoken:
                raise ValueError(f"Dictionary token contains a literal space: {pretoken!r}")
            output.write(f"{pretoken} {count}\n")

    print(f"Wrote {len(counts)} unique pretokens to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretokenize exactly the corpus files recorded in an official SuperBPE/BPE meta.json."
    )
    parser.add_argument("--mode", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--meta", required=True, help="Path to meta.json containing train_files.")
    parser.add_argument("--out", required=True, help="Output dictionary path.")
    args = parser.parse_args()

    write_dict_from_meta(Path(args.meta), Path(args.out), args.mode)


if __name__ == "__main__":
    main()
