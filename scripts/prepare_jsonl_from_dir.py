from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a directory of language-tagged .txt files into experiment JSONL."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing files like eng_Latn.txt.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--limit-per-language", type=int, default=0, help="Optional max non-empty lines per file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as output:
        for text_path in sorted(input_dir.glob("*.txt")):
            lang = text_path.stem
            count = 0
            with text_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    output.write(json.dumps({"lang": lang, "text": text}, ensure_ascii=False) + "\n")
                    rows_written += 1
                    count += 1
                    if args.limit_per_language and count >= args.limit_per_language:
                        break

    print(f"Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
