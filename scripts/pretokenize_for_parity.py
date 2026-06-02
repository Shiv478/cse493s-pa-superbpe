from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


STAGE1_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

STAGE2_REGEX = r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"


def get_pretokenizer(mode: str):
    try:
        from tokenizers import Regex, pre_tokenizers
        from tokenizers.pre_tokenizers import ByteLevel, Split
    except ImportError as exc:
        raise SystemExit(
            "This script needs the SuperBPE/tokenizers environment. "
            "Install vendor/superbpe requirements in a Python 3.12 + Rust environment first."
        ) from exc

    regex = STAGE1_REGEX if mode == "stage1" else STAGE2_REGEX
    return pre_tokenizers.Sequence(
        [
            Split(pattern=Regex(regex), behavior="isolated", invert=False),
            ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
        ]
    )


def write_dict(input_path: Path, output_path: Path, mode: str) -> None:
    pretokenizer = get_pretokenizer(mode)
    counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for pretoken, _span in pretokenizer.pre_tokenize_str(line):
                if pretoken:
                    counts[pretoken] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for pretoken, count in counts.most_common():
            if " " in pretoken:
                raise ValueError(
                    f"Pretoken contains a literal space, which parity-aware-bpe --dict-input cannot parse: {pretoken!r}"
                )
            output.write(f"{pretoken} {count}\n")

    print(f"{input_path} -> {output_path} ({len(counts)} unique pretokens)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply SuperBPE stage regex + ByteLevel and emit parity-aware-bpe dict-input files."
    )
    parser.add_argument("--mode", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--input-dir", required=True, help="Directory of <lang>.txt files.")
    parser.add_argument("--out-dir", required=True, help="Directory for <lang>.dict files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    for input_path in sorted(input_dir.glob("*.txt")):
        write_dict(input_path, out_dir / f"{input_path.stem}.dict", args.mode)


if __name__ == "__main__":
    main()

