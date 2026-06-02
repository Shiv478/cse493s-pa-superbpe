from __future__ import annotations

import argparse
import json
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


def read_merges(path: Path) -> list[str]:
    merges: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#version:"):
            continue
        merges.append(stripped)
    return merges


def build_vocab(merges: list[str]) -> dict[str, int]:
    from tokenizers.pre_tokenizers import ByteLevel

    vocab = {symbol: index for index, symbol in enumerate(ByteLevel.alphabet())}
    next_index = len(vocab)
    for merge in merges:
        left, right = merge.split()
        token = left + right
        if token not in vocab:
            vocab[token] = next_index
            next_index += 1
    return vocab


def make_pretokenizer(kind: str):
    from tokenizers import Regex, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel, Split, Whitespace

    if kind == "parity_bpe":
        return pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)])
    if kind == "pa_superbpe":
        return pre_tokenizers.Sequence(
            [
                Split(pattern=Regex(STAGE2_REGEX), behavior="isolated", invert=False),
                ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            ]
        )
    if kind == "stage1":
        return pre_tokenizers.Sequence(
            [
                Split(pattern=Regex(STAGE1_REGEX), behavior="isolated", invert=False),
                ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False),
            ]
        )
    raise ValueError(f"Unsupported tokenizer kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Hugging Face tokenizer.json from parity-aware BPE merges.")
    parser.add_argument("--merges", required=True, help="Path to merges.txt.")
    parser.add_argument("--out-dir", required=True, help="Directory to write tokenizer.json, merges.txt, vocab.json.")
    parser.add_argument(
        "--kind",
        choices=["parity_bpe", "pa_superbpe", "stage1"],
        required=True,
        help="Pretokenizer to store in tokenizer.json.",
    )
    parser.add_argument(
        "--expected-vocab-size",
        type=int,
        help="Fail if the assembled tokenizer vocab size does not match this value.",
    )
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE

    merges = read_merges(Path(args.merges))
    vocab = build_vocab(merges)
    if args.expected_vocab_size is not None and len(vocab) != args.expected_vocab_size:
        raise SystemExit(
            f"Expected vocab size {args.expected_vocab_size}, got {len(vocab)} "
            f"from {len(merges)} merges."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "merges.txt").write_text("#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[tuple(merge.split()) for merge in merges]))
    tokenizer.pre_tokenizer = make_pretokenizer(args.kind)
    tokenizer.decoder = ByteLevelDecoder(add_prefix_space=True, trim_offsets=True, use_regex=True)
    tokenizer.save(str(out_dir / "tokenizer.json"))

    print(f"Wrote {out_dir / 'tokenizer.json'}")
    print(f"vocab_size={len(vocab)} merges={len(merges)}")


if __name__ == "__main__":
    main()
