from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise SystemExit(f"Could not find expected patch target in {path}")
    return text.replace(old, new, 1)


def patch_utils(superbpe_dir: Path) -> None:
    path = superbpe_dir / "utils.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """def train_or_extend_tokenizer(
    text_files: str,
    vocab_size: int = 100000,
    regex_string: str = None,
):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(show_progress=True, vocab_size=vocab_size)
""",
        """def train_or_extend_tokenizer(
    text_files: str,
    vocab_size: int = 100000,
    regex_string: str = None,
    min_frequency: int = 2,
):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(show_progress=True, vocab_size=vocab_size, min_frequency=min_frequency)
""",
        path,
    )
    text = replace_once(
        text,
        """    print(f"Using regex string: {regex_string}", flush=True)
""",
        """    print(f"Using regex string: {regex_string}", flush=True)
    print(f"Using min_frequency: {min_frequency}", flush=True)
""",
        path,
    )
    path.write_text(text, encoding="utf-8")


def patch_train_tokenizer(superbpe_dir: Path) -> None:
    path = superbpe_dir / "train_tokenizer.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        """@click.option(
    "--regex_string",
    type=str,
    default=None,
    help="Regex for pretokenization.",
)
def main(
    output_dir: str,
    num_bytes: int,
    corpus_dir: str,
    vocab_size: int,
    regex_string: str,
):
""",
        """@click.option(
    "--regex_string",
    type=str,
    default=None,
    help="Regex for pretokenization.",
)
@click.option(
    "--min_frequency",
    type=int,
    default=2,
    help="Minimum frequency for a merge candidate.",
)
def main(
    output_dir: str,
    num_bytes: int,
    corpus_dir: str,
    vocab_size: int,
    regex_string: str,
    min_frequency: int,
):
""",
        path,
    )
    text = replace_once(
        text,
        """    tokenizer = train_or_extend_tokenizer(
        train_files,
        vocab_size=vocab_size,
        regex_string=regex_string,
    )
""",
        """    tokenizer = train_or_extend_tokenizer(
        train_files,
        vocab_size=vocab_size,
        regex_string=regex_string,
        min_frequency=min_frequency,
    )
""",
        path,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch official SuperBPE train_tokenizer.py to expose BpeTrainer(min_frequency=2)."
    )
    parser.add_argument("--superbpe-dir", required=True)
    args = parser.parse_args()

    superbpe_dir = Path(args.superbpe_dir)
    patch_utils(superbpe_dir)
    patch_train_tokenizer(superbpe_dir)
    print(f"Patched SuperBPE min_frequency support in {superbpe_dir}")


if __name__ == "__main__":
    main()
