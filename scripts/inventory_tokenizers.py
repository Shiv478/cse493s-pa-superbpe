from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path("vendor/superbpe/tokenizer_json")
    for path in sorted(root.iterdir()):
        tokenizer_json = path / "tokenizer.json"
        merges_txt = path / "merges.txt"
        if not path.is_dir() or not tokenizer_json.exists():
            continue
        tokenizer = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        vocab = tokenizer["model"].get("vocab", {})
        merges = tokenizer["model"].get("merges", [])
        merge_lines = sum(1 for _ in merges_txt.open(encoding="utf-8")) if merges_txt.exists() else 0
        print(f"{path.name}\tvocab={len(vocab)}\tjson_merges={len(merges)}\tmerges_txt_lines={merge_lines}")


if __name__ == "__main__":
    main()
