from __future__ import annotations

import argparse
from pathlib import Path


VERSION_PREFIX = "#version:"


def read_merge_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(VERSION_PREFIX):
            continue
        lines.append(stripped)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or combine BPE merges files while keeping exactly one #version header."
    )
    parser.add_argument("--stage1", required=True, help="Stage-1 merges file.")
    parser.add_argument("--stage2", help="Optional stage-2 merges file.")
    parser.add_argument("--out", required=True, help="Output merges file.")
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Write no #version header. Useful for parity-aware-bpe --preload compatibility.",
    )
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = read_merge_lines(Path(args.stage1))
    if args.stage2:
        lines.extend(read_merge_lines(Path(args.stage2)))

    output_lines = []
    if not args.no_header:
        output_lines.append("#version: 0.2")
    output_lines.extend(lines)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} merges to {output_path}")


if __name__ == "__main__":
    main()
