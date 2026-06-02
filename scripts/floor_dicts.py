from __future__ import annotations

"""Frequency-floor (and/or top-N cap) per-language pretoken dict files.

The parity learner's per-merge cost scales with the number of unique pretokens
(→ unique adjacent pairs). On fragmented scripts (zho/jpn/tha) the corpus has a
huge tail of pretokens that appear exactly once (hapaxes) — e.g. ~6.6M of zho's
7.24M. Those rarely contribute a mergeable pair (the learner already enforces
min_frequency>=2 on pairs), so dropping them barely changes the learned merges
while shrinking the pair universe several-fold and making training tractable.

This is a corpus-reduction step, applied IDENTICALLY to all languages and
disclosed in the writeup. It does not touch the algorithm.

Input/output format is the parity `.dict` format: lines `<token> <count>`,
token possibly containing the Ġ word-boundary marker, single-space separated
from an integer count. (The token itself never contains a space — the parity
dict-input parser splits on the last space, but pretokenize_for_parity.py
guarantees no internal spaces, so a simple rsplit is exact.)

Usage:
  # drop hapaxes only (recommended headline reduction)
  python scripts/floor_dicts.py --in-dir <stage1_dir> --out-dir <floored_dir> --min-count 2

  # aggressive floor for a fast reference safety-net run
  python scripts/floor_dicts.py --in-dir <stage1_dir> --out-dir <floored_dir> --min-count 8

  # optional per-language cap (keep the N most frequent; balanced across langs)
  python scripts/floor_dicts.py --in-dir <dir> --out-dir <out> --min-count 2 --top-n 250000
"""

import argparse
from pathlib import Path


def floor_one(in_path: Path, out_path: Path, min_count: int, top_n: int | None) -> tuple[int, int]:
    """Filter one dict file. Returns (kept, total)."""
    rows: list[tuple[str, int]] = []
    total = 0
    with in_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            token, _, count_s = line.rpartition(" ")
            if not token:
                # malformed / single-field line — skip defensively
                continue
            try:
                count = int(count_s)
            except ValueError:
                continue
            if count >= min_count:
                rows.append((token, count))

    # optional top-N most frequent (stable: sort by count desc, then token for determinism)
    if top_n is not None and len(rows) > top_n:
        rows.sort(key=lambda r: (-r[1], r[0]))
        rows = rows[:top_n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for token, count in rows:
            out.write(f"{token} {count}\n")
    return len(rows), total


def main() -> None:
    ap = argparse.ArgumentParser(description="Frequency-floor per-language parity .dict files.")
    ap.add_argument("--in-dir", required=True, help="Directory of <lang>.dict files.")
    ap.add_argument("--out-dir", required=True, help="Output directory for floored dicts.")
    ap.add_argument("--min-count", type=int, default=2, help="Keep tokens with count >= this (default 2 = drop hapaxes).")
    ap.add_argument("--top-n", type=int, default=None, help="Optional: keep only the N most frequent tokens per language.")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    dicts = sorted(in_dir.glob("*.dict"))
    if not dicts:
        raise SystemExit(f"No .dict files in {in_dir}")

    grand_kept = grand_total = 0
    print(f"Flooring {len(dicts)} dicts: min_count>={args.min_count}"
          + (f", top_n={args.top_n}" if args.top_n else ""))
    for d in dicts:
        kept, total = floor_one(d, out_dir / d.name, args.min_count, args.top_n)
        grand_kept += kept
        grand_total += total
        pct = (kept / total * 100) if total else 0.0
        print(f"  {d.stem:9s} {total:9d} -> {kept:9d}  ({pct:4.0f}%)")
    print(f"TOTAL {grand_total} -> {grand_kept} "
          f"({grand_kept / grand_total * 100:.1f}% kept)")


if __name__ == "__main__":
    main()
