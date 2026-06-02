from __future__ import annotations
"""Independent ground-truth check: fast-learner GLOBAL mode == textbook standard BPE.

This implementation shares NO code with the parity-aware lineage. It is vanilla
BPE (Sennrich et al. / GPT-2): start from the characters of each pretoken, then
repeatedly merge the globally-most-frequent adjacent pair (tie-break: highest
count, then lexicographically-largest pair tuple, matching the learner's
`max(stats, key=(count, pair))`). If its merge list matches the fast learner run
with --global-merges == --symbols, then global mode IS standard BPE.

Writes a small subset corpus (top-N pretokens/lang) so both finish in seconds.
"""
import argparse
from collections import defaultdict
from pathlib import Path


def read_dict(path: Path, top_n: int):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            tok, _, cnt = line.rpartition(" ")
            rows.append((tok, int(cnt)))
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows[:top_n]


def standard_bpe(corpus, num_merges, min_freq=2):
    """corpus: list of (symbols:tuple[str], count:int). Returns list of (a,b) merges."""
    words = [(list(sym), cnt) for sym, cnt in corpus]
    merges = []
    for _ in range(num_merges):
        stats = defaultdict(int)
        for sym, cnt in words:
            for a, b in zip(sym, sym[1:]):
                stats[(a, b)] += cnt
        if not stats:
            break
        # global most-frequent; tie-break = largest pair tuple (matches learner)
        best = max(stats, key=lambda p: (stats[p], p))
        if stats[best] < min_freq:
            break
        merges.append(best)
        a, b = best
        ab = a + b
        for sym, _cnt in words:
            i = 0
            while i < len(sym) - 1:
                if sym[i] == a and sym[i + 1] == b:
                    sym[i:i + 2] = [ab]
                else:
                    i += 1
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-dir", required=True)
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--merges", type=int, default=200)
    ap.add_argument("--out", required=True, help="write standard-BPE merges here")
    ap.add_argument("--subset-out", required=True, help="write the subset dict dir here")
    args = ap.parse_args()

    ddir = Path(args.dict_dir)
    sub = Path(args.subset_out)
    sub.mkdir(parents=True, exist_ok=True)
    corpus = []
    for dp in sorted(ddir.glob("*.dict")):
        rows = read_dict(dp, args.top_n)
        # write subset dict (identical format) for the learner to consume
        (sub / dp.name).write_text(
            "".join(f"{t} {c}\n" for t, c in rows), encoding="utf-8")
        for tok, cnt in rows:
            corpus.append((tuple(tok), cnt))

    merges = standard_bpe(corpus, args.merges)
    out = Path(args.out)
    out.write_text("#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in merges),
                   encoding="utf-8")
    print(f"standard BPE: wrote {len(merges)} merges to {out}")
    print(f"subset dicts ({args.top_n}/lang) -> {sub}")


if __name__ == "__main__":
    main()
