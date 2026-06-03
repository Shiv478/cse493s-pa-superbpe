from __future__ import annotations
"""Assemble the combined 40k results from the per-method evaluation outputs.

Reads outputs/final_results_40k/evaluation/<method>/compression_by_language.csv
for all six tokenizers and writes:
  - results_summary.csv          one row per method, both metric families
  - per_language_compression.csv per-language tokens & tokens/char, all methods

Two metric families are reported:
  * tokens-per-char  : tokens / Unicode-codepoint  (a compression measure, distorted across scripts)
  * token-count      : tokens to encode the SAME parallel FLORES content
                       (canonical cross-lingual fairness metric on parallel data)
Lower Gini / max-min = fairer. Run with the repo venv (no third-party deps needed).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "final_results_40k" / "evaluation"
OUT = ROOT / "outputs" / "final_results_40k"

# (label, eval-dir / short-key, stage-1 rule, stage-2 rule)
METHODS = [
    ("BPE",                         "bpe",                      "-",      "-"),
    ("SuperBPE",                    "superbpe",                 "global", "global"),
    ("PA-BPE",                      "pa_bpe",                   "parity", "-"),
    ("PA-SuperBPE",                 "pa_superbpe",              "parity", "parity"),
    ("Hybrid A (parity1->global2)", "hybrid_A_parity1_global2", "parity", "global"),
    ("Hybrid B (global1->parity2)", "hybrid_B_global1_parity2", "global", "parity"),
]
# short column key for the per-language table
SHORT = {"hybrid_A_parity1_global2": "hybrid_A", "hybrid_B_global1_parity2": "hybrid_B"}


def gini(values):
    xs = sorted(v for v in values if v >= 0)
    n, s = len(xs), sum(xs)
    return sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(xs)) / (n * s) if n and s else 0.0


def max_min(values):
    pos = [v for v in values if v > 0]
    return (max(pos) / min(pos)) if pos else 0.0


def load(key):
    rows = list(csv.DictReader((EVAL / key / "compression_by_language.csv").open()))
    return {
        r["language"]: {
            "chars": int(r["total_chars"]),
            "tokens": int(r["total_tokens"]),
            "tpc": float(r["tokens_per_char"]),
        }
        for r in rows
    }


def main():
    data = {key: load(key) for _, key, _, _ in METHODS}
    langs = sorted(next(iter(data.values())).keys())

    # ---- results_summary.csv ----
    with (OUT / "results_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "method", "stage1", "stage2", "mean_tokens_per_char",
            "gini_tokens_per_char", "maxmin_tokens_per_char",
            "gini_token_count", "maxmin_token_count",
            "eng_tokens", "eng_tokens_per_char",
        ])
        for label, key, s1, s2 in METHODS:
            tpc = [data[key][l]["tpc"] for l in langs]
            tok = [data[key][l]["tokens"] for l in langs]
            w.writerow([
                label, s1, s2, round(sum(tpc) / len(tpc), 4),
                round(gini(tpc), 4), round(max_min(tpc), 3),
                round(gini(tok), 4), round(max_min(tok), 3),
                data[key]["eng_Latn"]["tokens"], round(data[key]["eng_Latn"]["tpc"], 4),
            ])

    # ---- per_language_compression.csv ----
    with (OUT / "per_language_compression.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["language", "total_chars"]
        for _, key, _, _ in METHODS:
            col = SHORT.get(key, key)
            header += [f"{col}_tokens", f"{col}_tokens_per_char"]
        w.writerow(header)
        for l in langs:
            row = [l, data[METHODS[0][1]][l]["chars"]]
            for _, key, _, _ in METHODS:
                row += [data[key][l]["tokens"], round(data[key][l]["tpc"], 4)]
            w.writerow(row)

    # ---- console markdown ----
    print("Wrote results_summary.csv and per_language_compression.csv\n")
    print("| Method | s1 | s2 | mean t/c | Gini t/c | Gini tok-count | max/min tok-count | eng t/c |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for label, key, s1, s2 in METHODS:
        tpc = [data[key][l]["tpc"] for l in langs]
        tok = [data[key][l]["tokens"] for l in langs]
        print(f"| {label} | {s1} | {s2} | {sum(tpc)/len(tpc):.4f} | {gini(tpc):.4f} | "
              f"{gini(tok):.4f} | {max_min(tok):.3f} | {data[key]['eng_Latn']['tpc']:.4f} |")


if __name__ == "__main__":
    main()
