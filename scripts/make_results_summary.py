from __future__ import annotations
"""Assemble the combined 40k results from the per-method evaluation outputs.

Reads outputs/final_results_40k/evaluation/<method>/{summary.json,
compression_by_language.csv} for the four methods and writes:
  - results_summary.csv          one row per method, both metric families
  - per_language_compression.csv per-language tokens & tokens/char, all methods

Two metric families are reported:
  * tokens-per-char  : tokens / Unicode-codepoint  (the original disparity metric)
  * token-count      : tokens to encode the SAME parallel FLORES content
                       (canonical cross-lingual fairness metric on parallel data)
Lower Gini / max-min = fairer. Run with the repo venv (no third-party deps needed).
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "outputs" / "final_results_40k" / "evaluation"
OUT = ROOT / "outputs" / "final_results_40k"
METHODS = ["bpe", "superbpe", "pa_bpe", "pa_superbpe"]
LABEL = {"bpe": "BPE", "superbpe": "SuperBPE", "pa_bpe": "PA-BPE", "pa_superbpe": "PA-SuperBPE"}


def gini(values):
    xs = sorted(v for v in values if v >= 0)
    n, s = len(xs), sum(xs)
    if not n or s == 0:
        return 0.0
    return sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(xs)) / (n * s)


def max_min(values):
    pos = [v for v in values if v > 0]
    return (max(pos) / min(pos)) if pos else 0.0


def load(method):
    rows = list(csv.DictReader((EVAL / method / "compression_by_language.csv").open()))
    by_lang = {
        r["language"]: {
            "chars": int(r["total_chars"]),
            "tokens": int(r["total_tokens"]),
            "tpc": float(r["tokens_per_char"]),
        }
        for r in rows
    }
    return by_lang


def main():
    data = {m: load(m) for m in METHODS}
    langs = sorted(next(iter(data.values())).keys())

    # ---- results_summary.csv ----
    summ_path = OUT / "results_summary.csv"
    with summ_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "method", "mean_tokens_per_char",
            "gini_tokens_per_char", "maxmin_tokens_per_char",
            "gini_token_count", "maxmin_token_count",
        ])
        for m in METHODS:
            tpc = [data[m][l]["tpc"] for l in langs]
            tok = [data[m][l]["tokens"] for l in langs]
            w.writerow([
                LABEL[m],
                round(sum(tpc) / len(tpc), 4),
                round(gini(tpc), 4), round(max_min(tpc), 3),
                round(gini(tok), 4), round(max_min(tok), 3),
            ])

    # ---- per_language_compression.csv ----
    perlang_path = OUT / "per_language_compression.csv"
    with perlang_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["language", "total_chars"]
        for m in METHODS:
            header += [f"{m}_tokens", f"{m}_tokens_per_char"]
        w.writerow(header)
        for l in langs:
            row = [l, data[METHODS[0]][l]["chars"]]
            for m in METHODS:
                row += [data[m][l]["tokens"], round(data[m][l]["tpc"], 4)]
            w.writerow(row)

    # ---- console markdown (for RESULTS.md) ----
    print("Wrote", summ_path.name, "and", perlang_path.name, "\n")
    print("| Method | mean tok/char | Gini (tok/char) | max/min (tok/char) | "
          "Gini (token-count) | max/min (token-count) |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in METHODS:
        tpc = [data[m][l]["tpc"] for l in langs]
        tok = [data[m][l]["tokens"] for l in langs]
        print(f"| {LABEL[m]} | {sum(tpc)/len(tpc):.4f} | {gini(tpc):.4f} | "
              f"{max_min(tpc):.3f} | {gini(tok):.4f} | {max_min(tok):.3f} |")


if __name__ == "__main__":
    main()
