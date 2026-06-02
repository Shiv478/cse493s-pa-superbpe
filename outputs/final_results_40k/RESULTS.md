# Final results — four tokenizers at 40k vocab

All four tokenizers were trained on the **same** English-skewed FineWeb corpus, with the **same**
learner, at **40,000** vocabulary, and evaluated on the **same** FLORES-200 devtest split
(10 languages × 1,012 parallel sentences). The only thing that varies is the method.

## The four methods (the 2×2)

|                        | global-frequency selection | parity-aware selection |
|------------------------|----------------------------|------------------------|
| **within-word**        | `bpe`                      | `pa_bpe`               |
| **superword** (crosses whitespace) | `superbpe`     | `pa_superbpe`          |

- **Columns** = *how a merge is chosen.* Global picks the most frequent pair overall; parity
  picks the best pair for whichever language is currently worst-compressed (measured on FLORES dev).
- **Rows** = *what kind of token is allowed.* Superword tokens may span a space (e.g. `of the`).

## Headline numbers

Two fairness metrics (lower = fairer). Source: [`results_summary.csv`](results_summary.csv).

| Method | mean tok/char | Gini (tok/char) | max/min (tok/char) | **Gini (token-count)** | **max/min (token-count)** |
|---|---:|---:|---:|---:|---:|
| BPE         | 0.4561 | 0.3063 | 4.855 | 0.1142 | 2.272 |
| SuperBPE    | 0.4141 | 0.2350 | 3.699 | 0.0966 | 1.917 |
| PA-BPE      | 0.4061 | 0.2370 | 3.560 | **0.0051** | **1.027** |
| PA-SuperBPE | 0.4270 | 0.2356 | 3.564 | **0.0043** | **1.023** |

### Two metrics, and which one matters
- **token-count** = how many tokens each language needs to encode the **same** sentences (FLORES
  is parallel, so this is a like-for-like cost comparison). This is the standard cross-lingual
  fairness metric. **This is the headline.**
- **tokens-per-char** = tokens ÷ Unicode characters. Useful, but it divides by character count,
  which differs wildly by script (Chinese packs far more meaning per character than Spanish), so
  it distorts cross-script comparisons.

## What the results say (plain English)

1. **Parity nearly eliminates cross-lingual token-count disparity — in both rows.**
   On token-count, BPE makes Thai pay **2.27×** English's token bill for the same content;
   SuperBPE **1.92×**. Parity collapses this to **~1.02×** (Gini drops from ~0.10–0.11 to
   ~0.004 — roughly a **20–25× reduction**). See [`per_language_compression.csv`](per_language_compression.csv):
   under parity every language lands at ~40k–43k tokens; under BPE they range 29k–66k.

2. **The cost is English compression.** Parity stops over-serving English (the 50% bulk of the
   corpus): English goes from 29,250 tokens (BPE) to 41,237 (PA-BPE). That is the mechanism, not
   a bug — parity reallocates merges from the over-served language to the under-served ones.

3. **On tokens-per-char, parity clearly beats BPE but ties SuperBPE.** SuperBPE's superwords
   already remove much of BPE's per-char unfairness on their own (Gini 0.306 → 0.235), so adding
   parity on top doesn't lower the tok/char Gini further. But parity still wins on token-count,
   the metric that reflects equal cost for equal content.

## Folder contents

| Path | What it is |
|---|---|
| [`results_summary.csv`](results_summary.csv) | one row per method, both metric families (table above) |
| [`per_language_compression.csv`](per_language_compression.csv) | per-language chars, tokens, tokens/char for all four methods |
| `tokenizers/<method>/` | the finished tokenizer: `tokenizer.json` (loads in any HuggingFace stack) + `merges.txt` + `vocab.json` |
| `evaluation/<method>/` | `summary.json` (Gini, max/min, mean tok/char) + `compression_by_language.csv` |
| `merges/` | the raw learner outputs and two-stage intermediates (`*_stage1`, `*_stage2`, `*_preload`, combined). For reproducibility. |
| `logs/` | training, build, and evaluation logs for every method |

## How to reproduce
From the floored dicts in [`../training_input_dicts_floored/`](../training_input_dicts_floored/):
the baselines via `scripts/run_baselines_fast_40k.sh`, the parity methods via the parity learner;
build with `scripts/build_tokenizer_from_merges.py`; evaluate with
`scripts/evaluate_hf_tokenizer.py`; regenerate the two summary CSVs with
`scripts/make_results_summary.py`.

> Note on hypotheses (see the report): the headline **token-count** result strongly supports the
> project's fairness claim. The original H1 (SuperBPE *more* disparate than BPE) was **not**
> observed on this corpus — SuperBPE is slightly fairer than BPE. 200k and the language-model
> study (H3) are scoped as future work.
