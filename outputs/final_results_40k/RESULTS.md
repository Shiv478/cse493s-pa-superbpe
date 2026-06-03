# Final results — six tokenizers at 40k vocab

All tokenizers were trained on the **same** English-skewed FineWeb corpus, with the **same**
learner, at **40,000** vocabulary, and evaluated on the **same** FLORES-200 devtest split
(10 languages × 1,012 parallel sentences). The only thing that varies is the method.

## The methods

**Main 2×2** — selection rule × token type:

|                        | global-frequency selection | parity-aware selection |
|------------------------|----------------------------|------------------------|
| **within-word**        | `bpe`                      | `pa_bpe`               |
| **superword** (crosses whitespace) | `superbpe`     | `pa_superbpe`          |

**Stage ablation** — within the superword family, *which stage* uses parity:

| stage 1 ↓ / stage 2 → | global | parity |
|---|---|---|
| **global** | `superbpe` | `hybrid_B_global1_parity2` |
| **parity** | `hybrid_A_parity1_global2` | `pa_superbpe` |

## Headline numbers

Two fairness metrics (lower = fairer). Source: [`results_summary.csv`](results_summary.csv).

| Method | s1→s2 | mean tok/char | Gini (tok/char) | max/min (tok/char) | **Gini (token-count)** | **max/min (token-count)** | eng tok/char |
|---|---|---:|---:|---:|---:|---:|---:|
| BPE | – | 0.4561 | 0.3063 | 4.855 | 0.1142 | 2.272 | 0.2216 |
| SuperBPE | g→g | 0.4141 | 0.2350 | 3.699 | 0.0966 | 1.917 | 0.2212 |
| PA-BPE | parity | 0.4061 | 0.2370 | 3.560 | 0.0051 | 1.027 | 0.3125 |
| PA-SuperBPE | p→p | 0.4270 | 0.2356 | 3.564 | **0.0043** | **1.023** | 0.3273 |
| **Hybrid A** | p→g | 0.4089 | **0.2304** | **3.297** | 0.0662 | 1.442 | 0.2498 |
| **Hybrid B** | g→p | 0.4276 | 0.2556 | 4.185 | 0.0325 | 1.405 | 0.2413 |

- **token-count** = tokens to encode the *same* parallel content → the canonical cross-lingual
  fairness metric (what actually costs money/latency/context). **The headline.**
- **tokens-per-char** = tokens ÷ Unicode characters → a compression measure, distorted across
  scripts (Chinese packs far more per character than Spanish).

## What the results say

1. **Parity nearly eliminates token-count disparity** (Gini 0.114 → 0.004–0.005; worst/best
   2.27× → ~1.02×). Every language costs within ~2% to encode the same content. The cost is
   English compression (29k → 41–43k tokens).
2. **Hybrid A (parity stage 1 → global stage 2)** is the best-balanced tokenizer: **best
   per-character fairness of all six** (tok/char Gini 0.230) while keeping English fast (0.250).
   It applies fairness to the *base* and lets global superwords compress the dominant language —
   the variant that satisfies the project's original H2.
3. **Hybrid B (global stage 1 → parity stage 2)** gives the best **token-count** fairness among
   the English-preserving methods (0.033), but worse per-character fairness — because it
   equalizes counts, and char-dense scripts then look expensive per character.
4. **Design rule:** stage-1 parity → per-character fairness; stage-2 parity → token-count
   fairness; full parity → maximal count equality at the dominant language's expense.

Per-language token costs for all six: [`per_language_compression.csv`](per_language_compression.csv).

## Folder contents

| Path | What it is |
|---|---|
| [`results_summary.csv`](results_summary.csv) | one row per method, both metric families (table above) |
| [`per_language_compression.csv`](per_language_compression.csv) | per-language chars, tokens, tokens/char for all six methods |
| `tokenizers/<method>/` | the finished tokenizer: `tokenizer.json` (loads in any HuggingFace stack) + `merges.txt` + `vocab.json` |
| `evaluation/<method>/` | `summary.json` (Gini, max/min, mean tok/char) + `compression_by_language.csv` |
| `merges/` | raw learner outputs + two-stage intermediates (`*_stage1`, `*_stage2`, `*_preload`, combined) |
| `logs/` | training, build, and evaluation logs for every method |

Method folder names: `bpe`, `superbpe`, `pa_bpe`, `pa_superbpe`, `hybrid_A_parity1_global2`,
`hybrid_B_global1_parity2`.

## How to reproduce
From the floored dicts in [`../training_input_dicts_floored/`](../training_input_dicts_floored/):
each tokenizer has a self-contained recipe in [`../../scripts/tokenizer_recipes/`](../../scripts/tokenizer_recipes/)
(`1_bpe.sh` … `6_hybrid_B_global1_parity2.sh`). Regenerate these two summary CSVs with
`scripts/make_results_summary.py` (`make results`).

> The headline **token-count** result strongly supports the project's fairness claim. H1
> (SuperBPE *more* disparate than BPE) was **not** observed — SuperBPE is slightly fairer than
> BPE. H2 is satisfied by **Hybrid A**. 200k and the language-model study (H3) are future work.
