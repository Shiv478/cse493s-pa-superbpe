# Parity-Aware SuperBPE — fairer multilingual tokenization

**CSE 493S/599S project.** Chendur Jayavelu, Shiv Vinodshankar, Dhruti Vadlamudi
(Paul G. Allen School, University of Washington).

Byte-pair encoding (BPE) tokenizers maximize **global** compression, so on an English-heavy
corpus they compress English well and fragment other languages into far more tokens per
character. This project asks whether **parity-aware** merge selection — at each step, improve the
language that is currently *worst* compressed — produces fairer cross-lingual tokenization, and
how it interacts with **SuperBPE**'s "superword" merges that cross whitespace.

We compare four tokenizers — a 2×2 of *selection rule* × *token type* — at a matched **40,000**
vocabulary, all trained on the **same** English-skewed FineWeb corpus and evaluated on
**FLORES-200 devtest** (10 languages × 1,012 parallel sentences):

|                       | global-frequency selection | parity-aware selection |
|-----------------------|----------------------------|------------------------|
| **within-word**       | `bpe`                      | `pa_bpe`               |
| **superword** (crosses whitespace) | `superbpe`    | `pa_superbpe`          |

A follow-up **stage ablation** adds two hybrids that vary *which* stage uses parity
(`hybrid_A_parity1_global2`, `hybrid_B_global1_parity2`) — **six tokenizers in all**.

## Headline result

Full numbers and discussion: **[`outputs/final_results_40k/RESULTS.md`](outputs/final_results_40k/RESULTS.md)**.

On **token-count parity** (how many tokens each language needs for the *same* parallel content —
the standard cross-lingual fairness metric), parity-aware selection nearly eliminates the gap:

| Method | Gini (token-count) ↓ | max/min (token-count) ↓ |
|---|---:|---:|
| BPE | 0.114 | 2.27× |
| SuperBPE | 0.097 | 1.92× |
| **PA-BPE** | **0.005** | **1.03×** |
| **PA-SuperBPE** | **0.004** | **1.02×** |

Under BPE, Thai needs **2.27×** the tokens English does for the same sentences; parity collapses
this to **~1.02×** (a ~20–25× reduction in Gini), at the cost of some English compression. The
effect holds for both the within-word and superword rows.

**Best balance:** the stage ablation's **Hybrid A** (parity within-word base → global superwords)
keeps English fast *and* has the best per-character fairness of all six tokenizers — see
[`outputs/final_results_40k/RESULTS.md`](outputs/final_results_40k/RESULTS.md).

## Repository layout

```
scripts/      the pipeline: corpus prep, pretokenize, floor, train, build, evaluate, validate
src/pa_superbpe/   small Python library (incl. the Gini / max-min metric functions)
vendor/       official upstream repos used unmodified: superbpe/ and parity-aware-bpe/
data/         FineWeb training corpus (data/processed/fineweb_by_lang) + FLORES eval sets
outputs/      results — see outputs/README.md
  final_results_40k/   all six tokenizers, their evaluations, and combined tables
  training_input_dicts_floored/   the floored learner-input dicts
  prototype_demos/     small sanity runs (NOT the final results)
docs/         methodology and runbooks (see docs/fast_learner_methodology.md)
paper/        the report
tests/        equivalence / metric tests
```

## Setup (first time)

`vendor/` (the official SuperBPE + parity-aware-bpe repos and the build/eval venv) is **not**
committed to keep the repo small. Bootstrap it with:

```bash
git clone --recursive https://github.com/PythonNut/superbpe vendor/superbpe
git clone https://github.com/swiss-ai/parity-aware-bpe vendor/parity-aware-bpe
```

then create the SuperBPE Python 3.12 venv at `vendor/superbpe/.venv` (per their README) and
`pip install -e vendor/parity-aware-bpe`. The script `lightning_run_40k_exact.sh` automates this
full bootstrap (uv, Rust, clones, venv). Data is likewise re-created with the prep scripts below.

## How the results were produced

All six tokenizers are trained by one **validated fast learner**
([scripts/fast_parity_learn_bpe.py](scripts/fast_parity_learn_bpe.py)) — a ~600× faster
reimplementation that is **byte-identical** to the reference parity-aware BPE learner, and whose
global mode is byte-identical to an independent textbook BPE
([scripts/verify_global_is_standard_bpe.py](scripts/verify_global_is_standard_bpe.py)). Same
engine + same corpus for all six ⇒ differences come only from the method. See
[docs/fast_learner_methodology.md](docs/fast_learner_methodology.md).

Pipeline (uses the SuperBPE venv `vendor/superbpe/.venv/bin/python`, denoted `$PY`):

```bash
# 1. Build the English-skewed FineWeb corpus (~5 GB, 50% English + 9 languages)
$PY scripts/prepare_fineweb_language_shards.py --preset english_skew_5gb \
    --out-dir data/processed/fineweb_by_lang

# 2. Prepare FLORES eval + per-language parity-dev files
make prepare-flores && make export-flores-files

# 3. Pretokenize the corpus into per-language dicts (within-word + superword stages)
$PY scripts/pretokenize_for_parity.py ...        # -> stage1 / stage2 dicts
$PY scripts/floor_dicts.py --min-count 2 ...     # -> outputs/training_input_dicts_floored

# 4. Train + build + evaluate
bash scripts/run_baselines_fast_40k.sh           # BPE + SuperBPE (global mode)
bash scripts/train_parity_by_lang_40k.sh ...     # PA-BPE + PA-SuperBPE (parity mode)

# 5. Rebuild the combined results tables
make results                                     # scripts/make_results_summary.py
```

Reproduce the floored tokenizers without the raw corpus: everything in step 4 reads only the
dicts in `outputs/training_input_dicts_floored/`.

## Verify the implementation

```bash
vendor/superbpe/.venv/bin/python -m pytest tests/test_fast_learner.py -q   # fast == reference
vendor/superbpe/.venv/bin/python scripts/validate_fast_learner.py --fixture <dir> --symbols 1500
vendor/superbpe/.venv/bin/python scripts/verify_global_is_standard_bpe.py \
    --dict-dir outputs/training_input_dicts_floored/train_stage1 --out /tmp/std.txt --subset-out /tmp/sub
```

## Prototype demo (optional)

A tiny end-to-end check on FLORES text (not the final result):

```bash
make setup        # creates a local .venv (no third-party deps)
make smoke        # smallest run -> outputs/prototype_demos/smoke_test
make train-test   # 10-language demo -> outputs/prototype_demos/flores10_demo
```

## Status

- **Done (this report):** the four-way 40k comparison above, plus the validated fast learner.
- **Future work:** 200k vocabulary, and pretraining a small LM on a PA-SuperBPE tokenizer to
  measure downstream multilingual quality (the original H3).

## References

See [paper/references.bib](paper/references.bib): SuperBPE (Liu et al.), parity-aware BPE
(Foroutan et al.), tokenizer unfairness (Petrov et al.; Ahia et al.), BoundlessBPE (Schmidt et al.).
