# Methodology: the four tokenizers, and the validated fast parity learner

This document records (1) the 2×2 experimental design and the exact meaning of each
tokenizer's stages — in particular the easily-conflated distinction between the
**SuperBPE baseline** and **stage 2 of PA-SuperBPE** — and (2) the performance-optimized
parity learner used to make the parity methods tractable, together with the evidence that
it produces byte-identical results to the reference implementation.

---

## 1. The 2×2 design

The experiment compares four tokenizers, all at **vocab size 40,000**, trained on the same
English-skewed FineWeb corpus, evaluated on FLORES-200 devtest (tokens/char, Gini, max/min).

|                       | standard merge selection | parity-aware merge selection |
|-----------------------|--------------------------|------------------------------|
| **within-word**       | **BPE**                  | **PA-BPE**                   |
| **superword (cross-whitespace)** | **SuperBPE**  | **PA-SuperBPE**              |

- **Columns** differ in *how a merge is chosen*: standard BPE picks the globally
  most-frequent pair; parity picks the best pair for whichever language is currently
  worst-compressed on the FLORES **dev** reference.
- **Rows** differ in *what kind of token is allowed*: the bottom row ("superword") permits
  tokens that span whitespace (e.g. `of the` → one token); the top row does not.

The bottom-right cell, **PA-SuperBPE**, is the novel combination: SuperBPE's superword
mechanism driven by parity's merge selection.

---

## 2. Two different "two-stage" procedures — do not conflate

Both SuperBPE and PA-SuperBPE are built in two stages, but they are **different procedures
in different implementations**. This is the most common point of confusion.

### SuperBPE (baseline, bottom-left)
- Runs in the **official Rust trainer** (HuggingFace `tokenizers` BpeTrainer via
  `vendor/superbpe`), global-frequency selection.
- Stage 1: learn to a 16k base with the within-word (stage-1) regex.
- Stage 2: extend to 40k with the superword (stage-2) regex.
- No parity, no FLORES ruler. This is a separate `PHASE=baseline` run.

### PA-SuperBPE (parity method, bottom-right)
- Runs in the **parity learner** (Python; here the validated fast version), parity selection
  throughout.
- **Stage 1** (15,744 merges): parity-aware merges on the **stage-1 (within-word)**
  pretokenization → a parity 16k base.
- **Stage 2** (24,000 merges): parity-aware merges on the **stage-2 (superword)**
  pretokenization, *preloading* the stage-1 merges and applying the SuperBPE stage-2
  **filter** (`--superbpe-stage2-filter --max-stage2-words 4`: reject tokens containing
  `":Ġ"` or spanning >4 Ġ-words).

So **"stage 2" of PA-SuperBPE is the SuperBPE-style superword stage *of the parity method*** —
it uses SuperBPE's cross-whitespace tokenization and stage-2 filter, but every merge is still
selected by parity (the per-language target driven by the FLORES dev ruler). It is **not**
the SuperBPE baseline.

Vocabulary accounting (both SuperBPE and PA-SuperBPE):
`256 byte alphabet + 15,744 stage-1 merges + 24,000 stage-2 merges = 40,000`.

---

## 3. The performance problem

The reference parity learner (`vendor/parity-aware-bpe/.../parity_aware_learn_bpe.py`,
wrapped by `scripts/parity_aware_filtered_learn_bpe.py`) is pure Python and does, **every
merge**, two full linear scans:

1. **Pair selection** — `max(stats, key=lambda x: (stats[x][lang], x))` scans every unique
   adjacent pair (~tens of millions on the real corpus) to find the best pair for the target
   language.
2. **Dev-ruler update** — `replace_pair_dict` scans the *entire* FLORES-dev vocabulary
   (~61k words) to apply the merge and recompute each language's compression length.

Measured cost on the real corpus: **~370 s/merge** → 39,744 merges ≈ **5.6 months**.
Profiling showed (2) was ~72% of the time, and crucially it does **not** shrink with corpus
size — so reducing the training corpus alone could never have fixed it.

---

## 4. The fast learner (`scripts/fast_parity_learn_bpe.py`)

Same algorithm, same decisions, faster data structures. Nothing in the *selection logic*
changes — only how the data is stored and looked up.

**Fix 1 — per-language max-heaps (replaces scan #1).**
One max-heap **per language** (10 heaps; the parity target rotates, so a single global heap
would be wrong / would be standard BPE). Each merge:
`max_index = argmax(lengths)` (worst language on the FLORES ruler — identical to reference),
then peek `heaps[max_index]`. Entries are validated against the authoritative `stats` dict on
peek (**lazy deletion**): stale or filtered tops are discarded until a live top is found, which
is provably the same pair the reference's full scan would return. The tie-break (larger pair
tuple) is reproduced exactly (`_RevPair`).

**Fix 2 — inverted dev index (replaces scan #2).**
A `pair → set(dev words containing it)` index, so the per-language length update touches only
the few affected dev words instead of all 61k. Same per-language `length_change`, same
resulting `lengths` (the parity signal that picks the next target).

Unchanged and reused verbatim from the reference: language selection, `min_frequency=2`
stopping, periodic prune/resync, the superword stage-2 filter, the dict I/O and merges output
format.

Result: **~370 s/merge → ~0.6 s/merge** (~600×). The real run completed PA-SuperBPE stage 1
(15,744 merges) in ~17 minutes and PA-BPE (39,744) in well under an hour.

---

## 5. Evidence the fast learner is byte-identical to the reference

A tokenizer is fully determined by its ordered merge list, so byte-identical merges ⇒ identical
tokenizer ⇒ identical downstream metrics. This is verified, not assumed:

- **VERIFY mode** (`FAST_PARITY_VERIFY=1`): every merge computes the pick *both* ways (fast
  heap and reference full scan) and asserts equality. Passed for 300 consecutive merges on a
  4-language (incl. CJK + Arabic) corpus.
- **Unit tests** (`tests/test_fast_learner.py`, T1–T5): basic equivalence, the stage-2
  filter, deliberate count **ties** (tie-break fidelity), forced resync/pruning, and preload
  continuation — all produce identical merge files (and stop at the same merge).
- **Full 3-config harness** (`scripts/validate_fast_learner.py`) on **real corpus data**:
  PA-BPE, PA-SuperBPE stage 1, and PA-SuperBPE stage 2 (with preload + filter) — **all three
  byte-identical** to the reference (confirmed at 1,000 and 1,500–2,000 merges).

The comparison is against the **parity** reference (run with `--dev` FLORES sets), so identity
also proves the fast version is doing parity, not standard BPE — the two produce different
merges, and only parity matches.

**Live confirmation that it is parity, not BPE:** running the fast learner with `--verbose`
prints the target language (`max_index`) per merge. It rotates across the low-resource
languages and never selects English (already well-compressed), e.g. first 25 targets:
`8 3 8 5 3 5 8 8 8 5 8 3 5 8 5 8 8 8 5 3 0 8 5 8 3`
(8=Thai, 3=Hindi, 5=Russian, 0=Arabic; English=2 appears zero times). Standard BPE has no
per-language target at all and would spend these merges on English (50% of the corpus).

---

## 6. Reproducibility

- Reference path (default): `bash lightning_run_40k_exact.sh` (uses
  `scripts/parity_aware_filtered_learn_bpe.py`).
- Fast path (opt-in, identical output): set
  `PARITY_LEARNER=scripts/fast_parity_learn_bpe.py`.
- Corpus reduction (drop merge-ineligible hapaxes): `PRETOKEN_MIN_COUNT=2` floors the train
  dicts to count≥2 (retains 97.6% of token occurrences; dev reference left intact).
- Re-verify at any time: `vendor/superbpe/.venv/bin/python scripts/validate_fast_learner.py
  --fixture <dir> --symbols 1500` and `... tests/test_fast_learner.py`.
