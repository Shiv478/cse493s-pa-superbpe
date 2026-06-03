# Methodology Audit Against The Proposal And V1 Report

This file is the research-methods checkpoint. It separates the exact project requirement from the local prototype that was built first.

## What The Documents Require

The proposal and V1 report require:

1. Train tokenizers at matched vocabulary sizes: **40k** and **200k**.
2. Train on the same **5-10 GB English-skewed FineWeb corpus**, language-balanced as much as possible.
3. Evaluate on **FLORES-200 devtest**, restricted to 10 languages:
   `eng_Latn`, `spa_Latn`, `deu_Latn`, `rus_Cyrl`, `arb_Arab`, `hin_Deva`, `zho_Hans`, `jpn_Jpan`, `tha_Thai`, `swh_Latn`.
4. Compare four tokenizer methods:
   `BPE`, `SuperBPE`, `parity-aware BPE`, and `PA-SuperBPE`.
5. Use SuperBPE's two-stage setup:
   - Stage 1 regex: GPT-style regex with whitespace/category boundaries.
   - Stage 2 regex: whitespace-lifted regex that allows cross-word merges.
   - 40k split: 16k stage-1 vocab target, then extend to 40k final vocab.
   - 200k split: 80k stage-1 vocab target, then extend to 200k final vocab.
   - The same split must be used for vanilla SuperBPE and PA-SuperBPE, so the comparison isolates the merge-selection rule rather than stage budget.
6. For PA-SuperBPE, use the official parity-aware BPE learner:
   - `--variant base`
   - `--global-merges 0`
   - `--min-frequency 2`
   - FLORES `dev` as the parity reference
   - `--preload` stage-1 merges before stage 2
   - concatenate stage-1 merges and newly learned stage-2 merges into the final tokenizer merges file
7. Use the SuperBPE stage-2 merge filter:
   - veto tokens containing the SuperBPE colon marker
   - veto tokens spanning more than four whitespace-marked words
8. Report per-language `total_chars`, `total_tokens`, `mean_tokens_per_char`, Gini coefficient, and max/min ratio.
9. For H3, train small language models and evaluate FLORES perplexity and XNLI. This is explicitly compute-heavy.

## What Is Now Present In This Repo

| Requirement | Current status |
|---|---|
| FLORES dev/devtest prep | Present: `scripts/prepare_flores.py` |
| 10-language evaluation set | Present |
| Metric computation | Present |
| Official SuperBPE repo | Present in `vendor/superbpe/` |
| Official parity-aware BPE repo | Present in `vendor/parity-aware-bpe/` |
| 40k build commands | Present: per-tokenizer recipes in `scripts/tokenizer_recipes/` + `make results` |
| HF tokenizer evaluator | Present: `scripts/evaluate_hf_tokenizer.py` |
| Per-language text export for official parity repo | Present: `scripts/export_language_texts.py` |
| SuperBPE regex + ByteLevel dict pretokenization wrapper | Present: `scripts/pretokenize_for_parity.py` |
| PA-SuperBPE stage-2 colon/4-word filter wrapper | Present: `scripts/parity_aware_filtered_learn_bpe.py` |

## Update — what the final runs actually did

The reported **40k** results were produced by a validated **fast** parity learner
(`scripts/fast_parity_learn_bpe.py`), proven byte-identical to the reference parity learner and,
in global mode, to textbook BPE (see `docs/fast_learner_methodology.md` and
`scripts/verify_global_is_standard_bpe.py`). The completed deliverable is the **six-tokenizer**
40k comparison in `outputs/final_results_40k/` (the four-method 2×2 plus two stage hybrids).
**200k** and the **H3** language-model study are deferred to future work.

## What Is Still Not Exact

The first implementation under `src/pa_superbpe/` is a **prototype**, not the final exact research method.

It differs from the report in these ways:

1. It uses a simplified Unicode-category pretokenizer, not the exact Hugging Face `Split(Regex(...)) + ByteLevel(use_regex=False)` pipeline.
2. It implements parity selection directly instead of calling the vendored `parity_aware_bpe` learner.
3. It trains on FLORES for local convenience, but the report requires a 5-10 GB English-skewed FineWeb corpus for tokenizer training.
4. It produces custom JSON tokenizers, not final HuggingFace `PreTrainedTokenizerFast` tokenizers.
5. It does not train the H3 language models.

So: the prototype is useful for understanding and debugging, but it should not be claimed as the final research implementation.

## Correct Path Forward

Use the prototype for education and quick checks. Use the vendored official repos for final reported results.

The final experimental table should only use runs that satisfy:

- tokenizer training corpus: English-skewed FineWeb corpus, same data for all tokenizer methods
- evaluation: FLORES-200 devtest
- vocab sizes: 40k and 200k
- tokenizer implementations: official SuperBPE + official parity-aware BPE, with our PA-SuperBPE wrapper
