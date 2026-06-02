# Draft: Parity-Aware SuperBPE

Shiv Vinodshankar, Dhruti Vadlamudi, Chendur Jayavelu

**Status note:** this draft currently contains prototype/local results. Replace the Results section with exact 40k and 200k FineWeb-trained results before submitting as the final research paper. See `docs/methodology_audit.md`.

## Abstract

Byte-pair encoding tokenizers optimize global compression, which can overfit to high-resource languages and fragment underrepresented languages into more tokens per character. SuperBPE improves compression by adding a second stage that permits merges across whitespace boundaries, but this can further favor languages whose frequent phrases dominate the training corpus. This project investigates PA-SuperBPE, which combines SuperBPE's two-stage curriculum with parity-aware merge selection. The final experiment should compare BPE, SuperBPE, parity-aware BPE, and PA-SuperBPE at 40k and 200k vocabulary sizes using mean tokens per character, Gini coefficient, and max/min disparity ratio on FLORES-200 devtest.

## 1. Introduction

Tokenization determines how much context a language model can effectively use for each language. If one language requires far more tokens per character than another, speakers of that language pay more inference cost and receive a smaller effective context window. Prior work has shown that production tokenizers can create large cross-lingual disparities. The proposal for this project asks whether SuperBPE's cross-whitespace merges improve compression at the cost of fairness, and whether parity-aware merge selection can reduce that disparity.

## 2. Hypotheses

H1: At matched vocabulary size, SuperBPE-style two-stage training can increase cross-lingual compression disparity compared with ordinary BPE.

H2: PA-SuperBPE can preserve some of SuperBPE's compression benefits while reducing cross-lingual disparity below vanilla SuperBPE.

H3: In a larger future experiment, fairness-aware tokenization may improve downstream multilingual language-model metrics on underrepresented languages.

The tokenizer experiments focus on H1 and H2. H3 requires language-model pretraining compute and is treated as future work unless GPU budget is available.

## 3. Method

The exact method uses four tokenizers:

- BPE: official BPE baseline trained on the shared FineWeb subsample.
- SuperBPE: official SuperBPE two-stage baseline using the released regex + ByteLevel pretokenization pipeline.
- Parity-aware BPE: official parity-aware BPE learner using FLORES dev as the parity reference.
- PA-SuperBPE: our wrapper that applies parity-aware merge selection in both SuperBPE stages, preloading stage-1 merges before stage 2.

For each tokenizer, we compute total characters, total tokens, mean tokens per character, the Gini coefficient over per-language tokens per character, and the max/min tokens-per-character ratio.

## 4. Data

The exact experiment trains tokenizers on a shared 5-10 GB English-skewed FineWeb corpus and evaluates on FLORES-200 `devtest`. FLORES-200 `dev` is used as the parity reference for parity-aware methods. The experiment covers ten language tags: `eng_Latn`, `spa_Latn`, `deu_Latn`, `rus_Cyrl`, `arb_Arab`, `hin_Deva`, `zho_Hans`, `jpn_Jpan`, `tha_Thai`, and `swh_Latn`.

The prototype run included below trains on FLORES `dev` and tests on FLORES `devtest`. It is included as a local sanity check only.

## 5. Results

Prototype result, not final:

```bash
make train-test
```

The resulting compression summary was:

| method | mean tokens/char | Gini | max/min | eng_Latn tokens/char |
|---|---:|---:|---:|---:|
| parity_bpe | 0.9324 | 0.0079 | 1.05 | 0.9135 |
| pa_superbpe | 0.9112 | 0.0114 | 1.07 | 0.9063 |
| bpe | 0.8566 | 0.0714 | 1.39 | 0.7333 |
| superbpe | 0.8104 | 0.0930 | 1.57 | 0.6588 |

Suggested figure: `outputs/prototype_demos/flores10_demo/gini_bar.svg`.

In this prototype setting, the SuperBPE-style tokenizer achieved the best mean compression and the best English compression, but it also produced the highest cross-lingual disparity. This is useful as a sanity check, but final claims must be based on the exact 40k and 200k FineWeb-trained experiments.

## 6. Discussion

The main comparison is the Gini coefficient. A lower Gini means tokenization cost is more evenly distributed across languages. If PA-SuperBPE lowers Gini relative to SuperBPE while keeping English tokens per character near SuperBPE, that supports H2. If SuperBPE's Gini is above BPE, that supports H1.

The final discussion should compare exact 40k and 200k results. The key tests are whether `Gini(SuperBPE) > Gini(BPE)` for H1 and whether PA-SuperBPE both reduces Gini and preserves English compression within the threshold stated in the proposal for H2.

## 7. Limitations

The prototype result uses FLORES text directly rather than a 5-10 GB FineWeb training corpus, so those numbers should not be treated as the final benchmark. H3 is also not complete unless the project trains the proposed language models and evaluates FLORES perplexity and XNLI accuracy.

## 8. Conclusion

The intended contribution is a PA-SuperBPE training pipeline, a multilingual fairness evaluation harness, and 40k/200k comparisons against BPE, SuperBPE, and parity-aware BPE.

## References

See `paper/references.bib`.
