# PA-SuperBPE — Results & Findings (40k vocab)

> **TL;DR.** On an English-heavy corpus, standard **BPE makes Thai cost 2.27× more tokens than English for the same sentence.** Parity-aware merge selection cuts that gap to **~1.02×** (about a 25× drop in inequality). The catch: full parity does this by giving up English's compression. But if you apply parity **only in the within-word stage** and let global "superwords" do the rest (**Hybrid A**), you keep English fast **and** get the fairest tokenizer per character. **The fairness–compression trade-off is controllable, not fundamental.**

*All numbers come from `outputs/final_results_40k/` and `outputs/stage_hybrid_ablation_40k/` and are reproducible with the scripts in §11.*

**Contents:** 1. Problem · 2. Setup · 3. Methods · 4. Results · 5. Per-language cost · 6. Superword check · 7. Findings · 8. Hypotheses · 9. The English skew · 10. Limitations · 11. Reproduce

---

## 1. The problem

LLMs read and bill in **tokens**. On English-dominated training data, BPE-style tokenizers learn to compress English well and chop other languages into many more tokens for the *same content*. Because API cost, latency, and the usable context window all scale with token count, speakers of under-represented languages **pay more, wait longer, and get less usable context** for the same task (Petrov et al. 2023; Ahia et al. 2023).

Two existing tools pull in opposite directions. **SuperBPE** adds a strong new compression trick — "superword" merges that span whitespace — but keeps BPE's global-frequency objective, so that trick mostly serves the majority language. **Parity-aware BPE** fixes fairness by changing the rule (help the worst-compressed language), but it is within-word only and sacrifices the dominant language's compression.

So you seem forced to choose: **global = compressive but unfair**, or **parity = fair but anti-compression**.

> **Research question.** Is that trade-off fundamental, or just an artifact of applying fairness uniformly? SuperBPE's two-stage curriculum (within-word base → cross-whitespace superwords) gives a natural knob: *which stage should be fairness-driven?* Our idea: apply parity in the **within-word stage** (the base that reaches every language, even spaceless ones) and let **global frequency** drive the **superword stage** (recovering the dominant language) — getting both at once.

---

## 2. Setup

- **Training corpus:** English-skewed FineWeb / FineWeb-2, ~5 GB — **50% English** plus 9 other languages (~277 MB each). The skew is deliberate: it mirrors real pretraining mixes and makes the unfairness measurable (see §9).
- **Languages (10):** eng, spa, deu, rus, arb, hin, swh, and the three with **no spaces between words** — zho (Chinese), jpn (Japanese), tha (Thai).
- **Evaluation:** FLORES-200 **devtest** — 10 languages × **1,012 parallel sentences** (same content translated, so token counts are directly comparable). FLORES **dev** is the parity reference.
- **Vocabulary:** 40,000 for all. SuperBPE family = 15,744 within-word (stage 1) + 24,000 superword (stage 2) + 256 byte alphabet.
- **Engine:** one validated **fast learner** trains every tokenizer (proven byte-identical to the reference parity learner, and its global mode byte-identical to textbook BPE). Same engine + same corpus ⇒ any difference is the method, not the setup.

---

## 3. The methods

**Main comparison (2×2):** selection rule × token type.

| | global selection | parity selection |
|---|---|---|
| **within-word** | BPE | PA-BPE |
| **superword** | SuperBPE | PA-SuperBPE |

**Stage ablation:** within the superword family, *which stage* uses parity.

| stage 1 ↓ / stage 2 → | global | parity |
|---|---|---|
| **global** | SuperBPE | **Hybrid B** |
| **parity** | **Hybrid A** | PA-SuperBPE |

- **Hybrid A** = parity within-word base, then **global** superwords ("fair base, global icing").
- **Hybrid B** = global within-word base, then **parity** superwords.

---

## 4. Results

**Fairness** — lower is fairer. *Token-count* = tokens to encode the same content (the metric that costs money/latency). *Per-char* = tokens ÷ characters (a compression measure, distorted across scripts).

| Method | rule (s1→s2) | Token-count Gini ↓ | Token-count max/min ↓ | Per-char Gini ↓ |
|---|---|---:|---:|---:|
| BPE | – | 0.114 | 2.27× | 0.306 |
| SuperBPE | global→global | 0.097 | 1.92× | 0.235 |
| Hybrid B | global→parity | **0.033** | 1.41× | 0.256 |
| Hybrid A | parity→global | 0.066 | 1.44× | **0.230** |
| PA-BPE | parity | 0.005 | 1.03× | 0.237 |
| PA-SuperBPE | parity→parity | **0.004** | **1.02×** | 0.236 |

**Compression & English** — lower tok/char = more compression.

| Method | Mean tok/char ↓ | English tok/char | English tokens |
|---|---:|---:|---:|
| BPE | 0.456 | 0.222 | 29,250 |
| SuperBPE | 0.414 | 0.221 | 29,191 |
| Hybrid B | 0.428 | 0.241 | 31,838 |
| Hybrid A | 0.409 | 0.250 | 32,968 |
| PA-BPE | 0.406 | 0.313 | 41,237 |
| PA-SuperBPE | 0.427 | 0.327 | 43,194 |

> **Read it as:** the parity methods drive token-count inequality almost to zero (Gini 0.004–0.005). Full parity pays for it in English (29k → 41–43k tokens). **Hybrid A** is the balance point — best per-char fairness of all six, near-best compression, and English stays fast (0.250).

---

## 5. Per-language cost

Tokens to encode the 1,012 FLORES sentences in each language. Under BPE the spread is huge (29k–66k); under parity everyone converges to ~40–43k.

| language | BPE | SuperBPE | Hybrid B | Hybrid A | PA-BPE | PA-SuperBPE |
|---|---:|---:|---:|---:|---:|---:|
| eng_Latn | 29,250 | 29,191 | 31,838 | 32,968 | 41,237 | 43,194 |
| spa_Latn | 41,218 | 44,726 | 44,439 | 44,015 | 41,065 | 43,272 |
| deu_Latn | 41,167 | 45,894 | 44,722 | 44,326 | 41,117 | 43,387 |
| rus_Cyrl | 46,661 | 55,973 | 43,956 | 47,347 | 40,525 | 42,786 |
| arb_Arab | 40,468 | 48,016 | 44,355 | 47,484 | 40,887 | 43,213 |
| hin_Deva | 45,628 | 43,118 | 44,483 | 42,975 | 40,690 | 43,310 |
| swh_Latn | 35,265 | 35,316 | 42,152 | 36,090 | 41,135 | 43,228 |
| zho_Hans | 46,543 | 35,384 | 43,664 | 35,627 | 40,266 | 42,478 |
| jpn_Jpan | 52,716 | 41,986 | 43,828 | 42,071 | 41,279 | 42,720 |
| tha_Thai | 66,454 | 49,126 | 43,616 | 47,543 | 40,205 | 42,432 |

*(Per-character versions and char counts: `outputs/final_results_40k/per_language_compression.csv`.)*

---

## 6. Did the superwords really form?

The byte-level space marker is `Ġ`. A within-word token has `Ġ` only at the start (`Ġthe`); a **superword** has an *internal* `Ġ` (`ĠofĠthe` = "of the"). Counting vocab tokens with an internal `Ġ`:

| tokenizer | cross-space tokens | % of vocab |
|---|---:|---:|
| BPE | 1 | 0.0% |
| PA-BPE | 2 | 0.0% |
| SuperBPE | 5,845 | 14.6% |
| PA-SuperBPE | 5,717 | 14.3% |
| Hybrid A | 5,587 | 14.0% |
| Hybrid B | 4,483 | 11.2% |

And at encode time, on `" of the people in the United States of America"`:
- **BPE → 9 tokens**, none crossing a space.
- **SuperBPE / Hybrid A → 5 tokens**, fusing `of the`, `in the`, `United States`, `of America` into single space-spanning tokens.

> **Takeaway:** stage 2 genuinely created thousands of cross-space tokens (within-word tokenizers have ~0). Hybrid B has fewer (4,483) because its parity stage 2 also spends budget on *within-run* CJK merges — exactly why it equalizes spaceless-language counts.

---

## 7. Key findings

1. **Parity nearly eliminates token-count disparity** — Gini 0.114 → 0.004–0.005; worst/best ratio 2.27× → ~1.02×. Every language costs within ~2% to encode the same content.
2. **Full parity's cost is the dominant language** — English goes 29k → 41–43k tokens. Fairness and English-privilege conflict *only when parity is applied everywhere.*
3. **The trade-off is controllable by *where* parity goes.** Stage-1 parity (**Hybrid A**) gives the best **per-character** fairness while keeping English fast. Stage-2 parity (**Hybrid B**) gives the best **token-count** fairness among English-preserving methods, but worse per-character fairness.
4. **The two metrics disagree most on spaceless scripts** (CJK/Thai): equalizing token *counts* makes char-dense scripts look expensive *per character*. Stage-1 parity compresses them well; stage-2 parity equalizes their counts.
5. **Design rule:** *stage-1 parity → per-character fairness; stage-2 parity → token-count fairness; full parity → maximal count equality at the dominant language's expense.*

---

## 8. Hypothesis verdicts

- **H1** (SuperBPE *more* disparate than BPE): **Refuted.** SuperBPE is *fairer* than BPE on both metrics — the superword mechanism reduces disparity here, not widens it.
- **H2** (keep English within Δ≤0.05 of SuperBPE **and** beat both baselines on Gini): **Failed by full PA-SuperBPE** (English Δ = 0.106) but **achieved by Hybrid A** — English Δ = 0.029, and per-char Gini 0.230 < SuperBPE 0.235 < BPE 0.306. The stage-split hybrid is the variant that satisfies the original H2.
- **H3** (downstream LM perplexity / XNLI): **Not tested** — no model trained. Future work.

---

## 9. Does the 50%-English corpus drive this?

The skew is a deliberate, realistic choice. It changes the **magnitudes** but not the **conclusions**.

- **Caused by the skew:** how unfair the baselines look (more English ⇒ bigger BPE gap); *which* language gets the global superwords (English, as the most frequent); how much English compression parity must give up to equalize.
- **Independent of the skew:** parity equalizes token counts by construction; the token-count-vs-per-char divergence on CJK is about scripts; "stage-1 parity → per-char, stage-2 parity → count" is structural; the superword mechanism fires regardless of mix.

> A **corpus-composition sweep** (e.g. 30 / 50 / 70% English, or balanced) is the natural robustness check and is listed as future work.

---

## 10. Limitations

- One vocabulary size (**40k**); 200k not run.
- One corpus composition (**50% English**); a sweep would test robustness (§9).
- **No downstream LM** (H3) — gains shown are intrinsic (token counts), not yet model quality.
- Baselines use the validated fast learner (byte-identical to the reference learner and to textbook BPE), not the official Rust SuperBPE binary — absolute numbers may differ slightly, but the *relative* comparison is sound (shared engine + corpus).
- Spaceless scripts (CJK/Thai) can only be helped in the within-word stage; the superword stage's cross-space benefit doesn't reach them.

---

## 11. Reproduce

| step | script | output |
|---|---|---|
| build corpus | `prepare_fineweb_language_shards.py` | `data/processed/fineweb_by_lang/` |
| FLORES prep | `prepare_flores.py`, `export_language_texts.py` | `data/processed/flores10_*` |
| pretokenize → floor | `pretokenize_for_parity.py`, `floor_dicts.py` | `outputs/training_input_dicts_floored/` |
| BPE + SuperBPE | `run_baselines_fast_40k.sh` | `outputs/final_results_40k/` |
| PA-BPE + PA-SuperBPE | `train_parity_by_lang_40k.sh` | `outputs/final_results_40k/` |
| Hybrid A + B | `run_stage_hybrids_40k.sh` | `outputs/stage_hybrid_ablation_40k/` |
| combined tables | `make_results_summary.py` | `*_summary.csv` |
| equivalence proofs | `validate_fast_learner.py`, `verify_global_is_standard_bpe.py` | byte-identity checks |

Helpers: `build_tokenizer_from_merges.py`, `merge_merges_files.py`, `evaluate_hf_tokenizer.py`. Metric definitions: `src/pa_superbpe/evaluate.py`.
