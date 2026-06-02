# Tokenizer recipes — exactly how each of the 6 tokenizers is built

**Read this first if you haven't seen the code.** One learner builds *all six* tokenizers:
[`scripts/fast_parity_learn_bpe.py`](../fast_parity_learn_bpe.py). The only things that change
between tokenizers are (a) the **merge-selection rule** and (b) whether there is a second
**superword** stage. Each script in this folder spells one tokenizer out **in full** — even where
that repeats boilerplate — so you can read a single file and see precisely what makes that
tokenizer what it is.

> The actual experiments used the consolidated scripts `run_baselines_fast_40k.sh`,
> `train_parity_by_lang_40k.sh`, and `run_stage_hybrids_40k.sh` (same commands, parameterized).
> These per-tokenizer recipes are the *readable* version of the same thing.

## The two knobs that define a tokenizer

1. **Selection rule** — controlled by `--global-merges N`. The first `N` merges use **GLOBAL**
   selection (pick the most frequent pair overall = standard BPE); every merge after that uses
   **PARITY** selection (pick the best pair for the *currently worst-compressed* language).
   So: `--global-merges 0` ⇒ all parity; `--global-merges <symbols>` ⇒ all global.
2. **Token type** — one stage (**within-word**, tokens never cross a space) vs two stages (add a
   **superword** stage whose tokens may cross spaces). The superword stage adds
   `--superbpe-stage2-filter --max-stage2-words 4` (SuperBPE's rule: reject a token containing
   `:Ġ` or spanning more than 4 space-delimited words).

## Facts shared by every recipe

- **Vocabulary = 40,000** = 256 byte alphabet + 39,744 merges. Two-stage split = **15,744**
  within-word + **24,000** superword.
- `--dict-input` reads the floored frequency dictionaries in
  `outputs/training_input_dicts_floored/` (built once with `pretokenize_for_parity.py` →
  `floor_dicts.py`). Within-word stages read `train_stage1/dev_stage1`; superword stages read
  `train_stage2/dev_stage2`.
- `--min-frequency 2` ignores pairs seen fewer than twice.
- `--preload <file>` continues from an existing merge list (used to chain stage 1 → stage 2);
  the preload file has **no header** (`merge_merges_files.py --no-header`).
- `build_tokenizer_from_merges.py --kind`: `parity_bpe` selects the **within-word** pretokenizer,
  `pa_superbpe` selects the **superword** pretokenizer. **These names label the pretokenizer regex
  only — they do NOT imply parity selection.** (`--kind parity_bpe` is used for plain BPE too.)
- Evaluation is always FLORES-200 devtest via `evaluate_hf_tokenizer.py`.

## The six tokenizers

| # | script | selection (stage 1 → stage 2) | token type |
|---|---|---|---|
| 1 | `1_bpe.sh` | global (1 stage) | within-word |
| 2 | `2_superbpe.sh` | global → global | superword |
| 3 | `3_pa_bpe.sh` | parity (1 stage) | within-word |
| 4 | `4_pa_superbpe.sh` | parity → parity | superword |
| 5 | `5_hybrid_A_parity1_global2.sh` | **parity → global** | superword |
| 6 | `6_hybrid_B_global1_parity2.sh` | **global → parity** | superword |

Each script writes its tokenizer + evaluation to `outputs/recipes/<name>/`.
