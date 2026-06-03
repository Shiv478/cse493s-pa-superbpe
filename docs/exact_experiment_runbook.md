# Exact Experiment Runbook

This is the runbook for matching the proposal and V1 report as closely as possible.

## 1. Environment

SuperBPE requires a separate environment because it uses a custom Hugging Face `tokenizers` fork.

Recommended:

```bash
cd vendor/superbpe   # from the repo root
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ../parity-aware-bpe
pip install -e .
```

If `python3.12` or Rust is missing, install those first. The official SuperBPE README states that both are required.

## 2. Data

FLORES is already prepared by:

```bash
make prepare-flores
```

Export FLORES into per-language text files for the parity-aware BPE learner:

```bash
.venv/bin/python scripts/export_language_texts.py \
  --data data/processed/flores10_train.jsonl \
  --out-dir data/processed/flores10_dev_by_lang

.venv/bin/python scripts/export_language_texts.py \
  --data data/processed/flores10_test.jsonl \
  --out-dir data/processed/flores10_devtest_by_lang
```

For final tokenizer training, FLORES is not enough. Download or prepare the 5-10 GB English-skewed FineWeb corpus into per-language training files:

```text
data/processed/fineweb_by_lang/eng_Latn.txt
data/processed/fineweb_by_lang/spa_Latn.txt
...
```

The official SuperBPE README trains on an OLMo-2 sample; this project instead uses an English-skewed FineWeb corpus, built per-language by `scripts/prepare_fineweb_language_shards.py`.

For a valid four-tokenizer comparison, the concatenation of those per-language files is the shared training corpus. Train BPE and SuperBPE from the same directory that PA-BPE and PA-SuperBPE use for their per-language pair counts. Do not train the baselines on raw FineWeb files while training PA methods on a different language-sharded subset.

## 3. Baselines

Train BPE and SuperBPE using the official SuperBPE repository, not the prototype code.

The official scripts are:

```text
vendor/superbpe/scripts/train_tokenizer.sh
vendor/superbpe/scripts/extend_tokenizer.sh
```

For the report's target sizes:

```text
40k two-stage split:  stage 1 targets 16k vocab, then stage 2 extends to 40k vocab
200k two-stage split: stage 1 targets 80k vocab, then stage 2 extends to 200k vocab
min frequency: 2
```

This split is essential. Stage 2 is not independent. Stage 2 starts from the stage-1 merges and then adds the remaining vocabulary budget.

The SuperBPE repo supports this by seeding the stage-2 output directory with the first `num_inherit_merges` from the stage-1 `merges.txt` before calling `train_tokenizer.py` with the whitespace-lifted regex. The released 200k tokenizer folders include examples such as:

```text
fineweb_superbpe_run
```

That folder name means: inherit the 80k stage-1 portion, then extend to the 200k final tokenizer. This is the split we should mirror for PA-SuperBPE.

This repo includes a wrapper for the official SuperBPE baseline:

```bash
# 40k: 16k stage 1, extend to 40k
scripts/train_superbpe_official.sh \
  data/processed/fineweb_by_lang \
  outputs/exact_40k \
  40000 \
  16000

# 200k: 80k stage 1, extend to 200k
scripts/train_superbpe_official.sh \
  data/processed/fineweb_by_lang \
  outputs/exact_200k \
  200000 \
  80000
```

Note: Hugging Face `vocab_size` includes the byte alphabet, so the exact number of merge lines is below the nominal vocabulary size. For parity-aware scripts, `--symbols` means new merge operations. To build an exact 40k vocab from ByteLevel merges, PA-BPE uses `40000 - 256 = 39744` merges, PA-SuperBPE stage 1 uses `16000 - 256 = 15744` merges, and PA-SuperBPE stage 2 adds `40000 - 16000 = 24000` merges.

## 4. PA-SuperBPE Preparation

The report's PA-SuperBPE path pretokenizes with SuperBPE's exact stage regex + ByteLevel, then passes dict files into the official parity-aware learner.

Pretokenize stage 1:

```bash
python scripts/pretokenize_for_parity.py \
  --mode stage1 \
  --input-dir data/processed/fineweb_by_lang \
  --out-dir data/processed/pa_superbpe/stage1_train_dicts

python scripts/pretokenize_for_parity.py \
  --mode stage1 \
  --input-dir data/processed/flores10_dev_by_lang \
  --out-dir data/processed/pa_superbpe/stage1_dev_dicts
```

Pretokenize stage 2:

```bash
python scripts/pretokenize_for_parity.py \
  --mode stage2 \
  --input-dir data/processed/fineweb_by_lang \
  --out-dir data/processed/pa_superbpe/stage2_train_dicts

python scripts/pretokenize_for_parity.py \
  --mode stage2 \
  --input-dir data/processed/flores10_dev_by_lang \
  --out-dir data/processed/pa_superbpe/stage2_dev_dicts
```

## 5. PA-SuperBPE Training

Stage 1 uses the official parity-aware learner through our thin wrapper:

```bash
python scripts/parity_aware_filtered_learn_bpe.py \
  --dict-input \
  --global-merges 0 \
  --min-frequency 2 \
  --symbols 15744 \
  --input data/processed/pa_superbpe/stage1_train_dicts/*.dict \
  --dev data/processed/pa_superbpe/stage1_dev_dicts/*.dict \
  --output outputs/exact_40k/pa_superbpe_stage1_merges.txt
```

Prepare a headerless copy for `--preload`.

The official parity-aware learner's preload path expects merge pairs. To avoid treating the `#version` header as a merge, make a headerless preload file:

```bash
python scripts/merge_merges_files.py \
  --stage1 outputs/exact_40k/pa_superbpe_stage1_merges.txt \
  --out outputs/exact_40k/pa_superbpe_stage1_preload.txt \
  --no-header
```

Stage 2 preloads stage-1 merges and learns only the second split:

```bash
python scripts/parity_aware_filtered_learn_bpe.py \
  --dict-input \
  --global-merges 0 \
  --min-frequency 2 \
  --symbols 24000 \
  --preload outputs/exact_40k/pa_superbpe_stage1_preload.txt \
  --superbpe-stage2-filter \
  --max-stage2-words 4 \
  --input data/processed/pa_superbpe/stage2_train_dicts/*.dict \
  --dev data/processed/pa_superbpe/stage2_dev_dicts/*.dict \
  --output outputs/exact_40k/pa_superbpe_stage2_merges.txt
```

For the base variant, the stage-2 output contains the newly learned stage-2 merges. Build the final PA-SuperBPE merges file by concatenating stage 1 and stage 2:

```bash
python scripts/merge_merges_files.py \
  --stage1 outputs/exact_40k/pa_superbpe_stage1_merges.txt \
  --stage2 outputs/exact_40k/pa_superbpe_stage2_merges.txt \
  --out outputs/exact_40k/pa_superbpe_merges.txt
```

For 200k, use `--symbols 79744` for stage 1 and `--symbols 120000` for stage 2, then concatenate the same way.

Use `scripts/parity_aware_filtered_learn_bpe.py` for PA-SuperBPE stage 2 so the SuperBPE colon/4-word candidate filter is enforced without permanently modifying the vendored source.

## 6. Evaluation

Evaluate any Hugging Face tokenizer JSON on FLORES devtest:

```bash
python scripts/evaluate_hf_tokenizer.py \
  --tokenizer path/to/tokenizer.json \
  --data data/processed/flores10_test.jsonl \
  --out-dir outputs/exact_40k/eval_some_tokenizer
```

The final table should include one row per tokenizer method at each vocab size.
