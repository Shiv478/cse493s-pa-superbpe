#!/usr/bin/env bash
# ============================================================================
#  TOKENIZER 1 of 6 — BPE  (the standard baseline)
# ----------------------------------------------------------------------------
#  Selection rule : GLOBAL  — every merge is the most frequent pair overall.
#  Token type     : WITHIN-WORD — no token may cross a space.
#  Stages         : ONE stage, 39,744 merges.
#  What makes it BPE: global selection + within-word only = textbook byte-pair
#  encoding. (Our learner's global mode is verified byte-identical to a
#  from-scratch standard BPE, see scripts/verify_global_is_standard_bpe.py.)
# ============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PROJ"
PY="$PROJ/vendor/superbpe/.venv/bin/python"
LEARNER="$PROJ/scripts/fast_parity_learn_bpe.py"
DICTS="$PROJ/outputs/training_input_dicts_floored"
DATA="$PROJ/data/processed/flores10_test.jsonl"
OUT="$PROJ/outputs/recipes/bpe"; mkdir -p "$OUT"

# 1) TRAIN — 39,744 merges, ALL global  (--global-merges == --symbols => no parity)
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 39744 --symbols 39744 \
  --input "$DICTS"/train_stage1/*.dict \
  --dev   "$DICTS"/dev_stage1/*.dict \
  --output "$OUT/bpe_merges.txt"

# 2) BUILD the Hugging Face tokenizer  (--kind parity_bpe = WITHIN-WORD pretokenizer)
"$PY" "$PROJ/scripts/build_tokenizer_from_merges.py" \
  --merges "$OUT/bpe_merges.txt" --out-dir "$OUT/tokenizer" \
  --kind parity_bpe --expected-vocab-size 40000

# 3) EVALUATE on FLORES-200 devtest
"$PY" "$PROJ/scripts/evaluate_hf_tokenizer.py" \
  --tokenizer "$OUT/tokenizer" --data "$DATA" --out-dir "$OUT/evaluation"

echo "BPE done -> $OUT"
