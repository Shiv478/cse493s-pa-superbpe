#!/usr/bin/env bash
# ============================================================================
#  TOKENIZER 3 of 6 — PA-BPE  (parity-aware BPE)
# ----------------------------------------------------------------------------
#  Selection rule : PARITY  — every merge improves the worst-compressed
#                   language (measured on the FLORES dev "ruler").
#  Token type     : WITHIN-WORD — no token may cross a space.
#  Stages         : ONE stage, 39,744 merges.
#  Difference from BPE (tokenizer 1): ONLY the selection rule
#  (--global-merges 0 instead of 39744). Everything else is identical.
# ============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PROJ"
PY="$PROJ/vendor/superbpe/.venv/bin/python"
LEARNER="$PROJ/scripts/fast_parity_learn_bpe.py"
DICTS="$PROJ/outputs/training_input_dicts_floored"
DATA="$PROJ/data/processed/flores10_test.jsonl"
OUT="$PROJ/outputs/recipes/pa_bpe"; mkdir -p "$OUT"

# 1) TRAIN — 39,744 merges, ALL parity  (--global-merges 0)
#    --dev provides the per-language ruler that says which language is worst.
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 0 --symbols 39744 \
  --input "$DICTS"/train_stage1/*.dict \
  --dev   "$DICTS"/dev_stage1/*.dict \
  --output "$OUT/pa_bpe_merges.txt"

# 2) BUILD  (--kind parity_bpe = WITHIN-WORD pretokenizer)
"$PY" "$PROJ/scripts/build_tokenizer_from_merges.py" \
  --merges "$OUT/pa_bpe_merges.txt" --out-dir "$OUT/tokenizer" \
  --kind parity_bpe --expected-vocab-size 40000

# 3) EVALUATE on FLORES-200 devtest
"$PY" "$PROJ/scripts/evaluate_hf_tokenizer.py" \
  --tokenizer "$OUT/tokenizer" --data "$DATA" --out-dir "$OUT/evaluation"

echo "PA-BPE done -> $OUT"
