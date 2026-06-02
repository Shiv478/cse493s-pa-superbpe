#!/usr/bin/env bash
# ============================================================================
#  TOKENIZER 4 of 6 — PA-SuperBPE  (parity in BOTH stages)
# ----------------------------------------------------------------------------
#  Selection rule : PARITY in BOTH stages.
#  Token type     : SUPERWORD — stage 2 learns merges that cross spaces.
#  Stages         : stage 1 = 15,744 within-word merges (parity)
#                   stage 2 = 24,000 superword merges  (parity)  (total 39,744)
#  This is the "full parity" superword tokenizer: maximal token-count fairness,
#  but it gives up the dominant language's compression.
#  Difference from SuperBPE (tokenizer 2): both stages use --global-merges 0.
# ============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PROJ"
PY="$PROJ/vendor/superbpe/.venv/bin/python"
LEARNER="$PROJ/scripts/fast_parity_learn_bpe.py"
DICTS="$PROJ/outputs/training_input_dicts_floored"
DATA="$PROJ/data/processed/flores10_test.jsonl"
OUT="$PROJ/outputs/recipes/pa_superbpe"; mkdir -p "$OUT"

# 1) STAGE 1 — 15,744 within-word merges, PARITY (--global-merges 0)
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 0 --symbols 15744 \
  --input "$DICTS"/train_stage1/*.dict \
  --dev   "$DICTS"/dev_stage1/*.dict \
  --output "$OUT/stage1_merges.txt"

# 2) PRELOAD — reformat stage-1 merges (no header) for stage 2 to continue from
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --out "$OUT/stage1_preload.txt" --no-header

# 3) STAGE 2 — 24,000 SUPERWORD merges, PARITY (--global-merges 0) + SuperBPE filter
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 0 --symbols 24000 \
  --preload "$OUT/stage1_preload.txt" \
  --superbpe-stage2-filter --max-stage2-words 4 \
  --input "$DICTS"/train_stage2/*.dict \
  --dev   "$DICTS"/dev_stage2/*.dict \
  --output "$OUT/stage2_merges.txt"

# 4) COMBINE stage 1 + stage 2
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --stage2 "$OUT/stage2_merges.txt" \
  --out "$OUT/pa_superbpe_merges.txt"

# 5) BUILD  (--kind pa_superbpe = SUPERWORD pretokenizer)
"$PY" "$PROJ/scripts/build_tokenizer_from_merges.py" \
  --merges "$OUT/pa_superbpe_merges.txt" --out-dir "$OUT/tokenizer" \
  --kind pa_superbpe --expected-vocab-size 40000

# 6) EVALUATE on FLORES-200 devtest
"$PY" "$PROJ/scripts/evaluate_hf_tokenizer.py" \
  --tokenizer "$OUT/tokenizer" --data "$DATA" --out-dir "$OUT/evaluation"

echo "PA-SuperBPE done -> $OUT"
