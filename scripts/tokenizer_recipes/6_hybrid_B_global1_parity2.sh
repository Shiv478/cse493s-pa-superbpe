#!/usr/bin/env bash
# ============================================================================
#  TOKENIZER 6 of 6 — Hybrid B  (global stage 1  ->  parity stage 2)
# ----------------------------------------------------------------------------
#  Selection rule : GLOBAL in stage 1, PARITY in stage 2.
#  Token type     : SUPERWORD.
#  Stages         : stage 1 = 15,744 within-word merges (GLOBAL)
#                   stage 2 = 24,000 superword merges  (PARITY) (total 39,744)
#  The complement of Hybrid A. Because the bigger budget (stage 2) is the
#  parity one, this gives the best TOKEN-COUNT fairness while still keeping
#  English fast -- but worse PER-CHARACTER fairness than Hybrid A.
#  Difference from SuperBPE   (2): stage 2 is parity (--global-merges 0).
#  Difference from PA-SuperBPE (4): stage 1 is global (--global-merges 15744).
# ============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PROJ"
PY="$PROJ/vendor/superbpe/.venv/bin/python"
LEARNER="$PROJ/scripts/fast_parity_learn_bpe.py"
DICTS="$PROJ/outputs/training_input_dicts_floored"
DATA="$PROJ/data/processed/flores10_test.jsonl"
OUT="$PROJ/outputs/recipes/hybrid_B_global1_parity2"; mkdir -p "$OUT"

# 1) STAGE 1 — 15,744 within-word merges, GLOBAL (--global-merges 15744)  <-- standard base
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 15744 --symbols 15744 \
  --input "$DICTS"/train_stage1/*.dict \
  --dev   "$DICTS"/dev_stage1/*.dict \
  --output "$OUT/stage1_merges.txt"

# 2) PRELOAD — reformat stage-1 merges (no header) for stage 2
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --out "$OUT/stage1_preload.txt" --no-header

# 3) STAGE 2 — 24,000 SUPERWORD merges, PARITY (--global-merges 0) + filter  <-- fair superwords
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
  --out "$OUT/hybrid_B_merges.txt"

# 5) BUILD  (--kind pa_superbpe = SUPERWORD pretokenizer)
"$PY" "$PROJ/scripts/build_tokenizer_from_merges.py" \
  --merges "$OUT/hybrid_B_merges.txt" --out-dir "$OUT/tokenizer" \
  --kind pa_superbpe --expected-vocab-size 40000

# 6) EVALUATE on FLORES-200 devtest
"$PY" "$PROJ/scripts/evaluate_hf_tokenizer.py" \
  --tokenizer "$OUT/tokenizer" --data "$DATA" --out-dir "$OUT/evaluation"

echo "Hybrid B (global stage1 -> parity stage2) done -> $OUT"
