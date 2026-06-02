#!/usr/bin/env bash
# ============================================================================
#  TOKENIZER 5 of 6 — Hybrid A  (parity stage 1  ->  global stage 2)
# ----------------------------------------------------------------------------
#  Selection rule : PARITY in stage 1, GLOBAL in stage 2.
#  Token type     : SUPERWORD.
#  Stages         : stage 1 = 15,744 within-word merges (PARITY)
#                   stage 2 = 24,000 superword merges  (GLOBAL) (total 39,744)
#  Idea: build a FAIR within-word base (parity reaches every language, incl.
#  the spaceless ones), then let GLOBAL superwords recover the dominant
#  language's compression. This is the best-balanced tokenizer in our results:
#  best per-character fairness AND English stays fast.
#  Difference from PA-SuperBPE (4): stage 2 is global (--global-merges 24000).
#  Difference from SuperBPE   (2): stage 1 is parity (--global-merges 0).
# ============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PROJ"
PY="$PROJ/vendor/superbpe/.venv/bin/python"
LEARNER="$PROJ/scripts/fast_parity_learn_bpe.py"
DICTS="$PROJ/outputs/training_input_dicts_floored"
DATA="$PROJ/data/processed/flores10_test.jsonl"
OUT="$PROJ/outputs/recipes/hybrid_A_parity1_global2"; mkdir -p "$OUT"

# 1) STAGE 1 — 15,744 within-word merges, PARITY (--global-merges 0)  <-- fair base
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 0 --symbols 15744 \
  --input "$DICTS"/train_stage1/*.dict \
  --dev   "$DICTS"/dev_stage1/*.dict \
  --output "$OUT/stage1_merges.txt"

# 2) PRELOAD — reformat stage-1 merges (no header) for stage 2
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --out "$OUT/stage1_preload.txt" --no-header

# 3) STAGE 2 — 24,000 SUPERWORD merges, GLOBAL (--global-merges 24000) + filter  <-- global superwords
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 24000 --symbols 24000 \
  --preload "$OUT/stage1_preload.txt" \
  --superbpe-stage2-filter --max-stage2-words 4 \
  --input "$DICTS"/train_stage2/*.dict \
  --dev   "$DICTS"/dev_stage2/*.dict \
  --output "$OUT/stage2_merges.txt"

# 4) COMBINE stage 1 + stage 2
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --stage2 "$OUT/stage2_merges.txt" \
  --out "$OUT/hybrid_A_merges.txt"

# 5) BUILD  (--kind pa_superbpe = SUPERWORD pretokenizer)
"$PY" "$PROJ/scripts/build_tokenizer_from_merges.py" \
  --merges "$OUT/hybrid_A_merges.txt" --out-dir "$OUT/tokenizer" \
  --kind pa_superbpe --expected-vocab-size 40000

# 6) EVALUATE on FLORES-200 devtest
"$PY" "$PROJ/scripts/evaluate_hf_tokenizer.py" \
  --tokenizer "$OUT/tokenizer" --data "$DATA" --out-dir "$OUT/evaluation"

echo "Hybrid A (parity stage1 -> global stage2) done -> $OUT"
