#!/usr/bin/env bash
# ============================================================================
#  TOKENIZER 2 of 6 — SuperBPE  (the two-stage baseline)
# ----------------------------------------------------------------------------
#  Selection rule : GLOBAL in BOTH stages.
#  Token type     : SUPERWORD — stage 2 learns merges that cross spaces.
#  Stages         : stage 1 = 15,744 within-word merges
#                   stage 2 = 24,000 superword merges   (total 39,744)
#  What makes it SuperBPE: global selection + the second cross-whitespace stage.
# ============================================================================
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$PROJ"
PY="$PROJ/vendor/superbpe/.venv/bin/python"
LEARNER="$PROJ/scripts/fast_parity_learn_bpe.py"
DICTS="$PROJ/outputs/training_input_dicts_floored"
DATA="$PROJ/data/processed/flores10_test.jsonl"
OUT="$PROJ/outputs/recipes/superbpe"; mkdir -p "$OUT"

# 1) STAGE 1 — 15,744 within-word merges, ALL global
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 15744 --symbols 15744 \
  --input "$DICTS"/train_stage1/*.dict \
  --dev   "$DICTS"/dev_stage1/*.dict \
  --output "$OUT/stage1_merges.txt"

# 2) PRELOAD — reformat stage-1 merges (no header) so stage 2 continues from them
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --out "$OUT/stage1_preload.txt" --no-header

# 3) STAGE 2 — 24,000 SUPERWORD merges, ALL global, with the SuperBPE stage-2 filter
"$PY" "$LEARNER" --dict-input --min-frequency 2 \
  --global-merges 24000 --symbols 24000 \
  --preload "$OUT/stage1_preload.txt" \
  --superbpe-stage2-filter --max-stage2-words 4 \
  --input "$DICTS"/train_stage2/*.dict \
  --dev   "$DICTS"/dev_stage2/*.dict \
  --output "$OUT/stage2_merges.txt"

# 4) COMBINE stage 1 + stage 2 into the final 39,744-merge list
"$PY" "$PROJ/scripts/merge_merges_files.py" \
  --stage1 "$OUT/stage1_merges.txt" --stage2 "$OUT/stage2_merges.txt" \
  --out "$OUT/superbpe_merges.txt"

# 5) BUILD  (--kind pa_superbpe = SUPERWORD pretokenizer)
"$PY" "$PROJ/scripts/build_tokenizer_from_merges.py" \
  --merges "$OUT/superbpe_merges.txt" --out-dir "$OUT/tokenizer" \
  --kind pa_superbpe --expected-vocab-size 40000

# 6) EVALUATE on FLORES-200 devtest
"$PY" "$PROJ/scripts/evaluate_hf_tokenizer.py" \
  --tokenizer "$OUT/tokenizer" --data "$DATA" --out-dir "$OUT/evaluation"

echo "SuperBPE done -> $OUT"
