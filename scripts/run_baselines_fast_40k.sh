#!/usr/bin/env bash
# Baseline BPE + SuperBPE via the (validated) fast learner in GLOBAL mode.
# Global mode = every merge selected by aggregate frequency (max_index = -1) =
# standard BPE / SuperBPE. Same engine, same floored dicts, same min_frequency 2
# as the PA runs -> clean 2x2 (only difference is global vs parity selection).
set -euo pipefail

ROOT="/teamspace/studios/this_studio/2026-05-28/files-mentioned-by-the-user-cse"
cd "$ROOT"
PY="$ROOT/vendor/superbpe/.venv/bin/python"
LEARNER="$ROOT/scripts/fast_parity_learn_bpe.py"
DICTS="$ROOT/outputs/training_input_dicts_floored"
DATA="$ROOT/data/processed/flores10_test.jsonl"
OUT="$ROOT/outputs/run_40k/baseline"
LOG="$OUT/logs"
mkdir -p "$OUT" "$LOG"

TR1=( "$DICTS"/train_stage1/*.dict ); DV1=( "$DICTS"/dev_stage1/*.dict )
TR2=( "$DICTS"/train_stage2/*.dict ); DV2=( "$DICTS"/dev_stage2/*.dict )

run_bpe() {
  echo "[BPE] train (global, 39744 merges) $(date)"
  "$PY" "$LEARNER" --dict-input --min-frequency 2 \
    --global-merges 39744 --symbols 39744 \
    --input "${TR1[@]}" --dev "${DV1[@]}" \
    --output "$OUT/bpe_40000_merges.txt" > "$LOG/bpe_train.log" 2>&1
  echo "[BPE] build $(date)"
  "$PY" scripts/build_tokenizer_from_merges.py \
    --merges "$OUT/bpe_40000_merges.txt" --out-dir "$OUT/bpe_40000" \
    --kind parity_bpe --expected-vocab-size 40000 > "$LOG/bpe_build.log" 2>&1
  echo "[BPE] eval $(date)"
  "$PY" scripts/evaluate_hf_tokenizer.py \
    --tokenizer "$OUT/bpe_40000" --data "$DATA" \
    --out-dir "$OUT/eval_bpe_40000" > "$LOG/bpe_eval.log" 2>&1
  echo "[BPE] DONE $(date)"
}

run_superbpe() {
  echo "[SBPE] stage1 train (global, 15744 merges) $(date)"
  "$PY" "$LEARNER" --dict-input --min-frequency 2 \
    --global-merges 15744 --symbols 15744 \
    --input "${TR1[@]}" --dev "${DV1[@]}" \
    --output "$OUT/superbpe_stage1_16000_merges.txt" > "$LOG/sbpe_st1.log" 2>&1
  echo "[SBPE] build preload $(date)"
  "$PY" scripts/merge_merges_files.py \
    --stage1 "$OUT/superbpe_stage1_16000_merges.txt" \
    --out "$OUT/superbpe_stage1_preload.txt" --no-header > "$LOG/sbpe_preload.log" 2>&1
  echo "[SBPE] stage2 train (global, 24000 merges, filter) $(date)"
  "$PY" "$LEARNER" --dict-input --min-frequency 2 \
    --global-merges 24000 --symbols 24000 \
    --preload "$OUT/superbpe_stage1_preload.txt" \
    --superbpe-stage2-filter --max-stage2-words 4 \
    --input "${TR2[@]}" --dev "${DV2[@]}" \
    --output "$OUT/superbpe_stage2_24000_merges.txt" > "$LOG/sbpe_st2.log" 2>&1
  echo "[SBPE] combine $(date)"
  "$PY" scripts/merge_merges_files.py \
    --stage1 "$OUT/superbpe_stage1_16000_merges.txt" \
    --stage2 "$OUT/superbpe_stage2_24000_merges.txt" \
    --out "$OUT/superbpe_16000_extend_40000_merges.txt" > "$LOG/sbpe_combine.log" 2>&1
  echo "[SBPE] build $(date)"
  "$PY" scripts/build_tokenizer_from_merges.py \
    --merges "$OUT/superbpe_16000_extend_40000_merges.txt" \
    --out-dir "$OUT/superbpe_16000_extend_40000" \
    --kind pa_superbpe --expected-vocab-size 40000 > "$LOG/sbpe_build.log" 2>&1
  echo "[SBPE] eval $(date)"
  "$PY" scripts/evaluate_hf_tokenizer.py \
    --tokenizer "$OUT/superbpe_16000_extend_40000" --data "$DATA" \
    --out-dir "$OUT/eval_superbpe_40000" > "$LOG/sbpe_eval.log" 2>&1
  echo "[SBPE] DONE $(date)"
}

echo "==== baselines start $(date) ===="
run_bpe > "$LOG/bpe_pipeline.log" 2>&1 &
PID_BPE=$!
run_superbpe > "$LOG/sbpe_pipeline.log" 2>&1 &
PID_SBPE=$!
echo "BPE pid=$PID_BPE  SuperBPE pid=$PID_SBPE"

FAIL=0
wait $PID_BPE || { echo "[BPE] FAILED rc=$?"; FAIL=1; }
wait $PID_SBPE || { echo "[SBPE] FAILED rc=$?"; FAIL=1; }

echo "==== baselines end $(date) (FAIL=$FAIL) ===="
if [ "$FAIL" -eq 0 ]; then
  echo "ALL_BASELINES_OK" > "$OUT/STATUS"
else
  echo "BASELINES_FAILED" > "$OUT/STATUS"
fi
exit $FAIL
