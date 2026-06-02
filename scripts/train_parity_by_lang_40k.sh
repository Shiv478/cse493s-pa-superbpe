#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <fineweb_train_lang_dir> <flores_dev_lang_dir> <output_root>" >&2
  echo "Example: $0 data/processed/fineweb_by_lang data/processed/flores10_dev_by_lang outputs/run_40k" >&2
  exit 2
fi

TRAIN_LANG_DIR="$1"
DEV_LANG_DIR="$2"
OUTPUT_ROOT="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERBPE_DIR="${REPO_ROOT}/vendor/superbpe"
PYTHON="${SUPERBPE_PYTHON:-${SUPERBPE_DIR}/.venv/bin/python}"
# Default LEARNER is the trusted reference implementation. Set PARITY_LEARNER to
# scripts/fast_parity_learn_bpe.py to use the byte-identical fast reimplementation
# (validated via scripts/validate_fast_learner.py + tests/test_fast_learner.py).
LEARNER="${PARITY_LEARNER:-${REPO_ROOT}/scripts/parity_aware_filtered_learn_bpe.py}"
FINAL_VOCAB_SIZE="${FINAL_VOCAB_SIZE:-40000}"
STAGE1_VOCAB_SIZE="${STAGE1_VOCAB_SIZE:-16000}"
BYTE_ALPHABET_SIZE="${BYTE_ALPHABET_SIZE:-256}"
MIN_FREQUENCY="${MIN_FREQUENCY:-2}"
MIN_PARITY_LANGS="${MIN_PARITY_LANGS:-10}"

PA_BPE_MERGES=$((FINAL_VOCAB_SIZE - BYTE_ALPHABET_SIZE))
PA_STAGE1_MERGES=$((STAGE1_VOCAB_SIZE - BYTE_ALPHABET_SIZE))
PA_STAGE2_MERGES=$((FINAL_VOCAB_SIZE - STAGE1_VOCAB_SIZE))

if [[ "${PA_BPE_MERGES}" -le 0 || "${PA_STAGE1_MERGES}" -le 0 || "${PA_STAGE2_MERGES}" -le 0 ]]; then
  echo "Invalid vocab split: final=${FINAL_VOCAB_SIZE}, stage1=${STAGE1_VOCAB_SIZE}, alphabet=${BYTE_ALPHABET_SIZE}" >&2
  exit 2
fi

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

TRAIN_LANG_DIR_PATH="$(resolve_path "${TRAIN_LANG_DIR}")"
DEV_LANG_DIR_PATH="$(resolve_path "${DEV_LANG_DIR}")"
PRETOK_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}/pa_fineweb_by_lang_dicts"

if [[ ! -d "${TRAIN_LANG_DIR_PATH}" ]]; then
  echo "Missing per-language FineWeb training directory: ${TRAIN_LANG_DIR_PATH}" >&2
  echo "PA-BPE/PA-SuperBPE need one FineWeb training .txt per FLORES language." >&2
  exit 2
fi

mkdir -p "${PRETOK_ROOT}"

echo "Pretokenizing per-language FineWeb train shards"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage1 \
  --input-dir "${TRAIN_LANG_DIR_PATH}" \
  --out-dir "${PRETOK_ROOT}/train_stage1"

"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage2 \
  --input-dir "${TRAIN_LANG_DIR_PATH}" \
  --out-dir "${PRETOK_ROOT}/train_stage2"

echo "Pretokenizing FLORES dev parity reference"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage1 \
  --input-dir "${DEV_LANG_DIR_PATH}" \
  --out-dir "${PRETOK_ROOT}/dev_stage1"

"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage2 \
  --input-dir "${DEV_LANG_DIR_PATH}" \
  --out-dir "${PRETOK_ROOT}/dev_stage2"

# Optional corpus reduction: frequency-floor the TRAIN dicts (drop the rare-pretoken
# tail that can never form a min_frequency>=2 merge anyway). Applied identically to all
# languages and disclosed in the writeup. Dev dicts are NOT floored (parity reference must
# stay intact). Default off; set PRETOKEN_MIN_COUNT (e.g. 2 = drop hapaxes) to enable.
PRETOKEN_MIN_COUNT="${PRETOKEN_MIN_COUNT:-0}"
if [[ "${PRETOKEN_MIN_COUNT}" -gt 1 ]]; then
  echo "Frequency-flooring train dicts at count>=${PRETOKEN_MIN_COUNT} (corpus reduction)"
  for stage in train_stage1 train_stage2; do
    "${PYTHON}" "${REPO_ROOT}/scripts/floor_dicts.py" \
      --in-dir "${PRETOK_ROOT}/${stage}" \
      --out-dir "${PRETOK_ROOT}/${stage}_floored" \
      --min-count "${PRETOKEN_MIN_COUNT}"
    rm -rf "${PRETOK_ROOT}/${stage}"
    mv "${PRETOK_ROOT}/${stage}_floored" "${PRETOK_ROOT}/${stage}"
  done
fi

TRAIN_STAGE1_DICTS=()
TRAIN_STAGE2_DICTS=()
DEV_STAGE1_DICTS=()
DEV_STAGE2_DICTS=()

while IFS= read -r dev_stage1; do
  lang="$(basename "${dev_stage1}" .dict)"
  train_stage1="${PRETOK_ROOT}/train_stage1/${lang}.dict"
  train_stage2="${PRETOK_ROOT}/train_stage2/${lang}.dict"
  dev_stage2="${PRETOK_ROOT}/dev_stage2/${lang}.dict"
  if [[ ! -s "${train_stage1}" || ! -s "${train_stage2}" || ! -s "${dev_stage2}" ]]; then
    echo "Missing matching train/dev dict for language ${lang}" >&2
    echo "Expected ${train_stage1}, ${train_stage2}, and ${dev_stage2}" >&2
    exit 2
  fi
  TRAIN_STAGE1_DICTS+=("${train_stage1}")
  TRAIN_STAGE2_DICTS+=("${train_stage2}")
  DEV_STAGE1_DICTS+=("${dev_stage1}")
  DEV_STAGE2_DICTS+=("${dev_stage2}")
done < <(find "${PRETOK_ROOT}/dev_stage1" -maxdepth 1 -name '*.dict' -print | sort)

if [[ "${#DEV_STAGE1_DICTS[@]}" -lt "${MIN_PARITY_LANGS}" ]]; then
  echo "Expected at least ${MIN_PARITY_LANGS} language dicts, found ${#DEV_STAGE1_DICTS[@]}." >&2
  exit 2
fi

# PA-BPE and PA-SuperBPE are independent jobs: they only READ the shared
# pretokenized dicts and write disjoint outputs (parity_bpe_* vs pa_superbpe_*).
# Their merge loops are single-threaded, so running the two jobs concurrently is
# the only way to use >1 core here -- it roughly halves the parity-phase wall time
# at ~2x peak RAM. Results are identical to sequential (same inputs, same merges).
# Set PA_PARALLEL=0 to force sequential if RAM is tight.
PA_PARALLEL="${PA_PARALLEL:-1}"
PA_LOG_DIR="${REPO_ROOT}/${OUTPUT_ROOT}"

run_pa_bpe() {
  echo "[PA-BPE] Training parity-aware BPE ${FINAL_VOCAB_SIZE} (merges ${PA_BPE_MERGES}, min_freq ${MIN_FREQUENCY})"
  "${PYTHON}" "${LEARNER}" \
    --global-merges 0 \
    --min-frequency "${MIN_FREQUENCY}" \
    --dict-input \
    --symbols "${PA_BPE_MERGES}" \
    --input "${TRAIN_STAGE1_DICTS[@]}" \
    --dev "${DEV_STAGE1_DICTS[@]}" \
    --output "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}_merges.txt"

  "${PYTHON}" "${REPO_ROOT}/scripts/build_tokenizer_from_merges.py" \
    --merges "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}_merges.txt" \
    --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}" \
    --kind stage1 \
    --expected-vocab-size "${FINAL_VOCAB_SIZE}"
  echo "[PA-BPE] done"
}

run_pa_superbpe() {
  echo "[PA-SuperBPE] stage 1 (target ${STAGE1_VOCAB_SIZE}, merges ${PA_STAGE1_MERGES})"
  "${PYTHON}" "${LEARNER}" \
    --global-merges 0 \
    --min-frequency "${MIN_FREQUENCY}" \
    --dict-input \
    --symbols "${PA_STAGE1_MERGES}" \
    --input "${TRAIN_STAGE1_DICTS[@]}" \
    --dev "${DEV_STAGE1_DICTS[@]}" \
    --output "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt"

  "${PYTHON}" "${REPO_ROOT}/scripts/merge_merges_files.py" \
    --stage1 "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt" \
    --out "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_preload.txt" \
    --no-header

  echo "[PA-SuperBPE] stage 2 (add ${PA_STAGE2_MERGES} merges -> ${FINAL_VOCAB_SIZE}, superbpe filter on)"
  "${PYTHON}" "${LEARNER}" \
    --global-merges 0 \
    --min-frequency "${MIN_FREQUENCY}" \
    --dict-input \
    --symbols "${PA_STAGE2_MERGES}" \
    --preload "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_preload.txt" \
    --superbpe-stage2-filter \
    --max-stage2-words 4 \
    --input "${TRAIN_STAGE2_DICTS[@]}" \
    --dev "${DEV_STAGE2_DICTS[@]}" \
    --output "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage2_${PA_STAGE2_MERGES}_merges.txt"

  "${PYTHON}" "${REPO_ROOT}/scripts/merge_merges_files.py" \
    --stage1 "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt" \
    --stage2 "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage2_${PA_STAGE2_MERGES}_merges.txt" \
    --out "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_${STAGE1_VOCAB_SIZE}_extend_${FINAL_VOCAB_SIZE}_merges.txt"

  "${PYTHON}" "${REPO_ROOT}/scripts/build_tokenizer_from_merges.py" \
    --merges "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_${STAGE1_VOCAB_SIZE}_extend_${FINAL_VOCAB_SIZE}_merges.txt" \
    --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_${STAGE1_VOCAB_SIZE}_extend_${FINAL_VOCAB_SIZE}" \
    --kind pa_superbpe \
    --expected-vocab-size "${FINAL_VOCAB_SIZE}"
  echo "[PA-SuperBPE] done"
}

if [[ "${PA_PARALLEL}" == "1" ]]; then
  echo "Running PA-BPE and PA-SuperBPE concurrently (PA_PARALLEL=1). Per-job logs:"
  echo "  ${PA_LOG_DIR}/pa_bpe_job.log"
  echo "  ${PA_LOG_DIR}/pa_superbpe_job.log"
  run_pa_bpe      > "${PA_LOG_DIR}/pa_bpe_job.log"      2>&1 &
  pid_bpe=$!
  run_pa_superbpe > "${PA_LOG_DIR}/pa_superbpe_job.log" 2>&1 &
  pid_sbpe=$!
  rc=0
  wait "${pid_bpe}"  || { echo "PA-BPE job FAILED (see pa_bpe_job.log)" >&2; rc=1; }
  wait "${pid_sbpe}" || { echo "PA-SuperBPE job FAILED (see pa_superbpe_job.log)" >&2; rc=1; }
  echo "--- tail pa_bpe_job.log ---";      tail -n 5 "${PA_LOG_DIR}/pa_bpe_job.log"      || true
  echo "--- tail pa_superbpe_job.log ---"; tail -n 5 "${PA_LOG_DIR}/pa_superbpe_job.log" || true
  [[ "${rc}" -eq 0 ]] || exit 1
else
  echo "Running PA-BPE then PA-SuperBPE sequentially (PA_PARALLEL=0)"
  run_pa_bpe
  run_pa_superbpe
fi

echo "Done: parity-aware per-language ${FINAL_VOCAB_SIZE} artifacts in ${REPO_ROOT}/${OUTPUT_ROOT}"
