#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <train_lang_dir> <dev_lang_dir> <output_root>" >&2
  echo "Example: $0 data/processed/flores10_dev_by_lang data/processed/flores10_dev_by_lang outputs/exact_40k" >&2
  exit 2
fi

TRAIN_LANG_DIR="$1"
DEV_LANG_DIR="$2"
OUTPUT_ROOT="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERBPE_DIR="${REPO_ROOT}/vendor/superbpe"
PYTHON="${SUPERBPE_PYTHON:-${SUPERBPE_DIR}/.venv/bin/python}"
PARITY="${REPO_ROOT}/vendor/parity-aware-bpe/parity_aware_bpe/parity_aware_learn_bpe.py"
LEARNER="${REPO_ROOT}/scripts/parity_aware_filtered_learn_bpe.py"
FINAL_VOCAB_SIZE="${FINAL_VOCAB_SIZE:-40000}"
STAGE1_VOCAB_SIZE="${STAGE1_VOCAB_SIZE:-16000}"
BYTE_ALPHABET_SIZE="${BYTE_ALPHABET_SIZE:-256}"
MIN_FREQUENCY="${MIN_FREQUENCY:-2}"
PA_BPE_MERGES=$((FINAL_VOCAB_SIZE - BYTE_ALPHABET_SIZE))
PA_STAGE1_MERGES=$((STAGE1_VOCAB_SIZE - BYTE_ALPHABET_SIZE))
PA_STAGE2_MERGES=$((FINAL_VOCAB_SIZE - STAGE1_VOCAB_SIZE))

if [[ "${PA_BPE_MERGES}" -le 0 || "${PA_STAGE1_MERGES}" -le 0 || "${PA_STAGE2_MERGES}" -le 0 ]]; then
  echo "Invalid vocab split: final=${FINAL_VOCAB_SIZE}, stage1=${STAGE1_VOCAB_SIZE}, alphabet=${BYTE_ALPHABET_SIZE}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/${OUTPUT_ROOT}"

mapfile -t TRAIN_TXTS < <(find "${REPO_ROOT}/${TRAIN_LANG_DIR}" -maxdepth 1 -name '*.txt' -print | sort)
mapfile -t DEV_TXTS < <(find "${REPO_ROOT}/${DEV_LANG_DIR}" -maxdepth 1 -name '*.txt' -print | sort)

if [[ "${#TRAIN_TXTS[@]}" -eq 0 || "${#TRAIN_TXTS[@]}" -ne "${#DEV_TXTS[@]}" ]]; then
  echo "Expected matching non-empty per-language train/dev .txt files." >&2
  exit 1
fi

echo "Training parity-aware BPE ${FINAL_VOCAB_SIZE}"
"${PYTHON}" "${PARITY}" \
  --variant base \
  --global-merges 0 \
  --min-frequency "${MIN_FREQUENCY}" \
  --symbols "${PA_BPE_MERGES}" \
  --input "${TRAIN_TXTS[@]}" \
  --dev "${DEV_TXTS[@]}" \
  --output "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}_merges.txt"

"${PYTHON}" "${REPO_ROOT}/scripts/build_tokenizer_from_merges.py" \
  --merges "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}_merges.txt" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}" \
  --kind parity_bpe \
  --expected-vocab-size "${FINAL_VOCAB_SIZE}"

echo "Pretokenizing PA-SuperBPE stage 1 inputs"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage1 \
  --input-dir "${REPO_ROOT}/${TRAIN_LANG_DIR}" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage1_train"

"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage1 \
  --input-dir "${REPO_ROOT}/${DEV_LANG_DIR}" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage1_dev"

mapfile -t STAGE1_TRAIN_DICTS < <(find "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage1_train" -maxdepth 1 -name '*.dict' -print | sort)
mapfile -t STAGE1_DEV_DICTS < <(find "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage1_dev" -maxdepth 1 -name '*.dict' -print | sort)

echo "Training PA-SuperBPE stage 1"
"${PYTHON}" "${LEARNER}" \
  --global-merges 0 \
  --min-frequency "${MIN_FREQUENCY}" \
  --dict-input \
  --symbols "${PA_STAGE1_MERGES}" \
  --input "${STAGE1_TRAIN_DICTS[@]}" \
  --dev "${STAGE1_DEV_DICTS[@]}" \
  --output "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt"

"${PYTHON}" "${REPO_ROOT}/scripts/merge_merges_files.py" \
  --stage1 "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt" \
  --out "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_preload.txt" \
  --no-header

echo "Pretokenizing PA-SuperBPE stage 2 inputs"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage2 \
  --input-dir "${REPO_ROOT}/${TRAIN_LANG_DIR}" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage2_train"

"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage2 \
  --input-dir "${REPO_ROOT}/${DEV_LANG_DIR}" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage2_dev"

mapfile -t STAGE2_TRAIN_DICTS < <(find "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage2_train" -maxdepth 1 -name '*.dict' -print | sort)
mapfile -t STAGE2_DEV_DICTS < <(find "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_pretok/stage2_dev" -maxdepth 1 -name '*.dict' -print | sort)

echo "Training PA-SuperBPE stage 2"
"${PYTHON}" "${LEARNER}" \
  --global-merges 0 \
  --min-frequency "${MIN_FREQUENCY}" \
  --dict-input \
  --symbols "${PA_STAGE2_MERGES}" \
  --preload "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_preload.txt" \
  --superbpe-stage2-filter \
  --max-stage2-words 4 \
  --input "${STAGE2_TRAIN_DICTS[@]}" \
  --dev "${STAGE2_DEV_DICTS[@]}" \
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

echo "Done: parity-aware ${FINAL_VOCAB_SIZE} artifacts in ${REPO_ROOT}/${OUTPUT_ROOT}"
