#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <shared_fineweb_meta_json> <flores_dev_lang_dir> <output_root>" >&2
  echo "Example: $0 outputs/run_40k/bpe_40000/meta.json data/processed/flores10_dev_by_lang outputs/run_40k" >&2
  exit 2
fi

SHARED_META="$1"
DEV_LANG_DIR="$2"
OUTPUT_ROOT="$3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERBPE_DIR="${REPO_ROOT}/vendor/superbpe"
PYTHON="${SUPERBPE_PYTHON:-${SUPERBPE_DIR}/.venv/bin/python}"
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

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

SHARED_META_PATH="$(resolve_path "${SHARED_META}")"
DEV_LANG_DIR_PATH="$(resolve_path "${DEV_LANG_DIR}")"

mkdir -p "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts"

DEV_TXTS=()
while IFS= read -r path; do
  DEV_TXTS+=("${path}")
done < <(find "${DEV_LANG_DIR_PATH}" -maxdepth 1 -name '*.txt' -print | sort)
if [[ "${#DEV_TXTS[@]}" -eq 0 ]]; then
  echo "Expected non-empty per-language FLORES dev .txt files in ${DEV_LANG_DIR}." >&2
  exit 1
fi

echo "Pretokenizing shared FineWeb corpus for parity-aware BPE stage1"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_meta_for_parity.py" \
  --mode stage1 \
  --meta "${SHARED_META_PATH}" \
  --out "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/fineweb_stage1.dict"

echo "Pretokenizing shared FineWeb corpus for PA-SuperBPE stage2"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_meta_for_parity.py" \
  --mode stage2 \
  --meta "${SHARED_META_PATH}" \
  --out "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/fineweb_stage2.dict"

echo "Pretokenizing FLORES dev parity reference"
"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage1 \
  --input-dir "${DEV_LANG_DIR_PATH}" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/flores_dev_stage1"

"${PYTHON}" "${REPO_ROOT}/scripts/pretokenize_for_parity.py" \
  --mode stage2 \
  --input-dir "${DEV_LANG_DIR_PATH}" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/flores_dev_stage2"

DEV_STAGE1_DICTS=()
while IFS= read -r path; do
  DEV_STAGE1_DICTS+=("${path}")
done < <(find "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/flores_dev_stage1" -maxdepth 1 -name '*.dict' -print | sort)

DEV_STAGE2_DICTS=()
while IFS= read -r path; do
  DEV_STAGE2_DICTS+=("${path}")
done < <(find "${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/flores_dev_stage2" -maxdepth 1 -name '*.dict' -print | sort)

STAGE1_INPUTS=()
STAGE2_INPUTS=()
for _dev in "${DEV_STAGE1_DICTS[@]}"; do
  STAGE1_INPUTS+=("${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/fineweb_stage1.dict")
done
for _dev in "${DEV_STAGE2_DICTS[@]}"; do
  STAGE2_INPUTS+=("${REPO_ROOT}/${OUTPUT_ROOT}/pa_shared_dicts/fineweb_stage2.dict")
done

echo "Training parity-aware BPE ${FINAL_VOCAB_SIZE} from shared FineWeb corpus + FLORES parity dev"
echo "WARNING: shared-FineWeb parity training is exploratory only. Final PA runs need real per-language FineWeb shards." >&2
"${PYTHON}" "${LEARNER}" \
  --global-merges 0 \
  --min-frequency "${MIN_FREQUENCY}" \
  --dict-input \
  --symbols "${PA_BPE_MERGES}" \
  --input "${STAGE1_INPUTS[@]}" \
  --dev "${DEV_STAGE1_DICTS[@]}" \
  --output "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}_merges.txt"

"${PYTHON}" "${REPO_ROOT}/scripts/build_tokenizer_from_merges.py" \
  --merges "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}_merges.txt" \
  --out-dir "${REPO_ROOT}/${OUTPUT_ROOT}/parity_bpe_${FINAL_VOCAB_SIZE}" \
  --kind stage1 \
  --expected-vocab-size "${FINAL_VOCAB_SIZE}"

echo "Training PA-SuperBPE stage 1"
echo "  target stage-1 vocab size: ${STAGE1_VOCAB_SIZE}"
echo "  learned stage-1 merges: ${PA_STAGE1_MERGES}"
"${PYTHON}" "${LEARNER}" \
  --global-merges 0 \
  --min-frequency "${MIN_FREQUENCY}" \
  --dict-input \
  --symbols "${PA_STAGE1_MERGES}" \
  --input "${STAGE1_INPUTS[@]}" \
  --dev "${DEV_STAGE1_DICTS[@]}" \
  --output "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt"

"${PYTHON}" "${REPO_ROOT}/scripts/merge_merges_files.py" \
  --stage1 "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_merges.txt" \
  --out "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_preload.txt" \
  --no-header

echo "Training PA-SuperBPE stage 2"
echo "  additional stage-2 merges: ${PA_STAGE2_MERGES}"
"${PYTHON}" "${LEARNER}" \
  --global-merges 0 \
  --min-frequency "${MIN_FREQUENCY}" \
  --dict-input \
  --symbols "${PA_STAGE2_MERGES}" \
  --preload "${REPO_ROOT}/${OUTPUT_ROOT}/pa_superbpe_stage1_${STAGE1_VOCAB_SIZE}_preload.txt" \
  --superbpe-stage2-filter \
  --max-stage2-words 4 \
  --input "${STAGE2_INPUTS[@]}" \
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

echo "Done: parity-aware shared-FineWeb ${FINAL_VOCAB_SIZE} artifacts in ${REPO_ROOT}/${OUTPUT_ROOT}"
