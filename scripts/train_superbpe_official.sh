#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <corpus_dir> <output_root> <vocab_size> <stage1_vocab_size> [num_bytes]" >&2
  echo "Example 40k:  $0 data/processed/fineweb_by_lang outputs/exact_40k 40000 16000 5000000000" >&2
  echo "Example 200k: $0 data/processed/fineweb_by_lang outputs/exact_200k 200000 80000" >&2
  exit 2
fi

CORPUS_DIR="$1"
OUTPUT_ROOT="$2"
VOCAB_SIZE="$3"
STAGE1_VOCAB_SIZE="$4"
NUM_BYTES="${5:-}"
MIN_FREQUENCY="${SUPERBPE_MIN_FREQUENCY:-2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUPERBPE_DIR="${REPO_ROOT}/vendor/superbpe"
SUPERBPE_PYTHON="${SUPERBPE_PYTHON:-${SUPERBPE_DIR}/.venv/bin/python}"

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${REPO_ROOT}" "$1" ;;
  esac
}

STAGE1_REGEX='[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+'
STAGE2_REGEX='\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)'

STAGE1_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/superbpe_stage1_${STAGE1_VOCAB_SIZE}"
FINAL_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/superbpe_${STAGE1_VOCAB_SIZE}_extend_${VOCAB_SIZE}"
CORPUS_DIR_PATH="$(resolve_path "${CORPUS_DIR}")"

mkdir -p "${STAGE1_DIR}" "${FINAL_DIR}"

echo "Training official SuperBPE stage 1"
echo "  corpus: ${CORPUS_DIR_PATH}"
echo "  vocab_size: ${STAGE1_VOCAB_SIZE}"
echo "  min_frequency: ${MIN_FREQUENCY}"
if [[ -n "${NUM_BYTES}" ]]; then
  echo "  num_bytes: ${NUM_BYTES}"
fi
"${SUPERBPE_PYTHON}" "${REPO_ROOT}/scripts/patch_superbpe_min_frequency.py" \
  --superbpe-dir "${SUPERBPE_DIR}" >/dev/null
(
  cd "${SUPERBPE_DIR}"
  COMMAND=(
    "${SUPERBPE_PYTHON}" -m train_tokenizer
    --output_dir "${STAGE1_DIR}"
    --corpus_dir "${CORPUS_DIR_PATH}"
    --vocab_size "${STAGE1_VOCAB_SIZE}"
    --regex_string "${STAGE1_REGEX}"
    --min_frequency "${MIN_FREQUENCY}"
  )
  if [[ -n "${NUM_BYTES}" ]]; then
    COMMAND+=(--num_bytes "${NUM_BYTES}")
  fi
  "${COMMAND[@]}"
)

echo "Preparing official SuperBPE stage 2 seed merges"
head -n "${STAGE1_VOCAB_SIZE}" "${STAGE1_DIR}/merges.txt" > "${FINAL_DIR}/merges.txt"
cp "${STAGE1_DIR}/meta.json" "${FINAL_DIR}/meta.json"

echo "Training official SuperBPE stage 2"
echo "  inherit/stage1: ${STAGE1_VOCAB_SIZE}"
echo "  final vocab_size: ${VOCAB_SIZE}"
(
  cd "${SUPERBPE_DIR}"
  "${SUPERBPE_PYTHON}" -m train_tokenizer \
    --output_dir "${FINAL_DIR}" \
    --vocab_size "${VOCAB_SIZE}" \
    --regex_string "${STAGE2_REGEX}" \
    --min_frequency "${MIN_FREQUENCY}"
)

echo "Done: ${FINAL_DIR}"
