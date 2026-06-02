#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <corpus_dir> <output_root> <vocab_size> [num_bytes]" >&2
  echo "Example 40k: $0 data/processed/fineweb_by_lang outputs/exact_40k 40000 5000000000" >&2
  exit 2
fi

CORPUS_DIR="$1"
OUTPUT_ROOT="$2"
VOCAB_SIZE="$3"
NUM_BYTES="${4:-}"
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
OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_ROOT}/bpe_${VOCAB_SIZE}"
CORPUS_DIR_PATH="$(resolve_path "${CORPUS_DIR}")"

mkdir -p "${OUTPUT_DIR}"

echo "Training official BPE"
echo "  corpus: ${CORPUS_DIR_PATH}"
echo "  vocab_size: ${VOCAB_SIZE}"
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
    --output_dir "${OUTPUT_DIR}"
    --corpus_dir "${CORPUS_DIR_PATH}"
    --vocab_size "${VOCAB_SIZE}"
    --regex_string "${STAGE1_REGEX}"
    --min_frequency "${MIN_FREQUENCY}"
  )
  if [[ -n "${NUM_BYTES}" ]]; then
    COMMAND+=(--num_bytes "${NUM_BYTES}")
  fi
  "${COMMAND[@]}"
)

echo "Done: ${OUTPUT_DIR}"
