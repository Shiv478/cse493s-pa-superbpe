#!/usr/bin/env bash
set -euo pipefail

# Lightning-ready 40k tokenizer run via the OFFICIAL Rust BPE/SuperBPE trainers.
#
# NOTE: the project's final reported results were produced by the faster, validated
# fast-learner pipeline (scripts/run_baselines_fast_40k.sh + the parity learner) on
# the floored dicts. This script is the official-trainer alternate path; both train
# on the same English-skewed FineWeb corpus.
#
# Run from the project root inside a Lightning Studio or Lightning job:
#   bash lightning_run_40k_exact.sh
#
# Recommended machine:
#   16+ CPU cores, 128 GB RAM, 200+ GB disk.
#   GPU is not required for tokenizer training.
#
# Tunables:
#   NUM_BYTES=5000000000        # must stay in 5-10 GB range for the report.
#                               # With RUN_PARITY=1 this is a MINIMUM-size guard
#                               # (a floor), not a cap: all four tokenizers train
#                               # on the full per-language corpus you stage.
#   RAYON_NUM_THREADS=16        # lower to 8/4 if the SuperBPE stage-2 job is killed
#   OUTPUT_ROOT=outputs/run_40k
#   RUN_PARITY=1                # trains PA-BPE and PA-SuperBPE after BPE/SuperBPE
#   PARITY_TRAIN_DIR=data/processed/fineweb_by_lang

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

NUM_BYTES="${NUM_BYTES:-5000000000}"
VOCAB_SIZE="${VOCAB_SIZE:-40000}"
STAGE1_VOCAB_SIZE="${STAGE1_VOCAB_SIZE:-16000}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-16}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/run_40k}"
RUN_PARITY="${RUN_PARITY:-1}"
SUPERBPE_MIN_FREQUENCY="${SUPERBPE_MIN_FREQUENCY:-2}"
PARITY_TRAIN_DIR="${PARITY_TRAIN_DIR:-data/processed/fineweb_by_lang}"
# PHASE selects which tokenizers to train in this invocation, so the run can be
# split across machines/sessions without changing the corpus or any other knob:
#   PHASE=baseline -> only BPE + SuperBPE (+ their evals)   [Rust, scales with cores]
#   PHASE=parity   -> only PA-BPE + PA-SuperBPE (+ evals)   [Python, ~2 cores max]
#   PHASE=all      -> everything (default; original behavior)
# Env build, repo clones, FLORES prep, corpus + size guard, and the full-corpus
# (no --num_bytes cap) matching all run identically in every phase, so the four
# tokenizers stay trained on byte-identical data regardless of how you split.
PHASE="${PHASE:-all}"

SUPERBPE_DIR="${ROOT}/vendor/superbpe"
PARITY_DIR="${ROOT}/vendor/parity-aware-bpe"
PY="${SUPERBPE_DIR}/.venv/bin/python"
FINEWEB_DIR="${ROOT}/data/processed/fineweb_by_lang"
FLORES_DIR="${ROOT}/data/external/flores200_dataset"
LOG_DIR="${ROOT}/logs"

mkdir -p "${LOG_DIR}" data/external data/processed vendor

PARITY_LANGS=(
  arb_Arab
  deu_Latn
  eng_Latn
  hin_Deva
  jpn_Jpan
  rus_Cyrl
  spa_Latn
  swh_Latn
  tha_Thai
  zho_Hans
)

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${ROOT}" "$1" ;;
  esac
}

if [[ "${RUN_PARITY}" != "0" && "${RUN_PARITY}" != "1" ]]; then
  echo "RUN_PARITY must be 0 or 1, got ${RUN_PARITY}." >&2
  exit 2
fi

case "${PHASE}" in
  baseline|parity|all) ;;
  *) echo "PHASE must be baseline, parity, or all; got ${PHASE}." >&2; exit 2 ;;
esac

# The parity phase trains PA-* and so requires the per-language shards + RUN_PARITY=1.
if [[ "${PHASE}" == "parity" && "${RUN_PARITY}" != "1" ]]; then
  echo "PHASE=parity requires RUN_PARITY=1 (it trains the parity-aware tokenizers)." >&2
  exit 2
fi
# Decide once whether the parity tokenizers are in scope for this invocation.
DO_BASELINE=0; DO_PARITY=0
[[ "${PHASE}" == "baseline" || "${PHASE}" == "all" ]] && DO_BASELINE=1
if [[ "${RUN_PARITY}" == "1" && ( "${PHASE}" == "parity" || "${PHASE}" == "all" ) ]]; then
  DO_PARITY=1
fi

if [[ "${RUN_PARITY}" == "1" ]]; then
  PARITY_TRAIN_DIR_PATH="$(resolve_path "${PARITY_TRAIN_DIR}")"
  if [[ ! -d "${PARITY_TRAIN_DIR_PATH}" ]]; then
    echo "RUN_PARITY=1 requires per-language FineWeb training shards at:" >&2
    echo "  ${PARITY_TRAIN_DIR_PATH}" >&2
    echo "Expected files: ${PARITY_LANGS[*]/%/.txt}" >&2
    echo "Set RUN_PARITY=0 to run only BPE/SuperBPE baselines." >&2
    exit 2
  fi
  for lang in "${PARITY_LANGS[@]}"; do
    if [[ ! -s "${PARITY_TRAIN_DIR_PATH}/${lang}.txt" ]]; then
      echo "Missing or empty parity FineWeb shard: ${PARITY_TRAIN_DIR_PATH}/${lang}.txt" >&2
      exit 2
    fi
  done
fi

if [[ -z "${TOKENIZER_TRAIN_DIR:-}" ]]; then
  if [[ "${RUN_PARITY}" == "1" ]]; then
    TOKENIZER_TRAIN_DIR="${PARITY_TRAIN_DIR}"
  else
    TOKENIZER_TRAIN_DIR="data/processed/fineweb_by_lang"
  fi
fi
TOKENIZER_TRAIN_DIR_PATH="$(resolve_path "${TOKENIZER_TRAIN_DIR}")"

log "Installing/checking uv"
if ! need_cmd uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

log "Installing/checking Rust"
if ! need_cmd cargo; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  # shellcheck source=/dev/null
  source "${HOME}/.cargo/env"
fi

log "Cloning/checking official repos"
if [[ ! -d "${SUPERBPE_DIR}/.git" ]]; then
  git clone --recursive https://github.com/PythonNut/superbpe "${SUPERBPE_DIR}"
else
  (cd "${SUPERBPE_DIR}" && git submodule update --init --recursive)
fi

if [[ ! -d "${PARITY_DIR}/.git" ]]; then
  git clone https://github.com/swiss-ai/parity-aware-bpe "${PARITY_DIR}"
fi

log "Creating Python 3.12 SuperBPE environment"
uv python install 3.12
(cd "${SUPERBPE_DIR}" && uv venv --python 3.12 .venv)
# SuperBPE's requirements.txt has a RELATIVE editable path
# (`-e tokenizers_superbpe/bindings/python/`), so the install must run with the
# working directory set to SUPERBPE_DIR or uv resolves it against the wrong base.
(cd "${SUPERBPE_DIR}" && uv pip install --python "${PY}" -r requirements.txt)
uv pip install --python "${PY}" huggingface_hub
uv pip install --python "${PY}" -e "${PARITY_DIR}"

log "Downloading FLORES-200 if missing"
if [[ ! -d "${FLORES_DIR}" ]]; then
  curl -L -o data/external/flores200_dataset.tar.gz \
    https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
  tar -xzf data/external/flores200_dataset.tar.gz -C data/external
fi

log "Preparing FLORES eval/dev files"
"${PY}" scripts/prepare_flores.py \
  --source-dir data/external/flores200_dataset \
  --train-out data/processed/flores10_train.jsonl \
  --test-out data/processed/flores10_test.jsonl
"${PY}" scripts/export_language_texts.py \
  --data data/processed/flores10_train.jsonl \
  --out-dir data/processed/flores10_dev_by_lang
"${PY}" scripts/export_language_texts.py \
  --data data/processed/flores10_test.jsonl \
  --out-dir data/processed/flores10_devtest_by_lang

if [[ ! -d "${FINEWEB_DIR}" ]]; then
  log "FineWeb training corpus not found at ${FINEWEB_DIR}."
  log "Build it first (English-skewed, ~5 GB):"
  log "  ${PY} scripts/prepare_fineweb_language_shards.py --preset english_skew_5gb --out-dir data/processed/fineweb_by_lang"
  exit 2
fi
log "Using FineWeb tokenizer training corpus: ${TOKENIZER_TRAIN_DIR}"

log "Checking tokenizer training corpus size"
"${PY}" - <<PY
from pathlib import Path

train_dir = Path("${TOKENIZER_TRAIN_DIR_PATH}")
if not train_dir.is_dir():
    raise SystemExit(f"Missing tokenizer training directory: {train_dir}")
files = [
    path for path in train_dir.glob("*.txt")
    if "truncated" not in path.name and "split" not in path.name
]
size = sum(path.stat().st_size for path in files)
print(f"{len(files)} train files, {size / 1_000_000_000:.2f} GB")
if size < ${NUM_BYTES}:
    raise SystemExit(
        f"Training corpus has {size} bytes, but NUM_BYTES=${NUM_BYTES}. "
        "Do not let the official trainer loop over a smaller corpus for final results."
    )
PY

# Keep all four tokenizers on the IDENTICAL corpus.
# The parity-aware pretokenizer (scripts/pretokenize_for_parity.py) reads every
# per-language shard in full. The official trainer's --num_bytes does NOT: it
# picks whole shards at random until the byte budget is hit, truncating one shard
# to top up and looping/duplicating shards if the corpus is smaller than the
# budget (vendor/superbpe/utils.py:get_files_with_num_bytes, random.seed(0)).
# With 10 language shards that means some languages get dropped entirely (or
# duplicated) -- i.e. BPE/SuperBPE would train on a different, imbalanced corpus
# than PA. Omitting --num_bytes takes the "use all text data" branch so every
# method reads the full, balanced per-language corpus. NUM_BYTES still acts as
# the minimum-size guard checked above.
if [[ "${RUN_PARITY}" == "1" ]]; then
  BASELINE_NUM_BYTES=""
  BASELINE_CORPUS_DESC="full staged per-language corpus (no cap; matched to PA)"
else
  BASELINE_NUM_BYTES="${NUM_BYTES}"
  BASELINE_CORPUS_DESC="${NUM_BYTES} bytes"
fi

if [[ "${DO_BASELINE}" == "1" ]]; then
log "Training exact BPE ${VOCAB_SIZE} on ${BASELINE_CORPUS_DESC}"
SUPERBPE_MIN_FREQUENCY="${SUPERBPE_MIN_FREQUENCY}" \
SUPERBPE_PYTHON="${PY}" \
  scripts/train_bpe_official.sh \
  "${TOKENIZER_TRAIN_DIR}" \
  "${OUTPUT_ROOT}" \
  "${VOCAB_SIZE}" \
  "${BASELINE_NUM_BYTES}" \
  2>&1 | tee "${LOG_DIR}/bpe_${VOCAB_SIZE}_${NUM_BYTES}.log"

log "Training exact SuperBPE ${STAGE1_VOCAB_SIZE}->${VOCAB_SIZE} on ${BASELINE_CORPUS_DESC}"
log "Using RAYON_NUM_THREADS=${RAYON_NUM_THREADS}"
RAYON_NUM_THREADS="${RAYON_NUM_THREADS}" \
SUPERBPE_MIN_FREQUENCY="${SUPERBPE_MIN_FREQUENCY}" \
SUPERBPE_PYTHON="${PY}" \
  scripts/train_superbpe_official.sh \
  "${TOKENIZER_TRAIN_DIR}" \
  "${OUTPUT_ROOT}" \
  "${VOCAB_SIZE}" \
  "${STAGE1_VOCAB_SIZE}" \
  "${BASELINE_NUM_BYTES}" \
  2>&1 | tee "${LOG_DIR}/superbpe_${STAGE1_VOCAB_SIZE}_extend_${VOCAB_SIZE}_${NUM_BYTES}_rayon${RAYON_NUM_THREADS}.log"

log "Evaluating exact 40k baselines on FLORES devtest"
"${PY}" scripts/evaluate_hf_tokenizer.py \
  --tokenizer "${OUTPUT_ROOT}/bpe_${VOCAB_SIZE}" \
  --data data/processed/flores10_test.jsonl \
  --out-dir "${OUTPUT_ROOT}/eval_bpe_${VOCAB_SIZE}"

"${PY}" scripts/evaluate_hf_tokenizer.py \
  --tokenizer "${OUTPUT_ROOT}/superbpe_${STAGE1_VOCAB_SIZE}_extend_${VOCAB_SIZE}" \
  --data data/processed/flores10_test.jsonl \
  --out-dir "${OUTPUT_ROOT}/eval_superbpe_${VOCAB_SIZE}"
else
  log "PHASE=${PHASE}: skipping BPE/SuperBPE baselines this invocation."
fi

if [[ "${DO_PARITY}" == "1" ]]; then
  log "Training PA-BPE and PA-SuperBPE from per-language FineWeb shards + FLORES dev parity reference"
  FINAL_VOCAB_SIZE="${VOCAB_SIZE}" \
  STAGE1_VOCAB_SIZE="${STAGE1_VOCAB_SIZE}" \
  MIN_FREQUENCY="${SUPERBPE_MIN_FREQUENCY}" \
  SUPERBPE_PYTHON="${PY}" \
    scripts/train_parity_by_lang_40k.sh \
    "${PARITY_TRAIN_DIR}" \
    "data/processed/flores10_dev_by_lang" \
    "${OUTPUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/parity_by_lang_40k_${NUM_BYTES}.log"

  "${PY}" scripts/evaluate_hf_tokenizer.py \
    --tokenizer "${OUTPUT_ROOT}/parity_bpe_${VOCAB_SIZE}" \
    --data data/processed/flores10_test.jsonl \
    --out-dir "${OUTPUT_ROOT}/eval_parity_bpe_${VOCAB_SIZE}"

  "${PY}" scripts/evaluate_hf_tokenizer.py \
    --tokenizer "${OUTPUT_ROOT}/pa_superbpe_${STAGE1_VOCAB_SIZE}_extend_${VOCAB_SIZE}" \
    --data data/processed/flores10_test.jsonl \
    --out-dir "${OUTPUT_ROOT}/eval_pa_superbpe_${VOCAB_SIZE}"
else
  log "PHASE=${PHASE}: skipping parity-aware training this invocation."
fi

log "Checking tokenizer vocab sizes (for tokenizers produced so far)"
DO_BASELINE="${DO_BASELINE}" DO_PARITY="${DO_PARITY}" "${PY}" - <<PY
import os
from pathlib import Path
from tokenizers import Tokenizer

root = Path("${OUTPUT_ROOT}")
checks = {}
if os.environ["DO_BASELINE"] == "1":
    checks["bpe"] = root / "bpe_${VOCAB_SIZE}" / "tokenizer.json"
    checks["superbpe"] = root / "superbpe_${STAGE1_VOCAB_SIZE}_extend_${VOCAB_SIZE}" / "tokenizer.json"
if os.environ["DO_PARITY"] == "1":
    checks["parity_bpe"] = root / "parity_bpe_${VOCAB_SIZE}" / "tokenizer.json"
    checks["pa_superbpe"] = root / "pa_superbpe_${STAGE1_VOCAB_SIZE}_extend_${VOCAB_SIZE}" / "tokenizer.json"

for name, tokenizer_json in checks.items():
    if not tokenizer_json.exists():
        raise SystemExit(f"Missing tokenizer for {name}: {tokenizer_json}")
    size = Tokenizer.from_file(str(tokenizer_json)).get_vocab_size()
    print(f"{name}: vocab_size={size}")
    if size != ${VOCAB_SIZE}:
        raise SystemExit(f"{name} expected vocab size ${VOCAB_SIZE}, got {size}")
PY

log "Writing compact summary"
"${PY}" - <<PY
import csv, json
from pathlib import Path

root = Path("${OUTPUT_ROOT}")
rows = []
for path in sorted(root.glob("eval_*/summary.json")):
    payload = json.loads(path.read_text())
    rows.append({
        "run": path.parent.name,
        "tokenizer": payload["tokenizer"],
        "mean_tokens_per_char": payload["mean_tokens_per_char"],
        "gini": payload["gini"],
        "max_min_ratio": payload["max_min_ratio"],
    })

out = root / "summary_40k.csv"
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["run", "tokenizer", "mean_tokens_per_char", "gini", "max_min_ratio"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
for row in rows:
    print(row)
PY

log "Done. Outputs are in ${OUTPUT_ROOT}"
