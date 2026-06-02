#!/usr/bin/env bash
# Live progress dashboard for lightning_run_40k_exact.sh.
#
# Usage:
#   bash scripts/monitor_run.sh                 # print once
#   bash scripts/monitor_run.sh -w              # refresh every 15s
#   LOG=logs_train_run.log bash scripts/monitor_run.sh -w
#
# It infers progress from the artifacts the runner writes, so it works even if
# you started the run in another terminal. 12 milestones, in order.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUT="${OUTPUT_ROOT:-outputs/run_40k}"
LOG="${LOG:-logs_train_run.log}"
V="${VOCAB_SIZE:-40000}"
S1="${STAGE1_VOCAB_SIZE:-16000}"

# milestone: "label|path-that-exists-when-done"
MILESTONES=(
  "BPE ${V} trained|${OUT}/bpe_${V}/tokenizer.json"
  "SuperBPE stage1 ${S1}|${OUT}/superbpe_stage1_${S1}/tokenizer.json"
  "SuperBPE ${S1}->${V} final|${OUT}/superbpe_${S1}_extend_${V}/tokenizer.json"
  "eval BPE|${OUT}/eval_bpe_${V}/summary.json"
  "eval SuperBPE|${OUT}/eval_superbpe_${V}/summary.json"
  "PA-BPE ${V} merges|${OUT}/parity_bpe_${V}_merges.txt"
  "PA-BPE ${V} tokenizer|${OUT}/parity_bpe_${V}/tokenizer.json"
  "PA-SuperBPE stage1 merges|${OUT}/pa_superbpe_stage1_${S1}_merges.txt"
  "PA-SuperBPE ${S1}->${V} tokenizer|${OUT}/pa_superbpe_${S1}_extend_${V}/tokenizer.json"
  "eval PA-BPE|${OUT}/eval_parity_bpe_${V}/summary.json"
  "eval PA-SuperBPE|${OUT}/eval_pa_superbpe_${V}/summary.json"
  "summary_40k.csv (DONE)|${OUT}/summary_40k.csv"
)

dashboard() {
  local done=0 total=${#MILESTONES[@]} current=""
  printf '╔══════════════════════════════════════════════════════════════╗\n'
  printf '║  40k tokenizer run — progress                                  ║\n'
  printf '╚══════════════════════════════════════════════════════════════╝\n'
  printf 'time: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"

  for m in "${MILESTONES[@]}"; do
    local label="${m%%|*}" path="${m##*|}"
    if [[ -s "${path}" ]]; then
      printf '  [x] %s\n' "${label}"; done=$((done+1))
    else
      if [[ -z "${current}" ]]; then
        printf '  [>] %s   <= in progress\n' "${label}"; current="${label}"
      else
        printf '  [ ] %s\n' "${label}"
      fi
    fi
  done

  local pct=$(( done * 100 / total ))
  local filled=$(( done * 40 / total )) i bar=""
  for ((i=0;i<40;i++)); do [[ $i -lt $filled ]] && bar+="#" || bar+="."; done
  printf '\nprogress: [%s] %d/%d (%d%%)\n' "${bar}" "${done}" "${total}" "${pct}"

  if [[ -d "${OUT}" ]]; then
    printf 'output size: %s\n' "$(du -sh "${OUT}" 2>/dev/null | cut -f1)"
  fi

  if [[ -f "${LOG}" ]]; then
    printf '\n--- last 3 log lines (%s) ---\n' "${LOG}"
    tail -n 3 "${LOG}" 2>/dev/null | sed 's/^/  /'
    # surface the live tqdm merge counter if a parity learn step is running
    local merge
    merge="$(grep -aoE 'parity-aware BPE\.\.\.: *[0-9]+%[^|]*' "${LOG}" 2>/dev/null | tail -1)"
    [[ -n "${merge}" ]] && printf '  >> %s\n' "${merge}"
  else
    printf '\n(no log at %s yet)\n' "${LOG}"
  fi

  if pgrep -f lightning_run_40k_exact.sh >/dev/null 2>&1; then
    printf '\nrunner: RUNNING\n'
  elif [[ ${done} -eq ${total} ]]; then
    printf '\nrunner: COMPLETE ✅  -> see %s/summary_40k.csv\n' "${OUT}"
  else
    printf '\nrunner: not running (stopped or not started)\n'
  fi
}

if [[ "${1:-}" == "-w" || "${1:-}" == "--watch" ]]; then
  while true; do clear; dashboard; sleep 15; done
else
  dashboard
fi
