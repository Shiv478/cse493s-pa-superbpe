#!/usr/bin/env bash
# Stage-level ablation: which SuperBPE stage drives cross-lingual fairness?
#
# Fills the two missing cells of the stage 2x2 (selection rule per stage), reusing
# the EXISTING stage-1 bases so no stage-1 retraining is needed:
#
#   stage1 \ stage2 | global              | parity
#   ----------------+---------------------+--------------------
#   global          | SuperBPE (have)     | Hybrid B (build)
#   parity          | Hybrid A (build)    | PA-SuperBPE (have)
#
#   Hybrid A = parity within-word base  + GLOBAL superwords   (the "fair base,
#              then let superwords compress" idea)
#   Hybrid B = global within-word base  + PARITY  superwords
#
# Both stage-2 runs use the SuperBPE stage-2 filter and the 15,744 + 24,000 split
# (-> 40,000 vocab). Fast learner on the floored dicts; ~minutes, single core.
set -euo pipefail

ROOT="/teamspace/studios/this_studio/2026-05-28/files-mentioned-by-the-user-cse"
cd "$ROOT"
PY="$ROOT/vendor/superbpe/.venv/bin/python"
LEARNER="$ROOT/scripts/fast_parity_learn_bpe.py"
DICTS="$ROOT/outputs/training_input_dicts_floored"
SRC_MERGES="$ROOT/outputs/final_results_40k/merges"   # existing stage-1 bases
DATA="$ROOT/data/processed/flores10_test.jsonl"
OUT="$ROOT/outputs/stage_hybrid_ablation_40k"
mkdir -p "$OUT"/{tokenizers,evaluation,merges,logs}

TR2=( "$DICTS"/train_stage2/*.dict ); DV2=( "$DICTS"/dev_stage2/*.dict )
STAGE2_SYMBOLS=24000

# run_hybrid <name> <stage1_preload> <stage1_merges> <stage2_global_merges>
#   stage2_global_merges = 24000 -> all-global stage 2 ; 0 -> all-parity stage 2
run_hybrid() {
  local name="$1" preload="$2" stage1_merges="$3" global2="$4"
  local st2="$OUT/merges/${name}_stage2_merges.txt"
  local comb="$OUT/merges/${name}_merges.txt"
  local tok="$OUT/tokenizers/${name}"
  local ev="$OUT/evaluation/${name}"

  echo "[$name] stage-2 train (global_merges=${global2}/${STAGE2_SYMBOLS}) $(date)"
  "$PY" "$LEARNER" --dict-input --min-frequency 2 \
    --global-merges "$global2" --symbols "$STAGE2_SYMBOLS" \
    --preload "$preload" --superbpe-stage2-filter --max-stage2-words 4 \
    --input "${TR2[@]}" --dev "${DV2[@]}" \
    --output "$st2" > "$OUT/logs/${name}_stage2.log" 2>&1

  echo "[$name] combine stage1+stage2"
  "$PY" scripts/merge_merges_files.py \
    --stage1 "$stage1_merges" --stage2 "$st2" --out "$comb" > "$OUT/logs/${name}_combine.log" 2>&1

  echo "[$name] build tokenizer (assert vocab=40000)"
  "$PY" scripts/build_tokenizer_from_merges.py \
    --merges "$comb" --out-dir "$tok" --kind pa_superbpe \
    --expected-vocab-size 40000 > "$OUT/logs/${name}_build.log" 2>&1

  echo "[$name] evaluate on FLORES devtest"
  "$PY" scripts/evaluate_hf_tokenizer.py \
    --tokenizer "$tok" --data "$DATA" --out-dir "$ev" > "$OUT/logs/${name}_eval.log" 2>&1
  echo "[$name] DONE $(date)"
}

echo "==== stage-hybrid ablation start $(date) ===="
# Hybrid A (your idea): parity stage-1 base + GLOBAL stage-2 superwords
run_hybrid "hybrid_pa1_global2" \
  "$SRC_MERGES/pa_superbpe_stage1_preload.txt" \
  "$SRC_MERGES/pa_superbpe_stage1_merges.txt" 24000
# Hybrid B (complement): global stage-1 base + PARITY stage-2 superwords
run_hybrid "hybrid_global1_pa2" \
  "$SRC_MERGES/superbpe_stage1_preload.txt" \
  "$SRC_MERGES/superbpe_stage1_merges.txt" 0

echo "==== building 6-way comparison table $(date) ===="
"$PY" - <<'PY'
import csv
from pathlib import Path
ROOT = Path("/teamspace/studios/this_studio/2026-05-28/files-mentioned-by-the-user-cse")
FR = ROOT/"outputs/final_results_40k/evaluation"
AB = ROOT/"outputs/stage_hybrid_ablation_40k/evaluation"
# (label, eval_dir, stage1_rule, stage2_rule)
METHODS = [
    ("BPE",                         FR/"bpe",                 "-",      "-"),
    ("SuperBPE",                    FR/"superbpe",            "global", "global"),
    ("Hybrid B (global1->parity2)", AB/"hybrid_global1_pa2",  "global", "parity"),
    ("Hybrid A (parity1->global2)", AB/"hybrid_pa1_global2",  "parity", "global"),
    ("PA-BPE",                      FR/"pa_bpe",              "parity", "-"),
    ("PA-SuperBPE",                 FR/"pa_superbpe",         "parity", "parity"),
]
def gini(v):
    xs=sorted(x for x in v if x>=0); n=len(xs); s=sum(xs)
    return sum((2*(i+1)-n-1)*x for i,x in enumerate(xs))/(n*s) if n and s else 0.0
def mm(v):
    p=[x for x in v if x>0]; return max(p)/min(p) if p else 0.0
def load(d):
    rows=list(csv.DictReader((d/"compression_by_language.csv").open()))
    tpc={r["language"]:float(r["tokens_per_char"]) for r in rows}
    tok={r["language"]:int(r["total_tokens"]) for r in rows}
    return tpc, tok
out = ROOT/"outputs/stage_hybrid_ablation_40k/stage_ablation_summary.csv"
hdr=["method","stage1","stage2","mean_tpc","gini_tpc","maxmin_tpc",
     "gini_token_count","maxmin_token_count","eng_tokens","eng_tokens_per_char"]
with out.open("w",newline="") as fh:
    w=csv.writer(fh); w.writerow(hdr)
    print(f"{'method':30} {'s1':7} {'s2':7} {'mTPC':>6} {'Gtpc':>6} {'MMtpc':>6} "
          f"{'Gtok':>6} {'MMtok':>6} {'engTok':>7} {'engTPC':>7}")
    for label,d,s1,s2 in METHODS:
        tpc,tok=load(d)
        L=sorted(tpc)
        tpcv=[tpc[l] for l in L]; tokv=[tok[l] for l in L]
        row=[label,s1,s2,round(sum(tpcv)/len(tpcv),4),round(gini(tpcv),4),round(mm(tpcv),3),
             round(gini(tokv),4),round(mm(tokv),3),tok["eng_Latn"],round(tpc["eng_Latn"],4)]
        w.writerow(row)
        print(f"{label:30} {s1:7} {s2:7} {row[3]:6.4f} {row[4]:6.4f} {row[5]:6.3f} "
              f"{row[6]:6.4f} {row[7]:6.3f} {row[8]:7d} {row[9]:7.4f}")
print("\nwrote", out)
PY
echo "==== done $(date) ===="
