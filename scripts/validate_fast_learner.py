from __future__ import annotations

"""Validation harness: prove the fast learner is byte-identical to the reference
across all THREE production configurations on real (small) corpus data.

This is the research-defensibility artifact. If all three diffs are empty, the
fast learner is a provably-equivalent reimplementation — same merges, hence the
same tokenizers and the same experimental conclusions.

Configs (mirroring scripts/train_parity_by_lang_40k.sh):
  1. PA-BPE           : --global-merges 0 --min-frequency 2 --symbols S_BPE
  2. PA-SuperBPE st1  : same flags, --symbols S_ST1
  3. PA-SuperBPE st2  : --preload <st1 merges> --superbpe-stage2-filter --max-stage2-words 4
                        --symbols S_ST2, using train_stage2 / dev_stage2 dicts

Usage:
  vendor/superbpe/.venv/bin/python scripts/validate_fast_learner.py \
      --fixture <dir with train_stage1/2 + dev_stage1/2> --symbols 1500
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "scripts" / "parity_aware_filtered_learn_bpe.py"
FAST = ROOT / "scripts" / "fast_parity_learn_bpe.py"
MERGE = ROOT / "scripts" / "merge_merges_files.py"
PY = sys.executable


def dicts(d: Path):
    return sorted(str(p) for p in d.glob("*.dict"))


def run_learner(script: Path, out: Path, train, dev, symbols, preload=None, filt=False):
    cmd = [PY, str(script), "--global-merges", "0", "--min-frequency", "2",
           "--dict-input", "--symbols", str(symbols),
           "--input", *train, "--dev", *dev, "--output", str(out)]
    if preload:
        cmd += ["--preload", str(preload)]
    if filt:
        cmd += ["--superbpe-stage2-filter", "--max-stage2-words", "4"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"learner failed: {script.name}")
    return out


def identical(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, help="dir with train_stage1/2 + dev_stage1/2")
    ap.add_argument("--symbols", type=int, default=1500, help="merges per config (small for speed)")
    args = ap.parse_args()

    fx = Path(args.fixture)
    tr1, dv1 = dicts(fx / "train_stage1"), dicts(fx / "dev_stage1")
    tr2, dv2 = dicts(fx / "train_stage2"), dicts(fx / "dev_stage2")
    S = args.symbols
    work = Path(tempfile.mkdtemp(prefix="validate_fast_"))
    results = {}

    # Config 1: PA-BPE
    ref1 = run_learner(REF, work / "ref_pabpe.txt", tr1, dv1, S)
    fst1 = run_learner(FAST, work / "fast_pabpe.txt", tr1, dv1, S)
    results["PA-BPE"] = identical(ref1, fst1)

    # Config 2: PA-SuperBPE stage1
    ref2 = run_learner(REF, work / "ref_st1.txt", tr1, dv1, S)
    fst2 = run_learner(FAST, work / "fast_st1.txt", tr1, dv1, S)
    results["PA-SuperBPE-stage1"] = identical(ref2, fst2)

    # Config 3: PA-SuperBPE stage2 (preload stage1, filter on, stage2 dicts)
    # build preload from the (reference) stage1 merges, matching the shell's --no-header
    preload = work / "preload.txt"
    subprocess.run([PY, str(MERGE), "--stage1", str(ref2), "--out", str(preload), "--no-header"],
                   capture_output=True, text=True, check=True)
    ref3 = run_learner(REF, work / "ref_st2.txt", tr2, dv2, S, preload=preload, filt=True)
    fst3 = run_learner(FAST, work / "fast_st2.txt", tr2, dv2, S, preload=preload, filt=True)
    results["PA-SuperBPE-stage2(filter)"] = identical(ref3, fst3)

    print("\n=== VALIDATION: fast learner vs reference (byte-identical merges) ===")
    all_ok = True
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        all_ok = all_ok and ok
    print(f"\nRESULT: {'ALL BYTE-IDENTICAL ✓' if all_ok else 'DIVERGENCE DETECTED ✗'}")
    print(f"(work dir: {work})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
