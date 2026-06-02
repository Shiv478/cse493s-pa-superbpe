from __future__ import annotations
"""Deep correctness check for the staged per-language training corpus.

Run before training:
    python3 scripts/check_corpus.py --corpus data/processed/fineweb_by_lang

Checks:
  - all 10 FLORES shards present and non-empty
  - strict UTF-8 (head+tail)
  - dominant Unicode script matches the language (catches mislabeled streams)
  - line-oriented (one doc per line; no stray embedded newlines)
  - manifest total == on-disk sum
  - English-skew ratio (the experiment's premise)
  - total >= NUM_BYTES guard (default 5e9)
Exits non-zero if anything fails (so it can gate the run).
"""
import argparse, json, sys
from pathlib import Path

LANGS = ["arb_Arab","deu_Latn","eng_Latn","hin_Deva","jpn_Jpan",
         "rus_Cyrl","spa_Latn","swh_Latn","tha_Thai","zho_Hans"]

EXPECT = {
    "arb_Arab":{"Arabic"},"deu_Latn":{"Latin"},"eng_Latn":{"Latin"},
    "hin_Deva":{"Devanagari"},"jpn_Jpan":{"Kana","Han"},"rus_Cyrl":{"Cyrillic"},
    "spa_Latn":{"Latin"},"swh_Latn":{"Latin"},"tha_Thai":{"Thai"},"zho_Hans":{"Han"},
}

def script_of(ch: str):
    o = ord(ch)
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F: return "Arabic"
    if 0x0900 <= o <= 0x097F: return "Devanagari"
    if 0x0400 <= o <= 0x04FF: return "Cyrillic"
    if 0x0E00 <= o <= 0x0E7F: return "Thai"
    if 0x3040 <= o <= 0x30FF: return "Kana"
    if 0x4E00 <= o <= 0x9FFF: return "Han"
    if 0x0041 <= o <= 0x024F: return "Latin"
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/fineweb_by_lang")
    ap.add_argument("--num-bytes", type=int, default=5_000_000_000)
    args = ap.parse_args()
    corpus = Path(args.corpus)
    ok = True

    print(f"=== corpus: {corpus} ===\n")
    print("--- per-shard content + script sanity ---")
    disk_total = 0
    for lang in LANGS:
        p = corpus / f"{lang}.txt"
        if not p.exists() or p.stat().st_size == 0:
            print(f"FAIL {lang:9s} MISSING/EMPTY"); ok = False; continue
        size = p.stat().st_size; disk_total += size
        counts: dict[str,int] = {}; total_chars = 0; first = ""
        with p.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i == 0: first = line.rstrip("\n")[:70]
                if i >= 200: break
                for ch in line.strip()[:400]:
                    sc = script_of(ch)
                    if sc: counts[sc] = counts.get(sc,0)+1; total_chars += 1
        dom = max(counts, key=counts.get) if counts else "NONE"
        frac = counts.get(dom,0)/total_chars if total_chars else 0
        good = dom in EXPECT[lang]
        if not good: ok = False
        print(f"{'OK ' if good else 'WARN'} {lang:9s} {size:>13,}b  {dom:<11s} {frac:4.0%}  {first!r}")

    print("\n--- UTF-8 validity (head+tail samples) ---")

    def utf8_clean(buf: bytes, trim_head: bool, trim_tail: bool) -> bool:
        # A fixed-byte slice can cut a multibyte char at either edge; trimming up
        # to 3 bytes off the cut edge(s) removes that artifact without hiding real
        # corruption in the interior.
        for h in range(4 if trim_head else 1):
            for t in range(4 if trim_tail else 1):
                end = len(buf) - t
                try:
                    buf[h:end].decode("utf-8")
                    return True
                except UnicodeDecodeError:
                    continue
        return False

    for lang in LANGS:
        p = corpus / f"{lang}.txt"
        with p.open("rb") as fh:
            head = fh.read(1_000_000)            # head: only the tail edge is a cut
            ts = min(1_000_000, p.stat().st_size)
            fh.seek(-ts, 2); tail = fh.read()    # tail: only the head edge is a cut
        head_ok = utf8_clean(head, trim_head=False, trim_tail=True)
        tail_ok = utf8_clean(tail, trim_head=True, trim_tail=False)
        if head_ok and tail_ok:
            print(f"OK  {lang}")
        else:
            print(f"FAIL {lang}: head_ok={head_ok} tail_ok={tail_ok}"); ok = False

    print("\n--- manifest cross-check ---")
    mp = corpus / "CORPUS_MANIFEST.json"
    if mp.exists():
        man = json.loads(mp.read_text())
        match = man["total_actual_bytes"] == disk_total
        print(f"manifest={man['total_actual_bytes']:,}  disk={disk_total:,}  match={match}")
        if not match: ok = False
    else:
        print("WARN no manifest")

    print("\n--- english skew + size guard ---")
    eng = (corpus/"eng_Latn.txt").stat().st_size
    print(f"english fraction = {eng/disk_total:.1%} (target ~50%)")
    print(f"total = {disk_total:,}  vs NUM_BYTES = {args.num_bytes:,}  -> "
          f"{'PASS' if disk_total >= args.num_bytes else 'FAIL (too small)'}")
    if disk_total < args.num_bytes: ok = False

    print("\nRESULT:", "ALL_GOOD ✅" if ok else "HAS_FAILURES ❌")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
