from __future__ import annotations

"""Equivalence tests: the fast parity learner must produce BYTE-IDENTICAL merges
to the reference (scripts/parity_aware_filtered_learn_bpe.py) on every input we
can check. This is the research-defensibility core in unit-test form.

Run with the SuperBPE venv python (needs parity_aware_bpe + tokenizers):
    vendor/superbpe/.venv/bin/python -m pytest tests/test_fast_learner.py -q
or:
    vendor/superbpe/.venv/bin/python tests/test_fast_learner.py
"""

import io
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Force verify mode OFF here (we compare outputs directly); harness can set it on.
os.environ.setdefault("FAST_PARITY_VERIFY", "0")

import parity_aware_filtered_learn_bpe as ref  # noqa: E402
import fast_parity_learn_bpe as fast  # noqa: E402


def _write_dicts(tmp: Path, name: str, per_lang: dict[str, list[tuple[str, int]]]) -> list[str]:
    """Write <lang>.dict files; return sorted list of paths (stable lang order)."""
    paths = []
    for lang in sorted(per_lang):
        p = tmp / f"{name}_{lang}.dict"
        with p.open("w", encoding="utf-8") as fh:
            for tok, cnt in per_lang[lang]:
                fh.write(f"{tok} {cnt}\n")
        paths.append(str(p))
    return paths


def _run(learner_module, train_paths, dev_paths, symbols, **kw):
    """Run a learner's learn_bpe_* into a string buffer; return the merges text."""
    infiles = [open(p, encoding="utf-8") for p in train_paths]
    devfiles = [open(p, encoding="utf-8") for p in dev_paths]
    buf = io.StringIO()
    try:
        if learner_module is ref:
            ref.learn_bpe_filtered(
                infiles=infiles, outfile=buf, devfiles=devfiles,
                num_symbols=symbols, is_dict=True, **kw,
            )
        else:
            fast.learn_bpe_fast(
                infiles=infiles, outfile=buf, devfiles=devfiles,
                num_symbols=symbols, is_dict=True, **kw,
            )
    finally:
        for h in infiles + devfiles:
            h.close()
    return buf.getvalue()


class FastLearnerEquivalence(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(os.environ.get("PYTEST_TMP", "/tmp")) / f"fastlearn_{os.getpid()}"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def _assert_identical(self, train, dev, symbols, **kw):
        tp = _write_dicts(self.tmp, "train", train)
        dp = _write_dicts(self.tmp, "dev", dev)
        mf = kw.pop("min_frequency", 1)
        ref_out = _run(ref, tp, dp, symbols, min_frequency=mf, **kw)
        fast_out = _run(fast, tp, dp, symbols, min_frequency=mf, **kw)
        self.assertEqual(ref_out, fast_out,
                         f"\nREF:\n{ref_out}\nFAST:\n{fast_out}")
        return ref_out

    # NOTE on format: a real .dict line is `<pretoken> <count>` where <pretoken>
    # is ONE atomic string with NO internal spaces (e.g. "Ġthe"). BPE starts from
    # that pretoken's characters. So fixtures use single-token strings; the Ġ
    # word-boundary marker is a literal character inside the token.
    G = "Ġ"  # Ġ

    # T1 — basic multi-language equivalence
    def test_t1_basic_equivalence(self):
        G = self.G
        train = {
            "eng": [(f"{G}the", 50), (f"{G}quick", 12), (f"{G}fox", 9), (f"{G}then", 7)],
            "spa": [(f"{G}el", 40), (f"{G}rapido", 11), (f"{G}zorro", 8)],
            "zho": [("中文", 30), ("分词", 14), ("语言", 6)],
        }
        dev = {
            "eng": [(f"{G}the", 5), (f"{G}fox", 2)],
            "spa": [(f"{G}el", 4), (f"{G}zorro", 2)],
            "zho": [("中文", 3), ("语言", 2)],
        }
        self._assert_identical(train, dev, symbols=12)

    # T2 — SuperBPE stage-2 filter equivalence
    def test_t2_filter_equivalence(self):
        G = self.G
        # multi-word pretokens (Ġ-joined) so the >max_words and ":Ġ" rules bite
        train = {
            "eng": [(f"{G}a{G}b", 30), (f"{G}a{G}b{G}c{G}d{G}e", 20), (f"x:{G}y", 15), (f"{G}m{G}n", 9)],
            "zho": [(f"{G}p{G}q{G}r{G}s{G}t", 18), (f"{G}u{G}v", 12), (f"{G}w{G}x", 5)],
        }
        dev = {
            "eng": [(f"{G}a{G}b", 3), (f"x:{G}y", 2)],
            "zho": [(f"{G}u{G}v", 3), (f"{G}w{G}x", 2)],
        }
        self._assert_identical(train, dev, symbols=10,
                               superbpe_stage2_filter=True, max_stage2_words=4)

    # T3 — deliberate count ties across pairs (tie-break fidelity)
    def test_t3_tiebreak(self):
        G = self.G
        train = {
            "eng": [(f"{G}ab", 10), (f"{G}cd", 10), (f"{G}ef", 10), (f"{G}gh", 10), (f"{G}zz", 10)],
            "zho": [("莱", 10), ("筛", 10), ("译", 10), (f"{G}ab", 10)],
        }
        dev = {
            "eng": [(f"{G}ab", 2), (f"{G}zz", 2)],
            "zho": [("莱", 2), (f"{G}ab", 2)],
        }
        self._assert_identical(train, dev, symbols=8)

    # T4 — force resync (tiny counts + many symbols => pruning/resync paths)
    def test_t4_resync(self):
        G = self.G
        train = {
            "eng": [(f"{G}the", 6), (f"{G}and", 5), (f"{G}of", 4), (f"{G}to", 3), (f"{G}in", 2)],
            "spa": [(f"{G}de", 6), (f"{G}la", 5), (f"{G}en", 4), (f"{G}y", 3), (f"{G}que", 2)],
            "zho": [("的", 6), ("是", 5), ("在", 4), ("了", 3), ("和", 2)],
        }
        dev = {
            "eng": [(f"{G}the", 2), (f"{G}of", 1)],
            "spa": [(f"{G}de", 2), (f"{G}en", 1)],
            "zho": [("的", 2), ("在", 1)],
        }
        self._assert_identical(train, dev, symbols=20)

    # T5 — preload continuation (SuperBPE stage-2 style)
    def test_t5_preload(self):
        G = self.G
        train = {
            "eng": [(f"{G}the", 20), (f"{G}them", 8), (f"{G}there", 6)],
            "zho": [("中文", 15), ("中国", 7)],
        }
        dev = {"eng": [(f"{G}the", 2)], "zho": [("中文", 2)]}
        tp = _write_dicts(self.tmp, "ptrain", train)
        dp = _write_dicts(self.tmp, "pdev", dev)
        preload = self.tmp / "preload.txt"
        # preload lines are `<left> <right>` merges (no header, matches --no-header)
        preload.write_text(f"{G}t h\n{G}th e\n", encoding="utf-8")

        def run(mod):
            infiles = [open(p, encoding="utf-8") for p in tp]
            devfiles = [open(p, encoding="utf-8") for p in dp]
            bpe = open(preload, encoding="utf-8")
            buf = io.StringIO()
            try:
                if mod is ref:
                    ref.learn_bpe_filtered(infiles=infiles, outfile=buf, devfiles=devfiles,
                                           num_symbols=8, is_dict=True, min_frequency=1, bpe_file=bpe)
                else:
                    fast.learn_bpe_fast(infiles=infiles, outfile=buf, devfiles=devfiles,
                                        num_symbols=8, is_dict=True, min_frequency=1, bpe_file=bpe)
            finally:
                for h in infiles + devfiles:
                    h.close()
                bpe.close()
            return buf.getvalue()

        self.assertEqual(run(ref), run(fast))


if __name__ == "__main__":
    unittest.main(verbosity=2)
