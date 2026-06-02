from __future__ import annotations

"""Fast, drop-in-compatible parity-aware BPE learner.

Goal: produce BYTE-IDENTICAL merges to the reference learner
(scripts/parity_aware_filtered_learn_bpe.py), but fast enough to train at scale.

Why a reimplementation: the reference selects each merge with
    most_frequent = max(stats, key=lambda x: (stats[x][max_index], x))
i.e. a full O(#unique-pairs) Python scan EVERY merge. On the real corpus that is
~tens of millions of pairs → minutes per merge → months total. This script keeps
the EXACT same algorithm and the EXACT same vendored helpers
(preprocess_input_data, replace_pair, replace_pair_dict, prune_stats), and the
identical filtered tie-break rule, but accelerates the per-merge selection with
per-language lazy-deletion max-heaps. Everything else — the data, the updates,
the thresholds, the stopping rule — is unchanged.

Defense in depth (so byte-identity does not depend on the heap being perfect):
  * The authoritative counts always live in `stats` (same dict the reference uses).
  * The heap is only an accelerator. Each popped candidate is re-validated against
    `stats`; stale entries are skipped (lazy deletion).
  * If a heap fails to yield a live candidate, we FALL BACK to the reference
    full-scan `select_best_pair` for that step — guaranteeing correctness.
  * FAST_PARITY_VERIFY=1 asserts, every merge, that the heap pick equals the
    reference full-scan pick. Used by the validation harness / unit tests.

The selection/tie-break and the SuperBPE stage-2 filter are imported directly from
the reference wrapper so they are literally the same code.

CLI is identical to parity_aware_filtered_learn_bpe.py.
"""

import argparse
import codecs
import copy
import heapq
import operator
import os
import sys
from pathlib import Path

import numpy
from tqdm import tqdm

try:
    from parity_aware_bpe import parity_aware_learn_bpe as pabpe
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import parity_aware_bpe. Install vendor/parity-aware-bpe into the active env first."
    ) from exc

# Reuse the EXACT filter + reference selection from the trusted wrapper, so our
# fallback and tie-break are byte-for-byte the same code path.
from parity_aware_filtered_learn_bpe import (  # noqa: E402
    is_valid_superbpe_stage2_pair,
    select_best_pair,
)

VERIFY = os.environ.get("FAST_PARITY_VERIFY", "0") == "1"


class _RevPair:
    """Wrapper giving REVERSED tuple ordering, so a min-heap on (-count, _RevPair)
    pops the largest (count, pair) first — matching the reference's
    max(stats, key=(count, pair)) with its lexicographic pair tie-break."""

    __slots__ = ("p",)

    def __init__(self, p):
        self.p = p

    def __lt__(self, other):  # reversed: "smaller" in heap == larger pair tuple
        return self.p > other.p

    def __eq__(self, other):
        return self.p == other.p


def _push(heaps, pair, counts):
    """Push fresh entries for `pair` into every language heap with its current count."""
    rp = _RevPair(pair)
    for lang in range(len(heaps)):
        c = int(counts[lang])
        heapq.heappush(heaps[lang], (-c, rp, pair, c))


def _build_heaps(stats, array_length):
    heaps = [[] for _ in range(array_length)]
    for pair, counts in stats.items():
        rp = _RevPair(pair)
        for lang in range(array_length):
            c = int(counts[lang])
            heaps[lang].append((-c, rp, pair, c))
    for lang in range(array_length):
        heapq.heapify(heaps[lang])
    return heaps


def _heap_best(heaps, lang, stats, pair_filter):
    """Pop the heap for `lang` until a LIVE, filter-passing entry is found.
    Returns the pair, or None if the heap is exhausted (caller falls back)."""
    h = heaps[lang]
    while h:
        negc, _rp, pair, cached = h[0]
        # liveness: pair still present and its current count matches the cached one
        cur = stats.get(pair)
        if cur is None:
            heapq.heappop(h)
            continue
        try:
            cur_c = int(cur[lang])
        except (TypeError, IndexError):
            heapq.heappop(h)
            continue
        if cur_c != cached:
            heapq.heappop(h)  # stale (count changed); a fresh entry exists elsewhere
            continue
        if pair_filter is not None and not pair_filter(pair):
            heapq.heappop(h)  # filter is deterministic on the pair -> drop permanently
            continue
        # live + valid; leave it on the heap (it remains a candidate next round)
        return pair
    return None


def _best_threshold_count(heaps, lang, stats, pair_filter):
    """Top live count in language `lang` (for threshold recompute). 0 if none."""
    pair = _heap_best(heaps, lang, stats, pair_filter)
    if pair is None:
        return 0
    return int(stats[pair][lang])


import re  # noqa: E402


def build_dev_index(dev_vocab):
    """Inverted index: pair (a,b) -> set of dev word-tuples that contain it adjacently.
    Lets us touch only affected words per merge instead of rescanning all dev words."""
    index = {}
    for word in dev_vocab:
        for a, b in zip(word, word[1:]):
            index.setdefault((a, b), set()).add(word)
    return index


def replace_pair_dict_indexed(pair, dev_vocab, dev_index, array_length):
    """Indexed equivalent of pabpe.replace_pair_dict: same per-language length_change,
    same in-place merge of dev_vocab, but only processes words known to contain `pair`.

    Reproduces the reference arithmetic exactly: for each affected word, build the
    space-joined string, regex-substitute the pair, split back to a tuple, and add
    (len(old)-len(new))*freq to length_change. Also maintains dev_index incrementally."""
    first, second = pair
    pair_str = "".join(pair).replace("\\", "\\\\")
    pattern = re.compile(r"(?<!\S)" + re.escape(first + " " + second) + r"(?!\S)")

    affected = dev_index.get(pair)
    if not affected:
        return 0  # reference returns int 0 when nothing changed

    # IMPORTANT: length_change must be the DEV width (len(freq)), not array_length.
    # When num_global>0, array_length = n_langs+1 (extra aggregate column in stats),
    # but dev_vocab values are only n_langs wide. The reference sizes length_change
    # from len(freq), so we match that exactly (lazily, like the reference).
    length_change = None
    # snapshot to a list: we mutate dev_vocab/dev_index during iteration
    for word in list(affected):
        # word may have been merged away already in this same call? No: each word is
        # unique and processed once; but guard in case of duplicates in the set.
        if word not in dev_vocab:
            continue
        # the reference guard: first in word and second in word and pair in adjacent zip
        if not (first in word and second in word and pair in zip(word, word[1:])):
            continue
        freq = dev_vocab[word]
        new_word = " ".join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(" "))
        # remove old word from index (for all its adjacent pairs) and dev_vocab
        for a, b in set(zip(word, word[1:])):
            s = dev_index.get((a, b))
            if s is not None:
                s.discard(word)
        del dev_vocab[word]
        # mirror the reference exactly: overwrite (collisions are impossible because
        # the merged token is created for the first time in this call).
        dev_vocab[new_word] = freq
        for a, b in set(zip(new_word, new_word[1:])):
            dev_index.setdefault((a, b), set()).add(new_word)
        if length_change is None:
            length_change = numpy.zeros(len(freq), dtype=int)
        length_change += (len(word) - len(new_word)) * freq

    if length_change is None:
        length_change = 0  # reference returns int 0 when nothing actually changed
    return length_change


def update_pair_statistics_tracked(pair, changed, stats, indices, touched):
    """Byte-for-byte copy of vendored update_pair_statistics, additionally
    recording every pair whose count it touches into `touched`. Keeping the
    arithmetic identical is what preserves equivalence; `touched` lets us push
    fresh heap entries for exactly those pairs."""
    stats[pair] = 0
    indices[pair] = pabpe.defaultdict(int)
    touched.add(pair)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in changed:
        i = 0
        while True:
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            if i < len(old_word) - 1 and old_word[i + 1] == second:
                if i:
                    prev = old_word[i - 1:i + 1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                    touched.add(prev)
                if i < len(old_word) - 2:
                    if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                        nex = old_word[i + 1:i + 3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                        touched.add(nex)
                i += 2
            else:
                i += 1
        i = 0
        while True:
            try:
                i = word.index(new_pair, i)
            except ValueError:
                break
            if i:
                prev = word[i - 1:i + 1]
                stats[prev] += freq
                indices[prev][j] += 1
                touched.add(prev)
            if i < len(word) - 1 and word[i + 1] != new_pair:
                nex = word[i:i + 2]
                stats[nex] += freq
                indices[nex][j] += 1
                touched.add(nex)
            i += 1


def learn_bpe_fast(
    infiles,
    outfile,
    devfiles,
    num_symbols,
    min_frequency=2,
    verbose=False,
    is_dict=False,
    num_global=0,
    num_workers=1,
    bpe_file=None,
    superbpe_stage2_filter=False,
    max_stage2_words=4,
):
    if not is_dict:
        raise ValueError("This fast learner is intended for --dict-input runs.")

    outfile.write("#version: 0.2\n")
    pair_filter = (
        (lambda pair: is_valid_superbpe_stage2_pair(pair, max_stage2_words))
        if superbpe_stage2_filter
        else None
    )

    dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = (
        pabpe.preprocess_input_data(
            infiles, devfiles, is_dict=is_dict, total_symbols=False,
            num_global=num_global, num_workers=num_workers, bpe_file=bpe_file,
        )
    )

    heaps = _build_heaps(stats, array_length)
    dev_index = build_dev_index(dev_vocab)

    def pick(max_index):
        """Heap fast-path with full-scan fallback (+ optional verify)."""
        cand = _heap_best(heaps, max_index, stats, pair_filter)
        if cand is None:
            cand = select_best_pair(stats, max_index, pair_filter)  # reference fallback
        if VERIFY:
            ref = select_best_pair(stats, max_index, pair_filter)
            assert cand == ref, (
                f"heap pick {cand!r} != reference {ref!r} at lang {max_index}"
            )
        return cand

    for i in tqdm(range(num_symbols), desc="fast parity-aware BPE..."):
        if stats:
            if i < num_global:
                max_index = -1
                if verbose:
                    sys.stderr.write(f"lengths {lengths}: picking best subword based on concatenation\n")
            else:
                max_index, _max_value = max(enumerate(lengths), key=operator.itemgetter(1))
                if verbose:
                    sys.stderr.write(f"lengths {lengths}: picking best subword in corpus {max_index}\n")
            most_frequent = pick(max_index)
        else:
            most_frequent = None

        if most_frequent is None or (i and stats[most_frequent][max_index] < threshold[max_index]):
            pabpe.prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            heaps = _build_heaps(stats, array_length)  # rebuild after resync
            most_frequent = pick(max_index)

            if most_frequent is None:
                sys.stderr.write("no valid pair remains after applying the SuperBPE stage-2 filter\n")
                break

            for lang_index in range(array_length):
                threshold[lang_index] = (
                    _best_threshold_count(heaps, lang_index, stats, pair_filter)
                    * i / (i + 10000.0)
                )

            pabpe.prune_stats(stats, big_stats, threshold)

        if stats[most_frequent][max_index] < min_frequency:
            sys.stderr.write(
                f"no valid pair has frequency >= {min_frequency}. "
                f"Stopping for language {max_index} with length: {lengths}\n"
            )
            break

        if verbose:
            sys.stderr.write(
                "pair {0}: {1} {2} -> {1}{2} (frequency {3})\n".format(
                    i, most_frequent[0], most_frequent[1], stats[most_frequent]
                )
            )

        outfile.write("{0} {1}\n".format(*most_frequent))
        changes = pabpe.replace_pair(most_frequent, sorted_vocab, indices)

        length_change = replace_pair_dict_indexed(most_frequent, dev_vocab, dev_index, array_length)
        lengths -= length_change

        touched = set()
        update_pair_statistics_tracked(most_frequent, changes, stats, indices, touched)
        # push fresh heap entries for exactly the pairs whose counts changed
        for p in touched:
            cur = stats.get(p)
            if cur is not None and not numpy.isscalar(cur):
                _push(heaps, p, cur)

        if not i % 100:
            pabpe.prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)


def open_many(paths):
    return [codecs.open(path, encoding="utf-8") for path in paths]


def main():
    parser = argparse.ArgumentParser(
        description="Fast parity-aware BPE learner (byte-identical to the reference)."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--dev", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--symbols", type=int, required=True)
    parser.add_argument("--global-merges", type=int, default=0)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--preload")
    parser.add_argument("--dict-input", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--superbpe-stage2-filter", action="store_true")
    parser.add_argument("--max-stage2-words", type=int, default=4)
    args = parser.parse_args()

    if len(args.input) != len(args.dev):
        raise SystemExit("--input and --dev must contain the same number of files.")

    infiles = open_many(args.input)
    devfiles = open_many(args.dev)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpe_file = codecs.open(args.preload, encoding="utf-8") if args.preload else None

    try:
        with codecs.open(output_path, "w", encoding="utf-8") as outfile:
            learn_bpe_fast(
                infiles=infiles,
                outfile=outfile,
                devfiles=devfiles,
                num_symbols=args.symbols,
                min_frequency=args.min_frequency,
                verbose=args.verbose,
                is_dict=args.dict_input,
                num_global=args.global_merges,
                num_workers=args.num_workers,
                bpe_file=bpe_file,
                superbpe_stage2_filter=args.superbpe_stage2_filter,
                max_stage2_words=args.max_stage2_words,
            )
    finally:
        for handle in infiles + devfiles:
            handle.close()
        if bpe_file is not None:
            bpe_file.close()


if __name__ == "__main__":
    main()
