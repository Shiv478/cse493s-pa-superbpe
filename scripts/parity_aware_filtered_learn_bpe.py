from __future__ import annotations

import argparse
import codecs
import copy
import functools
import operator
import sys
from pathlib import Path

import numpy
from tqdm import tqdm

try:
    from parity_aware_bpe import parity_aware_learn_bpe as pabpe
except ImportError as exc:
    raise SystemExit(
        "Could not import parity_aware_bpe. Install vendor/parity-aware-bpe into the active environment first."
    ) from exc


def is_valid_superbpe_stage2_pair(pair: tuple[str, str], max_words: int = 4) -> bool:
    new_token = pair[0] + pair[1]
    if ":Ġ" in new_token:
        return False
    num_words = len([part for part in new_token.split("Ġ") if part])
    return num_words <= max_words


def select_best_pair(
    stats,
    max_index: int,
    pair_filter,
) -> tuple[str, str] | None:
    best_pair = None
    best_key = None
    for pair, counts in stats.items():
        if pair_filter is not None and not pair_filter(pair):
            continue
        key = (counts[max_index], pair)
        if best_key is None or key > best_key:
            best_key = key
            best_pair = pair
    return best_pair


def learn_bpe_filtered(
    infiles,
    outfile,
    devfiles,
    num_symbols: int,
    min_frequency: int = 2,
    verbose: bool = False,
    is_dict: bool = False,
    num_global: int = 0,
    ratio=None,
    num_workers: int = 1,
    bpe_file=None,
    superbpe_stage2_filter: bool = False,
    max_stage2_words: int = 4,
) -> None:
    if not is_dict:
        raise ValueError("This filtered wrapper is intended for --dict-input runs.")

    outfile.write("#version: 0.2\n")
    pair_filter = (
        (lambda pair: is_valid_superbpe_stage2_pair(pair, max_stage2_words))
        if superbpe_stage2_filter
        else None
    )

    dev_vocab, sorted_vocab, stats, indices, big_stats, threshold, lengths, array_length = (
        pabpe.preprocess_input_data(
            infiles,
            devfiles,
            is_dict=is_dict,
            total_symbols=False,
            num_global=num_global,
            num_workers=num_workers,
            bpe_file=bpe_file,
        )
    )

    if ratio is not None:
        initial_lengths = functools.reduce(numpy.add, [len(key) * value for key, value in sorted_vocab])
        lengths = numpy.copy(initial_lengths)

    for i in tqdm(range(num_symbols), desc="parity-aware BPE..."):
        if stats:
            if i < num_global:
                max_index = -1
                if verbose:
                    sys.stderr.write(f"lengths {lengths}: picking best subword based on concatenation\n")
            elif ratio is not None:
                compression_rates = initial_lengths / lengths
                adjusted_compression_rates = compression_rates / ratio
                max_index, _max_value = min(enumerate(adjusted_compression_rates), key=operator.itemgetter(1))
                if verbose:
                    sys.stderr.write(
                        f"compression rates {compression_rates}\nadjusted compression rates "
                        f"{adjusted_compression_rates}: picking best subword in corpus {max_index}\n"
                    )
            else:
                max_index, _max_value = max(enumerate(lengths), key=operator.itemgetter(1))
                if verbose:
                    sys.stderr.write(f"lengths {lengths}: picking best subword in corpus {max_index}\n")

            most_frequent = select_best_pair(stats, max_index, pair_filter)
        else:
            most_frequent = None

        if most_frequent is None or (i and stats[most_frequent][max_index] < threshold[max_index]):
            pabpe.prune_stats(stats, big_stats, threshold, full_sync=True)
            stats = copy.deepcopy(big_stats)
            most_frequent = select_best_pair(stats, max_index, pair_filter)

            if most_frequent is None:
                sys.stderr.write("no valid pair remains after applying the SuperBPE stage-2 filter\n")
                break

            for lang_index in range(array_length):
                best_for_lang = select_best_pair(stats, lang_index, pair_filter)
                if best_for_lang is None:
                    threshold[lang_index] = 0
                else:
                    threshold[lang_index] = stats[best_for_lang][lang_index] * i / (i + 10000.0)

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

        if ratio is not None:
            length_change = functools.reduce(numpy.add, [(len(c[2]) - len(c[1])) * c[3] for c in changes])
            lengths -= length_change
        else:
            length_change = pabpe.replace_pair_dict(most_frequent, dev_vocab)
            lengths -= length_change

        pabpe.update_pair_statistics(most_frequent, changes, stats, indices)

        if not i % 100:
            pabpe.prune_stats(stats, big_stats, threshold)

        stats[most_frequent] = numpy.zeros(array_length, dtype=int)


def open_many(paths: list[str]):
    return [codecs.open(path, encoding="utf-8") for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parity-aware BPE learner with optional SuperBPE stage-2 candidate filtering."
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
            learn_bpe_filtered(
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
