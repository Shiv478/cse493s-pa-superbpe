from __future__ import annotations

"""Stage per-language training shards from FineWeb / FineWeb-2 for the 40k
tokenizer-parity experiment.

Why this exists
---------------
The parity-aware tokenizers (PA-BPE / PA-SuperBPE) require one training .txt per
language, and all four tokenizers (BPE / SuperBPE / PA-BPE / PA-SuperBPE) train
on the *same* shard folder. FineWeb-2 provides explicit, documented per-language
configs, so provenance is defensible. FineWeb-2 deliberately excludes English
("For English data, please refer to the original FineWeb"), so English is pulled
from HuggingFaceFW/fineweb instead. FineWeb-2 keys Chinese as `cmn_Hani`; we save
that stream to `zho_Hans.txt` so it lines up with the FLORES eval codes.

The corpus is intentionally English-skewed (default 50% English) to model
web-scale pretraining imbalance and make BPE/SuperBPE compression unfairness
visible. The skew is controlled and reported (see CORPUS_MANIFEST.{json,md}) so
the paper can state the exact composition and run a skew family (e.g. 50/30/15%).

This script only WRITES corpus .txt shards + a manifest. It does not train.

Example
-------
    python -m pip install datasets huggingface_hub pyarrow
    python scripts/prepare_fineweb_language_shards.py \
        --out-dir data/processed/fineweb_by_lang \
        --preset english_skew_5gb \
        --overwrite

Then, unchanged downstream:
    RUN_PARITY=1 PARITY_TRAIN_DIR=data/processed/fineweb_by_lang \
        bash lightning_run_40k_exact.sh
"""

import argparse
import json
import sys
import time
from pathlib import Path

# FLORES evaluation codes -> these are the shard filenames the runner expects.
FLORES_LANGS = [
    "arb_Arab",
    "deu_Latn",
    "eng_Latn",
    "hin_Deva",
    "jpn_Jpan",
    "rus_Cyrl",
    "spa_Latn",
    "swh_Latn",
    "tha_Thai",
    "zho_Hans",
]

FINEWEB2 = "HuggingFaceFW/fineweb-2"
FINEWEB = "HuggingFaceFW/fineweb"

# Languages whose FineWeb-2 config name differs from the FLORES shard filename.
# (Chinese: FineWeb-2 uses cmn_Hani; we write it to zho_Hans.txt for FLORES.)
NON_DEFAULT_FW2_CONFIG = {
    "zho_Hans": "cmn_Hani",
}


def build_source_map(english_config: str) -> dict[str, tuple[str, str]]:
    """flores_code -> (repo_id, config_name)."""
    sources: dict[str, tuple[str, str]] = {}
    for lang in FLORES_LANGS:
        if lang == "eng_Latn":
            sources[lang] = (FINEWEB, english_config)  # FineWeb-2 has no English
        else:
            config = NON_DEFAULT_FW2_CONFIG.get(lang, lang)
            sources[lang] = (FINEWEB2, config)
    return sources


# Explicit named presets (bytes per FLORES code). Kept explicit so the paper can
# cite exact targets. `english_skew_5gb` = 50% English; the other 9 split the
# remaining 2.5 GB evenly (~277.78 MB each); total target 5,000,000,002 bytes.
def _even_split_preset(total_bytes: int, eng_fraction: float) -> dict[str, int]:
    eng_bytes = int(round(total_bytes * eng_fraction))
    others = [lang for lang in FLORES_LANGS if lang != "eng_Latn"]
    per_other = (total_bytes - eng_bytes) // len(others)
    targets = {lang: per_other for lang in others}
    targets["eng_Latn"] = eng_bytes
    return targets


PRESETS: dict[str, dict[str, int]] = {
    "english_skew_5gb": {
        "eng_Latn": 2_500_000_000,
        **{lang: 277_777_778 for lang in FLORES_LANGS if lang != "eng_Latn"},
    },
    "balanced_5gb": {lang: 500_000_000 for lang in FLORES_LANGS},
}


def resolve_targets(args: argparse.Namespace) -> dict[str, int]:
    """Decide per-language byte targets from --preset or --total-bytes/--eng-fraction."""
    if args.preset:
        if args.preset not in PRESETS:
            raise SystemExit(
                f"Unknown --preset {args.preset!r}. Known: {', '.join(sorted(PRESETS))}"
            )
        targets = dict(PRESETS[args.preset])
    else:
        if not (0.0 < args.eng_fraction < 1.0):
            raise SystemExit("--eng-fraction must be strictly between 0 and 1.")
        targets = _even_split_preset(args.total_bytes, args.eng_fraction)

    # Per-language explicit overrides (bytes), if provided.
    for override in args.set or []:
        try:
            lang, value = override.split("=", 1)
            targets[lang] = int(value)
        except ValueError as exc:
            raise SystemExit(f"--set expects LANG=BYTES, got {override!r}") from exc

    missing = [lang for lang in FLORES_LANGS if lang not in targets]
    if missing:
        raise SystemExit(f"No byte target for: {', '.join(missing)}")
    return targets


def human_gb(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000_000:.3f} GB"


def stream_language_shard(
    repo_id: str,
    config: str,
    out_path: Path,
    target_bytes: int,
    seed: int,
) -> tuple[int, int, bool]:
    """Stream `config` from `repo_id` and write whole documents to `out_path`
    until at least `target_bytes` UTF-8 bytes are written.

    Returns (actual_bytes, num_docs, reached_target). Each document is written on
    one line (internal CR/LF collapsed to spaces) to keep the file line-oriented
    for the existing line-by-line pretokenize/train steps.
    """
    from datasets import load_dataset  # imported lazily so --dry-run needs no deps

    try:
        dataset = load_dataset(repo_id, name=config, split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001 - surface the real loader error clearly
        raise SystemExit(
            f"Failed to open {repo_id}:{config} (split=train, streaming). "
            f"Is the config name correct and the dataset public? Underlying error: {exc}"
        ) from exc

    try:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    except Exception:  # noqa: BLE001 - shuffle is best-effort for reproducibility
        pass

    actual_bytes = 0
    num_docs = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in dataset:
            text = example.get("text")
            if not text:
                continue
            line = text.replace("\r", " ").replace("\n", " ").strip()
            if not line:
                continue
            payload = line + "\n"
            handle.write(payload)
            actual_bytes += len(payload.encode("utf-8"))
            num_docs += 1
            if actual_bytes >= target_bytes:
                break

    reached = actual_bytes >= target_bytes
    return actual_bytes, num_docs, reached


def write_manifest(out_dir: Path, preset_name: str, records: list[dict]) -> None:
    total_target = sum(r["target_bytes"] for r in records)
    total_actual = sum(r["actual_bytes"] for r in records)
    manifest = {
        "preset": preset_name,
        "total_target_bytes": total_target,
        "total_actual_bytes": total_actual,
        "languages": records,
    }
    (out_dir / "CORPUS_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# FineWeb corpus manifest",
        "",
        f"- Preset: `{preset_name}`",
        f"- Total target: {human_gb(total_target)} ({total_target:,} bytes)",
        f"- Total actual: {human_gb(total_actual)} ({total_actual:,} bytes)",
        "",
        "| FLORES shard | source repo | config | target bytes | actual bytes | docs | reached |",
        "|---|---|---|---:|---:|---:|:--:|",
    ]
    for r in records:
        lines.append(
            f"| {r['flores_code']} | {r['repo_id']} | {r['config']} | "
            f"{r['target_bytes']:,} | {r['actual_bytes']:,} | {r['num_docs']} | "
            f"{'yes' if r['reached_target'] else 'NO'} |"
        )
    lines.append("")
    (out_dir / "CORPUS_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage English-skewed per-language FineWeb/FineWeb-2 shards."
    )
    parser.add_argument("--out-dir", required=True, help="Output dir for <flores>.txt shards.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Named byte-target preset. Overrides --total-bytes/--eng-fraction.",
    )
    parser.add_argument(
        "--total-bytes", type=int, default=5_000_000_000,
        help="Total corpus target when no --preset (default 5e9).",
    )
    parser.add_argument(
        "--eng-fraction", type=float, default=0.5,
        help="Fraction of total given to English when no --preset (default 0.5).",
    )
    parser.add_argument(
        "--set", action="append", metavar="LANG=BYTES",
        help="Per-language byte override, e.g. --set swh_Latn=150000000 (repeatable).",
    )
    parser.add_argument(
        "--english-config", default="sample-10BT",
        help="HuggingFaceFW/fineweb config for English (default sample-10BT).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed for reproducibility.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download shards even if a non-empty .txt already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the source map + byte plan and exit. No downloads, no deps needed.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sources = build_source_map(args.english_config)
    targets = resolve_targets(args)
    preset_name = args.preset or f"eng_fraction={args.eng_fraction},total={args.total_bytes}"

    print("Corpus plan:")
    print(f"  out-dir: {out_dir}")
    print(f"  preset:  {preset_name}")
    total = sum(targets[lang] for lang in FLORES_LANGS)
    for lang in FLORES_LANGS:
        repo_id, config = sources[lang]
        print(
            f"  {lang:9s} <- {repo_id}:{config:14s} target {human_gb(targets[lang])}"
        )
    print(f"  TOTAL target: {human_gb(total)} ({total:,} bytes)")
    if total < 5_000_000_000:
        print(
            "  WARNING: total target < 5 GB; the runner's size guard "
            "(NUM_BYTES default 5e9) will reject this corpus.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("Dry run: no data downloaded.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    shortfalls: list[str] = []

    for lang in FLORES_LANGS:
        repo_id, config = sources[lang]
        out_path = out_dir / f"{lang}.txt"
        target = targets[lang]

        if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
            actual = out_path.stat().st_size
            reached = actual >= target
            print(f"[skip] {lang}: exists ({human_gb(actual)}); use --overwrite to refetch")
            records.append({
                "flores_code": lang, "repo_id": repo_id, "config": config,
                "target_bytes": target, "actual_bytes": actual,
                "num_docs": None, "reached_target": reached,
            })
            if not reached:
                shortfalls.append(lang)
            continue

        print(f"[fetch] {lang}: {repo_id}:{config} -> {out_path} (target {human_gb(target)})")
        start = time.time()
        actual, num_docs, reached = stream_language_shard(
            repo_id=repo_id, config=config, out_path=out_path,
            target_bytes=target, seed=args.seed,
        )
        elapsed = time.time() - start
        status = "ok" if reached else "SHORT"
        print(
            f"        {status}: {human_gb(actual)} from {num_docs:,} docs in {elapsed:.0f}s"
        )
        records.append({
            "flores_code": lang, "repo_id": repo_id, "config": config,
            "target_bytes": target, "actual_bytes": actual,
            "num_docs": num_docs, "reached_target": reached,
        })
        if not reached:
            shortfalls.append(lang)

    write_manifest(out_dir, preset_name, records)
    total_actual = sum(r["actual_bytes"] for r in records)
    print(f"Wrote manifest. Total corpus: {human_gb(total_actual)} ({total_actual:,} bytes)")

    if shortfalls:
        raise SystemExit(
            "Shortfall: these languages did not reach their byte target: "
            f"{', '.join(shortfalls)}. The stream may be exhausted for that "
            "config. Lower its target (--set LANG=BYTES) or shift bytes to "
            "higher-resource languages, then re-run. Final corpus must total "
            ">= NUM_BYTES (5e9 default) or the runner's size guard will reject it."
        )


if __name__ == "__main__":
    main()
