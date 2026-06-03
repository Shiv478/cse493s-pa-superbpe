# One-Week Finish Plan

> Historical planning doc. For the final results and write-up, see `paper/findings.md` and `outputs/final_results_40k/`.

## Day 1

- Run `make setup`, `make test`, and `make smoke`.
- Open `outputs/prototype_demos/smoke_test/summary.md` and sanity-check that all four methods produce metrics.
- Read `paper/final_report.md` once and mark any claims that need your class-specific wording.

## Day 2

- Replace or expand `data/sample_multilingual.jsonl` with more data.
- Best low-cost target: 50-200 sentences per language across the same 10 languages.
- Rerun `make smoke` or run `scripts/run_experiment.py` with a new output folder.

## Day 3

- Tune merge counts. For local runs, try `--stage1-merges 400 --stage2-merges 600`.
- Save the best run under `outputs/final_candidate`.
- Inspect tokenizers with `scripts/inspect_tokenizer.py outputs/final_candidate/tokenizers/pa_superbpe.json`.

## Day 4

- Fill in the Results section of `paper/final_report.md`.
- Insert the table from `outputs/final_candidate/summary.md`.
- Reference `outputs/final_candidate/gini_bar.svg` as the main figure.

## Day 5

- Tighten the Methodology section.
- Be explicit that the scaffold is a smoke-scale implementation of the report algorithm.
- Add limitations: small corpus, pure-Python trainer, no LM pretraining.

## Day 6

- Run final tests and final experiment.
- Freeze the outputs you will cite.
- Proofread the paper for consistency with the code.

## Day 7

- Package the repo, outputs, and final paper.
- Practice a 2-minute explanation: problem, algorithm, metric, result, limitation.

