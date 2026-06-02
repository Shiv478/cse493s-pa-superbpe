# `prototype_demos/` — sanity checks only (NOT final results)

These are tiny development runs used to verify the pipeline end-to-end. They train on a small
amount of data with a small merge budget, so their numbers are **not** the project's findings.
**For the real results see [`../final_results_40k/RESULTS.md`](../final_results_40k/RESULTS.md).**

| Subfolder | What it is |
|---|---|
| `flores10_demo/` | prototype run trained on FLORES-200 `dev` (not the FineWeb corpus), 80 stage-1 + 40 stage-2 merges. Used to check the metrics/plot pipeline. |
| `smoke_test/` | the smallest possible run (`make smoke`) to confirm the code executes. |

Each contains `summary.md/.csv`, `compression_by_language.csv`, a `gini_bar.svg` plot, and the
small `tokenizers/`. Reproduce with `make train-test` (flores10_demo) and `make smoke` (smoke_test).
