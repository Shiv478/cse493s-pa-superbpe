# `outputs/` — guide to everything in this folder

Three subfolders, in order of importance:

| Folder | What it is |
|---|---|
| [`final_results_40k/`](final_results_40k/) | **The actual results.** All four trained tokenizers (40k vocab), their FLORES-200 evaluations, the combined results tables, and training logs. Start at [`final_results_40k/RESULTS.md`](final_results_40k/RESULTS.md). |
| [`training_input_dicts_floored/`](training_input_dicts_floored/) | The frequency-floored pretoken dictionaries that the tokenizer learner reads as input. These reproduce every tokenizer **without** needing the 4.7 GB raw corpus. |
| [`prototype_demos/`](prototype_demos/) | Tiny sanity/demo runs used during development. **Not** the final results — kept only so the README walkthrough has reference outputs. |

The four tokenizers compared throughout are:

| name | merge selection | token type |
|---|---|---|
| `bpe` | global frequency | within-word |
| `superbpe` | global frequency | superword (crosses whitespace) |
| `pa_bpe` | parity-aware (helps worst language) | within-word |
| `pa_superbpe` | parity-aware | superword |

All four were trained on the **same** English-skewed FineWeb corpus, with the **same** learner,
at **40,000** vocab, and evaluated on the **same** FLORES-200 devtest set — so differences come
only from the method, not the setup.
