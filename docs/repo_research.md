# Repo Research And Fast Path

This project was scoped from the two provided PDFs:

- `CSE_493S_Final_Project_Version_1_Report.pdf`
- `document_pdf.pdf`

## Useful Upstream Repositories

### SuperBPE

- Website: https://superbpe.github.io/
- GitHub: https://github.com/PythonNut/superbpe

Why it matters: this is the official SuperBPE implementation. It documents the two-stage idea: stage 1 trains normal whitespace-bounded BPE, and stage 2 trains merges after loosening the whitespace boundary.

Practical issue: the repository requires Python 3.12, Rust/Cargo, and a forked Hugging Face `tokenizers` dependency. That is worth using for a final large-scale replication, but it is a risky first step when the deadline is one week away.

### Parity-Aware BPE

- GitHub: https://github.com/swiss-ai/parity-aware-bpe
- Paper page: https://arxiv.org/abs/2508.04796

Why it matters: this is the source for the parity-aware merge rule described in the V1 report. The key idea is to identify the currently worst-compressed language and pick a merge that reduces that language's token count.

Practical issue: this code is useful for the full version, but the fast scaffold implements the same core idea directly so the project is runnable without repo-specific setup surprises.

### BoundlessBPE

- Paper page: https://arxiv.org/abs/2504.00178

Why it matters: the proposal mentions BoundlessBPE as a related superword tokenizer and possible comparison arm. Keep it in related work unless there is spare time.

## Fast Implementation Decision

For the due-in-a-week version, this scaffold prioritizes:

1. A runnable local experiment.
2. Clear metrics: tokens per character, Gini coefficient, max/min ratio.
3. Four comparable tokenizers: BPE, SuperBPE-style two-stage BPE, parity-aware BPE, and PA-SuperBPE.
4. A transparent implementation that is easy to explain in the paper.

It intentionally does not train a 150M parameter LM. The final report can frame that as future work, matching the V1 report's H3 plan.

## Full-Scale Upgrade Path

1. Replace `data/sample_multilingual.jsonl` with a FLORES-200 dev/devtest extract in the same JSONL format.
2. Increase `--stage1-merges` and `--stage2-merges` to match the report's 16k/24k or 80k/120k split.
3. If local pure-Python training becomes slow, switch training to the official SuperBPE/parity-aware repos and keep this repository's evaluation harness.
4. Keep the paper centered on H1/H2: compression disparity and parity-aware mitigation. Treat H3 LM pretraining as future work unless compute is already available.

