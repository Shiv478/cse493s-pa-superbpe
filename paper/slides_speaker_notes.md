# PA-SuperBPE speaker notes


## Parity-Aware SuperBPE

Hi, we're Chendur, Shiv, and Dhruti. Our project, Parity-Aware SuperBPE, is about making tokenization fairer across languages without sacrificing how efficiently it compresses text. I'll cover the problem, how the tokenizers work, the system we built (including a 600x-faster learner we had to write to make this feasible), and then the results and findings.


## The problem: tokenization is unfair across languages

Language models don't see words, they see tokens, and everything is priced per token: the API bill, compute and latency, and how much fits in the context window. Tokenizers are trained on mostly-English text, so they compress English well and chop other languages into far more tokens. In our data, standard BPE makes Thai use 2.27 times as many tokens as English for the same sentence, so a Thai user literally pays more, waits longer, and gets less usable context.


## Background: two tools that pull in opposite directions

Two recent tools are relevant but pull in opposite directions. SuperBPE adds 'superword' tokens that merge across spaces, great for compression, but it picks merges by global frequency, so on English-heavy data those superwords mostly go to English. Parity-aware BPE does the opposite: it picks each merge to help whichever language is worst-compressed, fair, but it only works within words and gives up the dominant language's compression. So you seem stuck between efficient-but-unfair and fair-but-inefficient.


## How the two ideas work

Two pictures of the mechanisms. Top is token type: plain BPE never crosses a space, so 'of the people' is three tokens; SuperBPE's second stage merges across the space, so 'of the' becomes one token, but on English-heavy data that mostly helps English. Bottom is selection: under global BPE, Chinese is starved of merges because English pairs are more frequent, so the four characters stay split; parity instead spends merges on the worst-off language, so Chinese gets proper word units. Our project is about combining these two ideas well.


## Our research question

Our question is whether that trade-off is fundamental, or just because parity gets applied to everything at once. The key observation: SuperBPE already trains in two stages, a within-word stage then a cross-space superword stage. So we asked: what if we make only the first stage fair, and let the second stage do global compression? That one idea is what the whole experiment tests.


## The six tokenizers we built

We built six tokenizers. The main four are a two-by-two: standard versus parity selection, crossed with within-word versus superword tokens. Then two hybrids that split the rule by stage: Hybrid A is a fair base with global superwords, Hybrid B is the reverse. These six let us isolate exactly where fairness should live.


## The system: end-to-end pipeline

This is the whole system, the work behind the results. We download an English-skewed FineWeb corpus across ten languages; pre-tokenize it with SuperBPE's exact regex into per-language frequency dictionaries; floor those dictionaries to make training tractable; learn the merges with our fast parity learner; assemble a standard Hugging Face tokenizer; and evaluate on FLORES. Every tokenizer goes through the identical pipeline, so the only thing that varies is the method.


## Setup: a controlled comparison

Everything is held constant so the only variable is the method: the same training engine, the same English-skewed FineWeb corpus, the same 40,000 vocabulary, and the same FLORES evaluation across ten languages.


## Corpus flooring: drop pretokens seen only once (count >= 2)

Before training we floor the dictionaries: we drop any pretoken that appears only once. The type counts collapse, 78 percent of unique pretokens in stage 1 and 96 percent in stage 2, because language has a huge long tail of things seen once. But in terms of actual token mass we only lose 2.4 percent in stage 1. Stage 2, the superwords, is sparser, so we lose 31 percent there. The justification: anything seen once can't, by itself, form a merge, since a merge needs frequency at least two; and crucially all six tokenizers use the same floored data, so even if flooring shifts absolute numbers, the comparison between methods stays fair.


## Why we built a fast learner (the reference was infeasible)

Here's a big piece of the work. Parity re-ranks pairs by the worst language on every single merge, and the reference implementation is pure Python doing two full passes per merge. We measured it at about 370 seconds per merge, which for forty thousand merges is roughly five and a half months, completely infeasible. And about 72 percent of that time is rescanning the FLORES dev ruler, a cost that doesn't shrink if you use less data, so we couldn't just subsample our way out. So we reimplemented the hot loop with better data structures, and proved it produces byte-identical output. That turned months into hours and is what made this whole project possible.


## Fast learner architecture: same algorithm, better data structures

Here's how it works. The reference does two expensive things per merge: to pick the best pair it scans every unique adjacent pair, tens of millions of them; and to update the per-language ruler it rescans the entire FLORES dev set. We replaced the first with per-language max-heaps, so picking a merge is just a peek, and the second with an inverted index from a pair to the dev words containing it, so we only touch the few words a merge actually affects. The heaps use lazy deletion and we validate the top against the real counts before using it, so the tie-breaking and every decision are identical. Net result: about 600 times faster, byte-for-byte the same output.


## How we know it is correct (validation)

Results are only as good as the setup, so we validated heavily. The fast learner is byte-identical to the reference: five unit tests, a 300-merge step-by-step check where the fast pick equals the reference pick on every merge, and a three-configuration harness on real data. In standard mode it's also byte-identical to a completely independent from-scratch BPE we wrote. We separately verified the superword stage actually fires, which I'll show. And since all six share the engine, corpus, and evaluation, any difference is the method itself.


## What the evaluation metrics mean

One slide on the metrics. We tokenize the same FLORES sentences in all ten languages and count tokens per language. Tokens-per-character is raw compression, but it's distorted across scripts, because a Chinese character holds more meaning. Token-count is the fair one: because FLORES is parallel, it's the tokens to encode identical content, exactly what you pay. For example under BPE English is about 29,000 tokens and Thai about 66,000. We summarize the spread two ways, both lower-is-fairer: the max-over-min ratio where 1.0 is perfectly equal, and the Gini coefficient, like income inequality, where 0 is everyone equal.


## Headline: cross-lingual cost ratio (worst / best language, 1.0 = fair)

The headline. This is the worst-to-best ratio on token count; 1.0 is perfectly fair. BPE is 2.27, Thai pays more than double English. SuperBPE is 1.9. Both parity methods, in purple, collapse it to about 1.02, essentially equal cost for equal content. The hybrids, in gold, are in between. Axis starts at 1.0 because that's the natural floor.


## Full results: six tokenizers at 40k vocab

The full numbers; two columns matter most. Token-count fairness drops from 0.11 for BPE to about 0.004 for parity, roughly 25 times fairer. The far-right English column previews the catch: parity pushes English from 0.22 to 0.31 or 0.33. The highlighted row, Hybrid A, is strong on fairness while keeping English low.


## Finding 1: parity nearly eliminates the disparity, at a cost

Finding one: parity nearly eliminates the disparity, the ratio goes from 2.27 to about 1.02 and the Gini drops about 25-fold, every language within roughly 2 percent of every other for the same content. But it isn't free: parity spends merges away from English, which climbs from about 29,000 to 42,000 tokens. So the question becomes: can we keep the fairness without paying that price?


## The trade-off: English compression cost (tokens per character)

The trade-off as a picture: English's tokens-per-character, lower is better. The two baselines are low, full parity in purple is high, that's the cost, and the hybrids in gold sit much lower, close to the baselines. They recover most of English's compression. Axis is zoomed because the values are close.


## Finding 2: the trade-off is controllable, by WHERE parity goes

The key result: the trade-off is controllable by where you put parity. Hybrid A, fairness in the within-word base with global superwords on top, gives the best per-character fairness of all six while keeping English fast. More generally: stage-one parity buys per-character fairness, stage-two parity buys token-count fairness, and full parity maxes fairness at the dominant language's expense. So it's a design dial you set, not a law of nature.


## Per-character fairness (Gini): Hybrid A is best of all six

Same point visually: per-character fairness, lower is fairer, axis zoomed because the values are close. Hybrid A, the gold bar, is the lowest of all six, including the full-parity methods. So on the per-character metric the hybrid is our single fairest, best-balanced tokenizer.


## Superword mechanism: how many spaces each tokenizer merges

The space-merging result as a chart. It counts vocabulary tokens that cross a word boundary. BPE and PA-BPE are essentially zero, by design they never cross a space. The four superword methods each create between about 4,500 and 5,800 cross-space tokens. Notice Hybrid B has the fewest, because its parity stage spent budget on within-run merges for spaceless scripts like Chinese, which is exactly why it equalized those languages. So this also confirms the superword stage really fired.


## Space-merging, in detail

And the detailed table you asked for. Cross-space is how many vocabulary tokens span a space; percent of vocab; total spaces merged; and the breakdown of how many spaces each token merges, one through four. The within-word tokenizers are basically zero. The superword methods are in the thousands, and most are simple two-word tokens, the '1 space' column. The stage-2 filter caps it at four words, which you can see holds.


## What we asked, and what we found

Reframing around what we actually found. Q1: how unfair are standard tokenizers on realistic data? Substantially, about a 2.3-times gap. Q2: can parity fix it? Yes, near-perfect parity, about 25 times fairer. And Q3, the interesting one: does fairness have to cost the dominant language? No, putting parity only in the within-word stage keeps English fast and is the fairest per character. Honestly, our original proposal guessed SuperBPE would make things less fair; it didn't, and that pushed us to the sharper question of where fairness belongs.


## Why this matters

Why this matters: a simple, actionable design rule, put fairness in the within-word stage, that resolves a trade-off people treat as inherent. We show near-perfect cross-lingual parity is achievable, about 25 times fairer, at a cost you can dial. And it's fully reproducible: one validated, 600-times-faster engine, byte-identical to ground truth, with all the code, results, and slides public.


## Conclusion, limitations & next steps

To wrap up: parity-aware selection makes multilingual tokenization much fairer, and putting that fairness in the within-word stage keeps the dominant language efficient, captured by Hybrid A. Along the way we built the engine that made it feasible, a 600-times-faster learner proven byte-identical to the reference. Limitations are honest next steps: one vocab size, one corpus mix, the flooring approximation, and no language-model training yet. Everything is in the repo. Thank you, we're happy to take questions.
