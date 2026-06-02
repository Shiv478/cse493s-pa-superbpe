# Tokenization Explained

Language models do not read text directly as words or sentences. They read lists of token IDs.

The tokenizer is the tool that turns text into those pieces:

```text
"new york is big"
-> ["new", " york", " is", " big"]
-> [713, 4356, 318, 1263]
```

The model only sees the numbers. So the tokenizer decides how expensive a sentence is. If one language needs many more tokens than another language, that language gets less effective context and costs more to process.

## BPE

BPE means Byte Pair Encoding. The simple idea is:

1. Start with tiny pieces, usually characters or bytes.
2. Find the most common neighboring pair.
3. Glue that pair into one new token.
4. Repeat many times.

Example:

```text
n e w _ y o r k
```

BPE might learn:

```text
n + e -> ne
ne + w -> new
y + o -> yo
r + k -> rk
yo + rk -> york
```

Now the text becomes:

```text
new _ york
```

Normal BPE usually does not merge across the space, so it stops at:

```text
new | _ | york
```

## SuperBPE

SuperBPE keeps BPE's idea, but adds a second stage.

Stage 1 learns normal word/subword tokens:

```text
new | _ | york
```

Stage 2 allows merges across spaces:

```text
new + _ -> new_
new_ + york -> new_york
```

So SuperBPE can turn a phrase into one token:

```text
new_york
```

This is very useful for whitespace languages like English, Spanish, and German.

## Parity-Aware BPE

Normal BPE asks:

```text
What pair is most common overall?
```

Parity-Aware BPE asks:

```text
Which language is currently getting too many pieces?
What merge helps that language?
```

So if Chinese is being split like:

```text
北 | 京
```

Parity-Aware BPE may spend a merge on:

```text
北 + 京 -> 北京
```

It is trying to make tokenization cost more even across languages.

## PA-SuperBPE

PA-SuperBPE combines the two ideas:

```text
SuperBPE's two-stage/cross-space training
+ Parity-Aware BPE's fairness-aware merge choice
```

It can learn English phrase tokens:

```text
new_york
```

while also noticing if another language is being split too much:

```text
北京
```

That is the whole project: keep the compression power of SuperBPE, but reduce cross-language unfairness.

## What 40k And 200k Mean

`40k` and `200k` mean vocabulary size: roughly how many tokens the tokenizer is allowed to know.

A tiny tokenizer may know only:

```text
n, e, w, y, o, r, k
```

A 40k tokenizer might know many words and short phrases.

A 200k tokenizer can know many more words, scripts, names, and phrases.

In this project's setup, the report's split is:

```text
40k run:  stage 1 targets 16k vocab, then stage 2 extends to 40k vocab
200k run: stage 1 targets 80k vocab, then stage 2 extends to 200k vocab
```

For the parity-aware learner, `--symbols` counts merge operations, not final vocab size. Because ByteLevel starts with 256 byte tokens, the exact 40k PA-BPE run learns `40,000 - 256 = 39,744` merges.

For real 40k/200k results, the training corpus should be much bigger than FLORES alone. FLORES is excellent for evaluation, but it is small for training huge tokenizers.
