from __future__ import annotations
"""Generate the PA-SuperBPE slide deck (.pptx) with speaker notes.

Comprehensive deck: motivation, how the tokenizers work (token-chip diagrams),
the full system pipeline, corpus flooring, the fast learner (why + architecture),
validation, metrics, results (charts + tables incl. space-merging), reframed
findings. Native editable PPT charts/tables. No em dashes. Speaker script in every
slide's notes pane; also dumped to paper/slides_speaker_notes.md.

Run: vendor/superbpe/.venv/bin/python scripts/make_slides.py
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION

PURPLE = RGBColor(0x4B, 0x2E, 0x83); GOLD = RGBColor(0xB7, 0xA5, 0x7A)
DARK = RGBColor(0x23, 0x23, 0x23); GRAY = RGBColor(0x60, 0x60, 0x60)
LIGHT = RGBColor(0xF1, 0xEE, 0xF7); WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MIDGRAY = RGBColor(0x9A, 0x9A, 0x9A); FAINT = RGBColor(0xE6, 0xDF, 0xF2)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
CATS = ["BPE", "SuperBPE", "PA-BPE", "PA-SuperBPE", "Hybrid A", "Hybrid B"]
COLORS = [MIDGRAY, MIDGRAY, PURPLE, PURPLE, GOLD, GOLD]
NOTES_MD = ["# PA-SuperBPE speaker notes\n"]


def _clean(t): return t.replace(" — ", ", ").replace("—", "-")


def _box(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    return tb.text_frame


def _run(p, text, size, color, bold=False, italic=False):
    r = p.add_run(); r.text = _clean(text)
    f = r.font; f.size = Pt(size); f.bold = bold; f.italic = italic
    f.color.rgb = color; f.name = "Calibri"


def _p0(slide, l, t, w, h, text, size, color, bold=False, italic=False):
    p = _box(slide, l, t, w, h).paragraphs[0]; _run(p, text, size, color, bold, italic); return p


def _notes(slide, title, text):
    if text:
        text = _clean(text)
        slide.notes_slide.notes_text_frame.text = text
        NOTES_MD.append(f"\n## {_clean(title)}\n\n{text}\n")


def head(s, title):
    _p0(s, 0.7, 0.42, 12.2, 1.0, title, 28, PURPLE, bold=True)
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), Inches(1.28), Inches(11.8), Pt(2.5))
    rule.fill.solid(); rule.fill.fore_color.rgb = GOLD; rule.line.fill.background()


def chips_row(s, top, label, chips, color):
    _p0(s, 0.85, top + 0.04, 3.0, 0.5, label, 15, DARK, bold=True)
    x = 4.0
    for ch in chips:
        w = max(0.6, 0.32 + 0.17 * len(ch))
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(w), Inches(0.52))
        shp.fill.solid(); shp.fill.fore_color.rgb = color; shp.line.color.rgb = WHITE; shp.line.width = Pt(1.5)
        pp = shp.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; _run(pp, ch, 14, WHITE, bold=True)
        x += w + 0.13
    _p0(s, x + 0.05, top + 0.05, 2.6, 0.5, f"= {len(chips)} tokens", 14, GRAY, italic=True)


def title_slide(title, subtitle, authors, footer, notes):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = PURPLE
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(3.55), Inches(3.2), Pt(3))
    bar.fill.solid(); bar.fill.fore_color.rgb = GOLD; bar.line.fill.background()
    _p0(s, 0.9, 2.15, 11.5, 1.5, title, 46, WHITE, bold=True)
    _p0(s, 0.9, 3.75, 11.5, 1.0, subtitle, 22, GOLD, italic=True)
    tf = _box(s, 0.9, 5.3, 11.5, 1.2)
    _run(tf.paragraphs[0], authors, 18, WHITE)
    _run(tf.add_paragraph(), footer, 14, FAINT)
    _notes(s, title, notes)


def bullets_slide(title, items, notes):
    s = prs.slides.add_slide(BLANK); head(s, title)
    tf = _box(s, 0.85, 1.6, 11.7, 5.6); first = True
    for level, text, *style in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        p.level = level; p.space_after = Pt(6); p.space_before = Pt(2)
        bold = "b" in style; star = "star" in style
        glyph = "•  " if level == 0 else "–  "
        col = (PURPLE if bold else (DARK if level == 0 else GRAY))
        _run(p, glyph + text, 19 if level == 0 else 16, GOLD if star else col, bold=bold or star)
    _notes(s, title, notes)


def table_slide(title, headers, rows, notes, note=None, col_w=None, highlight=None, fs=12):
    s = prs.slides.add_slide(BLANK); head(s, title)
    nr, nc = len(rows) + 1, len(headers)
    tbl = s.shapes.add_table(nr, nc, Inches(0.66), Inches(1.7), Inches(12.0), Inches(0.38 * nr)).table
    if col_w:
        for i, w in enumerate(col_w):
            tbl.columns[i].width = Inches(w)
    for j, h in enumerate(headers):
        c = tbl.cell(0, j); c.fill.solid(); c.fill.fore_color.rgb = PURPLE
        p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
        _run(p, h, fs, WHITE, bold=True)
    for i, row in enumerate(rows, 1):
        hl = highlight is not None and (i - 1) in highlight
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0xEC, 0xE5, 0xD3) if hl else (LIGHT if i % 2 else WHITE)
            p = c.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER if j else PP_ALIGN.LEFT
            _run(p, str(val), fs, DARK, bold=hl)
    if note:
        _p0(s, 0.7, 1.7 + 0.4 * nr + 0.1, 12.0, 1.2, note, 13, GRAY, italic=True)
    _notes(s, title, notes)


def chart_slide(title, values, notes, num_fmt="0.000", lo=0.0, hi=None, note=None):
    s = prs.slides.add_slide(BLANK); head(s, title)
    cd = CategoryChartData(); cd.categories = CATS; cd.add_series("s", values, number_format=num_fmt)
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                               Inches(1.0), Inches(1.65), Inches(11.3), Inches(4.7), cd).chart
    chart.has_legend = False; chart.has_title = False
    plot = chart.plots[0]; plot.gap_width = 70; ser = plot.series[0]; ser.has_data_labels = True
    dl = ser.data_labels; dl.number_format = num_fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(12); dl.font.bold = True; dl.font.color.rgb = DARK
    for i, pt in enumerate(ser.points):
        pt.format.fill.solid(); pt.format.fill.fore_color.rgb = COLORS[i]
    va = chart.value_axis; va.has_major_gridlines = False; va.minimum_scale = lo
    if hi is not None:
        va.maximum_scale = hi
    va.tick_labels.font.size = Pt(10); va.tick_labels.number_format = num_fmt; va.tick_labels.number_format_is_linked = False
    chart.category_axis.tick_labels.font.size = Pt(12)
    if note:
        _p0(s, 0.9, 6.55, 11.5, 0.7, note, 13, GRAY, italic=True)
    _notes(s, title, notes)


def flow_slide(title, stages, caption, notes):
    s = prs.slides.add_slide(BLANK); head(s, title)
    n = len(stages); gap = 0.42; bw = (12.4 - gap * (n - 1)) / n
    x = 0.47; top = 2.55; bh = 1.7
    for i, (name, sub) in enumerate(stages):
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(bw), Inches(bh))
        shp.fill.solid(); shp.fill.fore_color.rgb = PURPLE; shp.line.color.rgb = GOLD; shp.line.width = Pt(1.5)
        tf = shp.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; _run(p, name, 13, WHITE, bold=True)
        for line in sub.split("|"):
            pp = tf.add_paragraph(); pp.alignment = PP_ALIGN.CENTER; _run(pp, line, 9, FAINT)
        if i < n - 1:
            ar = _box(s, x + bw - 0.04, top + bh / 2 - 0.3, gap + 0.1, 0.6).paragraphs[0]
            ar.alignment = PP_ALIGN.CENTER; _run(ar, "→", 22, GOLD, bold=True)
        x += bw + gap
    _p0(s, 0.85, top + bh + 0.5, 11.6, 1.2, caption, 15, GRAY, italic=True)
    _notes(s, title, notes)


# ============================================================================ deck
title_slide(
    "Parity-Aware SuperBPE",
    "Fairer multilingual tokenization, without giving up compression",
    "Chendur Jayavelu  ·  Shiv Vinodshankar  ·  Dhruti Vadlamudi",
    "University of Washington · CSE 493S/599S · github.com/Shiv478/cse493s-pa-superbpe",
    "Hi, we're Chendur, Shiv, and Dhruti. Our project, Parity-Aware SuperBPE, is about making "
    "tokenization fairer across languages without sacrificing how efficiently it compresses text. I'll "
    "cover the problem, how the tokenizers work, the system we built (including a 600x-faster learner we "
    "had to write to make this feasible), and then the results and findings.")

bullets_slide("The problem: tokenization is unfair across languages", [
    (0, "Language models read tokens, not words, and everything is priced per token: API cost, compute/latency, and the context window.", "b"),
    (0, "Tokenizers are trained on mostly-English data, so they compress English well and split other languages into many more tokens for the same text."),
    (0, "So speakers of under-represented languages pay more, wait longer, and get a smaller effective context."),
    (0, "In our data: standard BPE makes Thai use 2.27x the tokens of English for the exact same sentence.", "b"),
],
    "Language models don't see words, they see tokens, and everything is priced per token: the API bill, "
    "compute and latency, and how much fits in the context window. Tokenizers are trained on mostly-"
    "English text, so they compress English well and chop other languages into far more tokens. In our "
    "data, standard BPE makes Thai use 2.27 times as many tokens as English for the same sentence, so a "
    "Thai user literally pays more, waits longer, and gets less usable context.")

bullets_slide("Background: two tools that pull in opposite directions", [
    (0, "SuperBPE adds “superword” tokens that merge across spaces.", "b"),
    (1, "Great compression, but it picks merges by global frequency, so the superwords mostly serve English."),
    (0, "Parity-aware BPE picks each merge to help the worst-compressed language.", "b"),
    (1, "Fairer, but it only works within words and sacrifices the dominant language's compression."),
    (0, "The dilemma:  global = efficient but unfair  ·  parity = fair but anti-compression.", "b"),
],
    "Two recent tools are relevant but pull in opposite directions. SuperBPE adds 'superword' tokens that "
    "merge across spaces, great for compression, but it picks merges by global frequency, so on English-"
    "heavy data those superwords mostly go to English. Parity-aware BPE does the opposite: it picks each "
    "merge to help whichever language is worst-compressed, fair, but it only works within words and gives "
    "up the dominant language's compression. So you seem stuck between efficient-but-unfair and fair-but-"
    "inefficient.")

s = prs.slides.add_slide(BLANK); head(s, "How the two ideas work (▁ = a space)")
_p0(s, 0.85, 1.5, 11.6, 0.5, "1) Token type: “superwords” may merge across a space (SuperBPE):", 18, PURPLE, bold=True)
chips_row(s, 2.12, "BPE (within-word):", ["▁of", "▁the", "▁people"], MIDGRAY)
chips_row(s, 2.78, "SuperBPE (superword):", ["▁of▁the", "▁people"], GOLD)
_p0(s, 4.0, 3.42, 9.0, 0.5, "stage 2 merges across the space, fewer tokens (mostly English here)", 13, GRAY, italic=True)
_p0(s, 0.85, 4.05, 11.6, 0.5, "2) Selection: parity spends merges on the worst-compressed language:", 18, PURPLE, bold=True)
chips_row(s, 4.67, "Global (BPE), Chinese:", ["北", "京", "大", "学"], MIDGRAY)
chips_row(s, 5.33, "Parity, Chinese:", ["北京", "大学"], PURPLE)
_p0(s, 4.0, 5.97, 9.0, 0.5, "Chinese gets merges that global BPE would spend on English, fairer", 13, GRAY, italic=True)
_notes(s, "How the two ideas work",
    "Two pictures of the mechanisms. Top is token type: plain BPE never crosses a space, so 'of the "
    "people' is three tokens; SuperBPE's second stage merges across the space, so 'of the' becomes one "
    "token, but on English-heavy data that mostly helps English. Bottom is selection: under global BPE, "
    "Chinese is starved of merges because English pairs are more frequent, so the four characters stay "
    "split; parity instead spends merges on the worst-off language, so Chinese gets proper word units. "
    "Our project is about combining these two ideas well.")

bullets_slide("Our research question", [
    (0, "Is the compression-fairness trade-off fundamental, or just an artifact of applying fairness uniformly?", "b"),
    (0, "SuperBPE trains in two stages: a within-word base, then a cross-whitespace superword stage."),
    (0, "Idea: make only the right stage fairness-driven.", "b"),
    (1, "Put parity where it builds a fair base (within-word); let global superwords compress the dominant language."),
],
    "Our question is whether that trade-off is fundamental, or just because parity gets applied to "
    "everything at once. The key observation: SuperBPE already trains in two stages, a within-word stage "
    "then a cross-space superword stage. So we asked: what if we make only the first stage fair, and let "
    "the second stage do global compression? That one idea is what the whole experiment tests.")

bullets_slide("The six tokenizers we built", [
    (0, "Main 2x2: selection rule x token type:", "b"),
    (1, "within-word:  BPE (global)  /  PA-BPE (parity)"),
    (1, "superword:    SuperBPE (global)  /  PA-SuperBPE (parity)"),
    (0, "Stage ablation: which stage uses parity:", "b"),
    (1, "Hybrid A = parity within-word base  ->  global superwords"),
    (1, "Hybrid B = global within-word base  ->  parity superwords"),
],
    "We built six tokenizers. The main four are a two-by-two: standard versus parity selection, crossed "
    "with within-word versus superword tokens. Then two hybrids that split the rule by stage: Hybrid A is "
    "a fair base with global superwords, Hybrid B is the reverse. These six let us isolate exactly where "
    "fairness should live.")

flow_slide("The system: end-to-end pipeline",
    [("1. Corpus", "English-skewed|FineWeb 5 GB|10 languages"),
     ("2. Pretokenize", "SuperBPE regex|stage1 / stage2|-> per-lang dicts"),
     ("3. Floor", "drop count < 2|(min_freq 2)|smaller, faster"),
     ("4. Learn merges", "fast parity|learner|(our engine)"),
     ("5. Build", "HF tokenizer|vocab = 40,000"),
     ("6. Evaluate", "FLORES-200|Gini, max/min|tokens/char")],
    "Same engine + same corpus for all six tokenizers; only the merge-selection rule and the stage split change.",
    "This is the whole system, the work behind the results. We download an English-skewed FineWeb corpus "
    "across ten languages; pre-tokenize it with SuperBPE's exact regex into per-language frequency "
    "dictionaries; floor those dictionaries to make training tractable; learn the merges with our fast "
    "parity learner; assemble a standard Hugging Face tokenizer; and evaluate on FLORES. Every tokenizer "
    "goes through the identical pipeline, so the only thing that varies is the method.")

bullets_slide("Setup: a controlled comparison", [
    (0, "One engine for all six: our fast learner, validated byte-identical to the reference (and, in standard mode, to textbook BPE).", "b"),
    (0, "Same corpus: English-skewed FineWeb, ~5 GB (50% English + 9 languages)."),
    (0, "Same vocabulary: 40,000 (15,744 within-word + 24,000 superword)."),
    (0, "Same evaluation: FLORES-200 devtest, 10 languages x 1,012 parallel sentences."),
],
    "Everything is held constant so the only variable is the method: the same training engine, the same "
    "English-skewed FineWeb corpus, the same 40,000 vocabulary, and the same FLORES evaluation across ten "
    "languages.")

table_slide("Corpus flooring: drop pretokens seen only once (count >= 2)",
    ["Stage", "unique pretokens", "token occurrences"],
    [["Stage 1 (within-word)", "23.4M  ->  5.0M   (-78%)", "-2.4% dropped"],
     ["Stage 2 (superword)", "22.4M  ->  0.9M   (-96%)", "-31% dropped"]],
    note="Every dropped pretoken occurs exactly once, so it is individually merge-ineligible (a merge needs frequency >= 2). "
    "Stage-2 superwords are sparse, so more mass drops there. All six tokenizers use the same floored data, so the comparison stays controlled.",
    col_w=[3.6, 4.4, 4.0], fs=14,
    notes="Before training we floor the dictionaries: we drop any pretoken that appears only once. The "
    "type counts collapse, 78 percent of unique pretokens in stage 1 and 96 percent in stage 2, because "
    "language has a huge long tail of things seen once. But in terms of actual token mass we only lose "
    "2.4 percent in stage 1. Stage 2, the superwords, is sparser, so we lose 31 percent there. The "
    "justification: anything seen once can't, by itself, form a merge, since a merge needs frequency at "
    "least two; and crucially all six tokenizers use the same floored data, so even if flooring shifts "
    "absolute numbers, the comparison between methods stays fair.")

bullets_slide("Why we built a fast learner (the reference was infeasible)", [
    (0, "Parity re-ranks pairs by the worst language every merge, and the reference learner is pure Python doing two full passes per merge.", "b"),
    (0, "Measured cost: ~370 seconds per merge.  39,744 merges  ≈  5.6 MONTHS.", "b"),
    (0, "~72% of that time is rescanning the FLORES-dev “ruler”, and that cost does NOT shrink with corpus size, so using less data can't fix it."),
    (0, "So we reimplemented the hot loop with better data structures, proven byte-identical to the reference.", "b"),
    (0, "Result: hours instead of months, with the exact same merges.", "star"),
],
    "Here's a big piece of the work. Parity re-ranks pairs by the worst language on every single merge, "
    "and the reference implementation is pure Python doing two full passes per merge. We measured it at "
    "about 370 seconds per merge, which for forty thousand merges is roughly five and a half months, "
    "completely infeasible. And about 72 percent of that time is rescanning the FLORES dev ruler, a cost "
    "that doesn't shrink if you use less data, so we couldn't just subsample our way out. So we "
    "reimplemented the hot loop with better data structures, and proved it produces byte-identical "
    "output. That turned months into hours and is what made this whole project possible.")

table_slide("Fast learner architecture: same algorithm, better data structures",
    ["Per-merge step", "Reference learner (slow)", "Fast learner (ours)"],
    [["Pick best pair", "scan ALL unique pairs (tens of millions)", "per-language max-heap, peek O(log n)"],
     ["Update FLORES-dev ruler", "rescan ALL ~61k dev words (~72% of time)", "inverted pair->dev-word index, touch only affected"],
     ["Cost per merge", "~370 s", "~0.6 s   (≈ 600x faster)"],
     ["Output", "(ground truth)", "byte-identical to the reference"]],
    note="Same decisions kept exactly: tie-break, min_frequency, prune/resync, and the SuperBPE stage-2 filter. "
    "Heaps use lazy deletion, validated against the authoritative counts on peek. Only the lookup structures changed.",
    col_w=[2.7, 4.65, 4.65], highlight=[2], fs=12,
    notes="Here's how it works. The reference does two expensive things per merge: to pick the best pair "
    "it scans every unique adjacent pair, tens of millions of them; and to update the per-language ruler "
    "it rescans the entire FLORES dev set. We replaced the first with per-language max-heaps, so picking "
    "a merge is just a peek, and the second with an inverted index from a pair to the dev words "
    "containing it, so we only touch the few words a merge actually affects. The heaps use lazy deletion "
    "and we validate the top against the real counts before using it, so the tie-breaking and every "
    "decision are identical. Net result: about 600 times faster, byte-for-byte the same output.")

bullets_slide("How we know it is correct (validation)", [
    (0, "The fast learner is proven byte-identical to the reference parity learner:", "b"),
    (1, "5 unit tests (tie-breaks, the stage-2 filter, resync, preload)"),
    (1, "a 300-merge step-by-step exact check (fast pick == reference pick every merge)"),
    (1, "a 3-config harness on real data (PA-BPE, PA-SuperBPE stage 1, stage 2)"),
    (0, "Standard mode is byte-identical to an independent, from-scratch textbook BPE.", "b"),
    (0, "The superword stage really fires (next slides): within-word ≈ 0 cross-space tokens, superword thousands."),
    (0, "Same engine + corpus + evaluation for all six  =>  any difference is the method, not the setup.", "b"),
],
    "Results are only as good as the setup, so we validated heavily. The fast learner is byte-identical "
    "to the reference: five unit tests, a 300-merge step-by-step check where the fast pick equals the "
    "reference pick on every merge, and a three-configuration harness on real data. In standard mode it's "
    "also byte-identical to a completely independent from-scratch BPE we wrote. We separately verified "
    "the superword stage actually fires, which I'll show. And since all six share the engine, corpus, and "
    "evaluation, any difference is the method itself.")

bullets_slide("What the evaluation metrics mean", [
    (0, "We tokenize the SAME FLORES sentences in all 10 languages and count tokens per language.", "b"),
    (0, "tokens-per-character = tokens / characters  ->  raw compression.", "b"),
    (1, "distorted across scripts: Chinese packs more meaning per character, so equal cost can look unequal."),
    (0, "token-count = tokens to encode the SAME content (FLORES is parallel)  ->  the fair, apples-to-apples cost.", "b"),
    (1, "this is literally what you pay (money, latency, context).  Example (BPE): English ≈29k vs Thai ≈66k tokens."),
    (0, "Two summaries of the 10 per-language values (lower = fairer):", "b"),
    (1, "max/min ratio = worst language / best  (1.0 = perfectly equal)"),
    (1, "Gini = overall inequality across all 10 languages  (0 = everyone equal, like income inequality)"),
],
    "One slide on the metrics. We tokenize the same FLORES sentences in all ten languages and count "
    "tokens per language. Tokens-per-character is raw compression, but it's distorted across scripts, "
    "because a Chinese character holds more meaning. Token-count is the fair one: because FLORES is "
    "parallel, it's the tokens to encode identical content, exactly what you pay. For example under BPE "
    "English is about 29,000 tokens and Thai about 66,000. We summarize the spread two ways, both lower-"
    "is-fairer: the max-over-min ratio where 1.0 is perfectly equal, and the Gini coefficient, like "
    "income inequality, where 0 is everyone equal.")

chart_slide("Headline: cross-lingual cost ratio (worst / best language, 1.0 = fair)",
    [2.272, 1.917, 1.027, 1.023, 1.442, 1.405], num_fmt="0.00", lo=1.0, hi=2.4,
    note="Token count for identical content. 1.0 = perfectly fair. (Axis starts at 1.0 to show the differences.)",
    notes="The headline. This is the worst-to-best ratio on token count; 1.0 is perfectly fair. BPE is "
    "2.27, Thai pays more than double English. SuperBPE is 1.9. Both parity methods, in purple, collapse "
    "it to about 1.02, essentially equal cost for equal content. The hybrids, in gold, are in between. "
    "Axis starts at 1.0 because that's the natural floor.")

table_slide("Full results: six tokenizers at 40k vocab",
    ["Method", "s1->s2", "Gini tok/char", "Gini tok-count", "max/min tok-count", "eng tok/char"],
    [["BPE", "-", "0.306", "0.114", "2.27x", "0.222"],
     ["SuperBPE", "g->g", "0.235", "0.097", "1.92x", "0.221"],
     ["PA-BPE", "parity", "0.237", "0.005", "1.03x", "0.313"],
     ["PA-SuperBPE", "p->p", "0.236", "0.004", "1.02x", "0.327"],
     ["Hybrid A", "p->g", "0.230", "0.066", "1.44x", "0.250"],
     ["Hybrid B", "g->p", "0.256", "0.033", "1.41x", "0.241"]],
    note="token-count = fairness (costs money/latency/context).  eng tok/char = the English-compression trade-off.  Lower is better.",
    col_w=[2.2, 1.2, 2.25, 2.25, 2.4, 1.7], highlight=[4],
    notes="The full numbers; two columns matter most. Token-count fairness drops from 0.11 for BPE to "
    "about 0.004 for parity, roughly 25 times fairer. The far-right English column previews the catch: "
    "parity pushes English from 0.22 to 0.31 or 0.33. The highlighted row, Hybrid A, is strong on "
    "fairness while keeping English low.")

bullets_slide("Finding 1: parity nearly eliminates the disparity, at a cost", [
    (0, "Worst/best ratio 2.27x -> ~1.02x;  token-count Gini 0.114 -> 0.004  (≈ 25x fairer).", "b"),
    (0, "Every language ends up within ~2% of every other to encode the same content."),
    (0, "But it isn't free: parity spends merges away from English.", "b"),
    (1, "English: ~29,000 tokens (BPE)  ->  ~42,000 tokens (full parity)."),
],
    "Finding one: parity nearly eliminates the disparity, the ratio goes from 2.27 to about 1.02 and the "
    "Gini drops about 25-fold, every language within roughly 2 percent of every other for the same "
    "content. But it isn't free: parity spends merges away from English, which climbs from about 29,000 "
    "to 42,000 tokens. So the question becomes: can we keep the fairness without paying that price?")

chart_slide("The trade-off: English compression cost (tokens per character)",
    [0.2216, 0.2212, 0.3125, 0.3273, 0.2498, 0.2413], num_fmt="0.000", lo=0.20, hi=0.34,
    note="Lower = English compressed better. Full parity (purple) is costly; the hybrids (gold) stay near the baselines.",
    notes="The trade-off as a picture: English's tokens-per-character, lower is better. The two baselines "
    "are low, full parity in purple is high, that's the cost, and the hybrids in gold sit much lower, "
    "close to the baselines. They recover most of English's compression. Axis is zoomed because the "
    "values are close.")

bullets_slide("Finding 2: the trade-off is controllable, by WHERE parity goes", [
    (0, "Hybrid A (parity base -> global superwords) is the best-balanced tokenizer:", "b"),
    (1, "best per-character fairness of all six (Gini 0.230) AND English stays fast (0.250)."),
    (0, "The design rule we found:", "b"),
    (1, "stage-1 parity  ->  per-character fairness"),
    (1, "stage-2 parity  ->  token-count fairness"),
    (1, "full parity  ->  maximal count equality, at the dominant language's expense"),
    (0, "It's a design dial, not a law of nature.", "star"),
],
    "The key result: the trade-off is controllable by where you put parity. Hybrid A, fairness in the "
    "within-word base with global superwords on top, gives the best per-character fairness of all six "
    "while keeping English fast. More generally: stage-one parity buys per-character fairness, stage-two "
    "parity buys token-count fairness, and full parity maxes fairness at the dominant language's expense. "
    "So it's a design dial you set, not a law of nature.")

chart_slide("Per-character fairness (Gini): Hybrid A is best of all six",
    [0.3063, 0.2350, 0.2370, 0.2356, 0.2304, 0.2556], num_fmt="0.000", lo=0.20, hi=0.32,
    note="Lower = fairer per character. (Axis zoomed: values are close; Hybrid A, in gold, is the lowest.)",
    notes="Same point visually: per-character fairness, lower is fairer, axis zoomed because the values "
    "are close. Hybrid A, the gold bar, is the lowest of all six, including the full-parity methods. So "
    "on the per-character metric the hybrid is our single fairest, best-balanced tokenizer.")

chart_slide("Superword mechanism: how many spaces each tokenizer merges",
    [1, 5845, 2, 5717, 5587, 4483], num_fmt="0", lo=0, hi=6800,
    note="Cross-space tokens in the 40k vocab. Within-word (BPE, PA-BPE) ≈ 0 by design; superword methods merge ~4,500-5,800.",
    notes="The space-merging result as a chart. It counts vocabulary tokens that cross a word boundary. "
    "BPE and PA-BPE are essentially zero, by design they never cross a space. The four superword methods "
    "each create between about 4,500 and 5,800 cross-space tokens. Notice Hybrid B has the fewest, "
    "because its parity stage spent budget on within-run merges for spaceless scripts like Chinese, which "
    "is exactly why it equalized those languages. So this also confirms the superword stage really fired.")

table_slide("Space-merging, in detail",
    ["Tokenizer", "cross-space", "% vocab", "total spaces", "1 sp", "2", "3", "4"],
    [["BPE", "1", "0.0%", "1", "1", "0", "0", "0"],
     ["SuperBPE", "5,845", "14.6%", "7,260", "4,620", "1,046", "168", "11"],
     ["PA-BPE", "2", "0.0%", "2", "2", "0", "0", "0"],
     ["PA-SuperBPE", "5,717", "14.3%", "7,137", "4,577", "888", "224", "28"],
     ["Hybrid A", "5,587", "14.0%", "6,902", "4,444", "983", "148", "12"],
     ["Hybrid B", "4,483", "11.2%", "5,519", "3,617", "707", "148", "11"]],
    note="cross-space = vocab tokens spanning a space.  Columns 1-4 = how many spaces a token merges.  Within-word tokenizers ≈ 0 by design.",
    col_w=[2.3, 1.7, 1.2, 1.8, 1.1, 1.0, 1.0, 1.0], highlight=[4, 5], fs=12,
    notes="And the detailed table you asked for. Cross-space is how many vocabulary tokens span a space; "
    "percent of vocab; total spaces merged; and the breakdown of how many spaces each token merges, one "
    "through four. The within-word tokenizers are basically zero. The superword methods are in the "
    "thousands, and most are simple two-word tokens, the '1 space' column. The stage-2 filter caps it at "
    "four words, which you can see holds.")

bullets_slide("What we asked, and what we found", [
    (0, "Q1.  How unfair are standard tokenizers on realistic, English-heavy data?", "b"),
    (1, "Substantially: a 2.3x cost gap (Thai vs English). BPE and SuperBPE both leave large disparity."),
    (0, "Q2.  Can parity-aware selection fix it?", "b"),
    (1, "Yes: near-perfect token parity (~1.02x; ~25x lower Gini)."),
    (0, "Q3.  Does fairness have to cost the dominant language?", "b"),
    (1, "No: parity in the within-word stage (Hybrid A) keeps English fast AND is fairest per character.", "star"),
    (0, "(Honest note: our proposal guessed SuperBPE would worsen fairness; it didn't, which led us to the sharper question of WHERE fairness belongs.)"),
],
    "Reframing around what we actually found. Q1: how unfair are standard tokenizers on realistic data? "
    "Substantially, about a 2.3-times gap. Q2: can parity fix it? Yes, near-perfect parity, about 25 "
    "times fairer. And Q3, the interesting one: does fairness have to cost the dominant language? No, "
    "putting parity only in the within-word stage keeps English fast and is the fairest per character. "
    "Honestly, our original proposal guessed SuperBPE would make things less fair; it didn't, and that "
    "pushed us to the sharper question of where fairness belongs.")

bullets_slide("Why this matters", [
    (0, "A simple, actionable tokenizer-design rule: put fairness in the within-word stage.", "b"),
    (0, "It resolves a trade-off widely treated as inherent: fairness vs the dominant language's compression."),
    (0, "Near-perfect cross-lingual token parity is achievable (~25x fairer) at a controllable cost."),
    (0, "And it's reproducible: one validated, ~600x-faster engine, byte-identical to ground truth; code + results + slides public.", "b"),
],
    "Why this matters: a simple, actionable design rule, put fairness in the within-word stage, that "
    "resolves a trade-off people treat as inherent. We show near-perfect cross-lingual parity is "
    "achievable, about 25 times fairer, at a cost you can dial. And it's fully reproducible: one "
    "validated, 600-times-faster engine, byte-identical to ground truth, with all the code, results, and "
    "slides public.")

bullets_slide("Conclusion, limitations & next steps", [
    (0, "Parity-aware selection makes multilingual tokenization far fairer; putting parity in the within-word stage (Hybrid A) keeps the dominant language fast.", "b"),
    (0, "We also built the engine that made it feasible: a ~600x faster, byte-identical parity learner."),
    (0, "Limitations / future work:", "b"),
    (1, "40k vocab only (200k next); single 50%-English corpus (composition sweep); no LM yet."),
    (1, "Flooring drops 31% of sparse stage-2 mass (controlled, since shared by all six)."),
    (1, "Spaceless scripts (CJK/Thai) can only be helped in the within-word stage."),
    (0, "Code, results & these slides:  github.com/Shiv478/cse493s-pa-superbpe", "b"),
],
    "To wrap up: parity-aware selection makes multilingual tokenization much fairer, and putting that "
    "fairness in the within-word stage keeps the dominant language efficient, captured by Hybrid A. Along "
    "the way we built the engine that made it feasible, a 600-times-faster learner proven byte-identical "
    "to the reference. Limitations are honest next steps: one vocab size, one corpus mix, the flooring "
    "approximation, and no language-model training yet. Everything is in the repo. Thank you, we're happy "
    "to take questions.")

out = Path("paper/PA-SuperBPE_slides.pptx"); out.parent.mkdir(exist_ok=True)
prs.save(str(out))
Path("paper/slides_speaker_notes.md").write_text("\n".join(NOTES_MD), encoding="utf-8")
print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides) + paper/slides_speaker_notes.md")
