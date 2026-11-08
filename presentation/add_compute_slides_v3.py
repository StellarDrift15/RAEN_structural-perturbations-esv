#!/usr/bin/env python3
"""Add (or refresh) the two compute slides in ESV_Project_Proposal_v3.pptx.

Idempotent: it deletes any slide already carrying one of the two titles, rebuilds both from the numbers below and
inserts them after the results slide. Re-run it whenever the compute section of
`results/gsm8k/analysis/esv3_bottleneck_gsm8k.md` changes.

Every figure here comes from `analyze_esv3.py --benchmark gsm8k --models small [--verifier-model
openai/gpt-oss-120b]`, section "Compute cost", pooled over the three small solvers, all 198 pairs / 1,170
instances. The deck's visual system (sizes, fills, table style) is copied from the existing slides.

    python presentation/add_compute_slides_v3.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DECK = Path(__file__).resolve().parent / "ESV_Project_Proposal_v3.pptx"

INK = RGBColor(0x1A, 0x1B, 0x2E)
GREEN = RGBColor(0x2E, 0x7D, 0x5B)
MINT = RGBColor(0x7F, 0xD1, 0xA6)
AMBER = RGBColor(0xB8, 0x74, 0x0F)
RED = RGBColor(0x8F, 0x2F, 0x23)
LIGHT = RGBColor(0xD9, 0xD9, 0xD9)
GREY = RGBColor(0x8C, 0x8C, 0x8C)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PANEL_GREEN = RGBColor(0xDC, 0xEF, 0xE3)
PANEL_AMBER = RGBColor(0xFB, 0xEB, 0xD0)
TABLE_STYLE = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"
MONO = "Consolas"

TITLE_A = "Compute Profile: What Each Method Costs"
TITLE_B = "Accuracy vs Compute: the Trade-off"


# --------------------------------------------------------------------------- helpers
def textbox(slide, left, top, width, height, lines):
    """lines: (text, size, bold, color, font) tuples; '' starts a spacer paragraph."""
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(2)
        if not line[0]:
            continue
        text, size, bold, color = line[:4]
        font_name = line[4] if len(line) > 4 else None
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        if font_name:
            run.font.name = font_name
    return box


def panel(slide, left, top, width, height, fill, line=None, rounded=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    shape.line.width = Pt(1)
    shape.shadow.inherit = False
    if rounded:
        shape.adjustments[0] = 0.16667
    return shape


def grid(slide, left, top, width, widths, rows, row_h=0.36, size=12.5, mono_from=1,
         emphasis=None, header_fill=INK):
    """A table in the deck's style: dark header, white body, monospaced numeric columns.
    emphasis: {row_index: RGBColor} to recolour a whole body row."""
    emphasis = emphasis or {}
    shape = slide.shapes.add_table(len(rows), len(widths), Inches(left), Inches(top),
                                   Inches(width), Inches(row_h * len(rows)))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = True
    tbl._tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = TABLE_STYLE
    for ci, w in enumerate(widths):
        tbl.columns[ci].width = Inches(w)
    for ri, row in enumerate(rows):
        tbl.rows[ri].height = Inches(row_h)
        for ci, value in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.text = str(value)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = 0
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if ri == 0 else WHITE
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            for run in para.runs:
                run.font.size = Pt(size)
                run.font.bold = ri == 0 or ri in emphasis
                run.font.color.rgb = WHITE if ri == 0 else emphasis.get(ri, INK)
                if ri > 0 and ci >= mono_from:
                    run.font.name = MONO
    return shape


def footnote(slide, text):
    textbox(slide, 0.60, 6.95, 12.13, 0.48, [(text, 10, False, GREY)])


def title_of(slide) -> str:
    """This deck's title placeholder is idx=100, so `shapes.title` (which looks for idx=0) returns None."""
    for shape in slide.placeholders:
        if shape.placeholder_format.idx == 100:
            return shape.text_frame.text
    return ""


def new_slide(prs, title):
    layout = next(l for l in prs.slide_layouts if l.name == "TITLE_ONLY")
    slide = prs.slides.add_slide(layout)
    ph = slide.placeholders[100]
    ph.left, ph.top, ph.width, ph.height = Inches(0.60), Inches(0.30), Inches(12.13), Inches(0.90)
    ph.text_frame.text = title
    for run in ph.text_frame.paragraphs[0].runs:
        run.font.size = Pt(32)
        run.font.color.rgb = INK
    return slide


def guard_not_present(prs, titles):
    """Additive only. Dropping a slide in python-pptx leaves its part in the package, which then gets written
    twice, so this script refuses to run on a deck that already has the compute slides. To rebuild them:

        git checkout -- presentation/ESV_Project_Proposal_v3.pptx && python presentation/add_compute_slides_v3.py
    """
    clash = [t for t in titles if any(title_of(s) == t for s in prs.slides)]
    if clash:
        raise SystemExit(
            f"{DECK.name} already contains: {', '.join(clash)}.\n"
            "This script only adds slides. Restore the committed deck first, then re-run:\n"
            f"  git checkout -- presentation/{DECK.name} && python presentation/{Path(__file__).name}")


def move_slide(prs, from_idx, to_idx):
    lst = prs.slides._sldIdLst
    entry = list(lst)[from_idx]
    lst.remove(entry)
    lst.insert(to_idx, entry)


# --------------------------------------------------------------------------- slide A
def slide_compute_profile(prs):
    s = new_slide(prs, TITLE_A)
    textbox(s, 0.60, 1.18, 12.13, 0.52, [(
        "No method re-solves the problem: all four share the same 4 CoT samples and add at most one call, so the "
        "baseline is a constant and the extra call is the entire difference. serial = billed compute time; "
        "path = what a user waits for, since the 4 samples are issued concurrently.", 14, True, INK)])
    grid(s, 0.60, 1.74, 12.13,
         [2.95, 1.02, 0.80, 1.15, 1.18, 1.10, 1.15, 1.35, 1.43],
         [["method", "accuracy", "calls", "solver tok", "verifier tok", "total tok", "vs maj@4", "serial", "path"],
          ["Direct QA maj@4", "12.1%", "4.00", "1,095", "—", "1,095", "-54.0%", "2.4 s", "1.0 s"],
          ["CoT pass@1", "85.1%", "1.00", "595", "—", "595", "-75.0%", "4.8 s", "4.8 s"],
          ["CoT maj@4  (baseline)", "88.5%", "4.00", "2,379", "—", "2,379", "—", "19.2 s", "6.5 s"],
          ["Universal Self-Consistency", "89.3%", "4.26", "2,799", "—", "2,799", "+17.6%", "19.4 s", "6.7 s"],
          ["ESV, self-check", "85.1%", "4.26", "3,158", "—", "3,158", "+32.7%", "23.5 s", "10.8 s"],
          ["ESV, independent check", "93.5%", "4.26", "2,379", "1,038", "3,417", "+43.6%", "26.4 s", "13.7 s"],
          ["CoT pass@4  (oracle ceiling)", "94.5%", "4.00", "2,379", "—", "2,379", "—", "19.2 s", "6.5 s"]],
         row_h=0.33, size=12.5, emphasis={6: GREEN})

    panel(s, 0.60, 4.50, 5.95, 2.30, INK, INK, rounded=False)
    textbox(s, 0.82, 4.58, 5.51, 2.18, [
        ("The disagreement gate is the efficiency mechanism", 14, True, MINT),
        ("Stage 1 is code over answers already paid for: free, and it retires "
         "three-quarters of the work.", 12.5, False, LIGHT),
        ("fires on    303/1,170  (25.9%)", 12.5, True, WHITE, MONO),
        ("when fired  4,008 tok,   27.8 s", 12.5, True, WHITE, MONO),
        ("amortized   1,038 tok,    7.2 s", 12.5, True, WHITE, MONO),
        ("ungated     +168%, not +43.6%", 12.5, True, MINT, MONO),
        ("Gated out = the maj@4 answer untouched: 0 tokens, 0 seconds.", 12.5, False, LIGHT)])

    panel(s, 6.78, 4.50, 5.95, 2.30, PANEL_GREEN, GREEN)
    textbox(s, 7.00, 4.58, 5.51, 2.18, [
        ("+5.0 pp for +43.6% tokens and +0.26 calls", 15, True, GREEN),
        ("Not the 2x–5x a verify-everything pipeline costs. And the right answer is already in the 4 samples "
         "we paid for 94.5% of the time — ESV is buying selection, not reasoning, which is why its cost is a "
         "fraction of the baseline rather than a multiple of it.", 12.5, False, INK),
        ("The self-check arm is strictly dominated: +779 tokens and +4.3 s to lose 3.3 pp.", 12.5, True, RED)])
    footnote(s, "Pooled over 3 small solvers, all 198 pairs = 1,170 instances. Read off the usage and latency "
                "stored on every call, so it reproduces offline: analyze_esv3.py --benchmark gsm8k --models small "
                "[--verifier-model openai/gpt-oss-120b], section 'Compute cost'.")
    return s


# --------------------------------------------------------------------------- slide B
def slide_tradeoff(prs):
    s = new_slide(prs, TITLE_B)
    textbox(s, 0.60, 1.18, 12.13, 0.52, [(
        "net = fixes - breaks, so a method that trades a correct answer for a wrong one is charged for both. "
        "ESV costs 2.5x USC's token overhead and returns 5.9x the net fixes, so it is 2.4x more token-efficient "
        "per net fix than USC — the baseline it was pre-registered to beat.", 14, True, INK)])
    grid(s, 0.60, 1.74, 12.13,
         [3.30, 1.35, 1.45, 0.85, 1.48, 1.25, 2.45],
         [["method", "Δ accuracy", "fixes / breaks", "net", "extra tok", "extra wait", "tok per net fix"],
          ["Direct QA maj@4", "-76.3 pp", "7 / 900", "-893", "-1,284", "-5.5 s", "no net gain"],
          ["Universal Self-Consistency", "+0.9 pp", "19 / 9", "+10", "+420", "+0.2 s", "49,112"],
          ["ESV, self-check", "-3.3 pp", "18 / 57", "-39", "+779", "+4.3 s", "no net gain"],
          ["ESV, independent check", "+5.0 pp", "61 / 2", "+59", "+1,038", "+7.2 s", "20,585"],
          ["CoT pass@4  (oracle ceiling)", "+6.1 pp", "71 / 0", "+71", "0", "0 s", "0  (needs gold)"]],
         row_h=0.345, size=12.5, emphasis={4: GREEN})

    textbox(s, 0.60, 3.88, 5.95, 0.32,
            [("The spend is adaptive — it tracks where the gain is", 14, True, INK)])
    grid(s, 0.60, 4.22, 5.95, [1.95, 1.00, 0.95, 0.95, 1.10],
         [["", "gate", "maj@4", "ESV", "extra tok"],
          ["originals", "20.3%", "90.6%", "94.4%", "+33.2%"],
          ["strategy-breaking P'", "45.1%", "79.1%", "87.5%", "+72.7%"],
          ["preserving P'", "17.5%", "93.6%", "97.6%", "+28.0%"]],
         row_h=0.325, size=12, emphasis={2: AMBER})
    textbox(s, 0.60, 5.56, 5.95, 0.44, [(
        "Disagreement is a free proxy for \"the strategy may have broken\". Per solver the cost of a fix barely "
        "moves: 18.4k–26.8k tokens.", 12, False, INK)])

    panel(s, 6.78, 3.88, 5.95, 2.12, PANEL_AMBER, AMBER)
    textbox(s, 7.00, 3.94, 5.51, 2.00, [
        ("The honest caveat: the verifier is a bigger model", 14, True, AMBER),
        ("The 1,038 check tokens are billed by gpt-oss-120B, the 2,379 sample tokens by a 4–12B solver. "
         "With k = the verifier's price per token over the solver's:", 12, False, INK),
        ("k=1  3,417 tok  +43.6%  |  20,585 per fix", 12, True, INK, MONO),
        ("k=2  4,455 tok  +87.3%  |  41,170 per fix", 12, True, INK, MONO),
        ("k=5  7,569 tok  +218%   |  102,925 per fix", 12, True, INK, MONO),
        ("Break-even against USC is k = 2.4. No price list is assumed — read off your k.", 12, True, AMBER)])

    panel(s, 0.60, 6.06, 12.13, 0.78, INK, INK, rounded=False)
    textbox(s, 0.85, 6.14, 11.63, 0.66, [(
        "But the two are not interchangeable at any price: USC's total gain is capped at +10 net fixes against "
        "ESV's +59, so more USC cannot reach 93.5%. ESV buys accuracy USC cannot — k only sets its price.",
        14, True, MINT)])
    footnote(s, "Latency is ESV's real cost: the check is strictly sequential (it needs all 4 answers), so the wait "
                "goes 6.5 s -> 13.7 s overall and 34.3 s on the 25.9% of instances that fire. CoT pass@1 is the "
                "latency frontier: 85.1% for 595 tokens and 4.8 s.")
    return s


def refresh_pipeline_footnote(prs) -> bool:
    """The pipeline slide's cost footnote predates the full measurement: it rounded the gate rate to 73% and the
    extra calls to 0.27, and its token figure counted completion tokens only. Make it exact and point at the
    slides that now carry the whole bill."""
    want = ("Cost in practice: the gate retires 74.1% of instances at zero cost, so ESV adds 0.26 calls and 1,038 "
            "tokens per instance (+43.6%) rather than the 4,008 a verification call costs when it fires. "
            "Full accounting: the two compute slides after the results.")
    for slide in prs.slides:
        if not title_of(slide).startswith("The ESV v3 Pipeline"):
            continue
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.startswith("Cost in practice:"):
                run = shape.text_frame.paragraphs[0].runs[0]
                run.text = want
                for extra in shape.text_frame.paragraphs[0].runs[1:]:
                    extra.text = ""
                return True
    return False


def main() -> int:
    prs = Presentation(str(DECK))
    before = len(prs.slides)
    guard_not_present(prs, (TITLE_A, TITLE_B))
    print("pipeline-slide cost footnote refreshed:", refresh_pipeline_footnote(prs))
    anchor = next((i for i, s in enumerate(prs.slides) if title_of(s).startswith("Results:")), None)
    if anchor is None:
        raise SystemExit("could not find the results slide to insert after")
    slide_compute_profile(prs)
    slide_tradeoff(prs)
    n = len(prs.slides)
    move_slide(prs, n - 2, anchor + 1)   # compute profile
    move_slide(prs, n - 1, anchor + 2)   # trade-off
    prs.save(str(DECK))
    print(f"{DECK.name}: {before} slides -> {len(prs.slides)}; "
          f"compute slides are now {anchor + 2} and {anchor + 3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
