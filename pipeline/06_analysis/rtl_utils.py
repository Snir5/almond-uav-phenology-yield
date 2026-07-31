# -*- coding: utf-8 -*-
"""
rtl_utils.py: lightweight, from-scratch Hebrew RTL rendering helpers for
reportlab, since python-bidi / arabic-reshaper are not installable in this
offline sandbox and reportlab has no built-in bidi support.

Approach (verified against several rendering tests):
- Split each line into "blocks" of consecutive words that are either
  Hebrew-containing or not.
- Hebrew blocks: reverse the whole block's characters (this correctly
  reverses both word order and letter order within Hebrew, since it is
  mathematically identical to reversing word position + reversing each
  word's letters).
- Non-Hebrew blocks (English terms, numbers, formulas): left completely
  untouched, so multi-word English phrases keep their natural reading
  order (this matches standard bidi behaviour for embedded LTR runs).
- Reverse the order of blocks (not their contents) so the whole line
  reads correctly right-to-left when drawn by reportlab's left-to-right
  text engine.
- Word-wrapping must be done manually BEFORE reordering (using the exact
  same width reportlab will use to lay out the paragraph), because once a
  line is reordered, letting reportlab re-wrap it would scramble line
  order. Each wrapped line is reordered independently and lines are
  joined with <br/>, so line order (top to bottom) stays correct.
"""
from reportlab.pdfbase import pdfmetrics


def has_heb(word):
    return any('֐' <= c <= '׿' for c in word)


_MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}


def _mirror_brackets(s):
    # Bidi rule: bracket/paren GLYPHS must swap when the run containing them
    # is reversed for RTL display, otherwise "(" (which curves to indicate
    # "opens here reading left-to-right") ends up misreading as a close-paren
    # once its position is flipped to the other side of the text.
    return "".join(_MIRROR.get(c, c) for c in s)


def reorder_line(text):
    words = text.split(" ")
    blocks = []
    for w in words:
        h = has_heb(w)
        if blocks and blocks[-1][0] == h:
            blocks[-1][1].append(w)
        else:
            blocks.append([h, [w]])
    out_blocks = []
    for h, ws in blocks:
        joined = " ".join(ws)
        out_blocks.append(_mirror_brackets(joined[::-1]) if h else joined)
    out_blocks.reverse()
    return " ".join(out_blocks)


def wrap_words(text, font_name, font_size, max_width):
    words = text.split(" ")
    lines = []
    cur = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def rtl_html(text, font_name="Heb", font_size=10.6, max_width=504):
    lines = wrap_words(text, font_name, font_size, max_width)
    reordered = [reorder_line(l) for l in lines]
    return "<br/>".join(reordered)
