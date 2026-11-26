"""Render design-note.md to a PDF that fits inside the 6-page limit.

Tries a few type sizes and keeps the largest that fits. Run: python build_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
SRC = HERE / "design-note.md"
OUT = HERE / "design-note.pdf"
NAVY = colors.HexColor("#1F3864")
GREY = colors.HexColor("#5A5A5A")
RULE = colors.HexColor("#BFBFBF")


def esc(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="8">\1</font>', text)
    return text


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.match(r"^\|[\s:|-]+\|$", line.strip()):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def build(font_size: float, leading: float) -> int:
    body = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=font_size, leading=leading,
        alignment=TA_JUSTIFY, spaceAfter=3.2, textColor=colors.HexColor("#111111"),
    )
    h1 = ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=font_size + 5.4,
        leading=leading + 6, textColor=NAVY, spaceAfter=2,
    )
    sub = ParagraphStyle(
        "sub", fontName="Helvetica", fontSize=font_size - 0.6, leading=leading,
        textColor=GREY, spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=font_size + 1.8,
        leading=leading + 2.2, textColor=NAVY, spaceBefore=5, spaceAfter=3,
    )
    bullet = ParagraphStyle(
        "bullet", parent=body, leftIndent=10, bulletIndent=2, spaceAfter=1.8,
    )
    cell = ParagraphStyle(
        "cell", fontName="Helvetica", fontSize=font_size - 1.1,
        leading=leading - 1.2, textColor=colors.HexColor("#111111"),
    )
    cellh = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    caption = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique", fontSize=font_size - 1.4,
        leading=leading - 1, textColor=GREY, spaceAfter=5,
    )

    story: list = []
    lines = SRC.read_text(encoding="utf-8").split("\n")
    i = 0
    first_h2 = True
    last_section = ""
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("| "):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = parse_table(block)
            if rows:
                ncols = max(len(r) for r in rows)
                rows = [r + [""] * (ncols - len(r)) for r in rows]
                data = []
                for r_idx, row in enumerate(rows):
                    style = cellh if r_idx == 0 else cell
                    data.append([Paragraph(esc(c), style) for c in row])
                width = A4[0] - 3.0 * cm
                lens = [max(len(r[c]) for r in rows) for c in range(ncols)]
                total = sum(lens) or 1
                widths = [max(width * 0.10, width * ln / total) for ln in lens]
                scale = width / sum(widths)
                widths = [w * scale for w in widths]
                table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                        ]
                    )
                )
                story.append(Spacer(1, 2))
                story.append(table)
                story.append(Spacer(1, 5))
            continue
        if line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), h1))
        elif line.startswith("## "):
            title = line[3:]
            if not first_h2:
                from reportlab.platypus import PageBreak

                # The executive summary gets page 1 to itself (the case asks for
                # a one-page summary on top). After that the note flows, because
                # seven forced page breaks wasted half a page each.
                if "executive summary" in last_section.lower():
                    story.append(PageBreak())
            first_h2 = False
            last_section = title
            story.append(Paragraph(esc(title), h2))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceAfter=4))
        elif line.startswith("- "):
            story.append(Paragraph(esc(line[2:]), bullet, bulletText="\u2022"))
        elif line.startswith("*") and line.endswith("*") and len(line) > 2:
            story.append(Paragraph(esc(line.strip("*")), caption))
        else:
            story.append(Paragraph(esc(line), body))
        i += 1

    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.45 * cm, rightMargin=1.45 * cm,
        topMargin=1.15 * cm, bottomMargin=1.15 * cm,
        title="Private AI platform on one RTX 6000 - design note",
        author="Utkarsh Dwivedi",
    )

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(1.45 * cm, 0.55 * cm,
                          "CGPRR group / NEET.info / Hirrx - case response, 26 November 2026")
        canvas.drawRightString(A4[0] - 1.45 * cm, 0.55 * cm, f"Page {doc_.page} of 6")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return doc.page


def main() -> None:
    for size, leading in ((9.3, 11.4), (9.0, 11.0), (8.7, 10.6), (8.4, 10.2), (8.1, 9.9), (7.8, 9.6)):
        pages = build(size, leading)
        print(f"font {size}pt / leading {leading} -> {pages} pages")
        if pages <= 6:
            print(f"kept {size}pt")
            return
    print("still over 6 pages")


if __name__ == "__main__":
    main()
