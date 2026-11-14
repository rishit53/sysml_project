"""reportlab helpers: fonts, styles, figures, tables (kept separate so the report text stays readable)."""
from pathlib import Path

import matplotlib
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figures"
fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for n, f in (("DV", "DejaVuSans"), ("DV-B", "DejaVuSans-Bold"), ("DV-I", "DejaVuSans-Oblique"), ("DV-BI", "DejaVuSans-BoldOblique")):
    pdfmetrics.registerFont(TTFont(n, str(fd / f"{f}.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

NAVY, GREY, LIGHT = colors.HexColor("#1F3A5F"), colors.HexColor("#6B7280"), colors.HexColor("#F3F5F8")
W = A4[0] - 4 * cm
ST = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.2, leading=13.4, alignment=TA_JUSTIFY, spaceAfter=5),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=14.5, leading=18, textColor=NAVY, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=10.8, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10.6, textColor=GREY, spaceBefore=2, spaceAfter=9),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=9.8),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=7.8, leading=9.8, textColor=colors.white),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=21, leading=26, textColor=NAVY, spaceAfter=10),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=11, leading=15, textColor=GREY, spaceAfter=6),
    "abs": ParagraphStyle("abs", fontName="DV", fontSize=9.2, leading=13.4, alignment=TA_JUSTIFY),
    "note": ParagraphStyle("note", fontName="DV", fontSize=8.2, leading=11.4, textColor=colors.HexColor("#444444"), alignment=TA_JUSTIFY, spaceAfter=4),
}


def P(t, s="body"):
    return Paragraph(t, ST[s])


def H1(t):
    return P(t, "h1")


def H2(t):
    return P(t, "h2")


def bullets(items, s="body"):
    return ListFlowable([ListItem(P(i, s), leftIndent=12, value="•") for i in items], bulletType="bullet", start="•", leftIndent=12, bulletFontSize=8)


def fig(name, caption, width=W):
    p = FIGS / f"{name}.png"
    w, h = PILImage.open(p).size
    return KeepTogether([Image(str(p), width=width, height=width * h / w), P(caption, "cap")])


def table(df, widths=None, caption=None, align_first_left=True):
    rows = [[Paragraph(str(c), ST["cellb"]) for c in df.columns]]
    rows += [[Paragraph(str(x), ST["cell"]) for x in r] for r in df.itertuples(index=False)]
    n = len(df.columns)
    widths = widths or [W / n] * n
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DAE1")),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
    out = [t]
    if caption:
        out.append(P(caption, "cap"))
    else:
        out.append(Spacer(1, 6))
    return KeepTogether(out)


def callout(text):
    t = Table([[Paragraph(text, ST["abs"])]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.6, NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t
