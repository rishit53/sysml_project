"""Build report/M1_final_report.pdf from results/ (run `make all` first, or at least the experiments and figures).

    python report/build_report.py
Edit AUTHORS / INSTRUCTOR / COURSE below before submitting.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer  # noqa: E402
from rl import GREY, P, ST, W  # noqa: E402

import sections_baseline as B  # noqa: E402
import sections_design as D  # noqa: E402
from analysis import facts as N  # noqa: E402

AUTHORS = "Rishit, Shozab, Spandan R."          # EDIT
INSTRUCTOR = "Abhijit Sir"                      # EDIT
COURSE = "SysML course project P07: On-chip memory hierarchy design for LLM inference accelerators"
DATE = "November 2026"
TITLE = "Prefix-cached decode on accelerators: scratchpad, cache and a time-window multicast global buffer"
OUT = HERE / "M1_final_report.pdf"


def deco(c, d):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(GREY)
    if d.page > 1:
        c.drawString(2 * cm, A4[1] - 1.3 * cm, "Prefix-cached decode on accelerators")
        c.drawRightString(A4[0] - 2 * cm, A4[1] - 1.3 * cm, "P07 · SysML")
        c.setStrokeColor(GREY)
        c.setLineWidth(0.3)
        c.line(2 * cm, A4[1] - 1.45 * cm, A4[0] - 2 * cm, A4[1] - 1.45 * cm)
        c.drawCentredString(A4[0] / 2, 1.1 * cm, f"{d.page}")
    c.restoreState()


def cover(h):
    toc = ["1 Introduction", "2 Background", "3 How we found the gap", "4 Methodology", "5 Evaluating existing on-chip configurations",
           "6 Proposed design: a time-window multicast global buffer", "7 Discussion", "8 Conclusion", "Appendices: reproducibility, claims ledger, references"]
    return [Spacer(1, 2.2 * cm), P(TITLE, "title"), P(COURSE, "sub"), Spacer(1, 0.2 * cm), P(f"{AUTHORS}", "sub"), P(f"Instructor: {INSTRUCTOR} · {DATE}", "sub"),
            Spacer(1, 0.7 * cm), *D.abstract(h), Spacer(1, 0.6 * cm), P("<b>Contents</b>", "body")] + [P(t, "note") for t in toc] + [PageBreak()]


def main():
    h = N.headline()
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.1 * cm, bottomMargin=1.8 * cm,
                            title=TITLE, author=AUTHORS, subject=COURSE)
    story = cover(h) + B.intro() + B.background() + B.trail() + B.method(h) + B.baseline(h) + D.design(h) + D.discussion(h) + D.appendices(h)
    doc.build(story, onFirstPage=deco, onLaterPages=deco)
    print(f"wrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
