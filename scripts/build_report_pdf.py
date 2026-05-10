from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "final_report.md"
OUTPUT = ROOT / "docs" / "final_report.pdf"

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "DejaVu Sans",
    }
)

PAGE_W, PAGE_H = 8.27, 11.69
LEFT = 0.55
TOP = 10.82
BOTTOM = 0.62
LINE = 0.155
TITLE_COLOR = "#174A7C"
TEXT_COLOR = "#111827"
MUTED = "#64748B"


def clean_inline(text: str) -> str:
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    return text


def extract_title(markdown: str) -> str:
    for raw in markdown.splitlines():
        stripped = raw.strip()
        if stripped.startswith("# "):
            return clean_inline(stripped[2:])
    return "Heart Disease MLOps Pipeline"


def markdown_to_lines(markdown: str) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    in_code = False

    for raw in markdown.splitlines():
        stripped = raw.rstrip()

        if stripped.startswith("```"):
            in_code = not in_code
            if not in_code:
                lines.append(("blank", ""))
            continue

        if in_code:
            for wrapped in textwrap.wrap(stripped, width=98, replace_whitespace=False) or [""]:
                lines.append(("code", wrapped))
            continue

        if not stripped:
            if lines and lines[-1][0] != "blank":
                lines.append(("blank", ""))
            continue

        if stripped.startswith("# "):
            # The document title is already printed in the page header.
            continue

        if stripped.startswith("## "):
            lines.append(("h2", clean_inline(stripped[3:])))
            continue

        if stripped.startswith("### "):
            lines.append(("h3", clean_inline(stripped[4:])))
            continue

        if stripped.startswith("- "):
            item = clean_inline(stripped[2:])
            wrapped = textwrap.wrap(item, width=96, subsequent_indent="  ")
            for i, part in enumerate(wrapped):
                lines.append(("bullet", f"- {part}" if i == 0 else f"  {part}"))
            continue

        if stripped.startswith("|"):
            lines.append(("code", stripped))
            continue

        paragraph = clean_inline(stripped)
        wrapped = textwrap.wrap(paragraph, width=100, break_long_words=False)
        for part in wrapped:
            lines.append(("body", part))

    while lines and lines[-1][0] == "blank":
        lines.pop()
    return lines


def draw_page(pdf: PdfPages, title: str, page_lines: list[tuple[str, str]], page_number: int) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(
        LEFT / PAGE_W,
        11.25 / PAGE_H,
        title,
        fontsize=13,
        fontweight="bold",
        color=TITLE_COLOR,
        va="top",
    )
    ax.plot([LEFT / PAGE_W, 7.75 / PAGE_W], [11.0 / PAGE_H, 11.0 / PAGE_H], color="#2F7D5C", lw=1.0)

    y = TOP
    for kind, text in page_lines:
        if kind == "blank":
            y -= LINE * 0.55
            continue

        if kind == "h1":
            size, weight, color = 13.0, "bold", TITLE_COLOR
            y -= LINE * 0.35
        elif kind == "h2":
            size, weight, color = 11.2, "bold", TITLE_COLOR
            y -= LINE * 0.25
        elif kind == "h3":
            size, weight, color = 9.8, "bold", "#2F7D5C"
        elif kind == "code":
            size, weight, color = 7.5, "normal", "#334155"
        else:
            size, weight, color = 8.35, "normal", TEXT_COLOR

        ax.text(LEFT / PAGE_W, y / PAGE_H, text, fontsize=size, fontweight=weight, color=color, va="top")
        y -= LINE

    ax.plot([LEFT / PAGE_W, 7.75 / PAGE_W], [0.45 / PAGE_H, 0.45 / PAGE_H], color="#E5E7EB", lw=0.8)
    ax.text(LEFT / PAGE_W, 0.28 / PAGE_H, "Generated from docs/final_report.md", fontsize=7, color=MUTED)
    ax.text(7.75 / PAGE_W, 0.28 / PAGE_H, str(page_number), fontsize=7, color=MUTED, ha="right")

    pdf.savefig(fig)
    plt.close(fig)


def paginate(lines: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    pages: list[list[tuple[str, str]]] = []
    page: list[tuple[str, str]] = []
    y = TOP

    for line in lines:
        kind = line[0]
        needed = LINE * 0.55 if kind == "blank" else LINE
        if kind in {"h1", "h2"}:
            needed += LINE * 0.45

        if page and y - needed < BOTTOM:
            pages.append(page)
            page = []
            y = TOP

        page.append(line)
        y -= needed

    if page:
        pages.append(page)
    return pages


def build_pdf() -> Path:
    markdown = SOURCE.read_text(encoding="utf-8")
    title = extract_title(markdown)
    lines = markdown_to_lines(markdown)
    pages = paginate(lines)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        for number, page_lines in enumerate(pages, start=1):
            draw_page(pdf, title, page_lines, number)
    return OUTPUT


if __name__ == "__main__":
    print(build_pdf())
