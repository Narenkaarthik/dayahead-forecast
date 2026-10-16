"""Saving result tables: a CSV with full precision and a Markdown copy for the README.

scripts/report.py copies the Markdown files into README.md, so every number in the README
comes from one of these generated files and none is typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# A cell "looks like a number" if it starts with a digit or a minus sign ("61.6%",
# "-7.1 [-8.9, -5.4]", "0.0228"), or is the "–" placeholder for "not applicable".
NUMBER_LIKE = r"^(-?\d|–$)"


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub-style Markdown table: number columns right-aligned, label columns left-aligned."""
    numeric = [df[c].astype(str).str.match(NUMBER_LIKE).all() for c in df.columns]
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    align = "|" + "|".join("---:" if num else "---" for num in numeric) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, align, *body])


def save_table(stem: str | Path, data: pd.DataFrame, display: pd.DataFrame | None = None,
               note: str = "") -> None:
    """Write `data` to <stem>.csv and a Markdown table to <stem>.md.

    `display` is an optional copy with numbers already formatted as text (e.g. "61.6%");
    without it the Markdown shows `data` as is. `note` is a line printed above the table.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.6g", encoding="utf-8")
    shown = data if display is None else display
    text = (note + "\n\n" if note else "") + markdown_table(shown) + "\n"
    stem.with_suffix(".md").write_text(text, encoding="utf-8")
