#!/usr/bin/env python3
"""Export only the two Virchow2 rows and columns used in the paper table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cpathogen.endpoints.paper_xai import PAPER_COLUMNS, build_paper_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Endpoint-model output root containing models/virchow2.",
    )
    return parser.parse_args()


def _markdown_table(row: dict[str, str]) -> list[str]:
    header = "| " + " | ".join(PAPER_COLUMNS) + " |"
    divider = "|" + "|".join("---" for _ in PAPER_COLUMNS) + "|"
    values = "| " + " | ".join(row[column] for column in PAPER_COLUMNS) + " |"
    return [header, divider, values]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    model_dir = output_root / "models" / "virchow2"
    rows = build_paper_rows(model_dir)

    output_dir = model_dir / "paper_table"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown: list[str] = []
    latex: list[str] = []
    filenames = {
        "PAM50 Classification": "virchow2_pam50_row.csv",
        "Overall Survival": "virchow2_survival_row.csv",
    }
    for task, row in rows.items():
        pd.DataFrame([row], columns=PAPER_COLUMNS).to_csv(
            output_dir / filenames[task], index=False
        )
        markdown.extend((f"### {task}", "", *_markdown_table(row), ""))
        latex.extend(
            (
                f"% {task}",
                " & ".join(row[column] for column in PAPER_COLUMNS) + " \\\\",
            )
        )

    rendered = "\n".join(markdown).rstrip() + "\n"
    (output_dir / "virchow2_paper_rows.md").write_text(rendered, encoding="utf-8")
    (output_dir / "virchow2_paper_rows.tex").write_text(
        "\n".join(latex) + "\n", encoding="utf-8"
    )
    print(rendered, end="")


if __name__ == "__main__":
    main()
