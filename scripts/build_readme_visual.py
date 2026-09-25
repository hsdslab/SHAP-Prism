#!/usr/bin/env python3
"""Export the approved, normal-size graphical abstract for GitHub and PyPI.

The source PDF is supplied locally and is never copied into the public tree.
Install PyMuPDF to rebuild this documentation asset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "shap_prism_readme_overview.png"
WIDTH = 3000
HEIGHT = 1200


def build(source_pdf: Path, output: Path) -> Path:
    """Render the complete approved 5:2 PDF page without resizing its labels."""

    import fitz

    if "enlarged" in source_pdf.stem.casefold():
        raise ValueError("Use the normal graphical abstract, not the enlarged variant")
    with fitz.open(source_pdf) as document:
        if len(document) != 1:
            raise ValueError("The graphical abstract must contain exactly one page")
        page = document[0]
        if abs(page.rect.width / page.rect.height - WIDTH / HEIGHT) > 1e-6:
            raise ValueError("The graphical abstract must have a 5:2 aspect ratio")
        scale = WIDTH / page.rect.width
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(output)
    with Image.open(output) as rendered:
        rendered.verify()
    with Image.open(output) as rendered:
        if rendered.size != (WIDTH, HEIGHT):
            raise ValueError(f"Unexpected README visual size: {rendered.size}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build(args.source_pdf, args.output))
