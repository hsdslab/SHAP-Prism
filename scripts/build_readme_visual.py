#!/usr/bin/env python3
"""Build the compact README overview from the approved graphical-abstract assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "shap_prism_readme_overview.png"
STANDARD = (
    ROOT
    / "docs"
    / "assets"
    / "standard_shap_summary.png"
)
PRISM = (
    ROOT
    / "docs"
    / "assets"
    / "shap_prism_package_example.png"
)

WIDTH = 3000
HEIGHT = 1200
INK = "#202631"
MUTED = "#667085"
PURPLE = "#7655C5"
PINK = "#EF4D8B"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError as error:
        raise RuntimeError(
            "DejaVu Sans is required to rebuild the README visual"
        ) from error


def _contain(image: Image.Image, box: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(box, Image.Resampling.LANCZOS)
    return copy


def build(output: Path) -> Path:
    """Render and structurally verify the 5:2 README overview."""

    if not STANDARD.is_file() or not PRISM.is_file():
        raise FileNotFoundError("Approved graphical-abstract source assets are missing")

    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)

    standard = _contain(Image.open(STANDARD).convert("RGB"), (660, 850))
    standard_x = 54 + (700 - standard.width) // 2
    standard_y = 142 + (850 - standard.height) // 2
    canvas.paste(standard, (standard_x, standard_y))

    prism = _contain(Image.open(PRISM).convert("RGBA"), (1640, 1170))
    prism_x = WIDTH - prism.width - 34
    prism_y = (HEIGHT - prism.height) // 2
    canvas.paste(prism, (prism_x, prism_y), prism)

    draw.text(
        (54, 58),
        "STANDARD SHAP SUMMARY",
        font=_font(40, bold=True),
        fill=MUTED,
    )
    draw.multiline_text(
        (76, 1028),
        "category levels unlabeled\nsubgroup distributions pooled",
        font=_font(28),
        fill=MUTED,
        spacing=12,
    )

    center_left = 800
    center_right = prism_x - 68
    center = (center_left + center_right) // 2
    draw.text(
        (center, 336),
        "SHAP PRISM",
        anchor="mm",
        font=_font(52, bold=True),
        fill=INK,
    )

    arrow_y = 494
    draw.rounded_rectangle(
        (center_left + 12, arrow_y - 11, center_right - 42, arrow_y + 11),
        radius=11,
        fill=PURPLE,
    )
    draw.polygon(
        [
            (center_right - 44, arrow_y - 37),
            (center_right + 8, arrow_y),
            (center_right - 44, arrow_y + 37),
        ],
        fill=PURPLE,
    )
    draw.ellipse(
        (center_left - 2, arrow_y - 16, center_left + 30, arrow_y + 16),
        fill=PINK,
    )

    draw.text(
        (center, 570),
        "SAME SHAP VALUES",
        anchor="mm",
        font=_font(31, bold=True),
        fill=PURPLE,
    )
    draw.multiline_text(
        (center, 635),
        "reorganized by category\nand subgroup",
        anchor="ma",
        align="center",
        font=_font(27),
        fill=MUTED,
        spacing=8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    with Image.open(output) as rendered:
        rendered.verify()
    with Image.open(output) as rendered:
        if rendered.size != (WIDTH, HEIGHT):
            raise ValueError(f"Unexpected README visual size: {rendered.size}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args().output))
