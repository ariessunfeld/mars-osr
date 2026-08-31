#!/usr/bin/env python3
"""Compose the two independently generated Figure 8 panel families.

The scientific curves are rendered by ``figure_08_ring_densification.py`` and
``figure_08_ltan_rings.py``.  This script replaces the manuscript's manual
side-by-side image assembly with a deterministic, title-free composition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT = (
    REPOSITORY_ROOT
    / "simulation_outputs"
    / "manuscript_figures"
    / "figure_08a_ring_densification.png"
)
DEFAULT_RIGHT = (
    REPOSITORY_ROOT
    / "simulation_outputs"
    / "manuscript_figures"
    / "figure_08b_ltan_rings.png"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "figures"
    / "manuscript"
    / "generated"
    / "figure_08_constellation_scaling.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-px", type=int, default=36)
    parser.add_argument("--padding-px", type=int, default=8)
    return parser.parse_args()


def content_crop(image: Image.Image, padding_px: int) -> Image.Image:
    """Crop uniform white margins while retaining a small deterministic pad."""

    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background)
    box = difference.getbbox()
    if box is None:
        raise ValueError("component image contains no non-white content")
    left, top, right, bottom = box
    padded = (
        max(0, left - padding_px),
        max(0, top - padding_px),
        min(rgb.width, right + padding_px),
        min(rgb.height, bottom + padding_px),
    )
    return rgb.crop(padded)


def compose(left_path: Path, right_path: Path, output: Path, gap_px: int, padding_px: int) -> None:
    if gap_px < 0 or padding_px < 0:
        raise ValueError("gap and padding must be nonnegative")
    left = content_crop(Image.open(left_path), padding_px)
    right = content_crop(Image.open(right_path), padding_px)

    target_height = max(left.height, right.height)
    if left.height != target_height:
        left = left.resize(
            (round(left.width * target_height / left.height), target_height),
            Image.Resampling.LANCZOS,
        )
    if right.height != target_height:
        right = right.resize(
            (round(right.width * target_height / right.height), target_height),
            Image.Resampling.LANCZOS,
        )

    canvas = Image.new("RGB", (left.width + gap_px + right.width, target_height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap_px, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    print(f"wrote {output} ({canvas.width}x{canvas.height} px)")


def main() -> None:
    args = parse_args()
    compose(args.left, args.right, args.output, args.gap_px, args.padding_px)


if __name__ == "__main__":
    main()
