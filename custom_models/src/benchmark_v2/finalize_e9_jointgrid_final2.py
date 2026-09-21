"""Finalize the E9 joint-grid figures from the already-rendered final SVGs.

This is a graphics-only post-processing step.  It intentionally reads the
existing ``*_final.svg`` files, including their embedded scatter raster
layers, and never loads a checkpoint, runs inference, evaluates metrics, or
changes panel text.  The embedded point cloud is contrast-enhanced and
slightly dilated so the existing real points remain visible at paper scale.
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter


IMAGE_PATTERN = re.compile(
    r'(?P<prefix><image\b[^>]*?xlink:href="data:image/png;base64,)'
    r'(?P<data>[^"]+)'
    r'(?P<suffix>"[^>]*>)',
    re.DOTALL,
)
AXES_OPEN_PATTERN = re.compile(r'<g id="axes_(?P<number>\d+)">')


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _enhance_scatter_png(payload: str) -> str:
    raw = base64.b64decode(payload)
    image = Image.open(io.BytesIO(raw)).convert("RGBA")
    array = np.asarray(image).copy()
    rgb = array[:, :, :3].astype(np.float32)

    # The existing scatter layer is the GnBu point cloud composited over
    # white.  Work in distance-from-white space so the color family and the
    # shared colorbar meaning are unchanged while the point opacity becomes
    # visible.  The red regression line and neutral black geometry are not in
    # this raster layer, but the guards keep them safe if an exporter changes.
    distance = 255.0 - rgb
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    red_geometry = (rgb[:, :, 0] > rgb[:, :, 1] + 15.0) & (
        rgb[:, :, 0] > rgb[:, :, 2] + 15.0
    )
    point_mask = (
        (chroma >= 4.0)
        & (distance.mean(axis=2) >= 2.0)
        & ~red_geometry
    )

    # One source pixel is expanded to a small, paper-visible marker.  The
    # max-filter is applied channel-wise, preserving a density-dependent
    # green-to-blue appearance instead of painting synthetic uniform dots.
    dilated = maximum_filter(point_mask.astype(np.uint8), size=3, mode="nearest") > 0
    filtered_distance = np.stack(
        [maximum_filter(distance[:, :, channel], size=3, mode="nearest") for channel in range(3)],
        axis=2,
    )
    candidate = dilated & ~red_geometry & (filtered_distance.mean(axis=2) >= 2.0)
    boosted_distance = np.clip(filtered_distance * 1.70, 0.0, 255.0)
    enhanced_rgb = np.clip(255.0 - boosted_distance, 0.0, 255.0).astype(np.uint8)
    array[:, :, :3][candidate] = enhanced_rgb[candidate]

    return _encode_png(Image.fromarray(array, mode="RGBA"))


def _shift_panel_axes(svg: str) -> str:
    # Add only a small amount of breathing space.  The center column/row stays
    # fixed, so the compact overall composition and the 5 x 3 order remain.
    column_shift = (-3.0, 0.0, 3.0)
    row_shift = (-2.0, -1.0, 0.0, 1.0, 2.0)

    def replace(match: re.Match[str]) -> str:
        number = int(match.group("number"))
        if not 1 <= number <= 45:
            return match.group(0)
        panel_index = (number - 1) // 3
        row, column = divmod(panel_index, 3)
        dx = column_shift[column]
        dy = row_shift[row]
        return f'<g id="axes_{number}" transform="translate({dx:g} {dy:g})">'

    return AXES_OPEN_PATTERN.sub(replace, svg)


def _increase_readability(svg: str) -> str:
    # These are small typographic increases relative to the existing final
    # figure; positions, metric strings, panel labels, and axes are retained.
    replacements = (
        ("font-size: 14px", "font-size: 15px"),
        ("font-size: 11.4px", "font-size: 12px"),
        ("font-size: 10.2px", "font-size: 10.8px"),
        ("font-size: 10.8px", "font-size: 11.2px"),
        ("font-size: 9.5px", "font-size: 10px"),
        ("font-size: 11px", "font-size: 12px"),
        ("font-size: 9px", "font-size: 9.5px"),
    )
    for old, new in replacements:
        svg = svg.replace(old, new)
    return svg


def finalize_svg(input_path: Path, output_path: Path) -> dict[str, int | str]:
    svg = input_path.read_text(encoding="utf-8")
    matches = list(IMAGE_PATTERN.finditer(svg))
    if len(matches) < 16:
        raise RuntimeError(
            f"Expected 15 scatter images plus a colorbar image in {input_path}; "
            f"found {len(matches)} embedded PNGs."
        )

    image_index = 0

    def replace_image(match: re.Match[str]) -> str:
        nonlocal image_index
        current = image_index
        image_index += 1
        payload = match.group("data")
        if current < 15:
            payload = _enhance_scatter_png(payload)
        return match.group("prefix") + payload + match.group("suffix")

    svg = IMAGE_PATTERN.sub(replace_image, svg)
    svg = _shift_panel_axes(svg)
    svg = _increase_readability(svg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return {
        "source": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "embedded_pngs": len(matches),
        "enhanced_scatter_pngs": 15,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create E9 final2 SVGs by post-processing the existing final SVGs."
    )
    parser.add_argument("--figure-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    figure_dir = args.figure_dir.expanduser().resolve()
    stems = (
        "E9_Original_MaskedMSE_prediction_jointgrid_H3_H6_H10",
        "E9_MS_MG_DWU_prediction_jointgrid_H3_H6_H10",
    )
    for stem in stems:
        source = figure_dir / f"{stem}_final.svg"
        output = figure_dir / f"{stem}_final2.svg"
        if not source.is_file():
            raise SystemExit(f"Missing existing final SVG: {source}")
        result = finalize_svg(source, output)
        print(
            f"FINAL2_SVG {result['output']} "
            f"scatter_layers={result['enhanced_scatter_pngs']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
