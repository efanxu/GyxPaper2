from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create same-resolution visual comparisons for the ablation figure."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = Image.open(args.reference).convert("RGB")
    render = Image.open(args.render).convert("RGB")
    resized_reference = reference.resize(render.size, Image.Resampling.LANCZOS)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = args.output_dir / "reference_same_resolution.png"
    overlay_path = args.output_dir / "reference_render_overlay_50pct.png"
    difference_path = args.output_dir / "reference_render_difference.png"
    report_path = args.output_dir / "visual_qa_report.json"

    resized_reference.save(reference_path)
    Image.blend(resized_reference, render, 0.5).save(overlay_path)
    difference = ImageEnhance.Contrast(
        ImageChops.difference(resized_reference, render)
    ).enhance(2.0)
    difference.save(difference_path)

    report = {
        "reference_source": str(args.reference.resolve()),
        "render_source": str(args.render.resolve()),
        "reference_original_size_px": list(reference.size),
        "render_size_px": list(render.size),
        "reference_original_aspect_ratio": reference.width / reference.height,
        "render_aspect_ratio": render.width / render.height,
        "same_resolution_reference": str(reference_path.resolve()),
        "overlay_50pct": str(overlay_path.resolve()),
        "absolute_difference": str(difference_path.resolve()),
        "intentional_content_difference": (
            "The formal render omits the reference preview's Fig. X caption."
        ),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
