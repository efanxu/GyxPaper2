"""Generate the ST-MGPrompt paper figures as editable SVG, vector PDF, and PNG.

The layout is deliberately hand-controlled: the figures explain the model idea
and preserve the executable canonical configuration rather than mirroring class
names or implementation details.
"""

from __future__ import annotations

from html import escape, unescape
from pathlib import Path
import math
import re
import subprocess
import sys


OUT = Path(__file__).resolve().parent
FONT = "Arial"
FONT_BOLD = "Arial"

INK = "#253244"
MUTED = "#607086"
LINE = "#91A0B2"
GRID = "#D9E2EC"
SHARED = "#E8EEF5"
SHARED_STROKE = "#788DA5"
FINE = "#F4A340"
FINE_DARK = "#C97812"
FINE_FILL = "#FFF3DE"
COARSE = "#3E9C9A"
COARSE_DARK = "#277573"
COARSE_FILL = "#E5F4F1"
PURPLE = "#8C6BB1"
PURPLE_DARK = "#624486"
PURPLE_FILL = "#F1EBF8"
LOSS = "#B86770"
LOSS_FILL = "#FBECEE"
WHITE = "#FFFFFF"


class SVG:
    def __init__(self, width: int, height: int, title: str, physical_width_mm: float | None = None):
        self.width = width
        self.height = height
        self.parts: list[str] = []
        if physical_width_mm is None:
            display_width = f"{width}px"
            display_height = f"{height}px"
        else:
            display_width = f"{physical_width_mm:g}mm"
            display_height = f"{physical_width_mm * height / width:g}mm"
        self.parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        )
        self.parts.append(f"<title id=\"title\">{escape(title)}</title>")
        self.parts.append('<desc id="desc">Academic vector diagram for the ST-MGPrompt forecasting model.</desc>')
        self.parts.append(
            f'<style>text{{font-family:{FONT},Helvetica,Arial,sans-serif;fill:{INK};}} '
            f'.muted{{fill:{MUTED};}} .small{{font-size:13px;}} .tiny{{font-size:11px;}} '
            f'.label{{font-size:14px;font-weight:700;letter-spacing:.2px;}} '
            f'.title{{font-size:23px;font-weight:700;}} .section{{font-size:12px;font-weight:700;letter-spacing:1.1px;}} '
            f'.math{{font-family:{FONT},Arial,sans-serif;}}</style>'
        )
        self.parts.append(
            '<defs>'
            '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            '<path d="M0,0 L8,4 L0,8 Z" fill="#536579"/></marker>'
            '<marker id="arrowFine" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{FINE_DARK}"/></marker>'
            '<marker id="arrowCoarse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{COARSE_DARK}"/></marker>'
            '<marker id="arrowPurple" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{PURPLE_DARK}"/></marker>'
            '<marker id="arrowLoss" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L8,4 L0,8 Z" fill="{LOSS}"/></marker>'
            '</defs>'
        )

    def rect(self, x, y, w, h, fill=WHITE, stroke=LINE, sw=1.5, rx=10, dash=None, opacity=1):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"{da}/>'
        )

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.5, dash=None, marker=None, opacity=1):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{da}{mk}/>'
        )

    def path(self, d, stroke=LINE, sw=1.5, fill="none", dash=None, marker=None, opacity=1):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#{marker})"' if marker else ""
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{da}{mk}/>'
        )

    def circle(self, cx, cy, r, fill=WHITE, stroke=LINE, sw=1.2):
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    @staticmethod
    def visual_font_size(size):
        """Map the design-grid size to a publication-size SVG/PDF font.

        The figures use a 2400-unit design grid but are exported at 180 mm.
        Without this mapping, a visually convenient 22-unit annotation would
        become less than 5 pt in the placed PDF.  The tiers keep titles
        restrained while lifting secondary labels above the journal-readable
        threshold.
        """
        if size >= 50:
            return round(size * 1.20, 1)
        if size >= 30:
            return round(size * 1.20, 1)
        return max(35.5, round(size * 1.35, 1))

    def text(self, x, y, value, size=14, color=INK, weight=400, anchor="start", italic=False, letter=0, rotate=None):
        visual_size = self.visual_font_size(size)
        attrs = [f'x="{x}"', f'y="{y}"', f'font-size="{visual_size}px"', f'fill="{color}"', f'font-weight="{weight}"', f'text-anchor="{anchor}"']
        if italic:
            attrs.append('font-style="italic"')
        if letter:
            attrs.append(f'letter-spacing="{letter}px"')
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate} {x} {y})"')
        attr_text = " ".join(attrs)
        self.parts.append(f'<text {attr_text}>{escape(str(value))}</text>')

    def multiline(self, x, y, lines, size=14, color=INK, weight=400, anchor="middle", gap=None, italic=False):
        gap = gap or round(self.visual_font_size(size) * 1.22)
        for i, value in enumerate(lines):
            self.text(x, y + i * gap, value, size=size, color=color, weight=weight, anchor=anchor, italic=italic)

    def end(self):
        self.parts.append("</svg>")
        return "\n".join(self.parts)


def measure_text(value, font_size, weight=400, font_path=None):
    """Measure text with the actual publication font metrics.

    The design grid is expressed in SVG user units.  The returned values are
    therefore also in user units and can be used directly for bbox QA before
    the SVG is converted to a 180 mm PDF.
    """
    from PIL import ImageFont

    path = font_path or (r"C:\Windows\Fonts\arialbd.ttf" if int(weight) >= 700 else r"C:\Windows\Fonts\arial.ttf")
    try:
        font = ImageFont.truetype(path, max(1, int(round(font_size))))
        lines = str(value).splitlines() or [""]
        widths = [font.getlength(line) for line in lines]
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        return {
            "width": max(widths) if widths else 0.0,
            "height": line_height * len(lines),
            "ascent": ascent,
            "descent": descent,
            "line_height": line_height,
        }
    except Exception:
        lines = str(value).splitlines() or [""]
        return {
            "width": max((len(line) for line in lines), default=0) * font_size * 0.56,
            "height": font_size * 1.2 * len(lines),
            "ascent": font_size * 0.9,
            "descent": font_size * 0.25,
            "line_height": font_size * 1.2,
        }


class V3SVG(SVG):
    """SVG writer with publication-layout bbox bookkeeping for refined_v3."""

    def __init__(self, width: int, height: int, title: str, physical_width_mm: float = 180.0):
        super().__init__(width, height, title, physical_width_mm=physical_width_mm)
        self.qa_text = []
        self.qa_shapes = []
        self.qa_lines = []
        self._qa_index = 0

    def _next_id(self, prefix):
        self._qa_index += 1
        return f"{prefix}_{self._qa_index:04d}"

    def _record_text(self, x, y, value, size, weight, anchor):
        visual_size = self.visual_font_size(size)
        metrics = measure_text(value, visual_size, weight=weight)
        width = metrics["width"]
        if anchor == "middle":
            x_min, x_max = x - width / 2.0, x + width / 2.0
        elif anchor == "end":
            x_min, x_max = x - width, x
        else:
            x_min, x_max = x, x + width
        # PIL's ascent/descent are line-box metrics.  Use the measured font
        # metrics with an ink-area factor for collision geometry; this avoids
        # treating the whitespace above/below an SVG baseline as printed ink.
        self.qa_text.append({
            "element_id": self._next_id("text"),
            "element_type": "text",
            "value": str(value),
            "bbox": (x_min, y - metrics["ascent"] * 0.82, x_max, y + metrics["descent"] * 0.55),
            "font_px": visual_size,
        })

    def text(self, x, y, value, size=14, color=INK, weight=400, anchor="start", italic=False, letter=0, rotate=None):
        self._record_text(x, y, value, size, weight, anchor)
        return super().text(x, y, value, size=size, color=color, weight=weight, anchor=anchor,
                            italic=italic, letter=letter, rotate=rotate)

    def multiline(self, x, y, lines, size=14, color=INK, weight=400, anchor="middle", gap=None, italic=False):
        # Explicit v3 gaps are expressed on the design grid.  Keep them
        # proportional to the publication font so measured bboxes do not
        # overlap when the SVG is placed at 180 mm.
        actual_gap = self.visual_font_size(size) * 1.22 if gap is None else max(float(gap) * 1.20, self.visual_font_size(size) * 1.12)
        for i, value in enumerate(lines):
            self.text(x, y + i * actual_gap, value, size=size, color=color, weight=weight, anchor=anchor, italic=italic)

    def rect(self, x, y, w, h, fill=WHITE, stroke=LINE, sw=1.5, rx=10, dash=None, opacity=1):
        result = super().rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx, dash=dash, opacity=opacity)
        if float(x) == 0 and float(y) == 0 and float(w) == float(self.width) and float(h) == float(self.height):
            return result
        self.qa_shapes.append({
            "element_id": self._next_id("rect"),
            "element_type": "rect",
            "bbox": (float(x), float(y), float(x + w), float(y + h)),
            "radius": float(rx),
        })
        return result

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.5, dash=None, marker=None, opacity=1):
        result = super().line(x1, y1, x2, y2, stroke=stroke, sw=sw, dash=dash, marker=marker, opacity=opacity)
        self.qa_lines.append({
            "element_id": self._next_id("line"),
            "element_type": "arrow" if marker else "line",
            "segment": (float(x1), float(y1), float(x2), float(y2)),
            "sw": float(sw),
        })
        return result

    def path(self, d, stroke=LINE, sw=1.5, fill="none", dash=None, marker=None, opacity=1):
        result = super().path(d, stroke=stroke, sw=sw, fill=fill, dash=dash, marker=marker, opacity=opacity)
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
        if len(nums) >= 4:
            xs, ys = nums[0::2], nums[1::2]
            self.qa_shapes.append({
                "element_id": self._next_id("path"),
                "element_type": "path",
                "bbox": (min(xs) - float(sw), min(ys) - float(sw), max(xs) + float(sw), max(ys) + float(sw)),
            })
        return result

    def circle(self, cx, cy, r, fill=WHITE, stroke=LINE, sw=1.2):
        result = super().circle(cx, cy, r, fill=fill, stroke=stroke, sw=sw)
        # Tiny graph nodes and pooling dots are decorative sub-elements.  The
        # enclosing graph/card is the recorded main shape for layout QA.
        if float(r) <= 8.0:
            return result
        self.qa_shapes.append({
            "element_id": self._next_id("circle"),
            "element_type": "circle",
            "bbox": (float(cx - r), float(cy - r), float(cx + r), float(cy + r)),
            "center": (float(cx), float(cy)),
            "radius": float(r),
        })
        return result


def _bbox_intersects(a, b, eps=0.0):
    return not (a[2] <= b[0] + eps or b[2] <= a[0] + eps or a[3] <= b[1] + eps or b[3] <= a[1] + eps)


def _bbox_contains(outer, inner, pad=0.0):
    return (inner[0] >= outer[0] + pad and inner[1] >= outer[1] + pad and
            inner[2] <= outer[2] - pad and inner[3] <= outer[3] - pad)


def _circle_contains_bbox(shape, bbox):
    cx, cy = shape["center"]
    r = shape["radius"]
    return all((px - cx) ** 2 + (py - cy) ** 2 <= r ** 2 for px, py in (
        (bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3])
    ))


def _segment_hits_bbox(segment, bbox, padding=0.0):
    """Liang-Barsky segment/rectangle test for text-arrow QA."""
    x1, y1, x2, y2 = segment
    left, top, right, bottom = bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - left, right - x1, y1 - top, bottom - y1)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        t = qi / pi
        if pi < 0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)
    return u1 <= u2


def _segment_overlap_fraction(segment, bbox, padding=0.0):
    """Return the fraction of a segment covered by a padded bbox."""
    x1, y1, x2, y2 = segment
    left, top, right, bottom = bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - left, right - x1, y1 - top, bottom - y1)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return 0.0
            continue
        t = qi / pi
        if pi < 0:
            if t > u2:
                return 0.0
            u1 = max(u1, t)
        else:
            if t < u1:
                return 0.0
            u2 = min(u2, t)
    return max(0.0, u2 - u1) if u1 <= u2 else 0.0


def run_v3_collision_qa(svg: V3SVG, stem: str):
    text_text = []
    text_shape = []
    text_line = []
    for i, left in enumerate(svg.qa_text):
        for right in svg.qa_text[i + 1:]:
            if _bbox_intersects(left["bbox"], right["bbox"], eps=0.5):
                text_text.append((left["element_id"], right["element_id"], left["value"], right["value"]))
        containing_rects = [shape for shape in svg.qa_shapes
                            if shape["element_type"] == "rect" and _bbox_contains(shape["bbox"], left["bbox"], pad=2.0)]
        for shape in svg.qa_shapes:
            if not _bbox_intersects(left["bbox"], shape["bbox"], eps=0.2):
                continue
            if shape["element_type"] == "rect" and containing_rects:
                continue
            if shape["element_type"] == "circle" and _circle_contains_bbox(shape, left["bbox"]):
                continue
            if shape["element_type"] == "circle" and shape.get("radius", 0) >= 10 and _bbox_intersects(left["bbox"], shape["bbox"], eps=0.2):
                continue
            # Waveform paths are internal graphic texture; the explicit line
            # and rectangle checks still catch every label/arrow collision.
            if shape["element_type"] == "path":
                continue
            text_shape.append((left["element_id"], shape["element_id"], left["value"], shape["element_type"]))
        for line in svg.qa_lines:
            # Unmarked lines are waveform/coordinate baselines or other
            # decorative strokes.  Only routed information arrows are
            # collision-critical for publication layout.
            if line["element_type"] != "arrow":
                continue
            if _segment_overlap_fraction(line["segment"], left["bbox"], padding=line["sw"] / 2.0 + 1.0) > 0.75:
                text_line.append((left["element_id"], line["element_id"], left["value"], line["element_type"]))

    canvas_overflow = []
    canvas = (0.0, 0.0, float(svg.width), float(svg.height))
    for item in svg.qa_text + svg.qa_shapes:
        bbox = item["bbox"]
        if not _bbox_contains(canvas, bbox, pad=0.0):
            canvas_overflow.append((item["element_id"], item.get("value", item["element_type"]), bbox))
    box_overflows = list(text_shape)
    scale = (180.0 / 25.4 * 72.0) / float(svg.width)
    min_font_pt = min(item["font_px"] * scale for item in svg.qa_text) if svg.qa_text else 0.0
    report = {
        "TEXT_TEXT_COLLISIONS": len(text_text),
        "TEXT_SHAPE_COLLISIONS": len(text_shape),
        "TEXT_LINE_COLLISIONS": len(text_line),
        "BOX_OVERFLOWS": len(box_overflows),
        "CANVAS_OVERFLOWS": len(canvas_overflow),
        "MIN_FONT_PT": round(min_font_pt, 2),
        "text_text_examples": text_text[:10],
        "text_shape_examples": text_shape[:10],
        "text_line_examples": text_line[:10],
        "canvas_examples": canvas_overflow[:10],
    }
    qa_dir = OUT / "qa_v3"
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / f"{stem}_collision_qa.txt"
    lines = [f"{key}: {value}" for key, value in report.items()]
    qa_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def v3_panel(svg, x, y, w, h, title_lines, number, fill="#FBFCFD", stroke=GRID, title_size=44):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.2, rx=22, dash="10 8" if stroke == GRID else None)
    first = f"{number}  {title_lines[0]}" if number else title_lines[0]
    svg.text(x + 24, y + 42, first, size=title_size, color=stroke if stroke != GRID else INK, weight=700)
    if len(title_lines) > 1:
        svg.text(x + 24, y + 42 + title_size * 1.45, title_lines[1], size=title_size, color=stroke if stroke != GRID else INK, weight=700)


def v3_badge(svg, x, y, w, label, fill=SHARED, color=MUTED, size=30, h=40):
    svg.rect(x, y, w, h, fill=fill, stroke=fill, sw=0, rx=h / 2)
    svg.text(x + w / 2, y + h - 10, label, size=size, color=color, weight=700, anchor="middle")


def v3_sequence(svg, x, y, w, h, kind="fine", count=8, stroke=None, fill=WHITE):
    stroke = stroke or (FINE if kind == "fine" else COARSE)
    dark = FINE_DARK if kind == "fine" else COARSE_DARK
    gap = 8
    card_w = (w - gap * (count - 1)) / count
    for i in range(count):
        xx = x + i * (card_w + gap)
        svg.rect(xx, y, card_w, h, fill=fill, stroke=stroke, sw=1.6, rx=7)
        waveform(svg, xx + card_w * .15, y + h * .22, card_w * .70, h * .48,
                 kind=kind, stroke=dark, baseline=False, opacity=.95)


def v3_tensor(svg, x, y, w, h, label, color=SHARED_STROKE, fill=SHARED):
    for i in range(4, -1, -1):
        dx, dy = i * 9, -i * 7
        svg.rect(x + dx, y + dy, w, h, fill=WHITE if i else fill, stroke=color, sw=1.6, rx=7)
    waveform(svg, x + 18, y + h * .25, w - 36, h * .25, kind="coarse", stroke=color, baseline=False, opacity=.75)
    svg.text(x + w / 2 + 18, y + h * .78, label, size=34, color=INK, weight=700, anchor="middle")


def v3_tcn(svg, x, y, w, h, color, k, dilation):
    for i in range(4):
        svg.rect(x + i * 8, y + i * 5, w - i * 16, h / 4 - 4, fill=WHITE, stroke=color, sw=1.6, rx=4)
    svg.text(x + w / 2, y + h + 34, "TCN", size=32, color=color, weight=700, anchor="middle")
    svg.text(x + w / 2, y + h + 66, f"k={k}", size=21, color=MUTED, anchor="middle")
    svg.text(x + w / 2, y + h + 98, f"d={dilation}", size=18, color=MUTED, anchor="middle")


def v3_token_row(svg, x, y, labels, w=64, h=56, gap=10, fill=PURPLE_FILL, stroke=PURPLE):
    for i, label in enumerate(labels):
        xx = x + i * (w + gap)
        svg.rect(xx, y, w, h, fill=fill, stroke=stroke, sw=1.8, rx=7)
        svg.text(xx + w / 2, y + 37, label, size=30, color=PURPLE_DARK if stroke == PURPLE else INK, weight=700, anchor="middle")


def v3_bracket(svg, x1, x2, y, label, color=FINE_DARK):
    svg.line(x1, y, x1, y + 14, stroke=color, sw=2.0)
    svg.line(x1, y + 14, x2, y + 14, stroke=color, sw=2.0)
    svg.line(x2, y + 14, x2, y, stroke=color, sw=2.0)
    svg.text((x1 + x2) / 2, y + 52, label, size=28, color=color, weight=700, anchor="middle")


def v3_arrow(svg, x1, y1, x2, y2, color=LINE, marker="arrow", sw=2.2):
    svg.line(x1, y1, x2, y2, stroke=color, sw=sw, marker=marker)


def figure1_v3():
    W, H = 2400, 1380
    s = V3SVG(W, H, "Overall Framework of ST-MGPrompt")
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(40, 70, "ST-MGPrompt", size=64, weight=700)
    s.text(W - 40, 68, "Overall Framework", size=34, color=MUTED, weight=700, anchor="end")

    v3_panel(s, 30, 125, 300, 1015, ["History"], "", fill="#FBFCFD", title_size=30)
    v3_panel(s, 350, 125, 420, 1015, ["Dual Temporal", "Semantics"], "1", fill="#FBFCFD", title_size=36)
    v3_panel(s, 790, 125, 600, 1015, ["Graph-Temporal", "Encoding"], "2", fill="#FBFCFD", title_size=36)
    v3_panel(s, 1410, 125, 550, 1015, ["Cross-Granularity", "Fusion"], "3", fill=PURPLE_FILL, stroke=PURPLE, title_size=34)
    v3_panel(s, 1980, 125, 380, 1015, ["Task-Conditioned", "Forecasting"], "4", fill="#FBFCFD", title_size=24)

    # History and the shared projection glyph.
    v3_tensor(s, 75, 275, 170, 155, "X")
    s.multiline(180, 490, ["Wind-Farm", "History"], size=27, color=INK, weight=700, anchor="middle", gap=34)
    s.multiline(180, 585, ["134 turbines", "16 variables", "144 steps = 24 h"], size=25, color=MUTED, anchor="middle", gap=34)
    s.text(180, 720, "B×144×134×16", size=19, color=INK, anchor="middle")
    s.line(180, 745, 180, 785, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    s.path("M 105 785 L 255 785 L 235 845 L 125 845 Z", fill=SHARED, stroke=SHARED_STROKE, sw=1.8)
    s.text(180, 823, "16 -> 64", size=28, color=INK, weight=700, anchor="middle")
    s.text(180, 895, "shared projection", size=24, color=MUTED, anchor="middle")
    s.line(265, 815, 300, 815, stroke=SHARED_STROKE, sw=2.2)
    s.line(300, 815, 300, 450, stroke=SHARED_STROKE, sw=2.2)
    s.line(300, 450, 350, 450, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    s.line(300, 815, 300, 750, stroke=SHARED_STROKE, sw=2.2)
    s.line(300, 750, 350, 750, stroke=SHARED_STROKE, sw=2.2, marker="arrow")

    # Fine/coarse temporal semantics.
    v3_badge(s, 385, 255, 330, "same 144-step grid", fill=SHARED, color=MUTED, size=25)
    v3_badge(s, 385, 330, 100, "Fine", fill=FINE_FILL, color=FINE_DARK, size=30)
    v3_sequence(s, 385, 405, 340, 70, kind="fine", count=8, stroke=FINE, fill=WHITE)
    s.text(555, 515, "local variation", size=31, color=FINE_DARK, weight=700, anchor="middle")
    s.text(555, 555, "causal conv., k=3", size=26, color=MUTED, anchor="middle")
    v3_badge(s, 385, 665, 120, "Coarse", fill=COARSE_FILL, color=COARSE_DARK, size=30)
    v3_sequence(s, 385, 740, 340, 70, kind="coarse", count=8, stroke=COARSE, fill=WHITE)
    s.text(555, 850, "trend context", size=31, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(555, 890, "windows 6 / 18 / 36", size=26, color=MUTED, anchor="middle")
    v3_arrow(s, 725, 440, 790, 440, color=FINE_DARK, marker="arrowFine")
    v3_arrow(s, 725, 775, 790, 775, color=COARSE_DARK, marker="arrowCoarse")

    # Graph-temporal encoding: one compact visual grammar per lane.
    s.text(825, 270, "Micro", size=36, color=FINE_DARK, weight=700)
    s.text(825, 308, "dWspd + distance", size=25, color=FINE_DARK)
    graph_icon(s, 820, 370, 78, 72, FINE_DARK, mode="prior", node_r=6)
    s.text(925, 415, "x", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 950, 370, 78, 72, PURPLE_DARK, mode="learned", node_r=6)
    s.multiline(989, 485, ["adaptive", "weights"], size=21, color=FINE_DARK, anchor="middle", gap=30)
    v3_arrow(s, 1035, 406, 1065, 406, color=FINE_DARK, marker="arrowFine")
    bidiffusion(s, 1065, 406, 92, FINE_DARK, label="2-hop Bi-Diff.")
    v3_arrow(s, 1170, 406, 1235, 406, color=FINE_DARK, marker="arrowFine")
    v3_tcn(s, 1240, 370, 72, 72, FINE_DARK, 3, "{1,2}")

    s.text(825, 665, "Macro", size=36, color=COARSE_DARK, weight=700)
    s.text(825, 703, "Patv trend + distance", size=27, color=COARSE_DARK)
    graph_icon(s, 820, 765, 78, 72, COARSE_DARK, mode="prior", node_r=6)
    s.text(925, 810, "x", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 950, 765, 78, 72, PURPLE_DARK, mode="learned", node_r=6)
    s.multiline(989, 880, ["adaptive", "weights"], size=21, color=COARSE_DARK, anchor="middle", gap=30)
    v3_arrow(s, 1035, 801, 1065, 801, color=COARSE_DARK, marker="arrowCoarse")
    bidiffusion(s, 1065, 801, 92, COARSE_DARK, label="2-hop Bi-Diff.")
    v3_arrow(s, 1170, 801, 1235, 801, color=COARSE_DARK, marker="arrowCoarse")
    v3_tcn(s, 1240, 765, 72, 72, COARSE_DARK, 5, "{1,2,4,8}")

    # Prompt and source-asymmetric fusion.
    s.line(1277, 800, 1380, 800, stroke=COARSE_DARK, sw=2.2)
    s.line(1380, 800, 1380, 300, stroke=COARSE_DARK, sw=2.2)
    s.line(1380, 300, 1450, 300, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.text(1685, 270, "Macro-Trend Prompt", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    v3_token_row(s, 1535, 315, ["P1", "P2", "P3", "P4"], w=58, h=52, gap=9)
    s.text(1685, 405, "4 tokens / turbine", size=26, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1685, 425, 1685, 490, stroke=PURPLE_DARK, sw=2.0, marker="arrowPurple")
    s.text(1780, 470, "trend guidance", size=21, color=PURPLE_DARK, italic=True, anchor="middle")

    v3_sequence(s, 1445, 520, 125, 58, kind="fine", count=4, stroke=FINE, fill=WHITE)
    s.text(1508, 625, "Fine history", size=23, color=FINE_DARK, anchor="middle")
    relation_node(s, 1635, 550, "A", PURPLE_DARK, r=28)
    s.line(1575, 550, 1607, 550, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.line(1663, 550, 1705, 550, stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    residual_add(s, 1735, 550, FINE_DARK)
    s.line(1758, 550, 1795, 550, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    v3_sequence(s, 1795, 520, 90, 58, kind="fine", count=2, stroke=FINE, fill=FINE_FILL)
    s.text(1840, 625, "H'f", size=30, color=FINE_DARK, weight=700, anchor="middle")

    v3_sequence(s, 1445, 760, 125, 58, kind="coarse", count=4, stroke=COARSE, fill=WHITE)
    s.text(1535, 865, "Coarse history", size=23, color=COARSE_DARK, anchor="middle")
    s.text(1685, 930, "recent-detail correction", size=13, color=FINE_DARK, italic=True, anchor="middle")
    relation_node(s, 1635, 790, "A", PURPLE_DARK, r=28)
    s.line(1575, 790, 1607, 790, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.line(1663, 790, 1705, 790, stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    residual_add(s, 1735, 790, COARSE_DARK)
    s.line(1758, 790, 1795, 790, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    v3_sequence(s, 1795, 760, 90, 58, kind="coarse", count=2, stroke=COARSE, fill=COARSE_FILL)
    s.text(1840, 865, "H'c", size=30, color=COARSE_DARK, weight=700, anchor="middle")
    gate_diamond(s, 1690, 680, size=25)
    s.line(1690, 650, 1735, 578, stroke=PURPLE_DARK, sw=1.5, dash="7 6")
    s.line(1690, 710, 1735, 762, stroke=PURPLE_DARK, sw=1.5, dash="7 6")

    # Forecasting context, kept as three vertically separated objects.
    s.rect(2010, 260, 320, 205, fill=SHARED, stroke=SHARED_STROKE, sw=1.6, rx=14)
    s.multiline(2170, 315, ["Last-State", "Context"], size=27, color=INK, weight=700, anchor="middle", gap=34)
    s.multiline(2170, 390, ["Fine last", "Coarse last", "Macro mean"], size=20, color=MUTED, anchor="middle", gap=25)
    s.line(2170, 480, 2170, 515, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    v3_token_row(s, 2085, 515, ["[n]", "[h]", "[t]"], w=55, h=46, gap=11, fill=PURPLE_FILL, stroke=PURPLE)
    s.multiline(2170, 610, ["node / step", "fixed type"], size=15, color=PURPLE_DARK, anchor="middle", gap=25)
    s.rect(2040, 685, 260, 92, fill=PURPLE_FILL, stroke=PURPLE, sw=1.6, rx=12)
    s.multiline(2170, 728, ["Node-Horizon ST", "Prompt"], size=22, color=PURPLE_DARK, weight=700, anchor="middle", gap=29)
    s.line(2170, 795, 2170, 825, stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    s.rect(2040, 840, 260, 68, fill=WHITE, stroke=SHARED_STROKE, sw=1.6, rx=12)
    s.text(2170, 883, "Direct Decoder", size=27, color=INK, weight=700, anchor="middle")
    s.line(2170, 920, 2170, 945, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    s.text(2170, 970, "10-step forecast", size=22, color=INK, weight=700, anchor="middle")
    forecast_strip(s, 2080, 985, 180, 26, color=SHARED_STROKE, show_label=False)
    s.text(2170, 1045, "B×10×134", size=18, color=INK, anchor="middle")
    for i, label in enumerate(["H3", "H6", "H10"]):
        xx = 2085 + i * 62
        s.rect(xx, 1070, 52, 32, fill=PURPLE_FILL, stroke=PURPLE, sw=1.2, rx=6)
        s.text(xx + 26, 1092, label, size=19, color=PURPLE_DARK, weight=700, anchor="middle")

    # Compact training-only callout.
    s.rect(30, 1180, 1450, 160, fill=LOSS_FILL, stroke=LOSS, sw=1.2, rx=18, dash="10 8")
    s.text(55, 1222, "Training Only - MS-MG-DWU", size=36, color=LOSS, weight=700)
    s.text(1450, 1222, "not used in inference", size=22, color=LOSS, italic=True, anchor="end")
    s.text(150, 1290, "yhat, Y, M", size=27, color=LOSS, weight=700, anchor="middle")
    s.line(260, 1280, 330, 1280, stroke=LOSS, sw=2.2, marker="arrowLoss")
    s.multiline(530, 1270, ["Horizon difficulty", "+ Site difficulty"], size=23, color=LOSS, weight=700, anchor="middle", gap=30)
    s.line(700, 1280, 800, 1280, stroke=LOSS, sw=2.2, marker="arrowLoss")
    s.rect(840, 1245, 220, 62, fill=WHITE, stroke=LOSS, sw=1.6, rx=10)
    s.text(950, 1285, "Objective", size=29, color=LOSS, weight=700, anchor="middle")
    return s


def figure2_v3():
    W, H = 2400, 1360
    s = V3SVG(W, H, "Dual Semantic Graph Construction")
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(40, 70, "Dual Semantic Graph Construction", size=64, weight=700)
    s.text(W - 40, 68, "Train-Only Priors and Constrained Adaptation", size=30, color=MUTED, weight=700, anchor="end")
    v3_panel(s, 35, 125, 350, 1200, ["Data Sources"], "1", fill="#FBFCFD", title_size=30)
    v3_panel(s, 420, 125, 860, 1200, ["Semantic Priors"], "2", fill="#FBFCFD", title_size=30)
    v3_panel(s, 1310, 125, 1055, 1200, ["Prior-Constrained", "Adaptation"], "3", fill=PURPLE_FILL, stroke=PURPLE, title_size=30)

    # Data sources.
    s.text(70, 245, "train-only", size=29, color=LOSS, weight=700)
    v3_tensor(s, 80, 285, 240, 120, "Patv", color=COARSE, fill=COARSE_FILL)
    s.text(200, 450, "Patv", size=28, color=COARSE_DARK, weight=700, anchor="middle")
    v3_tensor(s, 80, 550, 240, 120, "Wspd", color=FINE, fill=FINE_FILL)
    s.text(200, 715, "Wspd", size=28, color=FINE_DARK, weight=700, anchor="middle")
    s.rect(80, 815, 240, 175, fill=SHARED, stroke=SHARED_STROKE, sw=1.6, rx=12)
    coordinate_icon(s, 115, 842, 170, 62, SHARED_STROKE)
    s.text(200, 930, "Coordinates", size=25, color=SHARED_STROKE, weight=700, anchor="middle")
    s.text(200, 970, "r_i=(x_i,y_i)", size=21, color=SHARED_STROKE, anchor="middle")
    s.line(320, 345, 450, 345, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.line(320, 610, 450, 610, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.line(320, 890, 450, 890, stroke=SHARED_STROKE, sw=2.2, marker="arrow")

    # Macro semantic prior.
    s.text(465, 225, "Macro / m", size=34, color=COARSE_DARK, weight=700)
    s.rect(465, 275, 185, 185, fill=COARSE_FILL, stroke=COARSE, sw=1.4, rx=12)
    waveform(s, 490, 300, 135, 40, kind="coarse", stroke=COARSE_DARK)
    s.text(557, 365, "Trend", size=27, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(557, 430, "36 steps ~ 6 h", size=19, color=MUTED, anchor="middle")
    s.line(650, 340, 690, 340, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.rect(690, 275, 135, 185, fill=WHITE, stroke=COARSE, sw=1.4, rx=12)
    similarity_icon(s, 705, 290, 105, 85, mode="pearson", color=COARSE_DARK, show_label=False)
    s.text(757, 435, "rho", size=21, color=COARSE_DARK, weight=700, anchor="middle")
    s.line(825, 340, 860, 340, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.rect(860, 275, 135, 185, fill=WHITE, stroke=COARSE, sw=1.4, rx=12)
    s.text(927, 338, "K=10", size=29, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(927, 430, "top-k", size=19, color=MUTED, anchor="middle")
    s.line(995, 340, 1030, 340, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.rect(1030, 275, 190, 185, fill=COARSE_FILL, stroke=COARSE, sw=1.4, rx=12)
    sparse_graph(s, 1065, 292, 120, 72, COARSE_DARK, k=10, dense=False, label_y=435, label="Trend graph")
    s.line(1125, 460, 1125, 480, stroke=COARSE_DARK, sw=1.8, marker="arrowCoarse")
    s.rect(1030, 480, 190, 135, fill=COARSE_FILL, stroke=COARSE, sw=1.4, rx=12)
    s.text(1125, 515, "Macro Prior", size=24, color=COARSE_DARK, weight=700, anchor="middle")
    s.multiline(1125, 555, ["Trend x 0.7", "+ Distance x 0.3"], size=19, color=COARSE_DARK, anchor="middle", gap=27)

    # Micro semantic prior.
    s.text(465, 650, "Micro / mu", size=34, color=FINE_DARK, weight=700)
    s.rect(465, 700, 185, 185, fill=FINE_FILL, stroke=FINE, sw=1.4, rx=12)
    waveform(s, 490, 725, 135, 40, kind="delta", stroke=FINE_DARK)
    s.text(557, 790, "dWspd", size=28, color=FINE_DARK, weight=700, anchor="middle")
    s.text(557, 870, "local change", size=19, color=MUTED, anchor="middle")
    s.line(650, 765, 690, 765, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.rect(690, 700, 135, 185, fill=WHITE, stroke=FINE, sw=1.4, rx=12)
    similarity_icon(s, 705, 715, 105, 85, mode="cosine", color=FINE_DARK, show_label=False)
    s.text(757, 870, "cos", size=21, color=FINE_DARK, weight=700, anchor="middle")
    s.line(825, 765, 860, 765, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.rect(860, 700, 135, 185, fill=WHITE, stroke=FINE, sw=1.4, rx=12)
    s.text(927, 763, "K=5", size=29, color=FINE_DARK, weight=700, anchor="middle")
    s.text(927, 855, "top-k", size=19, color=MUTED, anchor="middle")
    s.line(995, 765, 1030, 765, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.rect(1030, 700, 190, 185, fill=FINE_FILL, stroke=FINE, sw=1.4, rx=12)
    sparse_graph(s, 1065, 717, 120, 72, FINE_DARK, k=5, dense=False, label_y=870, label="Local graph")
    s.line(1125, 885, 1125, 905, stroke=FINE_DARK, sw=1.8, marker="arrowFine")
    s.rect(1030, 905, 190, 135, fill=FINE_FILL, stroke=FINE, sw=1.4, rx=12)
    s.text(1125, 940, "Micro Prior", size=24, color=FINE_DARK, weight=700, anchor="middle")
    s.multiline(1125, 980, ["Local x 0.3", "+ Distance x 0.7"], size=19, color=FINE_DARK, anchor="middle", gap=27)

    # Shared distance support without a long formula.
    s.rect(465, 1080, 430, 160, fill=SHARED, stroke=SHARED_STROKE, sw=1.4, rx=12)
    s.text(680, 1125, "Geographic distance", size=29, color=SHARED_STROKE, weight=700, anchor="middle")
    s.text(680, 1170, "Gaussian kernel", size=24, color=MUTED, anchor="middle")
    s.text(680, 1210, "K = 5", size=23, color=SHARED_STROKE, weight=700, anchor="middle")
    s.line(895, 1110, 970, 1110, stroke=SHARED_STROKE, sw=1.8, dash="8 7")
    s.line(970, 1110, 970, 545, stroke=SHARED_STROKE, sw=1.8, dash="8 7")
    s.line(970, 545, 1030, 545, stroke=SHARED_STROKE, sw=1.8, dash="8 7", marker="arrow")
    s.line(895, 1160, 990, 1160, stroke=SHARED_STROKE, sw=1.8, dash="8 7")
    s.line(990, 1160, 990, 970, stroke=SHARED_STROKE, sw=1.8, dash="8 7")
    s.line(990, 970, 1030, 970, stroke=SHARED_STROKE, sw=1.8, dash="8 7", marker="arrow")

    # Prior-constrained adaptation: topology, weights and effective graph.
    s.text(1340, 275, "Independent node embeddings", size=27, color=PURPLE_DARK, weight=700)
    s.text(1340, 315, "d_e = 10", size=23, color=MUTED)
    s.text(1360, 370, "Macro", size=32, color=COARSE_DARK, weight=700)
    graph_icon(s, 1430, 345, 120, 92, COARSE_DARK, mode="prior", node_r=6)
    s.text(1490, 475, "Prior support", size=22, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(1595, 400, "x", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1635, 345, 120, 92, PURPLE_DARK, mode="learned", node_r=6)
    s.text(1695, 475, "Weights", size=22, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1695, 508, "E1 / E2", size=20, color=MUTED, anchor="middle")
    s.text(1810, 400, "=", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1855, 337, 145, 105, INK, mode="effective", node_r=6)
    s.multiline(1975, 475, ["Effective", "graph"], size=25, color=INK, weight=700, anchor="middle", gap=32)
    s.text(1975, 555, "same support", size=22, color=MUTED, anchor="middle")
    s.text(1360, 685, "Micro", size=32, color=FINE_DARK, weight=700)
    graph_icon(s, 1430, 660, 120, 92, FINE_DARK, mode="prior", node_r=6)
    s.text(1490, 790, "Prior support", size=22, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1595, 715, "x", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1635, 660, 120, 92, PURPLE_DARK, mode="learned", node_r=6)
    s.text(1695, 790, "Weights", size=22, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1695, 823, "E1 / E2", size=20, color=MUTED, anchor="middle")
    s.text(1810, 715, "=", size=34, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1855, 652, 145, 105, INK, mode="effective", node_r=6)
    s.multiline(1975, 790, ["Effective", "graph"], size=25, color=INK, weight=700, anchor="middle", gap=32)
    s.text(1975, 870, "same support", size=22, color=MUTED, anchor="middle")

    s.rect(1410, 1000, 800, 260, fill=WHITE, stroke=PURPLE, sw=1.4, rx=14, dash="8 6")
    s.multiline(1810, 1050, ["same node set", "same support", "reweighted edges"], size=27, color=PURPLE_DARK, weight=700, anchor="middle", gap=38)
    graph_icon(s, 1535, 1145, 125, 92, SHARED_STROKE, mode="prior", node_r=6)
    graph_icon(s, 1950, 1145, 125, 92, INK, mode="effective", node_r=6)
    s.line(1685, 1190, 1920, 1190, stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")

    # Arrows from the two effective priors into the adaptation panel.
    s.line(1220, 545, 1310, 545, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    s.line(1220, 970, 1310, 970, stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    return s


def figure3_v3():
    W, H = 2400, 1380
    s = V3SVG(W, H, "Macro-Trend Prompt and Bidirectional Cross-Granularity Fusion")
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(40, 65, "Macro-Trend Prompt and Bidirectional Cross-Granularity Fusion", size=55, weight=700)
    s.text(W - 40, 108, "Fine and coarse historical representations remain separate", size=27, color=MUTED, weight=700, anchor="end")

    # Separate historical ribbons.
    v3_badge(s, 50, 150, 190, "Fine history", fill=FINE_FILL, color=FINE_DARK, size=30)
    v3_sequence(s, 55, 225, 370, 80, kind="fine", count=8, stroke=FINE, fill=WHITE)
    s.text(240, 345, "H_f  ·  144 positions", size=31, color=FINE_DARK, weight=700, anchor="middle")
    v3_bracket(s, 240, 425, 390, "recent 24", color=FINE_DARK)
    v3_badge(s, 2160, 150, 190, "Coarse history", fill=COARSE_FILL, color=COARSE_DARK, size=30)
    v3_sequence(s, 2045, 225, 290, 80, kind="coarse", count=4, stroke=COARSE, fill=WHITE)
    s.text(2190, 345, "H_c  ·  144 positions", size=31, color=COARSE_DARK, weight=700, anchor="middle")

    # Prompt extraction.
    s.rect(640, 150, 1160, 230, fill=PURPLE_FILL, stroke=PURPLE, sw=1.5, rx=20)
    s.text(1220, 195, "Node-Specific Macro-Trend Prompt", size=44, color=PURPLE_DARK, weight=700, anchor="middle")
    v3_badge(s, 720, 250, 120, "H_c", fill=COARSE_FILL, color=COARSE_DARK, size=31)
    s.line(850, 278, 930, 278, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    for i in range(8):
        s.circle(955 + i * 24, 270 + (i % 3) * 10, 6 + (i % 3), fill=PURPLE, stroke=WHITE, sw=1.0)
    s.text(1045, 345, "weighted pooling", size=26, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1140, 278, 1230, 278, stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    v3_token_row(s, 1240, 250, ["P1", "P2", "P3", "P4"], w=64, h=56, gap=12)
    s.text(1420, 350, "4 tokens / turbine", size=29, color=PURPLE_DARK, weight=700, anchor="middle")

    # Main interaction panel.
    s.rect(500, 430, 1380, 720, fill="#FCFAFE", stroke=PURPLE, sw=1.6, rx=24)
    s.text(1190, 480, "Bidirectional Source-Asymmetric Interaction", size=44, color=PURPLE_DARK, weight=700, anchor="middle")

    # Fine update row.
    s.text(610, 555, "Macro -> Fine", size=40, color=PURPLE_DARK, weight=700)
    s.text(610, 595, "Q: Fine", size=30, color=FINE_DARK)
    s.text(1015, 595, "K,V: Macro", size=28, color=PURPLE_DARK, weight=700, anchor="middle")
    v3_sequence(s, 610, 630, 280, 64, kind="fine", count=5, stroke=FINE, fill=WHITE)
    s.line(900, 662, 1088, 662, stroke=FINE_DARK, sw=2.4, marker="arrowFine")
    relation_node(s, 1130, 662, "A", PURPLE_DARK, r=38)
    s.text(1130, 735, "O_mf", size=28, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1175, 662, 1435, 662, stroke=PURPLE_DARK, sw=2.4, marker="arrowPurple")
    residual_add(s, 1480, 662, FINE_DARK)
    s.line(1505, 662, 1600, 662, stroke=FINE_DARK, sw=2.4, marker="arrowFine")
    v3_sequence(s, 1610, 630, 240, 64, kind="fine", count=5, stroke=FINE, fill=FINE_FILL)
    s.text(1730, 735, "Updated Fine H'_f", size=31, color=FINE_DARK, weight=700, anchor="middle")
    s.line(1420, 278, 1420, 510, stroke=PURPLE_DARK, sw=2.0)
    s.line(1420, 510, 1175, 620, stroke=PURPLE_DARK, sw=2.0, marker="arrowPurple")
    s.text(1500, 535, "trend guidance", size=24, color=PURPLE_DARK, italic=True, anchor="middle")

    # Coarse update row.
    s.text(610, 820, "Recent Fine -> Coarse", size=40, color=FINE_DARK, weight=700)
    s.text(610, 860, "Q: Coarse", size=30, color=COARSE_DARK)
    s.text(1260, 760, "K,V: Recent Fine · 24", size=28, color=FINE_DARK, weight=700, anchor="middle")
    v3_sequence(s, 610, 895, 280, 64, kind="coarse", count=5, stroke=COARSE, fill=WHITE)
    s.line(900, 927, 1088, 927, stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")
    relation_node(s, 1130, 927, "A", PURPLE_DARK, r=38)
    s.text(1130, 1000, "O_fc", size=28, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1175, 927, 1435, 927, stroke=PURPLE_DARK, sw=2.4, marker="arrowPurple")
    residual_add(s, 1480, 927, COARSE_DARK)
    s.line(1505, 927, 1600, 927, stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")
    v3_sequence(s, 1610, 895, 240, 64, kind="coarse", count=5, stroke=COARSE, fill=COARSE_FILL)
    s.text(1730, 1000, "Updated Coarse H'_c", size=31, color=COARSE_DARK, weight=700, anchor="middle")
    s.line(420, 445, 470, 445, stroke=FINE_DARK, sw=2.0)
    s.line(470, 445, 470, 875, stroke=FINE_DARK, sw=2.0)
    s.line(470, 875, 1030, 875, stroke=FINE_DARK, sw=2.0)
    s.line(1030, 875, 1098, 900, stroke=FINE_DARK, sw=2.0, marker="arrowFine")
    s.line(2045, 265, 1940, 265, stroke=COARSE_DARK, sw=2.0)
    s.line(1940, 265, 1940, 875, stroke=COARSE_DARK, sw=2.0)
    s.line(1940, 875, 1880, 875, stroke=COARSE_DARK, sw=2.0, marker="arrowCoarse")
    s.text(1810, 820, "recent-detail correction", size=23, color=FINE_DARK, italic=True, anchor="middle")

    # Shared gate between the two routes.
    gate_diamond(s, 1480, 795, size=36)
    s.text(1480, 875, "fusion gate", size=24, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1480, 759, 1480, 700, stroke=PURPLE_DARK, sw=1.5, dash="7 6")
    s.line(1480, 831, 1480, 890, stroke=PURPLE_DARK, sw=1.5, dash="7 6")

    # Formula band, separated from the interaction rows.
    s.line(760, 1040, 1620, 1040, stroke=GRID, sw=1.4)
    s.text(1190, 1085, "H'_f = LN(H_f + G*O_mf)", size=27, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1190, 1120, "H'_c = LN(H_c + (1-G)*O_fc)", size=27, color=COARSE_DARK, weight=700, anchor="middle")

    # Common decoder, without the long outer loop.
    s.line(1730, 760, 1730, 1180, stroke=SHARED_STROKE, sw=1.7)
    s.line(1730, 1030, 1730, 1180, stroke=SHARED_STROKE, sw=1.7)
    s.line(1730, 1180, 1350, 1180, stroke=SHARED_STROKE, sw=1.7, marker="arrow")
    s.rect(1030, 1160, 320, 72, fill=WHITE, stroke=SHARED_STROKE, sw=1.8, rx=12)
    s.text(1190, 1206, "Last-State Direct Decoder", size=32, color=INK, weight=700, anchor="middle")
    return s


def figure3_v3_publication():
    """Publication-layout pass for Fig. 3 with separated interaction rows."""
    W, H = 2400, 1380
    s = V3SVG(W, H, "Macro-Trend Prompt and Bidirectional Cross-Granularity Fusion")
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(40, 62, "Macro-Trend Prompt and Bidirectional Cross-Granularity Fusion", size=43, weight=700)
    s.text(W - 40, 105, "Fine and coarse historical representations remain separate", size=24, color=MUTED, weight=700, anchor="end")

    # Historical streams stay at the two sides of the interaction module.
    v3_badge(s, 45, 145, 220, "Fine history", fill=FINE_FILL, color=FINE_DARK, size=26)
    v3_sequence(s, 55, 215, 370, 72, kind="fine", count=8, stroke=FINE, fill=WHITE)
    s.text(240, 330, "H_f  -  144 positions", size=28, color=FINE_DARK, weight=700, anchor="middle")
    v3_bracket(s, 240, 425, 370, "recent 24", color=FINE_DARK)

    v3_badge(s, 2070, 145, 280, "Coarse history", fill=COARSE_FILL, color=COARSE_DARK, size=26)
    v3_sequence(s, 2045, 215, 290, 72, kind="coarse", count=4, stroke=COARSE, fill=WHITE)
    s.text(2190, 330, "H_c  -  144 positions", size=28, color=COARSE_DARK, weight=700, anchor="middle")

    # Macro-trend prompt extraction.
    s.rect(600, 145, 1200, 230, fill=PURPLE_FILL, stroke=PURPLE, sw=1.5, rx=20)
    s.text(1200, 195, "Node-Specific Macro-Trend Prompt", size=39, color=PURPLE_DARK, weight=700, anchor="middle")
    v3_badge(s, 690, 250, 120, "H_c", fill=COARSE_FILL, color=COARSE_DARK, size=29)
    s.line(820, 278, 900, 278, stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    for i in range(8):
        s.circle(925 + i * 24, 270 + (i % 3) * 10, 6 + (i % 3), fill=PURPLE, stroke=WHITE, sw=1.0)
    s.text(1015, 345, "weighted pooling", size=24, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1110, 278, 1190, 278, stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    v3_token_row(s, 1200, 250, ["P1", "P2", "P3", "P4"], w=60, h=52, gap=10)
    s.text(1380, 350, "4 tokens / turbine", size=26, color=PURPLE_DARK, weight=700, anchor="middle")

    # Main source-asymmetric interaction area.
    s.rect(500, 430, 1400, 700, fill="#FCFAFE", stroke=PURPLE, sw=1.6, rx=24)
    s.text(1200, 480, "Bidirectional Source-Asymmetric Interaction", size=39, color=PURPLE_DARK, weight=700, anchor="middle")

    # Coarse-to-fine: Q is fine history, K,V are macro prompt tokens.
    s.text(610, 555, "Macro -> Fine", size=36, color=PURPLE_DARK, weight=700)
    s.text(610, 595, "Q: Fine", size=27, color=FINE_DARK)
    s.text(1015, 595, "K,V: Macro", size=26, color=PURPLE_DARK, weight=700, anchor="middle")
    v3_sequence(s, 610, 630, 280, 60, kind="fine", count=5, stroke=FINE, fill=WHITE)
    s.line(900, 660, 1088, 660, stroke=FINE_DARK, sw=2.4, marker="arrowFine")
    relation_node(s, 1130, 660, "A", PURPLE_DARK, r=36)
    s.text(1130, 730, "O_mf", size=26, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1175, 660, 1435, 660, stroke=PURPLE_DARK, sw=2.4, marker="arrowPurple")
    residual_add(s, 1480, 660, FINE_DARK)
    s.line(1505, 660, 1600, 660, stroke=FINE_DARK, sw=2.4, marker="arrowFine")
    v3_sequence(s, 1610, 630, 240, 60, kind="fine", count=5, stroke=FINE, fill=FINE_FILL)
    s.text(1730, 730, "Updated Fine H'_f", size=28, color=FINE_DARK, weight=700, anchor="middle")
    s.line(1390, 302, 1390, 510, stroke=PURPLE_DARK, sw=2.0)
    s.line(1390, 510, 1175, 620, stroke=PURPLE_DARK, sw=2.0, marker="arrowPurple")
    s.text(1510, 535, "trend guidance", size=21, color=PURPLE_DARK, italic=True, anchor="middle")

    # Recent-fine-to-coarse: Q is coarse history, K,V are the recent fine slice.
    s.text(610, 815, "Recent Fine -> Coarse", size=36, color=FINE_DARK, weight=700)
    s.text(610, 855, "Q: Coarse", size=27, color=COARSE_DARK)
    s.text(1260, 770, "K,V: Recent Fine - 24", size=21, color=FINE_DARK, weight=700, anchor="middle")
    v3_sequence(s, 610, 885, 280, 60, kind="coarse", count=5, stroke=COARSE, fill=WHITE)
    s.line(900, 915, 1088, 915, stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")
    relation_node(s, 1130, 915, "A", PURPLE_DARK, r=36)
    s.text(1130, 985, "O_fc", size=24, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1175, 915, 1435, 915, stroke=PURPLE_DARK, sw=2.4, marker="arrowPurple")
    residual_add(s, 1480, 915, COARSE_DARK)
    s.line(1505, 915, 1600, 915, stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")
    v3_sequence(s, 1610, 885, 240, 60, kind="coarse", count=5, stroke=COARSE, fill=COARSE_FILL)
    s.text(1690, 985, "Updated Coarse H'_c", size=24, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(1510, 1020, "recent-detail correction", size=13, color=FINE_DARK, italic=True, anchor="middle")

    # External routes make the source of each row explicit without crossing text.
    s.line(425, 415, 470, 415, stroke=FINE_DARK, sw=2.0)
    s.line(470, 415, 470, 855, stroke=FINE_DARK, sw=2.0)
    s.line(470, 855, 1030, 855, stroke=FINE_DARK, sw=2.0)
    s.line(1030, 855, 1098, 888, stroke=FINE_DARK, sw=2.0, marker="arrowFine")
    s.line(2045, 250, 1940, 250, stroke=COARSE_DARK, sw=2.0)
    s.line(1940, 250, 1940, 855, stroke=COARSE_DARK, sw=2.0)
    s.line(1940, 855, 1880, 855, stroke=COARSE_DARK, sw=2.0, marker="arrowCoarse")

    # Shared gate and a separated formula band.
    gate_diamond(s, 1480, 785, size=22)
    s.text(1480, 840, "fusion gate", size=16, color=PURPLE_DARK, weight=700, anchor="middle")
    s.line(1480, 751, 1480, 700, stroke=PURPLE_DARK, sw=1.5, dash="7 6")
    s.line(1480, 819, 1480, 875, stroke=PURPLE_DARK, sw=1.5, dash="7 6")
    s.line(760, 1030, 1620, 1030, stroke=GRID, sw=1.4)
    s.text(1190, 1075, "H'_f = LN(H_f + G*O_mf)", size=25, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1190, 1110, "H'_c = LN(H_c + (1-G)*O_fc)", size=25, color=COARSE_DARK, weight=700, anchor="middle")

    # Both updated histories feed the common last-state decoder.
    s.line(1845, 660, 1870, 660, stroke=SHARED_STROKE, sw=1.7)
    s.line(1870, 660, 1870, 1180, stroke=SHARED_STROKE, sw=1.7)
    s.line(1845, 915, 1890, 915, stroke=SHARED_STROKE, sw=1.7)
    s.line(1890, 915, 1890, 1180, stroke=SHARED_STROKE, sw=1.7)
    s.line(1890, 1180, 1350, 1180, stroke=SHARED_STROKE, sw=1.7, marker="arrow")
    s.rect(930, 1160, 520, 72, fill=WHITE, stroke=SHARED_STROKE, sw=1.8, rx=12)
    s.text(1190, 1206, "Last-State Direct Decoder", size=29, color=INK, weight=700, anchor="middle")
    return s


def write_v3_outputs(svg: V3SVG, stem: str):
    svg_path = OUT / f"{stem}.svg"
    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}_300.png"
    svg_path.write_text(svg.end(), encoding="utf-8")
    convert_svg_to_pdf(svg_path, pdf_path)
    render_pdf_to_png(pdf_path, png_path, dpi=300)
    report = run_v3_collision_qa(svg, stem)
    print(f"{stem}: {report}")
    return report


def box(svg: SVG, x, y, w, h, title, lines=(), fill=WHITE, stroke=LINE, accent=None, title_size=14, radius=10, dashed=False):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.5, rx=radius, dash="7 5" if dashed else None)
    if accent:
        svg.rect(x, y, 5, h, fill=accent, stroke=accent, sw=0, rx=radius)
    svg.text(x + w / 2, y + 25, title, size=title_size, color=INK, weight=700, anchor="middle")
    if lines:
        start = y + 47 if len(lines) <= 2 else y + 43
        svg.multiline(x + w / 2, start, lines, size=12.2, color=MUTED, gap=16)


def section_band(svg: SVG, x, y, w, label, fill=SHARED, stroke=GRID):
    svg.rect(x, y, w, 25, fill=fill, stroke=stroke, sw=0.8, rx=7)
    svg.text(x + 12, y + 17, label.upper(), size=11, color=MUTED, weight=700, anchor="start", letter=1.1)


def tensor_icon(svg: SVG, x, y, w, h, color=SHARED_STROKE, layers=4, label=None):
    for i in range(layers - 1, -1, -1):
        dx = i * 7
        dy = -i * 5
        svg.rect(x + dx, y + dy, w, h, fill=WHITE if i else SHARED, stroke=color, sw=1.2, rx=4)
    if label:
        svg.text(x + w / 2 + (layers - 1) * 3.5, y + h / 2 + 5, label, size=13, color=INK, weight=700, anchor="middle")


def curve(svg: SVG, x, y, w, h, kind="fine", stroke=None):
    stroke = stroke or (FINE_DARK if kind == "fine" else COARSE_DARK)
    if kind == "fine":
        pts = [(0, .65), (.12, .35), (.22, .70), (.34, .20), (.47, .54), (.57, .31), (.68, .76), (.80, .25), (.91, .50), (1, .36)]
    else:
        pts = [(0, .72), (.16, .65), (.31, .52), (.46, .58), (.62, .42), (.79, .31), (1, .22)]
    d = "M " + " ".join(f"{x + px * w:.1f},{y + py * h:.1f}" for px, py in pts)
    svg.path(d, stroke=stroke, sw=2.2, fill="none")
    svg.line(x, y + h, x + w, y + h, stroke=GRID, sw=0.8)


def arrow(svg: SVG, x1, y1, x2, y2, color=LINE, sw=1.8, dash=None, marker="arrow"):
    svg.line(x1, y1, x2, y2, stroke=color, sw=sw, dash=dash, marker=marker)


# ---------------------------------------------------------------------------
# Refined visual grammar
# ---------------------------------------------------------------------------
# The original figures above are kept as a readable record of the first
# executable layout.  The primitives below are used by the refined_v1
# figures.  They deliberately give different object types different visual
# signatures so that the reader does not have to decode every object from its
# caption.

def tag(svg: SVG, x, y, label, fill=SHARED, color=MUTED, w=None, size=23):
    w = w or max(78, len(label) * size * .56 + 28)
    svg.rect(x, y, w, 34, fill=fill, stroke=fill, sw=0, rx=17)
    svg.text(x + w / 2, y + 24, label, size=size, color=color, weight=700, anchor="middle")
    return w


def stage(svg: SVG, x, y, w, h, number, title, fill=WHITE, stroke=GRID, dashed=True):
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.1, rx=22, dash="10 8" if dashed else None)
    svg.circle(x + 27, y + 29, 16, fill=stroke if stroke != GRID else SHARED_STROKE, stroke=WHITE, sw=1.0)
    svg.text(x + 27, y + 37, str(number), size=20, color=WHITE, weight=700, anchor="middle")
    title_color = INK if stroke == GRID else stroke
    if isinstance(title, (tuple, list)):
        svg.multiline(x + 55, y + 28, title, size=26, color=title_color, weight=700, anchor="start", gap=36)
    else:
        svg.text(x + 55, y + 37, title, size=30, color=title_color, weight=700)


def waveform(svg: SVG, x, y, w, h, kind="fine", stroke=None, baseline=True, opacity=1):
    stroke = stroke or (FINE_DARK if kind == "fine" else COARSE_DARK)
    if kind == "fine":
        pts = [(0.00, .66), (.08, .36), (.15, .72), (.23, .22), (.31, .57),
               (.39, .35), (.48, .78), (.57, .25), (.66, .59), (.76, .33),
               (.86, .70), (1.00, .42)]
    elif kind == "delta":
        pts = [(0.00, .52), (.10, .24), (.19, .72), (.29, .32), (.40, .66),
               (.51, .16), (.63, .74), (.74, .38), (.86, .62), (1.00, .43)]
    else:
        pts = [(0.00, .76), (.14, .68), (.28, .62), (.42, .54), (.57, .46),
               (.72, .38), (.86, .31), (1.00, .23)]
    d = "M " + " ".join(f"{x + px * w:.1f},{y + py * h:.1f}" for px, py in pts)
    svg.path(d, stroke=stroke, sw=3.0, fill="none", opacity=opacity)
    if baseline:
        svg.line(x, y + h, x + w, y + h, stroke=GRID, sw=1.0, opacity=opacity)


def sequence_ribbon(svg: SVG, x, y, w, h, kind="fine", count=8, stroke=None, fill=WHITE,
                    recent=0, label=None, compact=False):
    stroke = stroke or (FINE if kind in ("fine", "delta") else COARSE)
    dark = FINE_DARK if kind in ("fine", "delta") else COARSE_DARK
    gap = 9 if not compact else 6
    card_w = (w - gap * (count - 1)) / count
    for i in range(count):
        active = recent and i >= count - recent
        card_fill = FINE_FILL if kind in ("fine", "delta") and active else (COARSE_FILL if kind == "coarse" and active else fill)
        card_stroke = dark if active else stroke
        svg.rect(x + i * (card_w + gap), y, card_w, h, fill=card_fill, stroke=card_stroke, sw=1.6, rx=7)
        glyph_h = h * .42
        waveform(svg, x + i * (card_w + gap) + card_w * .15, y + h * .27,
                 card_w * .70, glyph_h, kind=kind, stroke=dark, baseline=False, opacity=.95)
    if label:
        svg.text(x + w / 2, y + h + 30, label, size=25, color=dark, weight=700, anchor="middle")


def tensor_stack(svg: SVG, x, y, w, h, color=SHARED_STROKE, fill=SHARED, layers=4,
                 label="X", sublabel=None):
    for i in range(layers - 1, -1, -1):
        dx, dy = i * 11, -i * 8
        svg.rect(x + dx, y + dy, w, h, fill=WHITE if i else fill, stroke=color, sw=1.8, rx=7)
    waveform(svg, x + 18, y + h * .37, w - 36, h * .25, kind="coarse", stroke=color, baseline=False, opacity=.8)
    svg.text(x + w / 2 + (layers - 1) * 5.5, y + h * .78, label, size=30, color=INK, weight=700, anchor="middle")
    if sublabel:
        svg.text(x + w / 2 + (layers - 1) * 5.5, y + h + 34, sublabel, size=25, color=MUTED, anchor="middle")


def graph_icon(svg: SVG, x, y, w, h, color, mode="prior", node_fill=WHITE, node_r=7):
    pts = [(0.10, .18), (.50, .10), (.90, .18), (.20, .72), (.52, .50),
           (.84, .72), (.50, .93)]
    nodes = [(x + px * w, y + py * h) for px, py in pts]
    edges = [(0, 1), (1, 2), (0, 4), (1, 4), (2, 4), (3, 4),
             (4, 5), (3, 6), (5, 6), (1, 6)]
    if mode == "prior":
        widths = [2.2, 2.2, 1.8, 1.8, 1.8, 1.8, 1.8, 2.2, 2.2, 1.4]
        alphas = [.65] * len(edges)
    elif mode == "learned":
        widths = [4.5, 1.5, 3.8, 2.0, 4.8, 1.6, 3.5, 2.2, 4.3, 1.2]
        alphas = [1.0, .45, .9, .55, .95, .48, .82, .55, .9, .38]
    else:
        widths = [3.8, 1.6, 3.2, 1.9, 4.1, 1.7, 3.0, 2.0, 3.8, 1.2]
        alphas = [1.0, .50, .95, .58, .98, .50, .85, .58, .9, .40]
    for (a, b), sw, op in zip(edges, widths, alphas):
        svg.line(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], stroke=color, sw=sw, opacity=op)
    for cx, cy in nodes:
        svg.circle(cx, cy, node_r, fill=node_fill, stroke=color, sw=2.0)


def sparse_graph(svg: SVG, x, y, w, h, color, k=5, dense=False, label_y=None, label=None):
    pts = [(0.12, .25), (.38, .12), (.72, .20), (.25, .78), (.62, .62), (.90, .78)]
    nodes = [(x + px * w, y + py * h) for px, py in pts]
    edges = [(0, 1), (1, 2), (0, 4), (1, 4), (2, 4), (3, 4), (4, 5), (3, 5)]
    if dense:
        edges += [(0, 3), (2, 5), (1, 5), (2, 3)]
    for i, (a, b) in enumerate(edges):
        svg.line(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], stroke=color,
                 sw=1.6 if dense else 2.5, opacity=.42 if dense else .85)
    for cx, cy in nodes:
        svg.circle(cx, cy, 6, fill=WHITE, stroke=color, sw=1.8)
    svg.text(x + w / 2, label_y if label_y is not None else y + h + 28,
             label or f"K={k}", size=24, color=color, weight=700, anchor="middle")


def tcn_stack(svg: SVG, x, y, w, h, color, title, detail):
    block_h = 17
    for i in range(4):
        yy = y + i * (block_h + 7)
        svg.rect(x + i * 9, yy, w - i * 18, block_h, fill=WHITE, stroke=color, sw=1.7, rx=4)
    svg.text(x + w / 2, y + h + 28, title, size=27, color=color, weight=700, anchor="middle")
    svg.text(x + w / 2, y + h + 58, detail, size=22, color=MUTED, anchor="middle")


def bidiffusion(svg: SVG, x, y, w, color, label="2-hop Bi-Diffusion"):
    left, mid, right = x + 20, x + w / 2, x + w - 20
    svg.circle(left, y, 7, fill=WHITE, stroke=color, sw=2.0)
    svg.circle(mid, y, 8, fill=color, stroke=color, sw=1.0)
    svg.circle(right, y, 7, fill=WHITE, stroke=color, sw=2.0)
    arrow(svg, left + 12, y, mid - 13, y, color=color, sw=2.4, marker="arrowFine" if color == FINE_DARK else "arrowCoarse")
    arrow(svg, mid + 13, y, right - 12, y, color=color, sw=2.4, marker="arrowFine" if color == FINE_DARK else "arrowCoarse")
    arrow(svg, mid - 2, y - 13, left + 10, y - 13, color=color, sw=1.5, marker="arrowFine" if color == FINE_DARK else "arrowCoarse")
    arrow(svg, right - 10, y + 13, mid + 2, y + 13, color=color, sw=1.5, marker="arrowFine" if color == FINE_DARK else "arrowCoarse")
    svg.text(x + w / 2, y + 43, label, size=18, color=color, weight=700, anchor="middle")


def prompt_tokens(svg: SVG, x, y, count=4, w=42, h=48, gap=9, label=True):
    for i in range(count):
        xx = x + i * (w + gap)
        svg.rect(xx, y, w, h, fill=PURPLE_FILL, stroke=PURPLE, sw=1.8, rx=6)
        if label:
            svg.text(xx + w / 2, y + h * .64, f"P{i + 1}", size=24, color=PURPLE_DARK, weight=700, anchor="middle")


def relation_node(svg: SVG, cx, cy, label="A", color=PURPLE_DARK, r=30):
    svg.circle(cx, cy, r, fill=WHITE, stroke=color, sw=2.5)
    svg.circle(cx, cy, r * .48, fill=PURPLE_FILL, stroke=color, sw=1.4)
    svg.text(cx, cy + 9, label, size=28, color=color, weight=700, anchor="middle")


def gate_diamond(svg: SVG, cx, cy, size=38):
    p = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
    for (x1, y1), (x2, y2) in zip(p, p[1:] + p[:1]):
        svg.line(x1, y1, x2, y2, stroke=PURPLE_DARK, sw=2.5)
    svg.circle(cx, cy, size * .52, fill=PURPLE_FILL, stroke=PURPLE_DARK, sw=1.2)
    svg.text(cx, cy + 9, "G", size=20, color=PURPLE_DARK, weight=700, anchor="middle")


def residual_add(svg: SVG, cx, cy, color):
    svg.circle(cx, cy, 23, fill=WHITE, stroke=color, sw=2.4)
    svg.line(cx - 11, cy, cx + 11, cy, stroke=color, sw=2.2)
    svg.line(cx, cy - 11, cx, cy + 11, stroke=color, sw=2.2)


def forecast_strip(svg: SVG, x, y, w, h, color=SHARED_STROKE, show_label=True):
    svg.line(x, y + h, x + w, y + h, stroke=GRID, sw=1.0)
    pts = [(0.00, .74), (.12, .67), (.24, .73), (.36, .50), (.49, .58),
           (.61, .37), (.74, .44), (.86, .20), (1.00, .28)]
    d = "M " + " ".join(f"{x + px * w:.1f},{y + py * h:.1f}" for px, py in pts)
    svg.path(d, stroke=color, sw=3.2, fill="none")
    for i in range(10):
        xx = x + i * w / 9
        svg.line(xx, y + h - 6, xx, y + h + 6, stroke=color, sw=1.0)
    if show_label:
        svg.text(x + w / 2, y + h + 33, "future horizon", size=23, color=MUTED, anchor="middle")


def similarity_icon(svg: SVG, x, y, w, h, mode="pearson", color=COARSE_DARK, show_label=True):
    waveform(svg, x, y, w, h * .55, kind="coarse" if mode == "pearson" else "delta", stroke=color, baseline=False)
    if mode == "pearson":
        svg.path(f"M {x + w*.10:.1f},{y + h*.78:.1f} C {x + w*.36:.1f},{y + h*.58:.1f} {x + w*.65:.1f},{y + h*.94:.1f} {x + w*.90:.1f},{y + h*.70:.1f}", stroke=color, sw=2.4)
        label = "rho"
    else:
        svg.path(f"M {x + w*.10:.1f},{y + h*.78:.1f} L {x + w*.34:.1f},{y + h*.40:.1f} L {x + w*.58:.1f},{y + h*.72:.1f} L {x + w*.90:.1f},{y + h*.34:.1f}", stroke=color, sw=2.4)
        label = "cos"
    if show_label:
        svg.circle(x + w / 2, y + h * .82, 17, fill=WHITE, stroke=color, sw=1.6)
        svg.text(x + w / 2, y + h * .88, label, size=20, color=color, weight=700, anchor="middle")


def coordinate_icon(svg: SVG, x, y, w, h, color=SHARED_STROKE):
    svg.line(x, y + h, x + w, y + h, stroke=color, sw=1.4)
    svg.line(x, y + h, x, y, stroke=color, sw=1.4)
    pts = [(0.18, .70), (.42, .30), (.68, .62), (.84, .18), (.32, .85)]
    for px, py in pts:
        svg.circle(x + px * w, y + py * h, 7, fill=WHITE, stroke=color, sw=2.0)


def draw_train_objective(svg: SVG, x, y, w, h):
    svg.rect(x, y, w, h, fill=LOSS_FILL, stroke=LOSS, sw=1.2, rx=20, dash="10 8")
    svg.text(x + 24, y + 38, "Training only · MS-MG-DWU", size=34, color=LOSS, weight=700)
    svg.text(x + w - 24, y + 38, "optimization branch; no forward-path input", size=24, color=LOSS, italic=True, anchor="end")
    # Prediction and target are shown as paired streams rather than generic
    # processing boxes.
    tag(svg, x + 28, y + 72, "Y-hat", fill=WHITE, color=LOSS, w=100, size=17)
    tag(svg, x + 142, y + 72, "Y + mask", fill=WHITE, color=LOSS, w=130, size=17)
    arrow(svg, x + 280, y + 89, x + 350, y + 89, color=LOSS, sw=2.4, marker="arrowLoss")
    tag(svg, x + 368, y + 72, "H3 / H6 / H10", fill=WHITE, color=LOSS, w=190, size=17)
    svg.text(x + 463, y + 132, "prefix difficulty", size=17, color=MUTED, anchor="middle")
    arrow(svg, x + 575, y + 89, x + 640, y + 89, color=LOSS, sw=2.4, marker="arrowLoss")
    tag(svg, x + 660, y + 72, "site difficulty", fill=WHITE, color=LOSS, w=170, size=17)
    svg.text(x + 745, y + 132, "dynamic weights", size=17, color=MUTED, anchor="middle")
    arrow(svg, x + 850, y + 89, x + 920, y + 89, color=LOSS, sw=2.4, marker="arrowLoss")
    svg.rect(x + 940, y + 61, 245, 64, fill=WHITE, stroke=LOSS, sw=1.8, rx=10)
    svg.text(x + 1062, y + 101, "MS-MG-DWU", size=22, color=LOSS, weight=700, anchor="middle")
    svg.text(x + 1295, y + 101, "objective only", size=17, color=LOSS, italic=True, anchor="middle")


def figure1_refined():
    W, H = 2400, 1390
    s = SVG(W, H, "Overall Framework of ST-MGPrompt", physical_width_mm=180.0)
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(42, 66, "ST-MGPrompt", size=64, weight=700)
    s.text(W - 42, 63, "Overall framework · train-only priors", size=31, color=MUTED, weight=700, anchor="end")

    # Four weak stage groups replace the six enclosing walls of the original.
    stage(s, 300, 118, 420, 845, 1, ("Dual Temporal", "Semantics"), fill="#FBFCFD", stroke=GRID)
    stage(s, 745, 118, 720, 845, 2, ("Graph-Temporal", "Encoding"), fill="#FBFCFD", stroke=GRID)
    stage(s, 1490, 118, 500, 845, 3, ("Cross-Granularity", "Fusion"), fill=PURPLE_FILL, stroke=PURPLE, dashed=False)
    stage(s, 2015, 118, 340, 845, 4, "Forecasting", fill="#FBFCFD", stroke=GRID)

    # Input representation.  A quiet text label is used here so the narrow
    # input margin does not turn a long label into a second visual panel.
    s.text(36, 158, "Historical input", size=19, color=MUTED, weight=700)
    tensor_stack(s, 80, 300, 150, 180, color=SHARED_STROKE, fill=SHARED, layers=4, label="X")
    s.text(155, 545, "Historical", size=19, color=INK, weight=700, anchor="middle")
    s.text(155, 575, "wind-farm data", size=19, color=INK, weight=700, anchor="middle")
    s.multiline(155, 625, ["134 turbines", "16 variables", "144 steps / 24 h"], size=23, color=MUTED, gap=31)
    s.multiline(155, 735, ["X ∈ R^{B×144}", "×134×16"], size=18, color=INK, anchor="middle", gap=34)
    # Projection funnel: a visual transformation rather than another box.
    s.line(155, 792, 155, 817, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    s.path("M 88 802 L 222 802 L 193 872 L 117 872 Z", stroke=SHARED_STROKE, sw=1.8, fill=SHARED)
    s.text(155, 845, "16 → 64", size=25, color=INK, weight=700, anchor="middle")
    s.text(155, 903, "shared projection", size=21, color=MUTED, anchor="middle")
    s.path("M 230 838 C 262 838, 274 520, 315 520", stroke=SHARED_STROKE, sw=2.4, marker="arrow")

    # Dual temporal semantics, same timeline and two distinct waveforms.
    s.text(510, 230, "same 144-step grid", size=20, color=MUTED, weight=700, anchor="middle")
    s.line(350, 250, 680, 250, stroke=GRID, sw=1.4)
    for i in range(8):
        xx = 350 + i * 47
        s.line(xx, 242, xx, 258, stroke=GRID, sw=1.2)
    tag(s, 346, 290, "Fine", fill=FINE_FILL, color=FINE_DARK, w=95, size=25)
    sequence_ribbon(s, 350, 350, 320, 92, kind="fine", count=8, stroke=FINE, fill=WHITE, label="local variation")
    s.text(515, 520, "causal local context", size=22, color=FINE_DARK, anchor="middle")
    tag(s, 346, 565, "Coarse", fill=COARSE_FILL, color=COARSE_DARK, w=115, size=25)
    sequence_ribbon(s, 350, 625, 320, 92, kind="coarse", count=8, stroke=COARSE, fill=WHITE, label="long-range trend")
    s.text(515, 785, "w = {6, 18, 36}", size=22, color=COARSE_DARK, anchor="middle")
    s.text(510, 823, "Fine / Coarse = 1/2 + 1/2", size=25, color=MUTED, weight=700, anchor="middle")
    s.path("M 680 396 C 708 396, 715 350, 760 350", stroke=FINE_DARK, sw=2.5, marker="arrowFine")
    s.path("M 680 671 C 708 671, 715 770, 760 770", stroke=COARSE_DARK, sw=2.5, marker="arrowCoarse")

    # Graph-temporal encoding: graph, adaptive edge thickness, propagation,
    # and TCN stack are no longer four identical boxes.
    s.text(790, 230, "two semantic views", size=20, color=MUTED, weight=700)
    s.text(790, 280, "Micro", size=31, color=FINE_DARK, weight=700)
    s.text(790, 312, "ΔWspd + distance", size=21, color=FINE_DARK)
    graph_icon(s, 805, 350, 135, 108, FINE_DARK, mode="adaptive")
    s.multiline(875, 474, ["local", "graph"], size=18, color=MUTED, anchor="middle", gap=34)
    s.text(985, 392, "×", size=40, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1015, 350, 135, 108, FINE_DARK, mode="effective")
    s.multiline(1110, 474, ["adaptive", "weights"], size=18, color=FINE_DARK, anchor="middle", gap=34)
    s.line(1168, 404, 1205, 404, stroke=FINE_DARK, sw=2.5, marker="arrowFine")
    bidiffusion(s, 1205, 404, 120, FINE_DARK, label="2-hop")
    tcn_stack(s, 1350, 352, 86, 89, FINE_DARK, "Fine TCN", "k=3 · Bi-diff.")
    s.text(1392, 535, "d={1,2}", size=14, color=MUTED, anchor="middle")
    s.line(1328, 404, 1346, 404, stroke=FINE_DARK, sw=2.5, marker="arrowFine")

    s.text(790, 620, "Macro", size=31, color=COARSE_DARK, weight=700)
    s.text(790, 652, "Patv trend + distance", size=21, color=COARSE_DARK)
    graph_icon(s, 805, 690, 135, 108, COARSE_DARK, mode="adaptive")
    s.multiline(875, 814, ["trend", "graph"], size=18, color=MUTED, anchor="middle", gap=34)
    s.text(985, 732, "×", size=40, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1015, 690, 135, 108, COARSE_DARK, mode="effective")
    s.multiline(1110, 814, ["adaptive", "weights"], size=18, color=COARSE_DARK, anchor="middle", gap=34)
    s.line(1168, 744, 1205, 744, stroke=COARSE_DARK, sw=2.5, marker="arrowCoarse")
    bidiffusion(s, 1205, 744, 120, COARSE_DARK, label="2-hop")
    tcn_stack(s, 1350, 692, 86, 89, COARSE_DARK, "Coarse TCN", "k=5 · Bi-diff.")
    s.text(1392, 875, "d={1,2,4,8}", size=14, color=MUTED, anchor="middle")
    s.line(1328, 744, 1346, 744, stroke=COARSE_DARK, sw=2.5, marker="arrowCoarse")

    # Macro prompt enters the fusion panel as a compact token object.
    s.path("M 1435 744 C 1470 744, 1472 270, 1510 270", stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")
    s.text(1570, 235, "Macro-trend prompt", size=23, color=PURPLE_DARK, weight=700, anchor="middle")
    prompt_tokens(s, 1495, 270, count=4, w=45, h=52, gap=10)
    s.text(1580, 350, "P = 4 tokens / turbine", size=21, color=PURPLE_DARK, weight=700, anchor="middle")

    # Main cross-fusion story. The purple stage is intentionally the largest
    # visual centre, while Q/K,V details remain for Figure 3.
    s.text(1740, 395, "Trend → Fine", size=27, color=PURPLE_DARK, weight=700, anchor="middle")
    sequence_ribbon(s, 1518, 430, 150, 64, kind="fine", count=4, stroke=FINE, fill=WHITE, compact=True)
    s.text(1590, 520, "Fine history", size=22, color=FINE_DARK, anchor="middle")
    relation_node(s, 1730, 464, "A", PURPLE_DARK, r=31)
    s.path("M 1655 462 L 1697 462", stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.path("M 1580 300 C 1635 320, 1694 350, 1730 424", stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    s.text(1768, 422, "Macro tokens", size=22, color=PURPLE_DARK)
    residual_add(s, 1860, 464, FINE_DARK)
    s.path("M 1762 464 L 1834 464", stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    s.path("M 1886 464 L 1950 464", stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    sequence_ribbon(s, 1900, 430, 78, 64, kind="fine", count=2, stroke=FINE, fill=FINE_FILL, compact=True)
    s.text(1939, 520, "H'f", size=26, color=FINE_DARK, weight=700, anchor="middle")

    s.text(1740, 610, "Recent Fine → Coarse", size=24, color=FINE_DARK, weight=700, anchor="middle")
    sequence_ribbon(s, 1518, 645, 150, 64, kind="coarse", count=4, stroke=COARSE, fill=WHITE, compact=True)
    s.text(1590, 735, "Coarse query", size=20, color=COARSE_DARK, anchor="middle")
    relation_node(s, 1730, 680, "A", PURPLE_DARK, r=31)
    s.path("M 1655 677 L 1697 677", stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    sequence_ribbon(s, 1600, 765, 124, 49, kind="fine", count=3, stroke=FINE, fill=FINE_FILL, compact=True)
    s.text(1660, 846, "Recent Fine = 24", size=21, color=FINE_DARK, anchor="middle")
    s.path("M 1660 790 C 1684 762, 1704 724, 1722 713", stroke=FINE_DARK, sw=2.1, marker="arrowFine")
    residual_add(s, 1860, 680, COARSE_DARK)
    s.path("M 1762 680 L 1834 680", stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    s.path("M 1886 680 L 1950 680", stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    sequence_ribbon(s, 1900, 646, 78, 64, kind="coarse", count=2, stroke=COARSE, fill=COARSE_FILL, compact=True)
    s.text(1939, 735, "H'c", size=26, color=COARSE_DARK, weight=700, anchor="middle")
    gate_diamond(s, 1810, 570, size=34)
    s.text(1880, 575, "gate G", size=17, color=PURPLE_DARK, weight=700, anchor="middle")
    s.path("M 1810 604 L 1860 440", stroke=PURPLE_DARK, sw=1.5, dash="7 6", marker="arrowPurple")
    s.path("M 1810 604 L 1860 657", stroke=PURPLE_DARK, sw=1.5, dash="7 6", marker="arrowPurple")
    s.text(1740, 895, "H'f = LN(Hf + G · O_m→f)", size=22, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1740, 925, "H'c = LN(Hc + (1−G) · O_f→c)", size=22, color=COARSE_DARK, weight=700, anchor="middle")

    # Task-conditioned forecasting and future trajectory.
    s.text(2185, 200, "Context", size=29, color=INK, weight=700, anchor="middle")
    s.path("M 1968 464 C 2030 464, 2042 530, 2088 530", stroke=FINE_DARK, sw=2.2, marker="arrowFine")
    s.path("M 1968 680 C 2030 680, 2042 590, 2088 590", stroke=COARSE_DARK, sw=2.2, marker="arrowCoarse")
    tag(s, 2080, 498, "last + macro mean", fill=SHARED, color=SHARED_STROKE, w=220, size=16)
    s.text(2190, 573, "context vector", size=23, color=MUTED, anchor="middle")
    s.multiline(2190, 670, ["Node-Horizon ST", "Prompt"], size=16, color=PURPLE_DARK, weight=700, anchor="middle", gap=25)
    s.rect(2080, 735, 220, 70, fill=WHITE, stroke=SHARED_STROKE, sw=1.8, rx=10)
    s.text(2190, 778, "Direct decoder · H=10", size=26, color=INK, weight=700, anchor="middle")
    s.line(2190, 808, 2190, 846, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    forecast_strip(s, 2090, 865, 200, 66, color=SHARED_STROKE)
    s.text(2190, 986, "Y-hat in R^{B×10×134}", size=23, color=INK, anchor="middle")
    s.text(2190, 1020, "H3     H6     H10", size=24, color=PURPLE_DARK, weight=700, anchor="middle")

    # Training-only objective is compact and explicitly outside inference.
    draw_train_objective(s, 36, 1080, 1425, 248)
    s.path("M 2190 1040 C 2190 1070, 1460 1050, 1438 1170", stroke=LOSS, sw=1.5, dash="8 7", marker="arrowLoss")
    return s


def figure2_refined():
    W, H = 2400, 1390
    s = SVG(W, H, "Dual Semantic Graph Construction", physical_width_mm=180.0)
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(42, 66, "Dual Semantic Graph Construction", size=64, weight=700)
    s.text(W - 42, 63, "train-only priors · constrained adaptation", size=31, color=MUTED, weight=700, anchor="end")
    stage(s, 35, 118, 365, 1160, 1, "Data sources", fill="#FBFCFD", stroke=GRID)
    stage(s, 425, 118, 835, 1160, 2, "Semantic priors", fill="#FBFCFD", stroke=GRID)
    stage(s, 1285, 118, 1080, 1160, 3, ("Prior-constrained", "adaptation"), fill=PURPLE_FILL, stroke=PURPLE, dashed=False)
    s.text(62, 205, "train-only statistics", size=26, color=LOSS, weight=700)
    s.text(62, 238, "No validation/test statistics", size=23, color=MUTED)

    # Three source types use three visual signatures.
    tensor_stack(s, 90, 315, 220, 118, color=COARSE, fill=COARSE_FILL, layers=4, label="Patv", sublabel="raw + valid mask")
    waveform(s, 125, 350, 150, 30, kind="coarse", stroke=COARSE_DARK, baseline=False)
    s.path("M 315 374 C 370 374, 390 290, 455 290", stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")
    tensor_stack(s, 90, 590, 220, 118, color=FINE, fill=FINE_FILL, layers=4, label="Wspd", sublabel="wind-speed history")
    waveform(s, 125, 625, 150, 30, kind="fine", stroke=FINE_DARK, baseline=False)
    s.path("M 315 649 C 370 649, 390 590, 455 590", stroke=FINE_DARK, sw=2.4, marker="arrowFine")
    s.rect(90, 890, 220, 118, fill=SHARED, stroke=SHARED_STROKE, sw=1.8, rx=12)
    coordinate_icon(s, 120, 912, 150, 60, SHARED_STROKE)
    s.text(200, 1040, "r_i=(x_i,y_i)", size=20, color=SHARED_STROKE, weight=700, anchor="middle")
    s.path("M 315 948 C 370 948, 392 1008, 455 1008", stroke=SHARED_STROKE, sw=2.4, marker="arrow")

    # Macro semantic path.
    s.text(465, 205, "Macro / m", size=32, color=COARSE_DARK, weight=700)
    s.text(465, 248, "trend semantics", size=24, color=MUTED)
    waveform(s, 470, 300, 175, 66, kind="coarse", stroke=COARSE_DARK)
    s.text(558, 403, "causal trend", size=24, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(558, 431, "MA = 36 ≈ 6 h", size=22, color=MUTED, anchor="middle")
    s.line(658, 340, 712, 340, stroke=COARSE_DARK, sw=2.3, marker="arrowCoarse")
    similarity_icon(s, 720, 290, 155, 116, mode="pearson", color=COARSE_DARK)
    s.text(797, 450, "Pearson", size=22, color=COARSE_DARK, weight=700, anchor="middle")
    s.line(885, 340, 930, 340, stroke=COARSE_DARK, sw=2.3, marker="arrowCoarse")
    sparse_graph(s, 940, 290, 125, 92, COARSE_DARK, k=10, dense=False,
                 label_y=438, label="K=10")
    s.text(1002, 472, "trend support", size=16, color=MUTED, anchor="middle")
    s.line(1075, 340, 1116, 340, stroke=COARSE_DARK, sw=2.3, marker="arrowCoarse")
    s.circle(1152, 340, 30, fill=COARSE_FILL, stroke=COARSE_DARK, sw=2.0)
    s.text(1152, 351, "Σ", size=29, color=COARSE_DARK, weight=700, anchor="middle")
    s.multiline(1152, 385, ["0.7 trend", "+ 0.3 distance"], size=15, color=COARSE_DARK, anchor="middle", gap=28)
    s.path("M 1185 340 C 1210 340, 1215 450, 1245 450", stroke=COARSE_DARK, sw=2.3, marker="arrowCoarse")
    graph_icon(s, 1115, 472, 130, 96, COARSE_DARK, mode="prior")
    s.text(1180, 602, "Macro prior", size=25, color=COARSE_DARK, weight=700, anchor="middle")

    # Micro semantic path.
    s.text(465, 540, "Micro / mu", size=32, color=FINE_DARK, weight=700)
    s.text(465, 583, "local-change semantics", size=24, color=MUTED)
    waveform(s, 470, 635, 175, 66, kind="delta", stroke=FINE_DARK)
    s.text(558, 738, "ΔWspd", size=24, color=FINE_DARK, weight=700, anchor="middle")
    s.text(558, 766, "first difference", size=22, color=MUTED, anchor="middle")
    s.line(658, 675, 712, 675, stroke=FINE_DARK, sw=2.3, marker="arrowFine")
    similarity_icon(s, 720, 625, 155, 116, mode="cosine", color=FINE_DARK)
    s.text(797, 766, "cosine", size=22, color=FINE_DARK, weight=700, anchor="middle")
    s.line(885, 675, 930, 675, stroke=FINE_DARK, sw=2.3, marker="arrowFine")
    sparse_graph(s, 940, 625, 125, 92, FINE_DARK, k=5, dense=False,
                 label_y=790, label="K=5")
    s.text(1002, 824, "local support", size=16, color=MUTED, anchor="middle")
    s.line(1075, 675, 1116, 675, stroke=FINE_DARK, sw=2.3, marker="arrowFine")
    s.circle(1152, 675, 30, fill=FINE_FILL, stroke=FINE_DARK, sw=2.0)
    s.text(1152, 686, "Σ", size=29, color=FINE_DARK, weight=700, anchor="middle")
    s.multiline(1152, 720, ["0.3 local", "+ 0.7 distance"], size=15, color=FINE_DARK, anchor="middle", gap=28)
    s.path("M 1185 675 C 1210 675, 1215 790, 1245 790", stroke=FINE_DARK, sw=2.3, marker="arrowFine")
    graph_icon(s, 1115, 812, 130, 96, FINE_DARK, mode="prior")
    s.text(1180, 942, "Micro prior", size=25, color=FINE_DARK, weight=700, anchor="middle")

    # Shared distance support enters both mixtures.
    s.text(465, 944, "Shared distance support", size=28, color=SHARED_STROKE, weight=700)
    s.text(465, 984, "A_d,ij = exp(−||r_i−r_j||² / sigma²)", size=22, color=MUTED)
    s.line(720, 1008, 785, 1008, stroke=SHARED_STROKE, sw=2.2, marker="arrow")
    sparse_graph(s, 800, 950, 135, 78, SHARED_STROKE, k=5, dense=False)
    s.path("M 940 1000 C 1010 1000, 1010 470, 1100 470", stroke=SHARED_STROKE, sw=1.7, dash="8 7", marker="arrow")
    s.path("M 940 1010 C 1010 1010, 1010 810, 1100 810", stroke=SHARED_STROKE, sw=1.7, dash="8 7", marker="arrow")

    # Adaptation: a single visual formula repeated for two branches, using
    # topology and edge thickness to make constrained reweighting factual.
    s.text(1325, 225, "Independent E1, E2 · dim = 10", size=25, color=PURPLE_DARK, weight=700)
    s.text(1325, 265, "reweight allowed edges only", size=21, color=MUTED)
    # Macro row
    s.text(1330, 345, "Macro", size=30, color=COARSE_DARK, weight=700)
    graph_icon(s, 1415, 280, 130, 100, COARSE_DARK, mode="prior")
    s.text(1480, 415, "A^m_prior", size=22, color=COARSE_DARK, anchor="middle")
    s.text(1585, 340, "×", size=38, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1640, 280, 130, 100, PURPLE_DARK, mode="learned")
    s.text(1705, 415, "A^m_adp", size=22, color=PURPLE_DARK, anchor="middle")
    s.line(1795, 330, 1845, 330, stroke=PURPLE_DARK, sw=2.5, marker="arrowPurple")
    graph_icon(s, 1870, 280, 145, 105, INK, mode="effective")
    s.multiline(1942, 400, ["Effective", "macro graph"], size=19, color=INK, weight=700, anchor="middle", gap=30)
    s.text(1942, 465, "normalize(prior × adaptive)", size=15, color=COARSE_DARK, anchor="middle")
    # Micro row
    s.text(1330, 660, "Micro", size=30, color=FINE_DARK, weight=700)
    graph_icon(s, 1415, 595, 130, 100, FINE_DARK, mode="prior")
    s.text(1480, 730, "A^mu_prior", size=22, color=FINE_DARK, anchor="middle")
    s.text(1585, 655, "×", size=38, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1640, 595, 130, 100, PURPLE_DARK, mode="learned")
    s.text(1705, 730, "A^mu_adp", size=22, color=PURPLE_DARK, anchor="middle")
    s.line(1795, 645, 1845, 645, stroke=PURPLE_DARK, sw=2.5, marker="arrowPurple")
    graph_icon(s, 1870, 595, 145, 105, INK, mode="effective")
    s.multiline(1942, 715, ["Effective", "micro graph"], size=19, color=INK, weight=700, anchor="middle", gap=30)
    s.text(1942, 785, "normalize(prior × adaptive)", size=15, color=FINE_DARK, anchor="middle")
    # Support constraint as a visual equality.
    s.line(1410, 850, 2015, 850, stroke=PURPLE, sw=1.6, dash="9 8")
    s.text(1712, 900, "same node set + same support", size=27, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1440, 940, 150, 115, SHARED_STROKE, mode="prior")
    s.text(1515, 1090, "prior edges", size=22, color=MUTED, anchor="middle")
    s.text(1670, 1000, "→", size=43, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_icon(s, 1770, 940, 150, 115, INK, mode="effective")
    s.text(1845, 1090, "changed thickness", size=22, color=INK, anchor="middle")
    s.text(2045, 1160, "edge strength changes; support fixed", size=17, color=MUTED, italic=True, anchor="middle")
    return s


def figure3_refined():
    W, H = 2400, 1320
    s = SVG(W, H, "Macro-Trend Prompt and Bidirectional Cross-Granularity Fusion", physical_width_mm=180.0)
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(42, 66, "Macro-Trend Prompt + Bidirectional Cross-Granularity Fusion", size=50, weight=700)
    s.text(W - 42, 63, "separate histories", size=17, color=MUTED, weight=700, anchor="end")

    # Side ribbons are deliberately light: the mechanism, not two giant
    # colored columns, is the visual subject.
    tag(s, 48, 132, "Fine history", fill=FINE_FILL, color=FINE_DARK, w=190, size=25)
    tag(s, 2162, 132, "Coarse history", fill=COARSE_FILL, color=COARSE_DARK, w=205, size=25)
    sequence_ribbon(s, 60, 230, 390, 84, kind="fine", count=8, stroke=FINE, fill=WHITE, label="Hf · 144 positions")
    sequence_ribbon(s, 2150, 230, 190, 84, kind="coarse", count=4, stroke=COARSE, fill=WHITE, label="Hc")
    # Compact history labels and recent bracket.
    s.text(255, 386, "local variation encoded", size=24, color=FINE_DARK, anchor="middle")
    s.text(2245, 386, "trend context", size=21, color=COARSE_DARK, anchor="middle")
    s.path("M 270 435 L 270 472 M 270 472 L 410 472 M 410 472 L 410 435", stroke=FINE_DARK, sw=2.0)
    s.text(340, 515, "recent Fine = 24 ≈ 4 h", size=25, color=FINE_DARK, weight=700, anchor="middle")
    sequence_ribbon(s, 280, 545, 130, 50, kind="fine", count=4, stroke=FINE, fill=FINE_FILL, compact=True)
    s.path("M 450 272 C 520 272, 580 555, 690 555", stroke=FINE_DARK, sw=2.4, marker="arrowFine")
    s.path("M 2150 272 C 2050 272, 1980 320, 1920 320", stroke=COARSE_DARK, sw=2.4, marker="arrowCoarse")

    # Prompt extraction at the top center: history -> attention weights -> 4
    # token cards.
    s.rect(690, 132, 1160, 250, fill=PURPLE_FILL, stroke=PURPLE, sw=1.4, rx=20)
    s.text(1270, 176, "Macro-Trend Prompt", size=26, color=PURPLE_DARK, weight=700, anchor="middle")
    tag(s, 760, 220, "Hc", fill=COARSE_FILL, color=COARSE_DARK, w=105, size=25)
    s.text(820, 287, "coarse history", size=22, color=MUTED, anchor="middle")
    s.line(880, 256, 1000, 256, stroke=COARSE_DARK, sw=2.3, marker="arrowCoarse")
    # Attention weights as dots, not another processing box.
    s.text(1080, 220, "attention", size=22, color=PURPLE_DARK, weight=700, anchor="middle")
    for i in range(8):
        s.circle(1000 + i * 23, 274 + (i % 3) * 9, 6 + (i % 3) * 2, fill=PURPLE, stroke=WHITE, sw=1.0)
    s.line(1005, 327, 1190, 327, stroke=PURPLE, sw=1.8, marker="arrowPurple")
    s.text(1275, 215, "P = 4", size=21, color=PURPLE_DARK, weight=700, anchor="middle")
    prompt_tokens(s, 1185, 250, count=4, w=52, h=58, gap=12)
    s.text(1275, 346, "trend tokens / turbine", size=19, color=PURPLE_DARK, weight=700, anchor="middle")

    # Central interaction mechanism.
    s.rect(560, 420, 1290, 650, fill="#FCFAFE", stroke=PURPLE, sw=1.6, rx=24)
    s.text(1205, 465, "Bidirectional, source-asymmetric interaction", size=29, color=PURPLE_DARK, weight=700, anchor="middle")

    # Upper direction: Macro Prompt -> Fine.
    s.text(705, 550, "Macro Prompt → Fine", size=31, color=PURPLE_DARK, weight=700)
    s.text(705, 585, "Q = Fine history", size=24, color=FINE_DARK)
    sequence_ribbon(s, 705, 615, 260, 62, kind="fine", count=5, stroke=FINE, fill=WHITE, compact=True)
    s.text(840, 714, "query tokens", size=22, color=FINE_DARK, anchor="middle")
    s.line(980, 646, 1122, 646, stroke=FINE_DARK, sw=2.5, marker="arrowFine")
    relation_node(s, 1170, 646, "A", PURPLE_DARK, r=38)
    s.text(1170, 700, "O_m→f", size=18, color=PURPLE_DARK, weight=700, anchor="middle")
    # The top prompt strip already establishes the token identity; the
    # interaction row therefore uses a single labelled source line instead of
    # repeating four cards in the narrow mechanism panel.
    s.text(1040, 520, "K,V = Macro Prompt", size=19, color=PURPLE_DARK, weight=700, anchor="middle")
    s.path("M 1080 535 L 1160 606", stroke=PURPLE_DARK, sw=2.2, marker="arrowPurple")
    residual_add(s, 1370, 646, FINE_DARK)
    s.line(1210, 646, 1344, 646, stroke=PURPLE_DARK, sw=2.5, marker="arrowPurple")
    s.line(1395, 646, 1510, 646, stroke=FINE_DARK, sw=2.5, marker="arrowFine")
    sequence_ribbon(s, 1518, 615, 240, 62, kind="fine", count=5, stroke=FINE, fill=FINE_FILL, compact=True)
    s.text(1640, 714, "H'f", size=28, color=FINE_DARK, weight=700, anchor="middle")

    # Lower direction: Recent Fine -> Coarse, intentionally not a mirror of
    # the upper source.
    s.text(705, 790, "Recent Fine → Coarse", size=31, color=FINE_DARK, weight=700)
    s.text(705, 825, "Q = Coarse history", size=24, color=COARSE_DARK)
    s.text(1260, 760, "K,V: Recent Fine · 24", size=11, color=FINE_DARK, weight=700, anchor="middle")
    s.path("M 1340 770 L 1210 850", stroke=FINE_DARK, sw=2.1, marker="arrowFine")
    sequence_ribbon(s, 705, 855, 260, 62, kind="coarse", count=5, stroke=COARSE, fill=WHITE, compact=True)
    s.text(840, 954, "query tokens", size=22, color=COARSE_DARK, anchor="middle")
    s.line(980, 886, 1122, 886, stroke=COARSE_DARK, sw=2.5, marker="arrowCoarse")
    relation_node(s, 1170, 886, "A", PURPLE_DARK, r=38)
    s.text(1170, 950, "O_f→c", size=24, color=PURPLE_DARK, weight=700, anchor="middle")
    residual_add(s, 1370, 886, COARSE_DARK)
    s.line(1210, 886, 1344, 886, stroke=PURPLE_DARK, sw=2.5, marker="arrowPurple")
    s.line(1395, 886, 1510, 886, stroke=COARSE_DARK, sw=2.5, marker="arrowCoarse")
    sequence_ribbon(s, 1518, 855, 240, 62, kind="coarse", count=5, stroke=COARSE, fill=COARSE_FILL, compact=True)
    s.text(1640, 954, "H'c", size=28, color=COARSE_DARK, weight=700, anchor="middle")

    # One actual gate controls both residual updates.
    gate_diamond(s, 1430, 775, size=36)
    s.text(1430, 830, "fusion gate", size=22, color=PURPLE_DARK, weight=700, anchor="middle")
    s.path("M 1430 739 L 1375 676", stroke=PURPLE_DARK, sw=1.7, dash="8 7", marker="arrowPurple")
    s.path("M 1430 811 L 1375 858", stroke=PURPLE_DARK, sw=1.7, dash="8 7", marker="arrowPurple")

    s.text(1205, 1015, "H'f = LN(Hf + G · O_m→f)", size=25, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1205, 1050, "H'c = LN(Hc + (1−G) · O_f→c)", size=25, color=COARSE_DARK, weight=700, anchor="middle")

    # Side outputs and direct-decoder continuation.
    s.line(450, 690, 505, 690, stroke=FINE_DARK, sw=2.0, marker="arrowFine")
    tag(s, 60, 690, "updated Fine H'f", fill=FINE_FILL, color=FINE_DARK, w=390, size=26)
    s.line(1850, 888, 2140, 888, stroke=COARSE_DARK, sw=2.0, marker="arrowCoarse")
    tag(s, 1980, 690, "updated Coarse H'c", fill=COARSE_FILL, color=COARSE_DARK, w=360, size=26)
    s.path("M 255 732 C 255 1115, 880 1148, 1080 1148", stroke=SHARED_STROKE, sw=1.6, dash="8 7", marker="arrow")
    s.path("M 2245 732 C 2245 1115, 1530 1148, 1330 1148", stroke=SHARED_STROKE, sw=1.6, dash="8 7", marker="arrow")
    s.rect(1080, 1110, 250, 66, fill=WHITE, stroke=SHARED_STROKE, sw=1.8, rx=10)
    s.text(1205, 1152, "last-state direct decoder", size=25, color=INK, weight=700, anchor="middle")
    s.text(1205, 1240, "Fine asks: what changed locally?     Coarse asks: what is the overall trend?", size=25, color=MUTED, italic=True, anchor="middle")
    return s


def write_svg_pdf_png(svg: SVG, stem: str, png_dpi=600):
    svg_path = OUT / f"{stem}.svg"
    pdf_path = OUT / f"{stem}.pdf"
    png_path = OUT / f"{stem}.png"
    svg_path.write_text(svg.end(), encoding="utf-8")
    # Inkscape is not assumed. ReportLab's SVG parser is not required because
    # the editable SVG is the source; the bundled Poppler converter renders it
    # through a small browser-independent SVG-to-PDF bridge below.
    convert_svg_to_pdf(svg_path, pdf_path)
    render_pdf_to_png(pdf_path, png_path, png_dpi)


def convert_svg_to_pdf(svg_path: Path, pdf_path: Path):
    """Convert SVG to PDF using the bundled Chromium print-to-PDF path if available.

    A minimal, dependency-free fallback writes a PDF by invoking the local
    browser runtime. The SVG remains the authoritative editable source.
    """
    # Prefer the bundled rsvg-convert if a future runtime exposes it.
    for candidate in (
        Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files\Inkscape\inkscape.exe"),
    ):
        if candidate.exists():
            subprocess.run([str(candidate), str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"], check=True)
            return
    # ReportLab can draw a high-fidelity vector mirror for the fixed layouts.
    # The dedicated PDF builders below are called by build_pdf_mirror().
    build_pdf_mirror(svg_path, pdf_path, target_width_mm=180.0)


def render_pdf_to_png(pdf_path: Path, png_path: Path, dpi=220):
    poppler = Path(r"C:\Users\12811\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin")
    pdftoppm = poppler / "pdftoppm.exe"
    prefix = png_path.with_suffix("")
    subprocess.run([str(pdftoppm), "-singlefile", "-png", "-r", str(dpi), str(pdf_path), str(prefix)], check=True)


def build_pdf_mirror(svg_path: Path, pdf_path: Path, target_width_mm: float = 180.0):
    """Create vector PDF from SVG using the SVG primitives through reportlab.

    This is intentionally a small parser for the controlled source emitted by
    this file. Text stays as text in PDF; boxes, lines, curves and arrows stay
    vector paths.
    """
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor, white

    pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
    txt = svg_path.read_text(encoding="utf-8")
    width = float(re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', txt).group(1))
    height = float(re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', txt).group(2))
    scale = (target_width_mm / 25.4 * 72.0) / width
    page_width = width * scale
    page_height = height * scale
    c = canvas.Canvas(str(pdf_path), pagesize=(page_width, page_height), pageCompression=1)
    c.scale(scale, scale)
    c.setFillColor(white)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    def col(v):
        return HexColor(v)

    def parse_style(tag):
        def get(name, default=None):
            m = re.search(rf'{name}="([^"]*)"', tag)
            return m.group(1) if m else default
        return get

    def y(v):
        return height - float(v)

    # Draw back-to-front in source order. Exclude <defs>: marker definitions
    # are instructions for SVG renderers, not visible page geometry.
    body = txt.split("</defs>", 1)[1] if "</defs>" in txt else txt
    for tag in re.findall(r'<rect\b[^>]*/>', body):
        get = parse_style(tag)
        x, yy, w, h = map(float, [get("x"), get("y"), get("width"), get("height")])
        fill, stroke = get("fill"), get("stroke")
        sw = float(get("stroke-width", "1"))
        rx = float(get("rx", "0"))
        c.setFillColor(col(fill) if fill and fill != "none" else white)
        c.setStrokeColor(col(stroke) if stroke and stroke != "none" else white)
        c.setLineWidth(sw)
        if get("stroke-dasharray"):
            c.setDash(5, 4)
        else:
            c.setDash()
        if rx:
            c.roundRect(x, y(yy + h), w, h, rx, fill=1 if fill != "none" else 0, stroke=1 if stroke != "none" else 0)
        else:
            c.rect(x, y(yy + h), w, h, fill=1 if fill != "none" else 0, stroke=1 if stroke != "none" else 0)
    for tag in re.findall(r'<line\b[^>]*/>', body):
        get = parse_style(tag)
        x1, yy1, x2, yy2 = map(float, [get("x1"), get("y1"), get("x2"), get("y2")])
        c.setStrokeColor(col(get("stroke", LINE)))
        c.setLineWidth(float(get("stroke-width", "1")))
        if get("stroke-dasharray"):
            c.setDash(5, 4)
        else:
            c.setDash()
        c.line(x1, y(yy1), x2, y(yy2))
        if get("marker-end"):
            draw_arrowhead(c, x1, y(yy1), x2, y(yy2), col(get("stroke", LINE)))
    for tag in re.findall(r'<path\b[^>]*/>', body):
        get = parse_style(tag)
        d = get("d", "")
        coords = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', d)]
        if len(coords) < 4:
            continue
        c.setStrokeColor(col(get("stroke", LINE)))
        c.setFillColor(col(get("fill")) if get("fill") not in (None, "none") else white)
        c.setLineWidth(float(get("stroke-width", "1")))
        p = c.beginPath()
        nums = iter(coords)
        # Controlled paths are either M x,y + L x,y or cubic-like simple M/L.
        p.moveTo(next(nums), y(next(nums)))
        rest = list(nums)
        for j in range(0, len(rest) - 1, 2):
            p.lineTo(rest[j], y(rest[j + 1]))
        c.drawPath(p, fill=0, stroke=1)
        if get("marker-end") and len(coords) >= 4:
            ex, ey = coords[-2], y(coords[-1])
            px, py = coords[-4], y(coords[-3])
            draw_arrowhead(c, px, py, ex, ey, col(get("stroke", LINE)))
    for tag in re.findall(r'<circle\b[^>]*/>', body):
        get = parse_style(tag)
        c.setFillColor(col(get("fill", WHITE)))
        c.setStrokeColor(col(get("stroke", LINE)))
        c.setLineWidth(float(get("stroke-width", "1")))
        c.circle(float(get("cx")), y(float(get("cy"))), float(get("r")), fill=1, stroke=1)
    for tag in re.findall(r'<text\b[^>]*>.*?</text>', body):
        get = parse_style(tag)
        value = re.sub(r'<[^>]+>', '', tag.split('>', 1)[1].rsplit('</', 1)[0])
        value = unescape(value)
        x = float(get("x")); yy = float(get("y")); size = float(get("font-size", "14px").replace("px", ""))
        color = col(get("fill", INK)); weight = get("font-weight", "400")
        anchor = get("text-anchor", "start")
        c.setFont("Arial-Bold" if weight in ("700", "bold") else "Arial", size)
        c.setFillColor(color)
        tw = c.stringWidth(value, "Arial-Bold" if weight in ("700", "bold") else "Arial", size)
        xx = x - tw / 2 if anchor == "middle" else (x - tw if anchor == "end" else x)
        c.drawString(xx, y(yy) - size * .28, value)
    c.save()


def draw_arrowhead(c, x1, y1, x2, y2, color):
    import math
    c.saveState()
    c.setFillColor(color)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 6
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - size * math.cos(angle - .45), y2 - size * math.sin(angle - .45))
    p.lineTo(x2 - size * math.cos(angle + .45), y2 - size * math.sin(angle + .45))
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def figure1():
    W, H = 2080, 990
    s = SVG(W, H, "Overall Framework of ST-MGPrompt", physical_width_mm=180.0)
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(36, 40, "ST-MGPrompt", size=25, weight=700)
    s.text(1935, 40, "Overall framework", size=14, color=MUTED, weight=700, anchor="end")

    # Main zones.
    zone_y, zone_h = 92, 635
    xs = [30, 240, 505, 935, 1375, 1780]
    ws = [180, 230, 390, 405, 375, 270]
    labels = ["HISTORICAL INPUT", "FIXED DUAL", "SEMANTIC SPATIAL-TEMPORAL ENCODING", "CROSS-GRANULARITY COLLABORATION", "TASK-CONDITIONED FORECASTING", "OUTPUT"]
    fills = [SHARED, SHARED, SHARED, PURPLE_FILL, SHARED, SHARED]
    for x, w, label, fill in zip(xs, ws, labels, fills):
        section_band(s, x, 61, w, label, fill=fill, stroke=GRID)
        s.rect(x, zone_y, w, zone_h, fill=fill if x == 1375 else WHITE, stroke=GRID if x != 935 else PURPLE, sw=1.2, rx=12)
    # Historical input.
    tensor_icon(s, 60, 202, 118, 150, layers=4, label="X")
    s.multiline(120, 394, ["Historical wind-farm", "observations"], size=14, weight=700)
    s.multiline(120, 440, ["134 turbines", "16 variables", "144 × 10 min = 24 h"], size=12, color=MUTED)
    s.text(120, 520, "X ∈ R^{B×144×134×16}", size=12.3, color=INK, anchor="middle")
    box(s, 48, 570, 145, 84, "Shared input", ["Projection", "16 → D = 64"], fill=SHARED, stroke=SHARED_STROKE, accent=SHARED_STROKE, title_size=13)
    arrow(s, 120, 540, 120, 570, color=SHARED_STROKE, marker="arrow")
    arrow(s, 193, 612, 242, 612, marker="arrow")

    # Fixed dual split.
    box(s, 270, 170, 170, 92, "Fine / F", ["same grid", "144 steps"], fill=FINE_FILL, stroke=FINE, accent=FINE, title_size=15)
    curve(s, 300, 238, 110, 18, "fine")
    box(s, 270, 490, 170, 92, "Coarse / C", ["same grid", "144 steps"], fill=COARSE_FILL, stroke=COARSE, accent=COARSE, title_size=15)
    curve(s, 300, 558, 110, 18, "coarse")
    s.text(355, 598, "w={6,18,36}", size=10.8, color=COARSE_DARK, weight=700, anchor="middle")
    arrow(s, 440, 612, 469, 612, marker="arrow")
    s.line(505, 238, 505, 540, stroke=GRID, sw=1.0)
    s.text(505, 332, "1/2", size=12, color=FINE_DARK, weight=700, anchor="middle")
    s.text(505, 467, "1/2", size=12, color=COARSE_DARK, weight=700, anchor="middle")
    arrow(s, 505, 612, 535, 612, marker="arrow")
    s.path("M 470 612 C 485 612, 482 216, 505 216", stroke=FINE_DARK, sw=1.5, marker="arrowFine")
    s.path("M 470 612 C 485 612, 482 536, 505 536", stroke=COARSE_DARK, sw=1.5, marker="arrowCoarse")
    s.text(620, 124, "Two semantic views on one 144-step timeline", size=12.5, color=MUTED, italic=True, anchor="middle")

    # Two aligned lanes.
    lane_y_f, lane_y_c = 162, 490
    lane_boxes = [
        (535, lane_y_f, "Micro Local Graph", ["ΔWspd + distance", "local change relations"], FINE_FILL, FINE, FINE_DARK),
        (535, lane_y_c, "Macro Trend Graph", ["Patv trend + distance", "long-range relations"], COARSE_FILL, COARSE, COARSE_DARK),
    ]
    for x, y0, title, lines, fill, stroke, dark in lane_boxes:
        box(s, x, y0, 170, 82, title, lines, fill=fill, stroke=stroke, accent=stroke, title_size=13)
        box(s, x + 190, y0, 180, 82, "Prior-constrained", ["adaptive reweighting", "prior × learned weights"], fill=WHITE, stroke=dark, accent=stroke, title_size=12.5)
        box(s, x + 390, y0, 165, 82, "Bi-diffusion", ["forward + reverse", "1/2-hop"], fill=WHITE, stroke=dark, accent=stroke, title_size=13)
        tcn_title = "Fine TCN" if y0 == lane_y_f else "Coarse TCN"
        tcn_lines = ["k=3 · d={1,2}", "causal context"] if y0 == lane_y_f else ["k=5 · d={1,2,4,8}", "causal context"]
        box(s, x + 575, y0, 150, 82, tcn_title, tcn_lines, fill=fill, stroke=stroke, accent=stroke, title_size=12.5)
        arrow(s, x + 170, y0 + 41, x + 190, y0 + 41, color=dark, marker="arrowFine" if y0 == lane_y_f else "arrowCoarse")
        arrow(s, x + 370, y0 + 41, x + 390, y0 + 41, color=dark, marker="arrowFine" if y0 == lane_y_f else "arrowCoarse")
        arrow(s, x + 555, y0 + 41, x + 575, y0 + 41, color=dark, marker="arrowFine" if y0 == lane_y_f else "arrowCoarse")
    s.text(610, 269, "Micro = local short-term dynamics", size=11.5, color=FINE_DARK, italic=True, anchor="middle")
    s.text(610, 597, "Macro = long-term trend context", size=11.5, color=COARSE_DARK, italic=True, anchor="middle")

    # Prompt from coarse history and cross-fusion.
    s.path("M 1330 532 C 1355 532, 1350 422, 1375 422", stroke=COARSE_DARK, sw=1.8, marker="arrowCoarse")
    box(s, 1160, 342, 180, 116, "Macro-Trend", ["Prompt"], fill=PURPLE_FILL, stroke=PURPLE, accent=PURPLE, title_size=14)
    for i in range(4):
        s.rect(1190 + i * 28, 404 - i * 2, 22, 31, fill=PURPLE_FILL, stroke=PURPLE, sw=1.2, rx=4)
        s.text(1201 + i * 28, 425 - i * 2, f"P{i+1}", size=9, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1250, 449, "P = 4 tokens / turbine", size=10.5, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1250, 329, "from coarse graph-temporal history", size=10.8, color=COARSE_DARK, italic=True, anchor="middle")
    s.path("M 1340 400 C 1370 400, 1370 265, 1400 265", stroke=PURPLE_DARK, sw=1.8, marker="arrowPurple")
    s.path("M 1340 425 C 1370 425, 1370 548, 1400 548", stroke=PURPLE_DARK, sw=1.8, marker="arrowPurple")
    # Cross-fusion central container.
    s.rect(1395, 158, 335, 450, fill=PURPLE_FILL, stroke=PURPLE, sw=1.5, rx=12)
    s.text(1562, 190, "Bidirectional Cross-Fusion", size=16, weight=700, anchor="middle", color=PURPLE_DARK)
    s.text(1430, 229, "Coarse → Fine", size=12.5, color=PURPLE_DARK, weight=700)
    s.text(1430, 249, "Macro Prompt as K,V", size=11.2, color=MUTED)
    s.text(1430, 269, "trend guidance", size=11.2, color=PURPLE_DARK, italic=True)
    s.text(1430, 407, "Fine → Coarse", size=12.5, color=PURPLE_DARK, weight=700)
    s.text(1430, 427, "Recent Fine as K,V", size=11.2, color=MUTED)
    s.text(1430, 447, "recent-detail correction", size=11.2, color=PURPLE_DARK, italic=True)
    box(s, 1510, 292, 105, 72, "Gated", ["fusion G"], fill=WHITE, stroke=PURPLE, accent=PURPLE, title_size=13)
    s.path("M 1450 290 C 1480 290, 1480 328, 1510 328", stroke=PURPLE_DARK, sw=1.5, marker="arrowPurple")
    s.path("M 1450 465 C 1480 465, 1480 340, 1510 340", stroke=PURPLE_DARK, sw=1.5, marker="arrowPurple")
    s.path("M 1615 328 C 1650 328, 1650 280, 1690 280", stroke=PURPLE_DARK, sw=1.6, marker="arrowPurple")
    s.path("M 1615 340 C 1650 340, 1650 445, 1690 445", stroke=PURPLE_DARK, sw=1.6, marker="arrowPurple")
    s.text(1660, 260, "H′f", size=15, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1660, 465, "H′c", size=15, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(1562, 577, "recent = 24 steps ≈ 4 h", size=11.2, color=MUTED, italic=True, anchor="middle")
    # Fine/coarse TCN outputs feed central box.
    s.path("M 1260 203 C 1320 203, 1320 245, 1395 245", stroke=FINE_DARK, sw=1.7, marker="arrowFine")
    s.path("M 1260 531 C 1320 531, 1320 420, 1395 420", stroke=COARSE_DARK, sw=1.7, marker="arrowCoarse")
    # Task-conditioned forecasting zone.
    box(s, 1418, 640, 175, 68, "Last-State Context", ["fine last + coarse last", "+ mean(4 macro tokens)"], fill=SHARED, stroke=SHARED_STROKE, accent=SHARED_STROKE, title_size=13)
    arrow(s, 1630, 280, 1630, 632, color=FINE_DARK, sw=1.5, dash="5 4", marker="arrowFine")
    arrow(s, 1680, 445, 1680, 632, color=COARSE_DARK, sw=1.5, dash="5 4", marker="arrowCoarse")
    arrow(s, 1505, 708, 1505, 735, color=SHARED_STROKE, marker="arrow")
    box(s, 1610, 640, 205, 68, "Node-Horizon ST Prompt", ["turbine n + future step h", "fixed type / granularity 0"], fill=PURPLE_FILL, stroke=PURPLE, accent=PURPLE, title_size=12.5)
    arrow(s, 1710, 708, 1710, 735, color=PURPLE_DARK, marker="arrowPurple")
    box(s, 1460, 756, 300, 54, "Prompt-conditioned direct decoder", ["non-autoregressive · H = 10"], fill=WHITE, stroke=SHARED_STROKE, accent=SHARED_STROKE, title_size=13)
    arrow(s, 1610, 810, 1610, 846, color=SHARED_STROKE, marker="arrow")
    # Output.
    s.rect(1785, 550, 245, 183, fill=SHARED, stroke=SHARED_STROKE, sw=1.5, rx=12)
    s.text(1908, 580, "10-step power forecast", size=15, weight=700, anchor="middle")
    s.text(1908, 605, "Ŷ ∈ R^{B×10×134}", size=12.5, color=MUTED, anchor="middle")
    s.rect(1810, 630, 194, 22, fill=FINE_FILL, stroke=FINE, sw=1.0, rx=5)
    s.text(1907, 646, "steps 1–3  →  H3", size=11.2, color=FINE_DARK, weight=700, anchor="middle")
    s.rect(1810, 658, 194, 22, fill=COARSE_FILL, stroke=COARSE, sw=1.0, rx=5)
    s.text(1907, 674, "steps 1–6  →  H6", size=11.2, color=COARSE_DARK, weight=700, anchor="middle")
    s.rect(1810, 686, 194, 22, fill=PURPLE_FILL, stroke=PURPLE, sw=1.0, rx=5)
    s.text(1907, 702, "steps 1–10 → H10", size=11.2, color=PURPLE_DARK, weight=700, anchor="middle")
    arrow(s, 1760, 846, 1785, 690, color=SHARED_STROKE, marker="arrow")

    # Train-only graph prior at top.
    s.rect(515, 6, 780, 43, fill=WHITE, stroke=SHARED_STROKE, sw=1.1, rx=8, dash="6 5")
    s.text(535, 24, "TRAIN-ONLY DUAL GRAPH PRIORS", size=11, color=MUTED, weight=700, letter=1.0)
    s.text(790, 24, "Wspd → local change", size=11, color=FINE_DARK, weight=700, anchor="middle")
    s.text(1000, 24, "+ coordinates", size=11, color=MUTED, weight=700, anchor="middle")
    s.text(1170, 24, "Patv_raw → causal trend", size=11, color=COARSE_DARK, weight=700, anchor="middle")
    s.path("M 610 49 C 610 95, 650 95, 650 162", stroke=FINE_DARK, sw=1.5, dash="6 5", marker="arrowFine")
    s.path("M 1210 49 C 1210 86, 518 86, 518 531 L 535 531", stroke=COARSE_DARK, sw=1.5, dash="6 5", marker="arrowCoarse")

    # Training objective, explicitly outside inference.
    s.rect(30, 770, 1360, 185, fill=LOSS_FILL, stroke=LOSS, sw=1.5, rx=12, dash="8 6")
    s.text(55, 800, "TRAINING ONLY · MS-MG-DWU", size=14, color=LOSS, weight=700, letter=1.0)
    box(s, 75, 825, 190, 82, "Prediction Ŷ", ["Target Y", "valid_target_mask"], fill=WHITE, stroke=LOSS, accent=LOSS, title_size=13)
    arrow(s, 265, 866, 325, 866, color=LOSS, marker="arrowLoss")
    box(s, 325, 825, 215, 82, "H3 / H6 / H10", ["difficulty rate", "prefix losses"], fill=WHITE, stroke=LOSS, accent=LOSS, title_size=13)
    arrow(s, 540, 866, 605, 866, color=LOSS, marker="arrowLoss")
    box(s, 605, 825, 210, 82, "Turbine difficulty", ["dynamic site weights"], fill=WHITE, stroke=LOSS, accent=LOSS, title_size=13)
    arrow(s, 815, 866, 880, 866, color=LOSS, marker="arrowLoss")
    box(s, 880, 825, 250, 82, "MS-MG-DWU", ["objective only", "not an inference input"], fill=LOSS_FILL, stroke=LOSS, accent=LOSS, title_size=15)
    s.text(1210, 865, "Dashed boundary = optimization", size=12, color=LOSS, italic=True, anchor="middle")
    s.text(1210, 885, "does not alter the forward path", size=12, color=MUTED, italic=True, anchor="middle")
    # Thin non-inference link.
    s.path("M 1908 735 C 1908 768, 1388 755, 1390 825", stroke=LOSS, sw=1.3, dash="6 5", marker="arrowLoss")
    return s


def graph_network(svg: SVG, x, y, w, h, color, mode="prior"):
    nodes = [(x + 22, y + 18), (x + w - 26, y + 18), (x + 20, y + h - 18), (x + w - 28, y + h - 18), (x + w / 2, y + h / 2)]
    edges = [(0, 1), (0, 4), (1, 4), (2, 4), (3, 4), (2, 3)]
    for a, b in edges:
        svg.line(nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1], stroke=color, sw=1.5 if mode == "prior" else 2.4, opacity=.65)
    for cx, cy in nodes:
        svg.circle(cx, cy, 5.5, fill=WHITE, stroke=color, sw=1.6)


def figure2():
    W, H = 1900, 1120
    s = SVG(W, H, "Dual Semantic Graph Construction", physical_width_mm=180.0)
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(34, 40, "Dual Semantic Graph Construction", size=25, weight=700)
    s.text(1860, 40, "train-only priors + constrained adaptation", size=14, color=MUTED, weight=700, anchor="end")
    # Columns and main containers.
    section_band(s, 34, 65, 340, "TRAIN-ONLY STATISTICS", fill=SHARED, stroke=GRID)
    section_band(s, 410, 65, 460, "PRIOR GRAPH CONSTRUCTION", fill=SHARED, stroke=GRID)
    section_band(s, 910, 65, 440, "PRIOR-CONSTRAINED ADAPTATION", fill=PURPLE_FILL, stroke=PURPLE)
    section_band(s, 1390, 65, 475, "EFFECTIVE GRAPH", fill=SHARED, stroke=GRID)
    s.rect(34, 92, 340, 920, fill=WHITE, stroke=GRID, sw=1.2, rx=12, dash="8 6")
    s.rect(410, 92, 460, 920, fill=WHITE, stroke=GRID, sw=1.2, rx=12)
    s.rect(910, 92, 440, 920, fill=PURPLE_FILL, stroke=PURPLE, sw=1.2, rx=12)
    s.rect(1390, 92, 475, 920, fill=WHITE, stroke=GRID, sw=1.2, rx=12)
    s.text(54, 126, "No validation/test statistics", size=12, color=LOSS, weight=700)
    s.text(54, 148, "No future target information", size=12, color=MUTED)
    # Inputs.
    box(s, 70, 195, 270, 86, "Train-only Patv_raw", ["+ valid_target_mask"], fill=COARSE_FILL, stroke=COARSE, accent=COARSE, title_size=14)
    box(s, 70, 430, 270, 86, "Train-only Wspd", ["wind-speed history"], fill=FINE_FILL, stroke=FINE, accent=FINE, title_size=14)
    box(s, 70, 665, 270, 86, "Turbine coordinates", ["rᵢ = (xᵢ, yᵢ)"], fill=SHARED, stroke=SHARED_STROKE, accent=SHARED_STROKE, title_size=14)
    # Macro lane.
    s.text(435, 123, "MACRO / m", size=13, color=COARSE_DARK, weight=700)
    box(s, 450, 168, 190, 80, "Causal trend", ["MA window = 36", "≈ 6 h"], fill=COARSE_FILL, stroke=COARSE, accent=COARSE, title_size=13)
    arrow(s, 340, 238, 450, 208, color=COARSE_DARK, marker="arrowCoarse")
    box(s, 665, 168, 170, 80, "Similarity", ["nonnegative", "Pearson"], fill=WHITE, stroke=COARSE, accent=COARSE, title_size=13)
    arrow(s, 640, 208, 665, 208, color=COARSE_DARK, marker="arrowCoarse")
    box(s, 450, 286, 190, 80, "K = 10", ["sparsification", "trend support"], fill=WHITE, stroke=COARSE, accent=COARSE, title_size=13)
    arrow(s, 750, 248, 545, 286, color=COARSE_DARK, marker="arrowCoarse")
    box(s, 665, 286, 170, 80, "Trend similarity", ["Aᵐ_trend"], fill=COARSE_FILL, stroke=COARSE, accent=COARSE, title_size=13)
    arrow(s, 640, 326, 665, 326, color=COARSE_DARK, marker="arrowCoarse")
    # Micro lane.
    s.text(435, 437, "MICRO / μ", size=13, color=FINE_DARK, weight=700)
    box(s, 450, 482, 190, 80, "Local change", ["ΔWspd", "first difference"], fill=FINE_FILL, stroke=FINE, accent=FINE, title_size=13)
    arrow(s, 340, 473, 450, 522, color=FINE_DARK, marker="arrowFine")
    box(s, 665, 482, 170, 80, "Similarity", ["nonnegative", "cosine"], fill=WHITE, stroke=FINE, accent=FINE, title_size=13)
    arrow(s, 640, 522, 665, 522, color=FINE_DARK, marker="arrowFine")
    box(s, 450, 600, 190, 80, "K = 5", ["sparsification", "local support"], fill=WHITE, stroke=FINE, accent=FINE, title_size=13)
    arrow(s, 750, 562, 545, 600, color=FINE_DARK, marker="arrowFine")
    box(s, 665, 600, 170, 80, "Local similarity", ["Aᵘ_local"], fill=FINE_FILL, stroke=FINE, accent=FINE, title_size=13)
    arrow(s, 640, 640, 665, 640, color=FINE_DARK, marker="arrowFine")
    # Distance lane shared.
    s.text(435, 755, "SHARED DISTANCE PRIOR", size=13, color=MUTED, weight=700)
    box(s, 450, 792, 190, 80, "Distance kernel", ["Aᵈᵢⱼ = exp(-‖rᵢ-rⱼ‖²/σ²)"], fill=SHARED, stroke=SHARED_STROKE, accent=SHARED_STROKE, title_size=12.5)
    arrow(s, 340, 708, 450, 832, color=SHARED_STROKE, marker="arrow")
    box(s, 665, 792, 170, 80, "K = 5", ["row normalization", "distance prior"], fill=WHITE, stroke=SHARED_STROKE, accent=SHARED_STROKE, title_size=13)
    arrow(s, 640, 832, 665, 832, color=SHARED_STROKE, marker="arrow")
    # Adaptation column.
    s.text(935, 130, "Each branch has independent E₁, E₂", size=12.5, color=PURPLE_DARK, weight=700)
    s.text(1130, 156, "dim = 10", size=11.5, color=MUTED, anchor="middle")
    # Macro adaptation.
    box(s, 950, 190, 175, 92, "Macro prior", ["A^m_prior"], fill=COARSE_FILL, stroke=COARSE, accent=COARSE, title_size=13)
    box(s, 1160, 190, 175, 92, "Macro E1, E2", ["similarity scores", "row-normalized"], fill=PURPLE_FILL, stroke=PURPLE, accent=PURPLE, title_size=13)
    arrow(s, 1125, 236, 1160, 236, color=PURPLE_DARK, marker="arrowPurple")
    s.text(1138, 304, "×", size=22, color=PURPLE_DARK, weight=700, anchor="middle")
    box(s, 950, 330, 385, 92, "Effective Macro Graph", ["A^m = normalize(A^m_prior * A^m_adp)"], fill=COARSE_FILL, stroke=COARSE_DARK, accent=COARSE, title_size=14)
    arrow(s, 1040, 282, 1040, 330, color=COARSE_DARK, marker="arrowCoarse")
    arrow(s, 1245, 282, 1245, 330, color=PURPLE_DARK, marker="arrowPurple")
    # Micro adaptation.
    box(s, 950, 550, 175, 92, "Micro prior", ["A^mu_prior"], fill=FINE_FILL, stroke=FINE, accent=FINE, title_size=13)
    box(s, 1160, 550, 175, 92, "Micro E1, E2", ["similarity scores", "row-normalized"], fill=PURPLE_FILL, stroke=PURPLE, accent=PURPLE, title_size=13)
    arrow(s, 1125, 596, 1160, 596, color=PURPLE_DARK, marker="arrowPurple")
    s.text(1138, 664, "×", size=22, color=PURPLE_DARK, weight=700, anchor="middle")
    box(s, 950, 690, 385, 92, "Effective Micro Graph", ["A^mu = normalize(A^mu_prior * A^mu_adp)"], fill=FINE_FILL, stroke=FINE_DARK, accent=FINE, title_size=14)
    arrow(s, 1040, 642, 1040, 690, color=FINE_DARK, marker="arrowFine")
    arrow(s, 1245, 642, 1245, 690, color=PURPLE_DARK, marker="arrowPurple")
    s.text(1138, 835, "Reweight allowed edges only", size=12.5, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1138, 856, "Adaptive weights do not create unsupported links", size=11.5, color=MUTED, italic=True, anchor="middle")
    # Effective graph right side.
    s.text(1420, 130, "Prior support", size=12.5, color=MUTED, weight=700)
    graph_network(s, 1420, 157, 120, 92, SHARED_STROKE, "prior")
    s.text(1563, 213, "×", size=24, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1632, 130, "Learned weights", size=12.5, color=PURPLE_DARK, weight=700, anchor="middle")
    graph_network(s, 1572, 157, 120, 92, PURPLE, "learned")
    arrow(s, 1698, 203, 1744, 203, color=PURPLE_DARK, marker="arrowPurple")
    s.text(1790, 130, "Effective graph", size=12.5, color=INK, weight=700, anchor="middle")
    graph_network(s, 1730, 157, 120, 92, INK, "effective")
    # Branch outcome cards.
    s.rect(1420, 338, 390, 240, fill=COARSE_FILL, stroke=COARSE, sw=1.2, rx=10)
    s.text(1442, 370, "MACRO / m", size=13, color=COARSE_DARK, weight=700)
    s.text(1442, 400, "Aᵐ_prior", size=16, color=COARSE_DARK, weight=700)
    s.text(1537, 400, "*", size=21, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1627, 400, "Aᵐ_adp", size=16, color=PURPLE_DARK, weight=700)
    arrow(s, 1490, 430, 1490, 470, color=COARSE_DARK, marker="arrowCoarse")
    arrow(s, 1665, 430, 1665, 470, color=PURPLE_DARK, marker="arrowPurple")
    s.text(1615, 512, "normalize → Aᵐ", size=13, color=COARSE_DARK, weight=700, anchor="middle")
    graph_network(s, 1690, 437, 90, 72, COARSE_DARK, "effective")
    s.rect(1420, 650, 390, 240, fill=FINE_FILL, stroke=FINE, sw=1.2, rx=10)
    s.text(1442, 682, "MICRO / μ", size=13, color=FINE_DARK, weight=700)
    s.text(1442, 712, "Aᵘ_prior", size=16, color=FINE_DARK, weight=700)
    s.text(1537, 712, "*", size=21, color=PURPLE_DARK, weight=700, anchor="middle")
    s.text(1627, 712, "Aᵘ_adp", size=16, color=PURPLE_DARK, weight=700)
    arrow(s, 1490, 742, 1490, 782, color=FINE_DARK, marker="arrowFine")
    arrow(s, 1665, 742, 1665, 782, color=PURPLE_DARK, marker="arrowPurple")
    s.text(1615, 824, "normalize → Aᵘ", size=13, color=FINE_DARK, weight=700, anchor="middle")
    graph_network(s, 1690, 749, 90, 72, FINE_DARK, "effective")
    # Mix weights explicit.
    s.text(690, 930, "Macro: 0.7 trend + 0.3 distance", size=12.2, color=COARSE_DARK, weight=700, anchor="middle")
    s.text(690, 952, "Micro: 0.3 local + 0.7 distance", size=12.2, color=FINE_DARK, weight=700, anchor="middle")
    s.path("M 835 326 C 870 326, 880 375, 950 375", stroke=COARSE_DARK, sw=1.5, dash="6 5", marker="arrowCoarse")
    s.path("M 835 640 C 870 640, 880 735, 950 735", stroke=FINE_DARK, sw=1.5, dash="6 5", marker="arrowFine")
    s.path("M 835 832 C 900 832, 900 255, 950 255", stroke=SHARED_STROKE, sw=1.4, dash="6 5", marker="arrow")
    s.path("M 835 832 C 900 832, 900 615, 950 615", stroke=SHARED_STROKE, sw=1.4, dash="6 5", marker="arrow")
    s.text(860, 770, "shared distance support", size=10.8, color=MUTED, italic=True, anchor="middle")
    return s


def figure3():
    W, H = 1900, 1040
    s = SVG(W, H, "Macro-Trend Prompt and Bidirectional Cross-Granularity Fusion", physical_width_mm=180.0)
    s.rect(0, 0, W, H, fill=WHITE, stroke=WHITE, sw=0, rx=0)
    s.text(34, 40, "Macro-Trend Prompt + Bidirectional Cross-Granularity Fusion", size=24, weight=700)
    s.text(1860, 40, "historical representations remain separate", size=14, color=MUTED, weight=700, anchor="end")
    # Inputs left and right.
    section_band(s, 40, 70, 410, "FINE HISTORY", fill=FINE_FILL, stroke=FINE)
    section_band(s, 1450, 70, 410, "COARSE HISTORY", fill=COARSE_FILL, stroke=COARSE)
    s.rect(40, 98, 410, 760, fill=FINE_FILL, stroke=FINE, sw=1.4, rx=12)
    s.rect(1450, 98, 410, 760, fill=COARSE_FILL, stroke=COARSE, sw=1.4, rx=12)
    # History cards.
    box(s, 80, 155, 330, 92, "Fine Graph-Temporal History", ["H_f ∈ R^{B×L×N×D}", "local variation encoded"], fill=WHITE, stroke=FINE, accent=FINE, title_size=14)
    box(s, 1490, 155, 330, 92, "Coarse Graph-Temporal History", ["H_c ∈ R^{B×L×N×D}", "trend context encoded"], fill=WHITE, stroke=COARSE, accent=COARSE, title_size=14)
    for i in range(6):
        s.rect(85 + i * 48, 305, 38, 52, fill=WHITE, stroke=FINE, sw=1.0, rx=4)
        curve(s, 92 + i * 48, 321, 24, 20, "fine", stroke=FINE_DARK)
        s.rect(1495 + i * 48, 305, 38, 52, fill=WHITE, stroke=COARSE, sw=1.0, rx=4)
        curve(s, 1502 + i * 48, 321, 24, 20, "coarse", stroke=COARSE_DARK)
    s.text(245, 386, "144 aligned time positions", size=12, color=FINE_DARK, italic=True, anchor="middle")
    s.text(1655, 386, "144 aligned time positions", size=12, color=COARSE_DARK, italic=True, anchor="middle")
    # recent fine highlight.
    s.rect(73, 415, 344, 116, fill=WHITE, stroke=FINE_DARK, sw=1.6, rx=8, dash="6 5")
    s.text(245, 444, "Recent Fine History", size=14, color=FINE_DARK, weight=700, anchor="middle")
    s.text(245, 470, "last 24 steps ≈ 4 h", size=12.5, color=MUTED, anchor="middle")
    for i in range(4):
        s.rect(135 + i * 55, 488, 42, 26, fill=FINE_FILL, stroke=FINE, sw=1.0, rx=4)
    s.path("M 417 472 C 500 472, 520 700, 650 700", stroke=FINE_DARK, sw=2.0, marker="arrowFine")
    # Macro prompt top center.
    s.rect(600, 92, 700, 190, fill=PURPLE_FILL, stroke=PURPLE, sw=1.5, rx=12)
    s.text(950, 122, "NODE-SPECIFIC MACRO-TREND PROMPT", size=14, color=PURPLE_DARK, weight=700, anchor="middle", letter=0.6)
    box(s, 650, 158, 170, 78, "H_c", ["coarse history"], fill=WHITE, stroke=COARSE, accent=COARSE, title_size=14)
    box(s, 860, 158, 170, 78, "Attention pool", ["over L positions"], fill=WHITE, stroke=PURPLE, accent=PURPLE, title_size=13)
    box(s, 1070, 158, 170, 78, "Projection", ["P = 4"], fill=WHITE, stroke=PURPLE, accent=PURPLE, title_size=13)
    arrow(s, 820, 197, 860, 197, color=PURPLE_DARK, marker="arrowPurple")
    arrow(s, 1030, 197, 1070, 197, color=PURPLE_DARK, marker="arrowPurple")
    s.path("M 1570 247 C 1570 285, 1300 290, 1238 290", stroke=COARSE_DARK, sw=2.0, marker="arrowCoarse")
    # Prompt cards under central header.
    s.text(950, 266, "4 trend tokens per turbine", size=11.5, color=PURPLE_DARK, italic=True, anchor="middle")
    for i in range(4):
        s.rect(850 + i * 55, 300, 43, 52, fill=PURPLE_FILL, stroke=PURPLE, sw=1.3, rx=5)
        s.text(871 + i * 55, 331, f"P{i+1}", size=13, color=PURPLE_DARK, weight=700, anchor="middle")
    # Cross-fusion central box.
    s.rect(560, 390, 780, 420, fill=WHITE, stroke=PURPLE, sw=1.6, rx=15)
    s.text(950, 424, "BIDIRECTIONAL · SOURCE-ASYMMETRIC", size=15, color=PURPLE_DARK, weight=700, anchor="middle", letter=0.8)
    # Upper direction.
    s.rect(625, 462, 650, 115, fill=PURPLE_FILL, stroke=PURPLE, sw=1.2, rx=10)
    s.text(655, 490, "Coarse → Fine", size=14, color=PURPLE_DARK, weight=700)
    s.text(655, 516, "Q = Fine history", size=12.2, color=MUTED)
    s.text(655, 541, "K,V = Macro Prompt", size=12.2, color=PURPLE_DARK, weight=700)
    box(s, 1055, 483, 170, 66, "Cross-attention", ["O_m→f"], fill=WHITE, stroke=PURPLE, accent=PURPLE, title_size=13)
    arrow(s, 945, 520, 1055, 520, color=PURPLE_DARK, marker="arrowPurple")
    s.text(950, 566, "Trend guidance", size=11.5, color=PURPLE_DARK, italic=True, anchor="middle")
    # Lower direction.
    s.rect(625, 615, 650, 115, fill=PURPLE_FILL, stroke=PURPLE, sw=1.2, rx=10)
    s.text(655, 643, "Fine → Coarse", size=14, color=PURPLE_DARK, weight=700)
    s.text(655, 669, "Q = Coarse history", size=12.2, color=MUTED)
    s.text(655, 694, "K,V = Recent Fine (24)", size=12.2, color=FINE_DARK, weight=700)
    box(s, 1055, 636, 170, 66, "Cross-attention", ["O_f→c"], fill=WHITE, stroke=PURPLE, accent=PURPLE, title_size=13)
    arrow(s, 945, 673, 1055, 673, color=PURPLE_DARK, marker="arrowPurple")
    s.text(950, 719, "Recent-detail correction", size=11.5, color=PURPLE_DARK, italic=True, anchor="middle")
    # Gate and equations.
    box(s, 730, 757, 215, 72, "Fusion Gate G", ["controls information", "amount"], fill=PURPLE_FILL, stroke=PURPLE, accent=PURPLE, title_size=14)
    s.path("M 1140 550 C 1140 740, 970 760, 945 780", stroke=PURPLE_DARK, sw=1.4, marker="arrowPurple")
    s.path("M 1140 700 C 1140 760, 970 790, 945 801", stroke=PURPLE_DARK, sw=1.4, marker="arrowPurple")
    s.text(965, 782, "H'f = LN(H_f + G * O_m->f)", size=12.5, color=FINE_DARK, weight=700)
    s.text(965, 809, "H'c = LN(H_c + (1-G) * O_f->c)", size=12.5, color=COARSE_DARK, weight=700)
    # Outputs.
    box(s, 80, 625, 330, 92, "Updated Fine History H′f", ["local representation retained"], fill=WHITE, stroke=FINE, accent=FINE, title_size=14)
    box(s, 1490, 625, 330, 92, "Updated Coarse History H′c", ["trend representation retained"], fill=WHITE, stroke=COARSE, accent=COARSE, title_size=14)
    s.path("M 560 540 C 500 540, 500 670, 410 670", stroke=FINE_DARK, sw=1.8, marker="arrowFine")
    s.path("M 1340 690 C 1400 690, 1400 670, 1490 670", stroke=COARSE_DARK, sw=1.8, marker="arrowCoarse")
    # Small decoder continuation.
    s.path("M 410 717 C 430 840, 1440 840, 1490 717", stroke=SHARED_STROKE, sw=1.3, dash="6 5", marker="arrow")
    s.text(950, 870, "→ last-state direct decoder", size=12, color=MUTED, italic=True, anchor="middle")
    s.path("M 1070 352 C 1150 380, 1220 400, 1260 462", stroke=PURPLE_DARK, sw=1.6, marker="arrowPurple")
    s.text(1160, 402, "Macro Prompt → Fine", size=11.5, color=PURPLE_DARK, italic=True, anchor="middle")
    s.path("M 1180 352 C 1290 352, 1330 870, 1490 870", stroke=PURPLE, sw=1.2, dash="6 5", marker="arrowPurple")
    s.text(950, 940, "Fine asks: what changed locally?     Coarse asks: what is the overall trend?", size=13, color=MUTED, italic=True, anchor="middle")
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Refined_v2 is a preserved reference set.  This entry point emits only
    # the new publication-layout pass so running the generator cannot replace
    # the v2 artifacts with a different visual version.
    write_v3_outputs(figure1_v3(), "Fig1_STMGPrompt_Overall_Framework_refined_v3")
    write_v3_outputs(figure2_v3(), "Fig2_Dual_Semantic_Graph_Construction_refined_v3")
    write_v3_outputs(figure3_v3_publication(), "Fig3_MacroPrompt_Bidirectional_CrossFusion_refined_v3")
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
