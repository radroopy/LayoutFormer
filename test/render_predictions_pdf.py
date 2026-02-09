from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typing import Any


try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF is required for PDF rendering. Install with: pip install pymupdf\n"
        f"Import error: {exc}"
    )

try:
    from PIL import Image, ImageChops, ImageDraw
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required for image operations. Install with: pip install pillow\n"
        f"Import error: {exc}"
    )

from boundary_resample import load_contour_points

NORM_RANGE = 10.0

@dataclass
class BBox:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return float(self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return float(self.max_y - self.min_y)


def _bbox(points: list[tuple[float, float]]) -> BBox:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _resolve_rel(project_root: Path, root: Path, value: Any) -> Path:
    p = Path(str(value))
    if p.is_absolute():
        return p
    cand = project_root / p
    if cand.exists():
        return cand
    return root / p


@lru_cache(maxsize=256)
def _render_pdf_first_page(pdf_path: str, render_dpi: int) -> Image.Image:
    """Render the first page of a PDF to an RGBA PIL image (cached)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        mat = fitz.Matrix(render_dpi / 72.0, render_dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=True)
        img = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
        return img
    finally:
        doc.close()


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()




def _trim_rendered_element(img: Image.Image) -> Image.Image:
    """Trim transparent/white margins to avoid large blank rectangles."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Trim fully transparent margins first (if any).
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Trim pure-white margins (common for PDFs with white page background).
    rgb = img.convert("RGB")
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, white)
    bbox2 = diff.getbbox()
    if bbox2:
        img = img.crop(bbox2)

    return img


def _has_white_background(img: Image.Image, thresh: int = 245) -> bool:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w == 0 or h == 0:
        return False
    samples = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((w - 1, 0)),
        rgb.getpixel((0, h - 1)),
        rgb.getpixel((w - 1, h - 1)),
    ]
    for r, g, b in samples:
        if (r + g + b) / 3.0 < thresh:
            return False
    return True


def _white_to_transparent(img: Image.Image, thresh: int = 12) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    rgb = img.convert("RGB")
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, white).convert("L")
    # keep non-white pixels (diff > thresh)
    mask = diff.point(lambda x: 0 if x <= thresh else 255)
    alpha = img.split()[-1]
    new_alpha = ImageChops.multiply(alpha, mask)
    img.putalpha(new_alpha)
    return img

def _compute_tgt_scale_from_contour(contour: list[tuple[float, float]]) -> float:
    bb = _bbox(contour)
    return max(bb.width, bb.height)


def _theta_deg_from_sin_cos(s: float, c: float) -> float:
    return math.degrees(math.atan2(float(s), float(c)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render LayoutFormer predictions into a multi-page PDF. "
            "Background: pattern piece boundary as vector path; "
            "Elements: rasterized PDFs pasted at predicted (x,y,w,h,theta)."
        )
    )
    parser.add_argument(
        "--pred",
        default=str(Path(__file__).resolve().parent / "result" / "predictions_test.json"),
        help="Path to predictions_*.json (from test/run_test.py)",
    )
    parser.add_argument(
        '--one-file',
        action='store_true',
        help='Write a single multi-page PDF instead of one PDF per sample.',
    )
    parser.add_argument(
        '--out-dir',
        default=str(Path(__file__).resolve().parent / 'result' / 'predictions_test'),
        help='Output directory (default mode: one PDF per sample).',
    )
    parser.add_argument(
        '--out-file',
        '--out',
        dest='out_file',
        default=str(Path(__file__).resolve().parent / 'result' / 'predictions_test_render.pdf'),
        help='Output PDF path when using --one-file (alias: --out).',
    )
    parser.add_argument(
        "--pattern-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Root used to resolve relative pattern json paths (default: project root)",
    )
    parser.add_argument(
        "--pdf-root",
        default=str(Path(__file__).resolve().parents[1] / "logo" / "logo"),
        help="Root used to resolve relative element pdf paths (default: <project>/logo/logo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of samples/pages (0 means all)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=20.0,
        help="Page margin (in the same coordinate units as json points)",
    )
    
    parser.add_argument(
        "--trim-white",
        action="store_true",
        help="Enable trimming white/transparent margins in element PDFs.",
    )
    parser.add_argument(
        "--no-trim-white",
        action="store_true",
        help="Disable trimming white/transparent margins in element PDFs (overrides --trim-white).",
    )
    parser.add_argument(
        "--render-dpi",
        type=int,
        default=300,
        help="DPI to rasterize element PDFs (higher = sharper but slower)",
    )
    parser.add_argument(
        "--embed-dpi",
        type=int,
        default=144,
        help=(
            "DPI assumed when converting (w,h) units to pixels for resizing. "
            "144 means 2 px per PDF point if you treat 1 unit as 1 point."
        ),
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        default=True,
        help="Disable applying predicted rotation to element PDFs (debug).",
    )
    parser.add_argument(
        "--draw-axes",
        action="store_true",
        help="Draw coordinate axes with ticks/labels for debugging.",
    )
    parser.add_argument(
        "--tick-step",
        type=float,
        default=100.0,
        help="Axis tick step in the same units as json coordinates.",
    )
    parser.add_argument(
        "--theta-sign",
        type=float,
        default=-1.0,
        help="Rotation sign for display (default -1 for y-up to y-down).",
    )
    parser.add_argument(
        "--logo-order",
        choices=["asc", "desc"],
        default="asc",
        help="Draw order by logo_level (asc draws smaller first; desc draws larger first).",
    )
    parser.add_argument(
        "--theta-offset",
        type=float,
        default=0.0,
        help="Additional rotation offset in degrees.",
    )
    parser.add_argument(
        "--draw-box",
        action="store_true",
        help="Draw the rotated (w,h) box outline for each element (debug)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    pattern_root = Path(args.pattern_root)
    pdf_root = Path(args.pdf_root)

    pred_path = Path(args.pred)
    if not pred_path.is_file():
        raise SystemExit(f"predictions json not found: {pred_path}")

    records = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit("predictions json must be a list")

    if args.limit and args.limit > 0:
        records = records[: args.limit]

    out_doc = None
    out_file = Path(args.out_file)
    out_dir = Path(args.out_dir)

    if args.one_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists():
            out_file.unlink()
        out_doc = fitz.open()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Cache contour/bbox per target json.
    contour_cache: dict[str, tuple[list[tuple[float, float]], BBox]] = {}

    missing_pdfs: set[str] = set()

    for idx, rec in enumerate(records):
        tgt_json = rec.get("tgt_json")
        if not tgt_json:
            continue

        tgt_json_path = _resolve_rel(project_root, pattern_root, tgt_json)
        if not tgt_json_path.is_file():
            raise FileNotFoundError(f"target json not found: {tgt_json_path}")

        key = str(Path(tgt_json))
        if key not in contour_cache:
            contour = load_contour_points(str(tgt_json_path))
            bb = _bbox(contour)
            contour_cache[key] = (contour, bb)
        else:
            contour, bb = contour_cache[key]

        width = bb.width
        height = bb.height
        margin = float(args.margin)

        doc = out_doc if out_doc is not None else fitz.open()
        page = doc.new_page(width=width + 2 * margin, height=height + 2 * margin)

        # Helper: convert (x,y) in json (origin bottom-left) -> PyMuPDF page coords (origin top-left).
        def to_page_xy(x: float, y: float) -> tuple[float, float]:
            x_rel = float(x - bb.min_x)
            y_rel = float(y - bb.min_y)
            return (margin + x_rel, margin + (height - y_rel))

        # Precompute boundary polyline and clipping mask.
        boundary_pts = [to_page_xy(x, y) for x, y in contour]
        if boundary_pts and boundary_pts[0] != boundary_pts[-1]:
            boundary_pts.append(boundary_pts[0])

        page_scale = float(args.embed_dpi) / 72.0
        page_px_w = max(1, int(round(page.rect.width * page_scale)))
        page_px_h = max(1, int(round(page.rect.height * page_scale)))
        mask_full = Image.new("L", (page_px_w, page_px_h), 0)
        if boundary_pts:
            poly_px = [(x * page_scale, y * page_scale) for x, y in boundary_pts]
            ImageDraw.Draw(mask_full).polygon(poly_px, fill=255)

        # Optional: draw axes and ticks (in original json coordinates).
        if args.draw_axes:
            axis_shape = page.new_shape()
            x0, y0 = to_page_xy(bb.min_x, bb.min_y)
            x1, _ = to_page_xy(bb.max_x, bb.min_y)
            x2, y2 = to_page_xy(bb.min_x, bb.max_y)
            axis_shape.draw_line((x0, y0), (x1, y0))
            axis_shape.draw_line((x0, y0), (x2, y2))
            axis_shape.finish(color=(0.4, 0.4, 0.4), width=0.7)
            axis_shape.commit()

            tick_step = float(args.tick_step)
            tick_len = 6.0
            tick_shape = page.new_shape()
            # X ticks
            if tick_step > 0:
                xv = math.floor(bb.min_x / tick_step) * tick_step
                while xv <= bb.max_x + 1e-6:
                    px, py = to_page_xy(xv, bb.min_y)
                    tick_shape.draw_line((px, py), (px, py + tick_len))
                    page.insert_text(
                        (px + 2, py + tick_len + 4),
                        f"{xv:.0f}",
                        fontsize=6,
                        color=(0.4, 0.4, 0.4),
                    )
                    xv += tick_step
            # Y ticks
            if tick_step > 0:
                yv = math.floor(bb.min_y / tick_step) * tick_step
                while yv <= bb.max_y + 1e-6:
                    px, py = to_page_xy(bb.min_x, yv)
                    tick_shape.draw_line((px, py), (px - tick_len, py))
                    page.insert_text(
                        (px - tick_len - 20, py + 2),
                        f"{yv:.0f}",
                        fontsize=6,
                        color=(0.4, 0.4, 0.4),
                    )
                    yv += tick_step
            tick_shape.finish(color=(0.4, 0.4, 0.4), width=0.5)
            tick_shape.commit()

        # Title text (metadata)
        src_json = rec.get("src_json")
        src_size = rec.get("src_size")
        tgt_size = rec.get("tgt_size")
        shape_id = rec.get("shape_id")
        title = f"{idx:05d}  {shape_id}  {src_size}->{tgt_size}"
        page.insert_text((margin, margin / 2), title, fontsize=10, color=(0, 0, 0))
        if src_json and tgt_json:
            page.insert_text((margin, margin / 2 + 12), f"src: {src_json}", fontsize=7, color=(0, 0, 0))
            page.insert_text((margin, margin / 2 + 22), f"tgt: {tgt_json}", fontsize=7, color=(0, 0, 0))

        # Determine target scale for denormalization.
        tgt_scale = rec.get("tgt_scale")
        if tgt_scale is None:
            tgt_scale = _compute_tgt_scale_from_contour(contour)
        tgt_scale = float(tgt_scale)

        pdf_paths = rec.get("pdf_paths") or []
        preds = rec.get("pred") or []
        valids = rec.get("valid") or []
        logo_levels = rec.get("logo_levels") or []

        def _lvl(v):
            try:
                return float(v)
            except Exception:
                return -1.0

        max_len = min(len(preds), len(pdf_paths), len(valids))
        if len(logo_levels) < max_len:
            logo_levels = list(logo_levels) + [-1] * (max_len - len(logo_levels))

        element_meta = {}
        for i in range(max_len):
            if float(valids[i]) <= 0.0:
                continue
            if not pdf_paths[i]:
                continue
            vals = preds[i]
            w = float(vals[2]) * tgt_scale / NORM_RANGE
            h = float(vals[3]) * tgt_scale / NORM_RANGE
            area = abs(w * h)
            element_meta[i] = (area, w, h)

        # Sort by area: large -> bottom (draw first), small -> top (draw later)
        order = sorted(element_meta.keys(), key=lambda i: (-element_meta[i][0], i))

        # Render each element and paste.
        for i in order:
            if float(valids[i]) <= 0.0:
                continue
            pdf_rel = pdf_paths[i]
            if not pdf_rel:
                continue

            # Resolve element PDF path.
            pdf_path = _resolve_rel(project_root, pdf_root, pdf_rel)
            if not pdf_path.is_file():
                missing_pdfs.add(str(pdf_rel))
                continue

            vals = preds[i]
            cx_n, cy_n, w_n, h_n = vals[0], vals[1], vals[2], vals[3]
            if len(vals) >= 6:
                s, c = vals[4], vals[5]
            else:
                s, c = 0.0, 1.0
            cx = float(cx_n) * tgt_scale / NORM_RANGE
            cy = float(cy_n) * tgt_scale / NORM_RANGE
            w = float(w_n) * tgt_scale / NORM_RANGE
            h = float(h_n) * tgt_scale / NORM_RANGE

            theta_yup_deg = _theta_deg_from_sin_cos(s, c)
            if args.no_rotate:
                theta_deg = 0.0
            else:
                theta_deg = args.theta_sign * theta_yup_deg + args.theta_offset

            # Rasterize base PDF page (cached) then resize/rotate to match (w,h,theta).
            base_img = _render_pdf_first_page(str(pdf_path), int(args.render_dpi))

            # Disable trimming by default; only trim when explicitly enabled.
            do_trim = args.trim_white and not args.no_trim_white
            if do_trim:
                base_img = _trim_rendered_element(base_img)

            # If the element has a white background, make it transparent.
            if _has_white_background(base_img):
                base_img = _white_to_transparent(base_img)

            # Convert desired (w,h) in page-units into pixels under embed_dpi.
            px_w = max(1, int(round(w * float(args.embed_dpi) / 72.0)))
            px_h = max(1, int(round(h * float(args.embed_dpi) / 72.0)))

            img = base_img.resize((px_w, px_h), resample=Image.BICUBIC)
            img = img.rotate(theta_deg, expand=True, resample=Image.BICUBIC)

            ins_w = img.width * 72.0 / float(args.embed_dpi)
            ins_h = img.height * 72.0 / float(args.embed_dpi)

            cx_p, cy_p = to_page_xy(cx, cy)
            rect = fitz.Rect(cx_p - ins_w / 2, cy_p - ins_h / 2, cx_p + ins_w / 2, cy_p + ins_h / 2)
            # Clip by contour mask: crop mask to element rect, apply as alpha.
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            x0_px = int(round(rect.x0 * page_scale))
            y0_px = int(round(rect.y0 * page_scale))
            x1_px = int(round(rect.x1 * page_scale))
            y1_px = int(round(rect.y1 * page_scale))
            mask_crop = mask_full.crop((x0_px, y0_px, x1_px, y1_px))
            if mask_crop.size != img.size:
                mask_crop = mask_crop.resize(img.size, resample=Image.NEAREST)
            alpha = img.split()[-1]
            img.putalpha(ImageChops.multiply(alpha, mask_crop))
            page.insert_image(rect, stream=_pil_to_png_bytes(img), overlay=True)

            if args.draw_box:
                # Draw the rotated predicted box outline as vector (debug).
                dx = w / 2.0
                dy = h / 2.0

                # Use the original (y-up) angle to rotate the box in the same coordinate system as the json points.
                theta_yup_deg = -theta_deg
                ang = math.radians(theta_yup_deg)
                cos_t = math.cos(ang)
                sin_t = math.sin(ang)

                corners = [
                    (-dx, -dy),
                    (dx, -dy),
                    (dx, dy),
                    (-dx, dy),
                    (-dx, -dy),
                ]
                poly = []
                for ox, oy in corners:
                    rx = ox * cos_t - oy * sin_t
                    ry = ox * sin_t + oy * cos_t
                    poly.append(to_page_xy(cx + rx, cy + ry))

                box_shape = page.new_shape()
                box_shape.draw_polyline(poly)
                box_shape.finish(color=(0, 0, 1), width=0.7)
                box_shape.commit()

        # Draw boundary as a vector polyline on top.
        if boundary_pts:
            shape = page.new_shape()
            shape.draw_polyline(boundary_pts)
            shape.finish(color=(0, 0, 0), width=1.0)
            shape.commit()

        if out_doc is None:
            src_tag = str(src_size) if src_size else 'NA'
            tgt_tag = str(tgt_size) if tgt_size else 'NA'
            tgt_p = Path(str(tgt_json))
            parts = list(tgt_p.parts)
            parts = parts[-3:] if len(parts) >= 3 else parts
            if parts:
                parts[-1] = Path(parts[-1]).stem
            json_tag = '_'.join(parts) if parts else 'NA'
            out_sample = out_dir / f'{idx:05d}_{src_tag}2{tgt_tag}_{json_tag}.pdf'
            if out_sample.exists():
                out_sample.unlink()
            doc.save(str(out_sample), garbage=4, deflate=True)
            doc.close()
            print(f"saved: {out_sample}")

    if out_doc is not None:
        out_doc.save(str(out_file), garbage=4, deflate=True)
        out_doc.close()

    if missing_pdfs:
        miss_path = out_file.with_suffix('.missing_pdfs.txt') if args.one_file else (out_dir / 'missing_pdfs.txt')
        miss_path.write_text('\n'.join(sorted(missing_pdfs)) + '\n', encoding='utf-8')
        print(f'WARNING: some element PDFs were missing. List saved to: {miss_path}')

    if args.one_file:
        print(f'saved: {out_file}')
    else:
        print(f'saved PDFs: {out_dir}')


if __name__ == "__main__":
    main()
