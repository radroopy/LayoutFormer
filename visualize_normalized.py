#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

DEFAULT_LINE_WIDTH = 2


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pick_png(json_path: Path):
    stem = json_path.stem
    if stem.endswith("-n"):
        stem = stem[:-2]
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = json_path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return None


def scale_point(pt, width, height, scale, img_w, img_h, normalized):
    if width == 0 or height == 0:
        return None
    x = pt[0]
    y = pt[1]
    if normalized:
        x *= scale
        y *= scale
    if img_w <= 1:
        x_px = 0.0
    else:
        x_px = x / width * (img_w - 1)
    if img_h <= 1:
        y_px = 0.0
    else:
        y_px = (1.0 - y / height) * (img_h - 1)
    return (x_px, y_px)


def choose_color(item):
    if item.get("isSewing"):
        return (0, 0, 255, 255)
    if item.get("isContour"):
        return (0, 0, 0, 255)
    if item.get("isGuides"):
        return (64, 64, 64, 255)
    return (64, 64, 64, 255)


def draw_path(draw, points, color, line_width, close=False):
    if len(points) < 2:
        return
    if close and points[0] != points[-1]:
        points = points + [points[0]]
    draw.line(points, fill=color, width=line_width)


def draw_item(draw, item, width, height, scale, img_w, img_h, line_width, normalized):
    color = choose_color(item)
    if isinstance(item.get("segments"), list) and item.get("segments"):
        for seg in item.get("segments"):
            if not isinstance(seg, list):
                continue
            pts = [
                scale_point(p, width, height, scale, img_w, img_h, normalized)
                for p in seg
                if isinstance(p, list) and len(p) == 2
            ]
            pts = [p for p in pts if p is not None]
            draw_path(draw, pts, color, line_width, close=False)
        return

    if isinstance(item.get("points"), list) and item.get("points"):
        pts = [
            scale_point(p, width, height, scale, img_w, img_h, normalized)
            for p in item.get("points")
            if isinstance(p, list) and len(p) == 2
        ]
        pts = [p for p in pts if p is not None]
        close = bool(item.get("isContour")) and len(pts) >= 3
        draw_path(draw, pts, color, line_width, close=close)


def render_one(
    json_path: Path,
    png_path: Path,
    out_path: Path,
    line_width: int,
    normalized: bool,
    overlay: bool,
    max_side: int,
):
    data = load_json(json_path)
    width = data.get("width")
    height = data.get("height")

    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        print(f"skip {json_path}: invalid width/height", file=sys.stderr)
        return False

    scale = max(width, height)
    if scale == 0:
        print(f"skip {json_path}: scale is 0", file=sys.stderr)
        return False

    if overlay:
        image = Image.open(png_path).convert("RGBA")
        img_w, img_h = image.size
    else:
        factor = 1.0
        if max_side and max_side > 0:
            factor = max_side / max(width, height)
        img_w = max(1, int(math.ceil(width * factor)))
        img_h = max(1, int(math.ceil(height * factor)))
        image = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                draw_item(
                    draw,
                    item,
                    width,
                    height,
                    scale,
                    img_w,
                    img_h,
                    line_width,
                    normalized,
                )

    image.save(out_path)
    print(out_path)
    return True


def iter_json_files(path: Path):
    if path.is_dir():
        yield from path.rglob("*-n.json")
    else:
        yield path


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild contours from -n.json (bottom-left origin) or overlay on PNG."
    )
    parser.add_argument(
        "path",
        help="Path to a -n.json file or a directory (recursively scans *-n.json).",
    )
    parser.add_argument("--png", help="PNG path (only for single file mode).")
    parser.add_argument("--out", help="Output image path (only for single file mode).")
    parser.add_argument("--line-width", type=int, default=DEFAULT_LINE_WIDTH, help="Line width.")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay contours on the original PNG instead of a blank canvas.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Max side length for rebuilt images (0 keeps original size).",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Treat points as original scale (skip normalization undo).",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    normalized = not args.original

    if path.is_file():
        if path.suffix.lower() != ".json" or not path.stem.endswith("-n"):
            print(f"Only -n.json files are supported: {path}", file=sys.stderr)
            return 1
        json_path = path
        png_path = None
        if args.overlay:
            png_path = Path(args.png) if args.png else pick_png(json_path)
            if not png_path or not png_path.exists():
                print(f"PNG not found for {json_path}", file=sys.stderr)
                return 1
        out_path = Path(args.out) if args.out else json_path.with_name(json_path.stem + "-vis.png")
        ok = render_one(
            json_path, png_path, out_path, args.line_width, normalized, args.overlay, args.max_side
        )
        return 0 if ok else 1

    for json_path in iter_json_files(path):
        png_path = None
        if args.overlay:
            png_path = pick_png(json_path)
            if not png_path or not png_path.exists():
                print(f"PNG not found for {json_path}", file=sys.stderr)
                continue
        out_path = json_path.with_name(json_path.stem + "-vis.png")
        render_one(
            json_path, png_path, out_path, args.line_width, normalized, args.overlay, args.max_side
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
