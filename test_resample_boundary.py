#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image, ImageDraw

from boundary_resample import resample_boundary


def main():
    # ????? JSON ??
    json_path = Path(r"C:\Users\Administrator\PycharmProjects\LayoutFormer\pattern\pattern\31103-Shirt\2XL\1-n.json")
    if not json_path.exists():
        raise SystemExit(f"json not found: {json_path}")

    k = 196
    radius = 4
    points = resample_boundary(json_path, k, ccw=True, rotate=True)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = data.get("width")
    height = data.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise SystemExit("invalid width/height in json")

    if json_path.stem.endswith("-n"):
        scale = max(width, height)
        points = [(x * scale, y * scale) for x, y in points]

    image = Image.new("RGBA", (int(round(width)), int(round(height))), (0, 0, 0, 0))
    img_w, img_h = image.size
    draw = ImageDraw.Draw(image)

    def to_pixel(x, y):
        x_px = x / width * (img_w - 1) if img_w > 1 else 0.0
        y_px = (1.0 - y / height) * (img_h - 1) if img_h > 1 else 0.0
        return (x_px, y_px)

    r = max(1, radius)
    for x, y in points:
        px, py = to_pixel(x, y)
        draw.ellipse(
            (px - r, py - r, px + r, py + r),
            outline=(255, 0, 0, 255),
            fill=(255, 0, 0, 255),
        )

    out_path = json_path.with_name(json_path.stem + "-resample.png")
    image.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
