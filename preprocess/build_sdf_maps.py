#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from boundary_resample import load_contour_points

try:
    from scipy.ndimage import distance_transform_edt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required for SDF generation. Install with: pip install scipy") from exc


def build_sdf(points, base_size=512, normalize=True):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError("invalid polygon bounds")

    scale = float(base_size) / max(width, height)
    w_px = max(2, int(round(width * scale)))
    h_px = max(2, int(round(height * scale)))

    # transform points into image coords (top-left origin)
    pts = []
    for x, y in points:
        x_n = (x - min_x) * scale
        y_n = (y - min_y) * scale
        y_img = (h_px - 1) - y_n
        pts.append((x_n, y_img))

    mask = Image.new("L", (w_px, h_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, outline=1, fill=1)
    mask_np = np.array(mask, dtype=np.uint8)

    dist_in = distance_transform_edt(mask_np)
    dist_out = distance_transform_edt(1 - mask_np)
    sdf = dist_out - dist_in  # inside negative, outside positive

    if normalize:
        denom = max(w_px, h_px)
        sdf = sdf / float(denom)

    return sdf.astype(np.float32), (w_px, h_px)


def main():
    parser = argparse.ArgumentParser(description="Build SDF maps for pattern pieces.")
    parser.add_argument(
        "--meta",
        default=str(Path(__file__).resolve().parents[1] / "data" / "pattern_piece_emb" / "pattern_points_meta.json"),
        help="pattern_points_meta.json path",
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "pattern" / "pattern"),
        help="Root for pattern json files",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "sdf_maps"),
        help="Output directory for SDF maps",
    )
    parser.add_argument("--base", type=int, default=512, help="Base size for longest side")
    parser.add_argument("--no-normalize", action="store_true", help="Do not normalize SDF values")
    args = parser.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.is_file():
        raise SystemExit(f"Not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    files = meta.get("files", [])

    project_root = Path(meta.get("project_root", meta_path.resolve().parents[2]))
    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    normalize = not args.no_normalize

    count = 0
    skipped = 0
    for entry in files:
        json_rel = entry.get("source_json")
        if not json_rel:
            skipped += 1
            continue

        json_path = Path(json_rel)
        if not json_path.is_absolute():
            candidate = project_root / json_rel
            if candidate.is_file():
                json_path = candidate
            else:
                json_path = root / json_rel

        if not json_path.is_file():
            skipped += 1
            continue

        points = load_contour_points(json_path)
        sdf, _ = build_sdf(points, base_size=args.base, normalize=normalize)

        rel_path = json_path.resolve().relative_to(project_root).as_posix()
        rel = Path(rel_path)
        out_path = out_root / rel.parent / (rel.stem + ".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, sdf)
        count += 1

    print(f"sdf_maps: {count}")
    print(f"skipped: {skipped}")
    print(f"output_dir: {out_root}")


if __name__ == "__main__":
    main()
