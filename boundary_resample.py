#!/usr/bin/env python3
import json
import math
from pathlib import Path


def _to_point_list(points):
    out = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        out.append((float(p[0]), float(p[1])))
    return out


def _dedup_consecutive(points):
    out = []
    prev = None
    for p in points:
        if prev is None or p != prev:
            out.append(p)
            prev = p
    return out


def flatten_segments(segments):
    flat = []
    for seg in segments:
        pts = _to_point_list(seg) if isinstance(seg, list) else []
        for p in pts:
            if flat and p == flat[-1]:
                continue
            flat.append(p)
    return _dedup_consecutive(flat)


def polygon_area(points):
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def ensure_ccw(points):
    if polygon_area(points) < 0:
        return list(reversed(points))
    return points


def resample_closed(points, k):
    if k <= 0:
        raise ValueError("k must be > 0")
    if len(points) < 2:
        raise ValueError("not enough points")

    pts = points
    if pts[0] != pts[-1]:
        pts = pts + [pts[0]]

    seg_lengths = []
    cum = [0.0]
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        length = math.hypot(dx, dy)
        seg_lengths.append(length)
        cum.append(cum[-1] + length)

    total = cum[-1]
    if total == 0:
        raise ValueError("total length is 0")

    step = total / k
    resampled = []
    seg_idx = 0

    for j in range(k):
        d = j * step
        while seg_idx < len(seg_lengths) - 1 and cum[seg_idx + 1] < d:
            seg_idx += 1
        seg_len = seg_lengths[seg_idx]
        if seg_len == 0:
            resampled.append(pts[seg_idx])
            continue
        t = (d - cum[seg_idx]) / seg_len
        x = pts[seg_idx][0] + t * (pts[seg_idx + 1][0] - pts[seg_idx][0])
        y = pts[seg_idx][1] + t * (pts[seg_idx + 1][1] - pts[seg_idx][1])
        resampled.append((x, y))

    return resampled


def rotate_to_min_xy(points):
    min_idx = min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    return points[min_idx:] + points[:min_idx]


def load_contour_points(json_path):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("items not found")

    contour = None
    for item in items:
        if isinstance(item, dict) and item.get("isContour"):
            contour = item
            break
    if contour is None:
        raise ValueError("no isContour item")

    segs = contour.get("segments")
    if isinstance(segs, list) and segs:
        points = flatten_segments(segs)
    else:
        pts = contour.get("points")
        if not isinstance(pts, list) or not pts:
            raise ValueError("no points/segments in contour")
        points = _dedup_consecutive(_to_point_list(pts))

    if len(points) < 3:
        raise ValueError("contour has too few points")
    return points


def resample_boundary(json_path, k, ccw=True, rotate=True):
    points = load_contour_points(json_path)
    if ccw:
        points = ensure_ccw(points)
    resampled = resample_closed(points, k)
    if rotate:
        resampled = rotate_to_min_xy(resampled)
    return resampled
