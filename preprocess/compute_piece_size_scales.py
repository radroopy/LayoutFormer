#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_resample import load_contour_points


def resolve_path(project_root: Path, root: Path, value) -> Path:
    rel = Path(str(value))
    if rel.is_absolute():
        return rel
    cand = project_root / rel
    if cand.exists():
        return cand
    return root / rel


def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not xs or not ys:
        raise ValueError("no points")
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return min_x, min_y, max_x, max_y


def _is_size_map(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    for v in obj.values():
        if isinstance(v, dict):
            return False
    return True


def normalize_lookup(payload: dict) -> tuple[dict, str]:
    """
    Normalize lookup into {shape_id: {base_json: size_map}}.
    - legacy: {base_json: size_map} -> "__all__"
    - new: {shape_id: {base_json: size_map}}
    """
    if not isinstance(payload, dict):
        return {}, "invalid"
    if isinstance(payload.get("lookup"), dict):
        payload = payload["lookup"]
    if payload and all(_is_size_map(v) for v in payload.values()):
        return {"__all__": payload}, "legacy"
    return payload, "shape"


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-piece size scale factors using contour points (real-world coords)."
    )
    parser.add_argument(
        "--lookup",
        help="Path to json_size_lookup.json (default: <project>/data/json_size_lookup.json).",
    )
    parser.add_argument(
        "--root",
        help="Root directory for relative JSON paths (defaults to project_root/pattern/pattern if exists).",
    )
    parser.add_argument(
        "--out",
        help="Output json path (default: <project>/data/size_scale_factors.json).",
    )
    args = parser.parse_args()

    project_root = ROOT
    default_root = project_root / "pattern" / "pattern"
    if not default_root.is_dir():
        default_root = project_root

    root = Path(args.root) if args.root else default_root

    lookup_path = Path(args.lookup) if args.lookup else project_root / "data" / "json_size_lookup.json"
    if not lookup_path.is_file():
        print(f"json_size_lookup.json not found: {lookup_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else project_root / "data" / "size_scale_factors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lookup_payload = json.loads(lookup_path.read_text(encoding="utf-8"))
    lookup, lookup_mode = normalize_lookup(lookup_payload)

    results = {}
    errors = []
    for shape_id, per_shape in lookup.items():
        if not isinstance(per_shape, dict) or not per_shape:
            continue
        shape_results = {}
        for base_json, size_map in per_shape.items():
            if not isinstance(size_map, dict) or not size_map:
                continue

            base_path = resolve_path(project_root, root, base_json)
            if not base_path.exists():
                errors.append(f"{shape_id}: base json not found: {base_json}")
                continue

            try:
                base_points = load_contour_points(base_path)
                min_x, min_y, max_x, max_y = bbox_from_points(base_points)
                base_w = max_x - min_x
                base_h = max_y - min_y
                if base_w <= 0 or base_h <= 0:
                    raise ValueError("invalid bbox")
            except Exception as exc:
                errors.append(f"{shape_id}: base {base_json} ({exc})")
                continue

            per_size = {}
            for size, json_rel in size_map.items():
                json_path = resolve_path(project_root, root, json_rel)
                if not json_path.exists():
                    errors.append(f"{shape_id}: {base_json}: json not found: {json_rel}")
                    continue
                try:
                    points = load_contour_points(json_path)
                    min_x, min_y, max_x, max_y = bbox_from_points(points)
                    width = max_x - min_x
                    height = max_y - min_y
                    if width <= 0 or height <= 0:
                        raise ValueError("invalid bbox")
                except Exception as exc:
                    errors.append(f"{shape_id}: {base_json}: {json_rel} ({exc})")
                    continue

                per_size[size] = {
                    "json": json_rel,
                    "width": float(width),
                    "height": float(height),
                    "scale_w": float(width / base_w),
                    "scale_h": float(height / base_h),
                }

            shape_results[base_json] = {
                "base_json": base_json,
                "base_width": float(base_w),
                "base_height": float(base_h),
                "sizes": per_size,
            }
        if shape_results:
            results[shape_id] = shape_results

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "lookup": str(lookup_path),
        "lookup_mode": lookup_mode,
        "json_root": str(root),
        "method": "bbox from contour points (raw json coords); scale relative to each base json",
        "results": results,
        "errors": errors,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base_count = sum(len(v) for v in results.values())
    print(f"shapes: {len(results)}")
    print(f"bases: {base_count}")
    print(f"errors: {len(errors)}")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
