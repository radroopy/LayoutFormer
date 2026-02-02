#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def _norm_point(pt, scale):
    return [pt[0] / scale, pt[1] / scale]


def _norm_points(points, scale):
    return [_norm_point(p, scale) for p in points]


def _norm_segments(segments, scale):
    return [[_norm_point(p, scale) for p in seg] for seg in segments]


def process_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    width = data.get("width")
    height = data.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        print(f"skip {path}: missing width/height", file=sys.stderr)
        return

    scale = max(width, height)
    if scale == 0:
        print(f"skip {path}: scale is 0", file=sys.stderr)
        return

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            pts = item.get("points")
            if isinstance(pts, list):
                item["points"] = _norm_points(pts, scale)

            segs = item.get("segments")
            if isinstance(segs, list):
                item["segments"] = _norm_segments(segs, scale)

    out_path = path.with_name(path.stem + "-n" + path.suffix)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)
        f.write("\n")

    print(out_path)


def iter_json_files(dir_path: Path, recursive: bool):
    if recursive:
        yield from dir_path.rglob("*.json")
    else:
        yield from dir_path.glob("*.json")


def main():
    parser = argparse.ArgumentParser(description="Normalize points/segments in JSON files.")
    parser.add_argument("directory", help="Target directory with JSON files.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not include subdirectories.")
    args = parser.parse_args()

    dir_path = Path(args.directory)
    if not dir_path.is_dir():
        print(f"Not a directory: {dir_path}", file=sys.stderr)
        return 1

    recursive = not args.no_recursive

    for p in iter_json_files(dir_path, recursive):
        if p.name.endswith("-n.json"):
            continue
        try:
            process_file(p)
        except Exception as e:
            print(f"error {p}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
