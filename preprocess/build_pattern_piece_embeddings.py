#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("numpy is required to run this script") from exc

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required to run this script") from exc

from boundary_resample import resample_boundary
from ly import FourierFeatureEncoder

NORM_RANGE = 10.0

def _make_rel_id(path: Path, project_root: Path, fallback_root: Path | None = None) -> str:
    # Prefer paths relative to LayoutFormer repo root (project_root).
    # If the file is outside project_root, fall back to fallback_root (scan root),
    # otherwise return an absolute posix path.
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except Exception:
        if fallback_root is not None:
            try:
                return path.resolve().relative_to(fallback_root).as_posix()
            except Exception:
                pass
        return path.resolve().as_posix()


def to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    return None


def normalize_json(path: Path, overwrite: bool):
    out_path = path.with_name(path.stem + "-n" + path.suffix)
    if out_path.exists() and not overwrite:
        return out_path, False

    data = json.loads(path.read_text(encoding="utf-8"))
    width = to_float(data.get("width"))
    height = to_float(data.get("height"))
    if width is None or height is None:
        raise ValueError(f"missing width/height: {path}")

    scale = max(width, height)
    if scale == 0:
        raise ValueError(f"scale is 0: {path}")

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            pts = item.get("points")
            if isinstance(pts, list):
                item["points"] = [[p[0] / scale * NORM_RANGE, p[1] / scale * NORM_RANGE] for p in pts]
            segs = item.get("segments")
            if isinstance(segs, list):
                item["segments"] = [
                    [[p[0] / scale * NORM_RANGE, p[1] / scale * NORM_RANGE] for p in seg] for seg in segs
                ]

    out_path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return out_path, True


def collect_json_files(root: Path):
    for path in root.rglob("*.json"):
        if path.name.endswith("-n.json"):
            continue
        if any(part == "__MACOSX" or part.startswith(".") for part in path.parts):
            continue
        if path.name.startswith("._"):
            continue
        yield path


def build_embedding(points, encoder):
    pts = torch.tensor(points, dtype=torch.float32)
    x = pts[:, 0].unsqueeze(0)
    y = pts[:, 1].unsqueeze(0)
    with torch.no_grad():
        fx = encoder(x)
        fy = encoder(y)
        raw = torch.cat([fx, fy], dim=-1)
    return raw.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize JSON, resample boundary points, and build Fourier embeddings."
    )
    parser.add_argument("root", help="Root directory containing pattern JSON files.")
    parser.add_argument("--k", type=int, default=196, help="Number of resampled points.")
    parser.add_argument("--num-bands", type=int, default=10, help="Fourier bands (L).")
    parser.add_argument("--max-freq", type=float, default=10.0, help="Max freq (kept for API).")
    parser.add_argument(
        "--out-npz",
        help="Output npz path (default: <project>/data/pattern_piece_emb/pattern_points_embed.npz).",
    )
    parser.add_argument(
        "--out-meta",
        help="Output metadata json path (default: <project>/data/pattern_piece_emb/pattern_points_meta.json).",
    )
    parser.add_argument(
        "--overwrite-n",
        action="store_true",
        help="Overwrite existing -n.json files.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    project_root = ROOT
    default_out_dir = project_root / "data" / "pattern_piece_emb"
    default_out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = Path(args.out_npz) if args.out_npz else default_out_dir / "pattern_points_embed.npz"
    out_meta = Path(args.out_meta) if args.out_meta else default_out_dir / "pattern_points_meta.json"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    encoder = FourierFeatureEncoder(num_bands=args.num_bands, max_freq=args.max_freq)

    # pattern_ids are normalized (-n.json) paths relative to the LayoutFormer repo root; use them as lookup keys.
    pattern_ids = []
    source_json_paths = []
    files = []
    embeds = []
    processed = 0
    normalized = 0

    for json_path in collect_json_files(root):
        try:
            n_path, wrote = normalize_json(json_path, args.overwrite_n)
            if wrote:
                normalized += 1

            points = resample_boundary(n_path, args.k, ccw=True, rotate=True)
            if len(points) != args.k:
                raise ValueError(f"resample returned {len(points)} points")

            embed = build_embedding(points, encoder)
            if embed.shape != (args.k, args.num_bands * 4):
                raise ValueError(f"unexpected embed shape {embed.shape}")

            source_id = _make_rel_id(json_path, project_root, fallback_root=root)
            norm_id = _make_rel_id(n_path, project_root, fallback_root=root)

            idx = processed
            pattern_ids.append(norm_id)
            source_json_paths.append(source_id)
            embeds.append(embed)
            files.append({"index": idx, "norm_json": norm_id, "source_json": source_id})
            processed += 1
        except Exception as exc:
            print(f"skip {json_path}: {exc}", file=sys.stderr)
            continue

    if not embeds:
        raise SystemExit("no embeddings generated")

    np.savez(
        out_npz,
        pattern_ids=np.array(pattern_ids, dtype=object),
        source_json_paths=np.array(source_json_paths, dtype=object),
        points_embed=np.stack(embeds),
    )

    meta = {
        "count": processed,
        "k": args.k,
        "num_bands": args.num_bands,
        "embed_dim": args.num_bands * 4,
        "norm_method": "scale=max(width,height), range=[0,10]",
        "norm_range": NORM_RANGE,
        "encoder": "FourierFeatureEncoder (2^k*pi)",
        "coord_origin": "bottom-left",
        "ordering": "ccw + resample along boundary + rotate_to_min_xy",
        "rotate_start": "min_xy",
        "ccw": True,
        "normalized_suffix": "-n.json",
        "source_root": str(root),
        "project_root": str(project_root),
        "paths_relative_to": "project_root",
        "output_dir": str(out_npz.parent),
        "npz_keys": {
            "pattern_ids": "pattern_ids",
            "source_json_paths": "source_json_paths",
            "points_embed": "points_embed",
        },
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "normalized_written": normalized,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    print(f"embeddings: {processed}")
    print(f"npz: {out_npz}")
    print(f"meta: {out_meta}")


if __name__ == "__main__":
    main()
