#!/usr/bin/env python3
import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build directed pair splits per shape (7:1:2).")
    parser.add_argument(
        "--shape-index",
        default=str(Path(__file__).resolve().parents[1] / "data" / "shape_size_index.json"),
        help="Path to shape_size_index.json",
    )

    parser.add_argument(
        "--size-scales",
        default=str(Path(__file__).resolve().parents[1] / "data" / "size_scale_factors.json"),
        help="Path to size_scale_factors.json (for target width/height/scale)",
    )

    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "pair_splits.json"),
        help="Output json path for splits",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.2, help="Test ratio")
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude same-size pairs (src_size==tgt_size). Default is to include them (e.g., L->L).",
    )
    args = parser.parse_args()

    shape_index_path = Path(args.shape_index)
    if not shape_index_path.is_file():
        raise SystemExit(f"Not found: {shape_index_path}")

    shapes = json.loads(shape_index_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)


    size_scale_path = Path(args.size_scales)
    json_wh = {}
    if size_scale_path.is_file():
        scale_payload = json.loads(size_scale_path.read_text(encoding="utf-8"))
        results = scale_payload.get("results", {})
        for _, info in results.items():
            for _, entry in info.get("sizes", {}).items():
                json_path = entry.get("json")
                if not json_path:
                    continue
                w = float(entry.get("width", 0.0))
                h = float(entry.get("height", 0.0))
                if json_path not in json_wh:
                    json_wh[json_path] = (w, h)

    splits = {"train": [], "val": [], "test": []}
    shape_stats = {}

    for shape_id, size_map in shapes.items():
        sizes = sorted(size_map.keys())
        pairs = []
        for src_size in sizes:
            for tgt_size in sizes:
                if args.exclude_self and src_size == tgt_size:
                    continue
                src_json = size_map[src_size]
                tgt_json = size_map[tgt_size]
                w_h = json_wh.get(tgt_json)
                tgt_w = None
                tgt_h = None
                tgt_scale = None
                if w_h is not None:
                    tgt_w, tgt_h = w_h
                    tgt_scale = max(tgt_w, tgt_h)
                pairs.append({
                    "shape_id": shape_id,
                    "src_size": src_size,
                    "tgt_size": tgt_size,
                    "src_json": src_json,
                    "tgt_json": tgt_json,
                    "tgt_width": tgt_w,
                    "tgt_height": tgt_h,
                    "tgt_scale": tgt_scale,
                })

        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val

        splits["train"].extend(pairs[:n_train])
        splits["val"].extend(pairs[n_train:n_train + n_val])
        splits["test"].extend(pairs[n_train + n_val:])

        shape_stats[shape_id] = {
            "num_sizes": len(sizes),
            "num_pairs": n,
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "shape_index": str(shape_index_path),
        "size_scale_factors": str(size_scale_path),
        "shapes": shape_stats,
        "splits": splits,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"shapes: {len(shapes)}")
    print(f"train pairs: {len(splits['train'])}")
    print(f"val pairs: {len(splits['val'])}")
    print(f"test pairs: {len(splits['test'])}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
