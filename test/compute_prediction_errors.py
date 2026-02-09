#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_NORM_RANGE = 10.0


def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_elements(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    elem_ids = [to_str(x) for x in data["element_ids"].tolist()]
    lookup = {}
    lookup_by_pdf_json = {}
    for idx, elem_id in enumerate(elem_ids):
        entry = {
            "center_x": float(data["center_x"][idx]),
            "center_y": float(data["center_y"][idx]),
            "w": float(data["w"][idx]),
            "h": float(data["h"][idx]),
            "sin": float(data["sin"][idx]) if "sin" in data else 0.0,
            "cos": float(data["cos"][idx]) if "cos" in data else 1.0,
            "logo_level": int(data["logo_level"][idx]) if "logo_level" in data else None,
            "label": int(data["labels"][idx]) if "labels" in data else None,
            "pdf_path": to_str(data["pdf_paths"][idx]) if "pdf_paths" in data else None,
            "pattern_json_path": to_str(data["pattern_json_paths"][idx])
            if "pattern_json_paths" in data
            else None,
        }
        lookup[elem_id] = entry
        key = (entry["pdf_path"], entry["pattern_json_path"])
        lookup_by_pdf_json.setdefault(key, []).append((elem_id, entry))
    return lookup, lookup_by_pdf_json


def main():
    parser = argparse.ArgumentParser(description="Compute per-element errors from predictions_test.json")
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("test/result/predictions_test.json"),
        help="Path to predictions_test.json",
    )
    parser.add_argument(
        "--elements",
        type=Path,
        default=Path("data/element_emb/elements_embed.npz"),
        help="Path to elements_embed.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("test/result/prediction_element_errors.json"),
        help="Output json path",
    )
    parser.add_argument("--norm-range", type=float, default=DEFAULT_NORM_RANGE)
    args = parser.parse_args()

    pred_items = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    elem_lookup, lookup_by_pdf_json = load_elements(args.elements)

    results = []
    all_abs_values = []
    per_dim_values = {}
    element_count = 0
    for item in pred_items:
        tgt_scale = float(item.get("tgt_scale", 1.0))
        pred = item.get("pred")
        pred_denorm = item.get("pred_denorm")
        if pred_denorm is None and pred is not None:
            pred_denorm = []
            for row in pred:
                row = list(row)
                if len(row) >= 4:
                    row[:4] = [v * tgt_scale / args.norm_range for v in row[:4]]
                pred_denorm.append(row)

        pdf_paths = item.get("pdf_paths") or []
        element_ids = item.get("element_ids") or []
        logo_levels = item.get("logo_levels") or []
        valid = item.get("valid") or [True] * len(pdf_paths)

        element_errors = {}
        for i in range(len(pdf_paths)):
            if i >= len(valid) or not bool(valid[i]):
                continue
            elem_id = to_str(element_ids[i]) if i < len(element_ids) and element_ids[i] else None
            gt = None
            candidates = None
            if elem_id and elem_id in elem_lookup:
                gt = elem_lookup[elem_id]
            else:
                key = (pdf_paths[i] if i < len(pdf_paths) else None, item.get("tgt_json"))
                candidates = lookup_by_pdf_json.get(key, [])
                if candidates:
                    # choose first but record ambiguity if multiple
                    elem_id, gt = candidates[0]

            if not gt:
                continue
            gt_denorm = [
                gt["center_x"] * tgt_scale / args.norm_range,
                gt["center_y"] * tgt_scale / args.norm_range,
                gt["w"] * tgt_scale / args.norm_range,
                gt["h"] * tgt_scale / args.norm_range,
            ]
            # sin/cos are dimensionless
            if pred_denorm and len(pred_denorm[i]) >= 6:
                gt_denorm += [gt["sin"], gt["cos"]]

            pred_row = pred_denorm[i] if pred_denorm else None
            if pred_row is None:
                continue

            dim = min(len(pred_row), len(gt_denorm))
            abs_err = [abs(pred_row[d] - gt_denorm[d]) for d in range(dim)]
            all_abs_values.extend(abs_err)
            for d, val in enumerate(abs_err):
                per_dim_values.setdefault(d, []).append(val)
            element_count += 1
            pdf_path = pdf_paths[i] if i < len(pdf_paths) else None
            if pdf_path in element_errors:
                if isinstance(element_errors[pdf_path][0], list):
                    element_errors[pdf_path].append(abs_err)
                else:
                    element_errors[pdf_path] = [element_errors[pdf_path], abs_err]
            else:
                element_errors[pdf_path] = abs_err

        results.append(
            {
                "shape_id": item.get("shape_id"),
                "src_json": item.get("src_json"),
                "tgt_json": item.get("tgt_json"),
                "src_size": item.get("src_size"),
                "tgt_size": item.get("tgt_size"),
                "tgt_scale": tgt_scale,
                "element_errors": element_errors,
            }
        )

    stats = {
        "max_abs_error": float(max(all_abs_values)) if all_abs_values else 0.0,
        "min_abs_error": float(min(all_abs_values)) if all_abs_values else 0.0,
        "mean_abs_error": float(sum(all_abs_values) / len(all_abs_values)) if all_abs_values else 0.0,
        "count": int(len(all_abs_values)),
        "element_count": int(element_count),
        "item_count": int(len(results)),
        "per_dim": {},
    }
    for d, vals in per_dim_values.items():
        stats["per_dim"][str(d)] = {
            "max": float(max(vals)) if vals else 0.0,
            "min": float(min(vals)) if vals else 0.0,
            "mean": float(sum(vals) / len(vals)) if vals else 0.0,
            "count": int(len(vals)),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": stats, "items": results}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
