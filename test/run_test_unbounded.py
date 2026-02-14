#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.LayoutFormer_unbounded import LayoutFormer
from train.train_layoutformer_unbounded import (
    load_pattern_points,
    load_elements,
    build_scale_lookup,
    PairDataset,
    collate_batch,
)

NORM_RANGE = 10.0

def main():
    parser = argparse.ArgumentParser(description="Test LayoutFormer_unbounded: load weights, run prediction, save results.")
    parser.add_argument(
        "--split",
        default=str(Path(__file__).resolve().parents[1] / "train" / "pair_splits.json"),
        help="Path to pair_splits.json",
    )
    parser.add_argument(
        "--split-name",
        default="test",
        choices=["train", "val", "test"],
        help="Split to use",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Data directory",
    )
    parser.add_argument(
        "--ckpt",
        default=str(Path(__file__).resolve().parents[1] / "adjust" / "1" / "model" / "best.pt"),
        help="Checkpoint path",
    )
    parser.add_argument(
        "--sdf-dir",
        default=str(Path(__file__).resolve().parents[1] / "sdf_maps"),
        help="Directory of SDF maps (required)",
    )
    parser.add_argument("--sdf-ext", default=".npy")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument(
        "--max-elements",
        type=int,
        default=40,
        help="Maximum number of elements per layout (must match training/checkpoint setting).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "result_unbounded"),
        help="Output directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pattern_npz = data_dir / "pattern_piece_emb" / "pattern_points_embed.npz"
    element_npz = data_dir / "element_emb" / "elements_embed.npz"
    type_vocab = data_dir / "element_emb" / "type_vocab.json"
    scale_json = data_dir / "size_scale_factors.json"

    points_embed, boundary_index = load_pattern_points(pattern_npz)
    layouts = load_elements(element_npz)
    scale_lookup = build_scale_lookup(scale_json)

    vocab = json.loads(type_vocab.read_text(encoding="utf-8"))
    num_types = int(vocab.get("num_types", len(vocab.get("type_to_id", {}))))

    splits = json.loads(Path(args.split).read_text(encoding="utf-8"))
    pairs = splits["splits"][args.split_name]

    sdf_dir = Path(args.sdf_dir)
    if not sdf_dir.is_dir():
        raise SystemExit("sdf-dir is required and must exist")

    dataset = PairDataset(
        pairs,
        layouts,
        points_embed,
        boundary_index,
        scale_lookup,
        max_elements=args.max_elements,
        strict=True,
        sdf_dir=sdf_dir,
        sdf_ext=args.sdf_ext,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    model = LayoutFormer(
        num_element_types=num_types,
        max_elements=args.max_elements,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        boundary_seq_len=points_embed.shape[1],
        fourier_bands=10,
    ).to(args.device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    idx = 0

    # running sums for mean absolute error
    mae_sum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mae_count = 0
    with torch.no_grad():
        for batch in loader:
            (
                src_types,
                src_geom,
                src_pos,
                src_bnd,
                tgt_bnd,
                scale_factors,
                tgt_geom,
                valid,
                sdf_maps,
                sdf_hw,
            ) = batch
            src_types = src_types.to(args.device)
            src_geom = src_geom.to(args.device)
            src_pos = src_pos.to(args.device)
            src_bnd = src_bnd.to(args.device)
            tgt_bnd = tgt_bnd.to(args.device)
            scale_factors = scale_factors.to(args.device)

            pred = model(
                src_types,
                src_geom,
                src_bnd,
                tgt_bnd,
                scale_factors,
                return_postprocess=False,
                src_pos_embed=src_pos,
            )

            pred_np = pred.cpu().numpy()
            valid_np = valid.numpy()

            # accumulate mean absolute error over valid elements
            tgt_np = tgt_geom.numpy()
            pred_dim = pred_np.shape[-1]
            tgt_dim = tgt_np.shape[-1]
            cmp_dim = min(pred_dim, tgt_dim)
            for bi in range(pred_np.shape[0]):
                mask = valid_np[bi] > 0.0
                if mask.any():
                    pred_vals = pred_np[bi][mask][:, :cmp_dim]
                    tgt_vals = tgt_np[bi][mask][:, :cmp_dim]
                    diff = abs(pred_vals - tgt_vals)
                    for i in range(min(4, cmp_dim)):
                        mae_sum[i] += float(diff[:, i].sum())
                    if cmp_dim >= 6:
                        mae_sum[4] += float(diff[:, 4].sum())
                        mae_sum[5] += float(diff[:, 5].sum())
                    mae_count += int(mask.sum())

            for b in range(pred_np.shape[0]):
                pair = pairs[idx]

                # 关键：pred[i] 对应的是该布局里第 i 个 element token。
                # element 的顺序来自 elements_embed.npz 的稳定排序 (pdf_path, element_id)。
                # 因此这里把同样顺序的 pdf_paths 取出来并 pad 到 max_elements，便于你在结果里按索引对齐：
                #   pdf_paths[i] <-> pred[i]
                shape_id = pair.get("shape_id")
                src_json = pair.get("src_json")
                layout = layouts.get((shape_id, src_json))
                if layout is None:
                    raise KeyError(f"missing layout for shape_id={shape_id} src_json={src_json}")
                pdf_paths = list(layout.get("pdf_paths", []))
                element_ids = list(layout.get("element_ids", []))
                logo_levels = [int(v) if v is not None else None for v in layout.get("logo_level", [])]
                if logo_levels and len(logo_levels) != len(pdf_paths):
                    raise ValueError(f"logo_levels length mismatch for {src_json}")
                if element_ids and len(element_ids) != len(pdf_paths):
                    raise ValueError(f"element_ids length mismatch for {src_json}")
                if not element_ids:
                    element_ids = [None] * len(pdf_paths)
                if not logo_levels:
                    logo_levels = [-1] * len(pdf_paths)
                max_elems = int(pred_np.shape[1])
                if len(pdf_paths) > max_elems:
                    raise ValueError(f"element count {len(pdf_paths)} exceeds max_elements {max_elems} for {src_json}")
                pdf_paths = pdf_paths + [None] * (max_elems - len(pdf_paths))
                element_ids = element_ids + [None] * (max_elems - len(element_ids))
                logo_levels = logo_levels + [-1] * (max_elems - len(logo_levels))
                tgt_scale = pair.get("tgt_scale")
                if tgt_scale is not None:
                    pred_denorm = pred_np[b].copy()
                    pred_denorm[:, :4] = pred_denorm[:, :4] * float(tgt_scale) / NORM_RANGE
                else:
                    pred_denorm = None

                results.append({
                    "shape_id": pair.get("shape_id"),
                    "src_json": pair.get("src_json"),
                    "tgt_json": pair.get("tgt_json"),
                    "src_size": pair.get("src_size"),
                    "tgt_size": pair.get("tgt_size"),
                    "tgt_scale": tgt_scale,
                    "pdf_paths": pdf_paths,
                    "element_ids": element_ids,
                    "logo_levels": logo_levels,
                    "pred": pred_np[b].tolist(),
                    "pred_denorm": pred_denorm.tolist() if pred_denorm is not None else None,
                    "valid": valid_np[b].tolist(),
                })
                idx += 1

    if mae_count > 0:
        mae = [v / mae_count for v in mae_sum]
        if mae_sum[4] == 0.0 and mae_sum[5] == 0.0:
            print(
                "MAE (x,y,w,h): "
                f"{mae[0]:.6f} {mae[1]:.6f} {mae[2]:.6f} {mae[3]:.6f}"
            )
        else:
            print(
                "MAE (x,y,w,h,sin,cos): "
                f"{mae[0]:.6f} {mae[1]:.6f} {mae[2]:.6f} {mae[3]:.6f} {mae[4]:.6f} {mae[5]:.6f}"
            )

    out_path = out_dir / f"predictions_{args.split_name}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
