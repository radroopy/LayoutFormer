#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.LayoutFormer import LayoutFormer


def to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_pattern_points(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    points_embed = data["points_embed"]  # (N, K, 4L)
    source_json_paths = [to_str(p) for p in data["source_json_paths"].tolist()]
    mapping = {p: i for i, p in enumerate(source_json_paths)}
    return points_embed, mapping


def load_elements(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    element_ids = [to_str(v) for v in data["element_ids"].tolist()]
    pdf_paths = [to_str(v) for v in data["pdf_paths"].tolist()]
    pattern_json_paths = [to_str(v) for v in data["pattern_json_paths"].tolist()]
    labels = data["labels"].astype(np.int64)
    center_x = data["center_x"].astype(np.float32)
    center_y = data["center_y"].astype(np.float32)
    w = data["w"].astype(np.float32)
    h = data["h"].astype(np.float32)
    sin = data["sin"].astype(np.float32)
    cos = data["cos"].astype(np.float32)
    element_embed = data["element_embed"].astype(np.float32)

    by_json = {}
    for idx, json_path in enumerate(pattern_json_paths):
        by_json.setdefault(json_path, []).append(idx)

    layouts = {}
    for json_path, idxs in by_json.items():
        sorted_idxs = sorted(idxs, key=lambda i: (pdf_paths[i], element_ids[i]))
        layouts[json_path] = {
            "indices": sorted_idxs,
            "pdf_paths": [pdf_paths[i] for i in sorted_idxs],
            "labels": labels[sorted_idxs],
            "geom": np.stack(
                [
                    center_x[sorted_idxs],
                    center_y[sorted_idxs],
                    w[sorted_idxs],
                    h[sorted_idxs],
                    sin[sorted_idxs],
                    cos[sorted_idxs],
                ],
                axis=-1,
            ),
            "pos_embed": element_embed[sorted_idxs],
        }
    return layouts


def build_scale_lookup(scale_json_path: Path):
    payload = json.loads(scale_json_path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    lookup = {}
    for base_json, info in results.items():
        sizes = info.get("sizes", {})
        for _, entry in sizes.items():
            tgt_json = entry["json"]
            lookup.setdefault(base_json, {})[tgt_json] = (
                float(entry["scale_w"]),
                float(entry["scale_h"]),
            )
    return lookup




def collate_batch(batch):
    (src_types, src_geom, src_pos, src_bnd, tgt_bnd, scale_factors, tgt_geom, valid, sdf_maps, sdf_hw) = zip(*batch)
    src_types = torch.stack(src_types, dim=0)
    src_geom = torch.stack(src_geom, dim=0)
    src_pos = torch.stack(src_pos, dim=0)
    src_bnd = torch.stack(src_bnd, dim=0)
    tgt_bnd = torch.stack(tgt_bnd, dim=0)
    scale_factors = torch.stack(scale_factors, dim=0)
    tgt_geom = torch.stack(tgt_geom, dim=0)
    valid = torch.stack(valid, dim=0)
    sdf_hw = torch.stack(sdf_hw, dim=0)  # (B, 2) -> (h, w)

    max_h = int(sdf_hw[:, 0].max().item())
    max_w = int(sdf_hw[:, 1].max().item())
    padded = []
    for sdf in sdf_maps:
        # sdf: (1, h, w)
        h = sdf.shape[1]
        w = sdf.shape[2]
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h < 0 or pad_w < 0:
            raise ValueError("SDF padding error")
        sdf_p = F.pad(sdf, (0, pad_w, 0, pad_h), value=1.0)
        padded.append(sdf_p)
    sdf_batch = torch.stack(padded, dim=0)

    return (
        src_types,
        src_geom,
        src_pos,
        src_bnd,
        tgt_bnd,
        scale_factors,
        tgt_geom,
        valid,
        sdf_batch,
        sdf_hw,
    )

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.dim() == 3:
        mask = mask.unsqueeze(-1)
    masked = values * mask
    denom = mask.sum().clamp(min=1)
    return masked.sum() / denom


def compute_losses(pred, gt, mask, sdf_maps, sdf_hw):
    pred_xy = pred[..., :2]
    pred_wh = pred[..., 2:4]
    pred_sc = pred[..., 4:6]

    gt_xy = gt[..., :2]
    gt_wh = gt[..., 2:4]
    gt_sc = gt[..., 4:6]

    l_pos = masked_mean((pred_xy - gt_xy) ** 2, mask)
    l_dim = masked_mean((pred_wh - gt_wh).abs(), mask)
    l_rot = masked_mean((pred_sc - gt_sc) ** 2, mask)
    l_reg = masked_mean((pred_sc.pow(2).sum(dim=-1) - 1.0).abs(), mask)

    l_shape = torch.tensor(0.0, device=pred.device)
    if sdf_maps is not None:
        # sample SDF at element centers only
        # sdf_hw is (B,2) -> (h, w) of original (unpadded) maps
        h = sdf_hw[:, 0].clamp(min=2).float()
        w = sdf_hw[:, 1].clamp(min=2).float()
        h_max = float(sdf_maps.shape[2] - 1)
        w_max = float(sdf_maps.shape[3] - 1)
        x_scale = (w - 1.0) / max(w_max, 1.0)
        y_scale = (h - 1.0) / max(h_max, 1.0)
        grid_x = pred_xy[..., 0] * x_scale.view(-1, 1) * 2 - 1
        grid_y = (1 - pred_xy[..., 1]) * y_scale.view(-1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
        sampled = F.grid_sample(
            sdf_maps,
            grid,
            mode="bilinear",
            align_corners=True,
        )  # (B, 1, N, 1)
        sampled = sampled.squeeze(1).squeeze(-1)  # (B, N)
        l_shape = masked_mean(F.relu(sampled), mask)

    return l_pos, l_dim, l_rot, l_reg, l_shape


class PairDataset(Dataset):
    def __init__(
        self,
        pairs,
        layouts,
        boundary_embed,
        boundary_index,
        scale_lookup,
        max_elements=20,
        strict=True,
        sdf_dir: Path | None = None,
        sdf_ext: str = ".npy",
    ):
        self.pairs = pairs
        self.layouts = layouts
        self.boundary_embed = boundary_embed
        self.boundary_index = boundary_index
        self.scale_lookup = scale_lookup
        self.max_elements = max_elements
        self.strict = strict
        if sdf_dir is None:
            raise ValueError("sdf_dir is required (L_shape must be used)")
        self.sdf_dir = sdf_dir
        self.sdf_ext = sdf_ext

    def __len__(self):
        return len(self.pairs)

    def _pad(self, arr, target, pad_value=0):
        if arr.shape[0] > target:
            raise ValueError(f"element count {arr.shape[0]} exceeds max_elements {target}")
        if arr.shape[0] == target:
            return arr
        pad_shape = (target - arr.shape[0],) + arr.shape[1:]
        pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    def _load_sdf(self, json_path: str):
        rel = Path(json_path)
        sdf_path = self.sdf_dir / rel.parent / (rel.stem + self.sdf_ext)
        if not sdf_path.is_file():
            raise FileNotFoundError(f"Missing SDF: {sdf_path}")
        sdf = np.load(sdf_path)
        if sdf.ndim == 2:
            sdf = sdf[None, ...]
        return sdf.astype(np.float32)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src_json = pair["src_json"]
        tgt_json = pair["tgt_json"]

        if src_json not in self.layouts or tgt_json not in self.layouts:
            raise KeyError(f"missing layout for {src_json} or {tgt_json}")

        src_layout = self.layouts[src_json]
        tgt_layout = self.layouts[tgt_json]

        if self.strict and src_layout["pdf_paths"] != tgt_layout["pdf_paths"]:
            raise ValueError(
                "element set mismatch between source and target\n"
                f"src={src_json}\n"
                f"tgt={tgt_json}\n"
            )

        src_labels = src_layout["labels"].astype(np.int64)
        src_geom = src_layout["geom"].astype(np.float32)
        src_pos = src_layout["pos_embed"].astype(np.float32)

        tgt_geom = tgt_layout["geom"].astype(np.float32)

        valid = np.ones((src_labels.shape[0],), dtype=np.float32)

        src_labels = self._pad(src_labels, self.max_elements, pad_value=0)
        src_geom = self._pad(src_geom, self.max_elements, pad_value=0.0)
        src_pos = self._pad(src_pos, self.max_elements, pad_value=0.0)
        tgt_geom = self._pad(tgt_geom, self.max_elements, pad_value=0.0)
        valid = self._pad(valid, self.max_elements, pad_value=0.0)

        if src_json not in self.boundary_index or tgt_json not in self.boundary_index:
            raise KeyError(f"missing boundary embedding for {src_json} or {tgt_json}")

        src_bnd = self.boundary_embed[self.boundary_index[src_json]]
        tgt_bnd = self.boundary_embed[self.boundary_index[tgt_json]]

        scale_entry = self.scale_lookup.get(src_json, {}).get(tgt_json)
        if scale_entry is None:
            raise KeyError(f"missing scale factor for {src_json} -> {tgt_json}")
        scale_factors = np.array(scale_entry, dtype=np.float32)

        sdf_map = self._load_sdf(tgt_json)
        sdf_hw = np.array([sdf_map.shape[1], sdf_map.shape[2]], dtype=np.int64)

        return (
            torch.from_numpy(src_labels),
            torch.from_numpy(src_geom),
            torch.from_numpy(src_pos),
            torch.from_numpy(src_bnd),
            torch.from_numpy(tgt_bnd),
            torch.from_numpy(scale_factors),
            torch.from_numpy(tgt_geom),
            torch.from_numpy(valid),
            torch.from_numpy(sdf_map),
            torch.from_numpy(sdf_hw),
        )


def main():
    parser = argparse.ArgumentParser(description="Train LayoutFormer on graded layout pairs.")
    parser.add_argument(
        "--split",
        default=str(Path(__file__).resolve().parent / "pair_splits.json"),
        help="Path to pair_splits.json",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        choices=["train", "val", "test"],
        help="Split to use for training (val used for validation)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data"),
        help="Data directory",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-pos", type=float, default=1.0)
    parser.add_argument("--lambda-dim", type=float, default=1.0)
    parser.add_argument("--lambda-rot", type=float, default=1.0)
    parser.add_argument("--lambda-shape", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--max-elements", type=int, default=20)
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parents[1] / "result" / "model"))
    parser.add_argument("--sdf-dir", default=str(Path(__file__).resolve().parents[1] / "sdf_maps"), help="Directory of SDF maps (required)")
    parser.add_argument("--sdf-ext", default=".npy", help="SDF file extension (default: .npy)")
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
    num_types = len(vocab)

    splits = json.loads(Path(args.split).read_text(encoding="utf-8"))
    train_pairs = splits["splits"][args.split_name]
    val_pairs = splits["splits"].get("val", [])

    sdf_dir = Path(args.sdf_dir) if args.sdf_dir else None
    if sdf_dir is None or not sdf_dir.is_dir():
        raise SystemExit("sdf-dir is required and must exist (L_shape is mandatory)")

    train_ds = PairDataset(
        train_pairs,
        layouts,
        points_embed,
        boundary_index,
        scale_lookup,
        max_elements=args.max_elements,
        strict=True,
        sdf_dir=sdf_dir,
        sdf_ext=args.sdf_ext,
    )
    val_ds = PairDataset(
        val_pairs,
        layouts,
        points_embed,
        boundary_index,
        scale_lookup,
        max_elements=args.max_elements,
        strict=True,
        sdf_dir=sdf_dir,
        sdf_ext=args.sdf_ext,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    model = LayoutFormer(
        num_element_types=num_types,
        max_elements=args.max_elements,
        d_model=256,
        nhead=8,
        num_layers=4,
        boundary_seq_len=points_embed.shape[1],
        fourier_bands=10,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
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
            tgt_geom = tgt_geom.to(args.device)
            valid = valid.to(args.device)
            sdf_maps = sdf_maps.to(args.device)
            sdf_hw = sdf_hw.to(args.device)

            pred = model(
                src_types,
                src_geom,
                src_bnd,
                tgt_bnd,
                scale_factors,
                return_postprocess=False,
                src_pos_embed=src_pos,
            )

            l_pos, l_dim, l_rot, l_reg, l_shape = compute_losses(
                pred, tgt_geom, valid, sdf_maps=sdf_maps, sdf_hw=sdf_hw
            )
            loss = (
                args.lambda_pos * l_pos
                + args.lambda_dim * l_dim
                + args.lambda_rot * l_rot
                + args.lambda_shape * l_shape
                + args.lambda_reg * l_reg
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_loss = running / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
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
                tgt_geom = tgt_geom.to(args.device)
                valid = valid.to(args.device)
                sdf_maps = sdf_maps.to(args.device)
                sdf_hw = sdf_hw.to(args.device)

                pred = model(
                    src_types,
                    src_geom,
                    src_bnd,
                    tgt_bnd,
                    scale_factors,
                    return_postprocess=False,
                    src_pos_embed=src_pos,
                )
                l_pos, l_dim, l_rot, l_reg, l_shape = compute_losses(
                    pred, tgt_geom, valid, sdf_maps=sdf_maps, sdf_hw=sdf_hw
                )
                loss = (
                    args.lambda_pos * l_pos
                    + args.lambda_dim * l_dim
                    + args.lambda_rot * l_rot
                    + args.lambda_shape * l_shape
                    + args.lambda_reg * l_reg
                )
                val_loss += loss.item()

        val_loss = val_loss / max(1, len(val_loader))
        print(f"epoch {epoch}: train_loss={avg_loss:.6f} val_loss={val_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, save_dir / "best.pt")


if __name__ == "__main__":
    main()
