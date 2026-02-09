#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

from models.LayoutFormer import LayoutFormer

NORM_RANGE = 10.0


def safe_torch_save(obj: dict, path: Path):
    """
    安全保存模型权重，避免 Windows 上文件被占用/映射导致 1224 失败。
    逻辑：
      1) 先写入临时文件
      2) 尝试 replace 到目标
      3) 若失败（例如目标文件被占用），则改用带时间戳的文件名保存
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        try:
            os.replace(tmp, path)
        except Exception as exc:
            fallback = path.with_name(f"{path.stem}_{int(time.time())}{path.suffix}")
            os.replace(tmp, fallback)
            print(f"[WARN] could not replace {path} ({exc}); saved to {fallback}")
    except Exception as exc:
        print(f"[WARN] failed to save {path}: {exc}")
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

def to_str(x):
    """
    将 npz 里读出来的 object/bytes 统一转成 Python str。
    
    说明：np.savez 在保存 dtype=object 的字符串时，有时会以 bytes 的形式读出；
    为了后续用 json 路径做 key（字典索引）稳定一致，这里统一转成 str。
    """
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


# 读取边界嵌入：points_embed (N,K,4L) + source_json_paths 映射
def load_pattern_points(npz_path: Path):
    """
    读取 pattern piece 的边界嵌入（已离线计算好的 Fourier 特征）。
    
    输入：pattern_points_embed.npz
      - points_embed: (num_pieces, K, 4L) float32
      - source_json_paths: (num_pieces,) object/str
    
    返回：
      - points_embed: np.ndarray (num_pieces, K, 4L)
      - mapping: dict {source_json_path -> index}
    
    备注：训练时我们不会再喂 (B,K,2) 的原始点坐标，而是直接喂 (B,K,4L)。
    模型内部会做 Linear(4L -> d_model) + 可学习位置编码。
    """
    data = np.load(npz_path, allow_pickle=True)
    points_embed = data["points_embed"]  # (N, K, 4L)
    source_json_paths = [to_str(p) for p in data["source_json_paths"].tolist()]
    mapping = {p: i for i, p in enumerate(source_json_paths)}
    return points_embed, mapping


# 读取元素嵌入：按 json 聚合元素，并保持与 pdf_path + element_id 的稳定排序
def load_elements(npz_path: Path):
    """
    读取 element（图案/标注）的预处理结果，并按 pattern json 聚合成布局。
    
    输入：elements_embed.npz（由 build_element_embeddings.py 生成）
      - labels: 每个 element 的 type_id（整数）
      - center_x/center_y/w/h/sin/cos: (N,) 几何信息（已经归一化）
      - element_embed: (N, 4L) 只包含 x/y 的 Fourier 特征（用于 emb_pos 分支）
      - pattern_json_paths: (N,) 每个 element 属于哪个 pattern piece(json)
      - pdf_paths + element_ids: 用来给 element 做稳定排序（跨尺码对齐）
    
    输出：layouts: dict
      key: pattern_json_path
      value: {
        pdf_paths: list[str]  # 稳定排序后的 pdf 序列（用于跨尺码对齐）
        labels: np.ndarray (Ni,) int64
        geom: np.ndarray (Ni, 6) float32  # [x,y,w,h,sin,cos]
        pos_embed: np.ndarray (Ni, 4L) float32  # 位置 Fourier 特征
      }
    
    关键点：这里用 (pdf_path, element_id) 做排序，保证 src/tgt 可以逐元素对齐后再做监督。
    """
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
    if "logo_level" in data:
        logo_level = data["logo_level"].astype(np.int64)
    else:
        logo_level = np.full(labels.shape[0], -1, dtype=np.int64)
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
            "logo_level": logo_level[sorted_idxs],
        }
    return layouts


# 读取 size_scale_factors.json，构建 src_json -> tgt_json -> (scale_w, scale_h)
def build_scale_lookup(scale_json_path: Path):
    """
    读取 size_scale_factors.json，构建 src_json -> tgt_json -> (scale_w, scale_h)。
    
    scale_w / scale_h 的计算方式来自你给的公式：
      W_A = max(xA) - min(xA), H_A = max(yA) - min(yA)
      W_B = max(xB) - min(xB), H_B = max(yB) - min(yB)
      scale_w = W_B / W_A, scale_h = H_B / H_A
    
    训练时 (scale_w, scale_h) 会作为 Scale Token 融合进 Target Boundary tokens。
    """
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




# 自定义 collate：对不同尺寸的 SDF 进行 padding，并携带原始 (H,W)
def collate_batch(batch):
    """
    DataLoader 的 collate_fn：把 Dataset 返回的若干条样本拼成一个 batch。
    
    特殊点：SDF 的尺寸 (H,W) 会随 piece 的宽高比变化，默认 stack 会报错。
    因此这里对 sdf_map 做 batch 内 padding，并携带原始尺寸 sdf_hw。
    
    padding 规则：
      - pad 到当前 batch 的 (max_h, max_w)
      - pad 区域填 1.0（正值，表示轮廓外部，能产生惩罚）
    """
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

    # 返回：10 个张量，顺序与 collate_batch / 训练循环解包顺序一致
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
    """
    对带 mask 的张量求平均。
    
    mask: (B,N) 取值 0/1，1 表示有效 element，0 表示 padding element。
    values: 可以是 (B,N) 或 (B,N,C)。若是 (B,N,C)，会自动把 mask 扩展到最后一维。
    """
    if values.dim() == 3:
        mask = mask.unsqueeze(-1)
    masked = values * mask
    denom = mask.sum().clamp(min=1)
    return masked.sum() / denom


# 计算损失：位置/尺寸/旋转 + 正则 + L_shape(中心点采样SDF)
def compute_losses(pred, gt, mask, sdf_maps, sdf_hw, log_dim=False, log_pos=False, log_eps=1e-6):
    """
    计算训练/验证用的各项损失（与你给的公式对齐）。
    
    pred / gt: (B,N,6)，6=[x,y,w,h,sin,cos]（都在归一化空间里计算 loss）。
    mask: (B,N)，有效元素为 1，padding 为 0。
    
    L_pos: 位置损失（x,y 的 MSE）
    L_dim: 尺寸损失（w,h 的 L1）
    L_rot: 方向损失（sin,cos 的 MSE）
    L_reg: 旋转一致性正则（|sin^2 + cos^2 - 1|）
    L_shape: 包含损失（中心点采样 SDF，ReLU(dist)）
    
    注意：grid_sample 的坐标是 [-1,1] 且图像坐标左上为原点；
    我们数据坐标左下为原点，所以 y 需要做 (1 - y) 翻转。
    """
    pred_xy = pred[..., :2]
    pred_wh = pred[..., 2:4]
    # pred_sc = pred[..., 4:6]

    gt_xy = gt[..., :2]
    gt_wh = gt[..., 2:4]
    # gt_sc = gt[..., 4:6]

    if log_pos:
        pred_xy_log = torch.log1p(pred_xy.clamp(min=0.0) + log_eps)
        gt_xy_log = torch.log1p(gt_xy.clamp(min=0.0) + log_eps)
        l_pos = masked_mean((pred_xy_log - gt_xy_log) ** 2, mask)
    else:
        l_pos = masked_mean((pred_xy - gt_xy) ** 2, mask)

    if log_dim:
        pred_wh_log = torch.log1p(pred_wh.clamp(min=0.0) + log_eps)
        gt_wh_log = torch.log1p(gt_wh.clamp(min=0.0) + log_eps)
        l_dim = masked_mean((pred_wh_log - gt_wh_log).abs(), mask)
    else:
        l_dim = masked_mean((pred_wh - gt_wh).abs(), mask)
    # l_rot = masked_mean((pred_sc - gt_sc) ** 2, mask)
    # l_reg = masked_mean((pred_sc.pow(2).sum(dim=-1) - 1.0).abs(), mask)
    l_rot = torch.tensor(0.0, device=pred.device)
    l_reg = torch.tensor(0.0, device=pred.device)

    if sdf_maps is None or sdf_hw is None:
        raise ValueError("sdf_maps and sdf_hw are required (L_shape is mandatory).")

    # sample SDF at element centers only
    # sdf_hw is (B,2) -> (h, w) of original (unpadded) maps
    h = sdf_hw[:, 0].clamp(min=2).float()
    w = sdf_hw[:, 1].clamp(min=2).float()            #原始尺寸
    h_max = float(sdf_maps.shape[2] - 1)
    w_max = float(sdf_maps.shape[3] - 1)              #batch内最大尺寸
    x_scale = (w - 1.0) / max(w_max, 1.0)
    y_scale = (h - 1.0) / max(h_max, 1.0)
    pred_xy_norm = pred_xy / NORM_RANGE
    grid_x = pred_xy_norm[..., 0] * x_scale.view(-1, 1) * 2 - 1       #映射到grad_sample期望的[-1,1]
    grid_y = (1 - pred_xy_norm[..., 1]) * y_scale.view(-1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # (B, N, 1, 2)      #对每个 element 的中心点采样一次,找到对应的SDF值
    sampled = F.grid_sample(
        sdf_maps,
        grid,
        mode="bilinear",
        align_corners=True,
    )  # (B, 1, N, 1)                             # SDF 上双线性插值采样每个中心点对应的 SDF 值
    sampled = sampled.squeeze(1).squeeze(-1)  # (B, N)
    l_shape = masked_mean(F.relu(sampled), mask)
    return l_pos, l_dim, l_rot, l_reg, l_shape


# 数据集：根据 pair_splits 取 (src_json, tgt_json) 的布局与边界嵌入
class PairDataset(Dataset):
    """
    PairDataset：每条样本是一对布局 (Source A -> Target B)。
    
    样本来自 pair_splits.json 的一个 entry，包含：
      - src_json / tgt_json: 同一 piece 的不同尺码 json 路径
      - src_size / tgt_size: 尺码名（如 M -> XL）
    
    每条样本返回 10 个张量（与 collate_batch / 训练循环解包顺序一致）：
      1) src_types      : (Nmax,) int64   元素 type_id
      2) src_geom       : (Nmax,6) float32 源布局 [x,y,w,h,sin,cos]
      3) src_pos_embed  : (Nmax,4L) float32 预计算的位置 Fourier 特征（x/y）
      4) src_bnd_embed  : (K,4L) float32   源边界 Fourier 特征
      5) tgt_bnd_embed  : (K,4L) float32   目标边界 Fourier 特征
      6) scale_factors  : (2,) float32     [scale_w, scale_h]
      7) tgt_geom(gt)   : (Nmax,6) float32 目标布局真值
      8) valid_mask     : (Nmax,) float32  1=真实元素,0=padding
      9) sdf_map        : (1,H,W) float32  目标 piece 的 SDF（用于 L_shape）
     10) sdf_hw         : (2,) int64       原始 SDF 的 (H,W)
    
    strict=True：要求 src/tgt 的元素列表完全一致（按 pdf_paths 对齐）。
    因为这里的监督是逐元素的，如果不一致会导致 element 对不上。
    """
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
        """
        参数说明：
          pairs: list[dict]，pair_splits.json 中的样本列表
          layouts: dict，由 load_elements() 构建的 {json -> layout} 映射
          boundary_embed: np.ndarray (num_pieces,K,4L)，所有 json 的边界嵌入表
          boundary_index: dict {json -> index}，用于从 boundary_embed 里取某个 json 的 Kx4L
          scale_lookup: dict {src_json -> {tgt_json -> (scale_w,scale_h)}}
          max_elements: 固定 element token 长度（不足 padding，超过报错）
          strict: 是否强制 src/tgt 元素集合一致（建议训练时保持 True）
          sdf_dir/sdf_ext: SDF 文件根目录与后缀名（L_shape 强制使用，因此必须存在）
        """
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
        """
        把可变长的 element 序列 padding 到固定长度 max_elements。
        
        - 如果真实元素数量 > max_elements：直接报错（模型输入固定 20）。
        - pad_value：不同字段 padding 值不同：
            labels 用 0（也对应 type_vocab 里的 PAD 类）
            geom/pos_embed 用 0.0
            valid_mask 用 0.0（表示无效）
        """
        if arr.shape[0] > target:
            raise ValueError(f"element count {arr.shape[0]} exceeds max_elements {target}")
        if arr.shape[0] == target:
            return arr
        pad_shape = (target - arr.shape[0],) + arr.shape[1:]
        pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=0)

    def _load_sdf(self, json_path: str):
        """
        根据 tgt_json 路径加载对应的 SDF map。
        
        约定：SDF 文件与 json 共享相同的相对目录结构，例如：
          json: pattern/pattern/10052/2XL/6.json
          sdf : <sdf_dir>/pattern/pattern/10052/2XL/6.npy
        
        返回：np.ndarray (1,H,W) float32；内部为负/0，外部为正。
        """
        rel = Path(json_path)
        sdf_path = self.sdf_dir / rel.parent / (rel.stem + self.sdf_ext)
        if not sdf_path.is_file():
            raise FileNotFoundError(f"Missing SDF: {sdf_path}")
        sdf = np.load(sdf_path)
        if sdf.ndim == 2:
            sdf = sdf[None, ...]
        return sdf.astype(np.float32)

    # 单条样本：返回 src_types/src_geom/src_pos/src_bnd/tgt_bnd/scale/gt/valid/sdf/sdf_hw
    def __getitem__(self, idx):
        """
        取出一条 pair 样本，并组装成模型输入/监督所需的张量。
        
        处理步骤（按代码顺序）：
          1) 从 pairs[idx] 取 src_json/tgt_json
          2) 从 layouts 里取源/目标布局（labels/geom/pos_embed）
          3) strict 模式下检查两边元素是否一一对应（pdf_paths 列表必须相同）
          4) padding 到 max_elements，并生成 valid_mask
          5) 从 boundary_embed 里取 src/tgt 的边界嵌入 (K,4L)
          6) 从 scale_lookup 里取 (scale_w, scale_h)
          7) 加载目标 SDF，并记录原始 (H,W) 用于 batch padding/坐标映射
        """
        # 1) 当前样本是一对同形状不同尺码的有向布局：Source(A) -> Target(B)
        pair = self.pairs[idx]
        src_json = pair["src_json"]
        tgt_json = pair["tgt_json"]

        if src_json not in self.layouts or tgt_json not in self.layouts:
            raise KeyError(f"missing layout for {src_json} or {tgt_json}")

        src_layout = self.layouts[src_json]
        tgt_layout = self.layouts[tgt_json]

        # strict=True：要求 src/tgt 的元素序列严格一致（按 pdf_paths 对齐），否则无法做逐元素监督
        if self.strict and src_layout["pdf_paths"] != tgt_layout["pdf_paths"]:
            raise ValueError(
                "element set mismatch between source and target\n"
                f"src={src_json}\n"
                f"tgt={tgt_json}\n"
            )

        # 2) 读取源布局元素：type_id(label) + 几何向量(geom) + 位置 Fourier 特征(pos_embed)
        src_labels = src_layout["labels"].astype(np.int64)
        src_geom = src_layout["geom"].astype(np.float32)
        src_pos = src_layout["pos_embed"].astype(np.float32)

        tgt_geom = tgt_layout["geom"].astype(np.float32)

        # 3) valid_mask：真实 element=1，padding=0；后续 loss 只统计 valid=1 的位置
        valid = np.ones((src_labels.shape[0],), dtype=np.float32)

        # 4) padding 到固定长度 max_elements(默认 20)：不足补 0，超过直接报错（模型输入长度固定）
        src_labels = self._pad(src_labels, self.max_elements, pad_value=0)
        src_geom = self._pad(src_geom, self.max_elements, pad_value=0.0)
        src_pos = self._pad(src_pos, self.max_elements, pad_value=0.0)
        tgt_geom = self._pad(tgt_geom, self.max_elements, pad_value=0.0)
        valid = self._pad(valid, self.max_elements, pad_value=0.0)

        # 5) 读取 src/tgt 的边界嵌入 (K,4L)：作为 Transformer 的 boundary tokens 输入
        if src_json not in self.boundary_index or tgt_json not in self.boundary_index:
            raise KeyError(f"missing boundary embedding for {src_json} or {tgt_json}")

        src_bnd = self.boundary_embed[self.boundary_index[src_json]]
        tgt_bnd = self.boundary_embed[self.boundary_index[tgt_json]]

        # 6) 读取尺度因子 (scale_w, scale_h)：用于构建 Scale Token，并广播加到 Target boundary tokens 上
        scale_entry = self.scale_lookup.get(src_json, {}).get(tgt_json)
        if scale_entry is None:
            raise KeyError(f"missing scale factor for {src_json} -> {tgt_json}")
        scale_factors = np.array(scale_entry, dtype=np.float32)

        # 7) 加载目标 SDF (1,H,W)：用于 L_shape；同时记录原始 (H,W)，batch 内 collate 再做 padding
        sdf_map = self._load_sdf(tgt_json)
        sdf_hw = np.array([sdf_map.shape[1], sdf_map.shape[2]], dtype=np.int64)

        # 返回：10 个张量，顺序与 collate_batch / 训练循环解包顺序一致
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


# 训练入口：加载数据、构建 DataLoader、训练与验证
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate every N epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parents[1] / "result" / "logs"),
        help="TensorBoard log directory",
    )
    parser.add_argument("--lambda-pos", type=float, default=10)
    parser.add_argument("--lambda-dim", type=float, default=10)
    parser.add_argument("--lambda-rot", type=float, default=1.0)
    parser.add_argument("--lambda-shape", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--max-elements", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument(
        "--log-dim-loss",
        dest="log_dim_loss",
        action="store_true",
        default=False,
        help="Use log1p loss for w/h (default: off).",
    )
    parser.add_argument(
        "--no-log-dim-loss",
        dest="log_dim_loss",
        action="store_false",
        help="Disable log1p loss for w/h.",
    )
    parser.add_argument(
        "--log-pos-loss",
        dest="log_pos_loss",
        action="store_true",
        default=False,
        help="Use log1p loss for x/y (default: off).",
    )
    parser.add_argument(
        "--no-log-pos-loss",
        dest="log_pos_loss",
        action="store_false",
        help="Disable log1p loss for x/y.",
    )
    parser.add_argument("--log-eps", type=float, default=1e-6)
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parents[1] / "result" / "model"))
    parser.add_argument("--sdf-dir", default=str(Path(__file__).resolve().parents[1] / "sdf_maps"), help="Directory of SDF maps (required)")
    parser.add_argument("--sdf-ext", default=".npy", help="SDF file extension (default: .npy)")
    args = parser.parse_args()

    # 数据加载：
    # 1) pattern_points_embed.npz 提供每个 json 的边界嵌入 (K,4L)
    # 2) elements_embed.npz 提供每个 json 的元素信息（type/几何/pos_embed）
    # 3) size_scale_factors.json 提供 src->tgt 的缩放因子 (scale_w, scale_h)
    # 4) pair_splits.json 决定训练/验证/测试的有向对样本
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

    # 训练开始前打印样本数量，便于确认划分/数据是否正确
    print(f"pairs: train={len(train_pairs)} val={len(val_pairs)} (split_name={args.split_name})")

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

    print(f"dataset: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    model = LayoutFormer(
        num_element_types=num_types,
        max_elements=args.max_elements,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        boundary_seq_len=points_embed.shape[1],
        fourier_bands=10,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    writer = None
    if SummaryWriter is None:
        print("TensorBoard not available: torch.utils.tensorboard is missing.", file=sys.stderr)
    else:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=str(log_dir / run_name))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_pos = 0.0
        running_dim = 0.0
        running_rot = 0.0
        running_reg = 0.0
        running_shape = 0.0
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
                pred,
                tgt_geom,
                valid,
                sdf_maps=sdf_maps,
                sdf_hw=sdf_hw,
                log_dim=args.log_dim_loss,
                log_pos=args.log_pos_loss,
                log_eps=args.log_eps,
            )
            loss = (
                args.lambda_pos * l_pos
                + args.lambda_dim * l_dim
                # + args.lambda_rot * l_rot
                + args.lambda_shape * l_shape
                # + args.lambda_reg * l_reg
            )

            running_pos += l_pos.item()
            running_dim += l_dim.item()
            running_rot += l_rot.item()
            running_shape += l_shape.item()
            running_reg += l_reg.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        denom = max(1, len(train_loader))
        avg_loss = running / denom
        avg_pos = running_pos / denom
        avg_dim = running_dim / denom
        avg_rot = running_rot / denom
        avg_shape = running_shape / denom
        avg_reg = running_reg / denom

        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/pos", avg_pos, epoch)
            writer.add_scalar("train/dim", avg_dim, epoch)
            writer.add_scalar("train/rot", avg_rot, epoch)
            writer.add_scalar("train/shape", avg_shape, epoch)
            writer.add_scalar("train/reg", avg_reg, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        val_loss = None
        geom_sum = None
        if epoch % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_pos = 0.0
            val_dim = 0.0
            val_rot = 0.0
            val_shape = 0.0
            val_reg = 0.0
            geom_sum = 0.0
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
                        # + args.lambda_rot * l_rot
                        + args.lambda_shape * l_shape
                        # + args.lambda_reg * l_reg
                    )
                    val_loss += loss.item()
                    val_pos += l_pos.item()
                    val_dim += l_dim.item()
                    val_rot += l_rot.item()
                    val_shape += l_shape.item()
                    val_reg += l_reg.item()
                    geom_sum += (l_pos + l_dim + l_rot).item()

            val_denom = max(1, len(val_loader))
            val_loss = val_loss / val_denom
            val_pos = val_pos / val_denom
            val_dim = val_dim / val_denom
            val_rot = val_rot / val_denom
            val_shape = val_shape / val_denom
            val_reg = val_reg / val_denom
            geom_sum = geom_sum / val_denom
            print(
                f"epoch {epoch} train: loss={avg_loss:.6f} pos={avg_pos:.6f} dim={avg_dim:.6f} rot={avg_rot:.6f} "
                f"shape={avg_shape:.6f} reg={avg_reg:.6f}"
            )
            print(
                f"epoch {epoch} val:   loss={val_loss:.6f} pos={val_pos:.6f} dim={val_dim:.6f} rot={val_rot:.6f} "
                f"shape={val_shape:.6f} reg={val_reg:.6f} geom_sum={geom_sum:.6f}"
            )

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/pos", val_pos, epoch)
                writer.add_scalar("val/dim", val_dim, epoch)
                writer.add_scalar("val/rot", val_rot, epoch)
                writer.add_scalar("val/shape", val_shape, epoch)
                writer.add_scalar("val/reg", val_reg, epoch)
                writer.add_scalar("val/geom_sum", geom_sum, epoch)

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "geom_sum": geom_sum,
                "args": vars(args),
            }
            safe_torch_save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                safe_torch_save(ckpt, save_dir / "best.pt")
        else:
            print(
                f"epoch {epoch} train: loss={avg_loss:.6f} pos={avg_pos:.6f} dim={avg_dim:.6f} rot={avg_rot:.6f} "
                f"shape={avg_shape:.6f} reg={avg_reg:.6f} (eval skipped)"
            )

        if epoch == 5:
            best_path = save_dir / "best.pt"
            out_5 = save_dir / "5_best.pt"
            if best_path.exists():
                shutil.copy2(best_path, out_5)
            else:
                ckpt_5 = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "geom_sum": geom_sum,
                    "args": vars(args),
                }
                safe_torch_save(ckpt_5, out_5)

        if epoch == 200:
            best_path = save_dir / "best.pt"
            out_200 = save_dir / "200_best.pt"
            if best_path.exists():
                shutil.copy2(best_path, out_200)
            else:
                ckpt_200 = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "geom_sum": geom_sum,
                    "args": vars(args),
                }
                safe_torch_save(ckpt_200, out_200)

        if epoch == 500:
            best_path = save_dir / "best.pt"
            out_500 = save_dir / "500_best.pt"
            if best_path.exists():
                shutil.copy2(best_path, out_500)
            else:
                ckpt_500 = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "geom_sum": geom_sum,
                    "args": vars(args),
                }
                safe_torch_save(ckpt_500, out_500)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
