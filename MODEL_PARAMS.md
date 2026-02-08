# LayoutFormer 参数与当前默认值

> 来源：`models/LayoutFormer.py` 与 `train/train_layoutformer.py` 的当前默认配置。
> 如有命令行覆盖，以运行时参数为准。

## 1) 模型结构参数（Transformer）

**核心维度**
- `d_model = 256`
- `nhead = 8`
- `num_layers = 4`
- `max_elements = 20`
- `boundary_seq_len = K`（由 `pattern_points_embed.npz` 的 K 决定，运行时 `points_embed.shape[1]`）
- `fourier_bands = 10`（原始 Fourier 维度 `raw_fourier_dim = 4L = 40`）
- `num_element_types`：由 `data/element_emb/type_vocab.json` 的词表大小决定

**边界与缩放嵌入**
- `boundary_proj`: Linear(40 → 256)
- `boundary_pos_enc`: 可学习位置编码 (1, K, 256)
- `scale_proj`: Linear(2 → 256)

**元素嵌入（解耦后融合）**
- `emb_type`: Embedding(num_element_types → 64)
- `emb_pos`: MLP(40 → 128 → 128)
- `emb_size`: Linear(2 → 32)
- `emb_rot`: Linear(2 → 32)
- `fusion_proj`: Linear(64 + 128 + 32 + 32 = 256 → 256)

**Transformer 主体（PyTorch 默认值）**
- `TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)`
- `dim_feedforward`：**默认 2048**
- `dropout`：**默认 0.1**
- `activation`：**默认 ReLU**

**输出头**
- `geo_head`: Linear(256 → 128) → ReLU → Linear(128 → 4) → Sigmoid
- `rot_head`: Linear(256 → 128) → ReLU → Linear(128 → 2) → Tanh

## 2) 训练超参数（当前默认值）

- `batch_size = 128`
- `epochs = 2000`
- `eval_interval = 5`
- `lr = 1e-4`
- `num_workers = 0`
- `device = cuda (自动选择)`

**Loss 权重**
- `lambda_pos = 10`
- `lambda_dim = 10`
- `lambda_rot = 0.0`  ← 目前设为关闭
- `lambda_shape = 1.0`
- `lambda_reg = 1.0`

**其它**
- `max_elements = 20`
- `sdf_dir = sdf_maps`（必需）
- `sdf_ext = .npy`
- `save_dir = result/model`

## 3) 数据输入约定（简要）

- 边界输入：`(B, K, 4L)` 或 `(B, K, 2)`（若为 2 则内部做 Fourier）
- 元素输入：`src_types (B, N)`, `src_geom (B, N, 6)`
- 预测输出：`(B, N, 6)`，顺序 `[x, y, w, h, sin, cos]`
