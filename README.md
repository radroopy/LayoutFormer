# LayoutFormer

LayoutFormer 是一个用于服装版片跨尺码布局迁移的 Transformer 管线。
输入源尺码布局和源/目标版片边界，输出目标尺码上的元素几何位置与尺寸。

仓库主要包含：
- 数据预处理脚本（embedding、切分、SDF）
- 训练与验证代码
- 测试推理、误差统计与 PDF 可视化脚本

## 1. 项目流程

完整流程如下：

1. 准备版片 JSON、级放表格（xlsx）和元素 PDF
2. 执行预处理（`preprocess/run_preprocess.py`）
3. 训练模型（`train/train_layoutformer.py`）
4. 测试推理（`test/run_test.py`）
5. 误差统计与可视化（`test/compute_prediction_errors.py`、`test/render_predictions_pdf.py`）

## 2. 目录说明

关键路径：

- `models/LayoutFormer.py`：主模型定义
- `preprocess/`：预处理流程
- `train/train_layoutformer.py`：训练入口
- `test/run_test.py`：测试入口
- `test/compute_prediction_errors.py`：预测误差统计
- `test/render_predictions_pdf.py`：预测结果渲染为 PDF
- `pattern/pattern/`：版片 JSON 数据
- `logo/logo/`：元素 PDF 数据
- `sdf_maps/`：SDF 地图（用于形状约束损失）
- `result/`：训练日志和模型权重（运行生成）
- `data/`：预处理生成产物（`.gitignore` 中已忽略）

## 3. 环境依赖

建议环境：

- Python 3.10+
- PyTorch（可选 CUDA）

安装依赖：

```bash
pip install torch numpy openpyxl pillow scipy pymupdf
```

可选（TensorBoard）：

```bash
pip install tensorboard
```

## 4. 数据前提

默认假设如下：

- 版片 JSON 位于 `pattern/pattern/`
- 有一份用于训练的级放 xlsx，列结构满足预处理脚本要求
- xlsx 中引用的元素 PDF 可被解析（通常位于 `logo/logo/`）

注意：
- `data/` 和 `train/pair_splits.json` 是生成文件，不在 git 中跟踪。
- 新 clone 仓库后，必须先跑预处理，再训练/测试。

## 5. 预处理

一键按顺序执行预处理：

```bash
python preprocess/run_preprocess.py \
  --pattern-root pattern/pattern \
  --xlsx "<你的级放表.xlsx>" \
  --sdf-base 512
```

如果需要基于旧表格构造测试集划分：

```bash
python preprocess/run_preprocess.py \
  --pattern-root pattern/pattern \
  --xlsx "<你的级放表.xlsx>" \
  --old-xlsx "<旧级放表.xlsx>"
```

主要会生成：

- `data/pattern_piece_emb/pattern_points_embed.npz`
- `data/element_emb/elements_embed.npz`
- `data/element_emb/type_vocab.json`
- `data/shape_element_map.json`
- `data/shape_size_index.json`
- `data/json_size_lookup.json`
- `data/size_scale_factors.json`
- `train/pair_splits.json`
- `sdf_maps/**/*.npy`

## 6. 训练

基础训练命令：

```bash
python train/train_layoutformer.py \
  --split train/pair_splits.json \
  --split-name train \
  --data-dir data \
  --sdf-dir sdf_maps \
  --batch-size 256 \
  --epochs 2000 \
  --eval-interval 10 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --max-elements 20 \
  --save-dir result/model \
  --log-dir result/logs
```

说明：
- 权重会保存到 `result/model/last.pt` 和 `result/model/best.pt`。
- 当前训练脚本把 `L_shape` 作为必选项，因此 `--sdf-dir` 必须有效。
- 当前总损失主要使用位置、尺寸和形状项；旋转项代码保留但在损失组合中默认关闭。
- `test/run_test.py` 里当前固定 `PairDataset(..., max_elements=20)`。如果样本元素数超过 20，需要同步调整测试脚本和训练参数，保持一致。

## 7. 测试推理

按 split 执行预测：

```bash
python test/run_test.py \
  --split train/pair_splits.json \
  --split-name test \
  --data-dir data \
  --ckpt result/model/best.pt \
  --sdf-dir sdf_maps \
  --batch-size 8 \
  --d-model 256 \
  --num-layers 4 \
  --nhead 8 \
  --out-dir test/result
```

输出文件：

- `test/result/predictions_test.json`

## 8. 后处理

### 8.1 误差统计

```bash
python test/compute_prediction_errors.py \
  --pred test/result/predictions_test.json \
  --elements data/element_emb/elements_embed.npz \
  --out test/result/prediction_element_errors.json
```

### 8.2 渲染预测 PDF

输出单个多页 PDF：

```bash
python test/render_predictions_pdf.py \
  --pred test/result/predictions_test.json \
  --one-file \
  --out test/result/predictions_test_render.pdf
```

每条样本输出一个 PDF：

```bash
python test/render_predictions_pdf.py \
  --pred test/result/predictions_test.json \
  --out-dir test/result/predictions_test
```

## 9. 常见问题

- `File not found: data/...`
  - 先执行预处理，或检查 `--data-dir` 是否指向正确产物目录。

- `Missing SDF: ...`
  - 检查 `sdf_maps/**/*.npy` 是否存在，且 `--sdf-dir` 是否正确。

- `Checkpoint not found`
  - 检查训练时 `--save-dir` 与测试时 `--ckpt` 是否一致。

- `TensorBoard not available`
  - 安装 `tensorboard`；若不看可视化日志可忽略。

- 可视化时部分元素未绘制
  - 检查 xlsx 中 PDF 路径与 `test/render_predictions_pdf.py` 的 `--pdf-root` 设置。

## 10. 快速检查清单

首次跑通建议按以下顺序：

1. 安装依赖。
2. 确认版片 JSON、xlsx、PDF 都已就位。
3. 执行 `preprocess/run_preprocess.py`。
4. 确认 `data/` 与 `train/pair_splits.json` 已生成。
5. 训练并拿到 `result/model/best.pt`。
6. 执行测试和后处理脚本。
