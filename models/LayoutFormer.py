#!/usr/bin/env python3
import torch
import torch.nn as nn

from ly import FourierFeatureEncoder


class LayoutFormer(nn.Module):
    """
    Hybrid Transformer for multi-size grading (A -> B).

    Inputs (data loader provides 7 items):
      1) src_types:           (B, Nmax=20)       source type ids
      2) src_geometries:        (B, Nmax=20, 6)    [x, y, w, h, sin, cos] normalized to source A
      3) src_boundary_points:   (B, K, 2) or (B, K, 4L)  source A boundary (points or pre-encoded)
      4) target_boundary_points:(B, K, 2) or (B, K, 4L)  target B boundary (points or pre-encoded)
      5) scale_factors:         (B, 2)          [scale_w, scale_h] computed from real-world coords
      6) gt_geometries:         (B, Nmax=20, 6)    target B ground truth (used in loss)
      7) sdf_maps:              (B, 1, H, W)    target B SDF maps (used in loss)

    The model uses (1)-(5) and predicts target geometries (B, Nmax=20, 6).

    By default (return_postprocess=True), provide target_scale (S=max(W_target,H_target)) to get denormalized coords; sin/cos stay unchanged.

    Padding: if src_mask is not provided, rows with type_id==0 and all-zero geometry are masked.
    """

    def __init__(
        self,
        num_element_types: int = 10,
        max_elements: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        boundary_seq_len: int = 196,
        fourier_bands: int = 10,
    ):
        super().__init__()
        if d_model % 8 != 0:
            raise ValueError("d_model must be divisible by 8")

        self.boundary_seq_len = boundary_seq_len
        self.d_model = d_model
        self.max_elements = max_elements

        # Fourier encoders
        self.fourier_enc = FourierFeatureEncoder(num_bands=fourier_bands)
        raw_fourier_dim = (fourier_bands * 2) * 2  # sin/cos and x/y
        self.raw_fourier_dim = raw_fourier_dim

        # Boundary encoders
        self.boundary_proj = nn.Linear(raw_fourier_dim, d_model)
        self.boundary_pos_enc = nn.Parameter(torch.randn(1, boundary_seq_len, d_model))
        # self.boundary_type_emb = nn.Embedding(2, d_model)  # 0=src, 1=target

        # Scale token
        self.scale_proj = nn.Linear(2, d_model)
        # self.scale_type_emb = nn.Parameter(torch.randn(1, 1, d_model))

        # Element embeddings (disentangled)
        self.emb_type = nn.Embedding(num_element_types, d_model // 4)
        self.emb_pos = nn.Sequential(
            nn.Linear(raw_fourier_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        self.emb_size = nn.Linear(2, d_model // 8)
        self.emb_rot = nn.Linear(2, d_model // 8)
        fusion_dim = (d_model // 4) + (d_model // 2) + (d_model // 8) + (d_model // 8)
        self.fusion_proj = nn.Linear(fusion_dim, d_model)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.geo_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),
            nn.Sigmoid(),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
            nn.Tanh(),
        )

    def _encode_boundary(self, boundary: torch.Tensor, boundary_type: int) -> torch.Tensor:
        # boundary: (B, K, 2) points OR (B, K, 4L) pre-encoded
        if boundary.dim() != 3:
            raise ValueError("boundary must be 3D (B, K, C)")

        last_dim = boundary.shape[-1]
        if last_dim == 2:
            bx = boundary[:, :, 0]
            by = boundary[:, :, 1]
            raw = torch.cat([self.fourier_enc(bx), self.fourier_enc(by)], dim=-1)
        elif last_dim == self.raw_fourier_dim:
            raw = boundary
        else:
            raise ValueError(
                f"boundary last dim must be 2 or {self.raw_fourier_dim}, got {last_dim}"
            )

        tokens = self.boundary_proj(raw)

        seq_len = tokens.shape[1]
        if seq_len > self.boundary_pos_enc.shape[1]:
            raise ValueError(f"boundary_seq_len too small: {seq_len} > {self.boundary_pos_enc.shape[1]}")

        # Add learnable positional encoding: (1, K, d_model) broadcast over batch
        tokens = tokens + self.boundary_pos_enc[:, :seq_len, :]

        type_ids = torch.full((tokens.shape[0], seq_len), boundary_type, device=tokens.device, dtype=torch.long)
        # tokens = tokens + self.boundary_type_emb(type_ids)
        return tokens

    def _decode_elements(self, features: torch.Tensor, tgt_len: int, src_len: int) -> torch.Tensor:
        """
        Decode transformer outputs into element geometries.

        Sequence slicing:
          - drop first 2K tokens (target boundary + source boundary)
          - keep N element tokens
        """
        elem_start = tgt_len + src_len
        element_features = features[:, elem_start:, :]
        pred_geo = self.geo_head(element_features)  # (B, N, 4) in [0,1]
        pred_rot = self.rot_head(element_features)  # (B, N, 2) in [-1,1]
        return torch.cat([pred_geo, pred_rot], dim=-1)

    @staticmethod
    def denormalize(pred: torch.Tensor, target_scale: torch.Tensor) -> torch.Tensor:
        """
        Denormalize predicted geometries using isotropic scale S = max(W_target, H_target).

        pred: (B, N, 6) where x,y,w,h are normalized to [0,1]. Only first 4 dims are scaled.
        target_scale: (B,) or (B,1) or (B,1,1)
        """
        if target_scale.dim() == 1:
            scale = target_scale.view(-1, 1, 1)
        elif target_scale.dim() == 2 and target_scale.shape[1] == 1:
            scale = target_scale.view(-1, 1, 1)
        elif target_scale.dim() == 3 and target_scale.shape[1:] == (1, 1):
            scale = target_scale
        else:
            raise ValueError("target_scale must be (B,), (B,1), or (B,1,1)")

        out = pred.clone()
        out[..., :4] = out[..., :4] * scale
        return out

    @staticmethod
    def decode_orientation(pred: torch.Tensor) -> torch.Tensor:
        """
        Decode orientation from sin/cos. Returns theta in radians.
        pred: (B, N, 6) with sin at index 4, cos at index 5.
        """
        sin = pred[..., 4]
        cos = pred[..., 5]
        return torch.atan2(sin, cos)

    def forward(
        self,
        src_types: torch.Tensor,
        src_geometries: torch.Tensor,
        src_boundary_points: torch.Tensor,
        target_boundary_points: torch.Tensor,
        scale_factors: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        target_scale: torch.Tensor | None = None,
        return_postprocess: bool = True,
        src_pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Ensure element sequence length is fixed to max_elements
        if src_types.shape[1] != self.max_elements:
            bsz, cur_n = src_types.shape
            if cur_n > self.max_elements:
                raise ValueError(f"src_types length {cur_n} exceeds max_elements {self.max_elements}")
            pad_n = self.max_elements - cur_n
            pad_elem = torch.zeros((bsz, pad_n), device=src_types.device, dtype=src_types.dtype)
            src_types = torch.cat([src_types, pad_elem], dim=1)
            pad_geo = torch.zeros((bsz, pad_n, src_geometries.shape[-1]), device=src_geometries.device, dtype=src_geometries.dtype)
            src_geometries = torch.cat([src_geometries, pad_geo], dim=1)
            if src_pos_embed is not None:
                pad_pos = torch.zeros((bsz, pad_n, src_pos_embed.shape[-1]), device=src_pos_embed.device, dtype=src_pos_embed.dtype)
                src_pos_embed = torch.cat([src_pos_embed, pad_pos], dim=1)
            if src_mask is not None:
                pad_mask = torch.ones((bsz, pad_n), device=src_mask.device, dtype=torch.bool)
                src_mask = torch.cat([src_mask, pad_mask], dim=1)

        # Auto-build src_mask from padding (class==0 and geometry all-zero)
        if src_mask is None:
            pad_geom = src_geometries.abs().sum(dim=-1) == 0
            pad_cls = src_types == 0
            src_mask = pad_geom & pad_cls

        # Boundary tokens (source + target)
        src_boundary_tokens = self._encode_boundary(src_boundary_points, boundary_type=0)
        tgt_boundary_tokens = self._encode_boundary(target_boundary_points, boundary_type=1)

        # Scale embedding (add to target boundary tokens)
        scale_token = self.scale_proj(scale_factors).unsqueeze(1)  # (B,1,d)
        tgt_boundary_tokens = tgt_boundary_tokens + scale_token

        # Element tokens (source)
        ex = src_geometries[:, :, 0]
        ey = src_geometries[:, :, 1]
        ew_eh = src_geometries[:, :, 2:4]
        esin_ecos = src_geometries[:, :, 4:6]

        feat_cls = self.emb_type(src_types)
        if src_pos_embed is not None:
            if src_pos_embed.shape[-1] != self.raw_fourier_dim:
                raise ValueError(f"src_pos_embed last dim must be {self.raw_fourier_dim}")
            raw_pos = src_pos_embed
        else:
            raw_pos = torch.cat([self.fourier_enc(ex), self.fourier_enc(ey)], dim=-1)
        feat_pos = self.emb_pos(raw_pos)
        feat_size = self.emb_size(ew_eh)
        feat_rot = self.emb_rot(esin_ecos)
        concat_feats = torch.cat([feat_cls, feat_pos, feat_size, feat_rot], dim=-1)
        element_tokens = self.fusion_proj(concat_feats)

        # Concatenate: [src boundary | target boundary | scale | elements]
        combined_tokens = torch.cat(
            [tgt_boundary_tokens, src_boundary_tokens, element_tokens], dim=1
        )

        if src_mask is not None:
            bsz = src_mask.shape[0]
            prefix_len = tgt_boundary_tokens.shape[1] + src_boundary_tokens.shape[1]
            prefix_mask = torch.zeros((bsz, prefix_len), device=src_mask.device, dtype=torch.bool)
            full_mask = torch.cat([prefix_mask, src_mask], dim=1)
        else:
            full_mask = None

        features = self.transformer(combined_tokens, src_key_padding_mask=full_mask)
        pred = self._decode_elements(features, tgt_boundary_tokens.shape[1], src_boundary_tokens.shape[1])
        if not return_postprocess:
            return pred
        if target_scale is None:
            raise ValueError("target_scale is required when return_postprocess=True")
        pred_denorm = self.denormalize(pred, target_scale)
        # keep sin/cos unchanged; output stays (B, N, 6)
        return pred_denorm
