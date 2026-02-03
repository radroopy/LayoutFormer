import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FourierFeatureEncoder(nn.Module):
    """
    Encodes a coordinate (x or y) into a high-dimensional vector
    using sinusoidal functions at different frequencies.
    """
    def __init__(self, num_bands=10, max_freq=10.0):
        super().__init__()
        self.num_bands = num_bands
        # Create frequencies: 2^0, 2^1, ..., 2^(L-1)
        # We use powers of two to span the frequencies
        freqs = torch.arange(num_bands, dtype=torch.float32)
        self.register_buffer('freqs', torch.pow(2.0, freqs) * math.pi)

    def forward(self, x):
        # Input x: (Batch, Seq_Len) or (Batch, Seq_Len, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # args: (Batch, Seq_Len, num_bands)
        args = x * self.freqs
        
        sin_feat = torch.sin(args)
        cos_feat = torch.cos(args)
        
        # Output: (Batch, Seq_Len, num_bands * 2)
        return torch.cat([sin_feat, cos_feat], dim=-1)

class SDFContainer(nn.Module):
    """
    Computes Signed Distance Field (SDF) loss using GPU-accelerated Grid Sampling.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_coords, sdf_maps):
        """
        Args:
            pred_coords: (Batch, N, 2) [x, y] in range [0, 1]
            sdf_maps: (Batch, 1, H, W) Signed Distance Fields (Positive = Outside)
        """
        # 1. Remap [0, 1] -> [-1, 1] for grid_sample
        grid_coords = pred_coords * 2.0 - 1.0
        
        # 2. Reshape to (Batch, 1, N, 2) to treat points as a "1xN image"
        batch_size, n_elements, _ = pred_coords.shape
        sample_grid = grid_coords.unsqueeze(1)
        
        # 3. Sample Distance Values
        # padding_mode='border' ensures points far outside get the edge distance
        sampled_dist = F.grid_sample(
            sdf_maps, 
            sample_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=False
        )
        
        # 4. Flatten back to (Batch, N)
        sampled_dist = sampled_dist.view(batch_size, n_elements)
        
        # 5. Penalize only positive values (Outside)
        return F.relu(sampled_dist).mean()

class ShapeAwareLayoutTransformer(nn.Module):
    def __init__(self, 
                 num_classes=10, 
                 d_model=256, 
                 nhead=8, 
                 num_layers=4, 
                 boundary_seq_len=50,
                 fourier_bands=10):
        """
        Hybrid Transformer with Disentangled Embeddings.
        
        Guide Section 2.C.2 Implementation:
        - Class: Embedding
        - Position: MLP(Fourier(x,y))
        - Size: Linear
        - Rot: Linear
        """
        super().__init__()
        self.boundary_seq_len = boundary_seq_len
        self.d_model = d_model
        
        # --- 1. Encoders & Projections ---
        self.fourier_enc = FourierFeatureEncoder(num_bands=fourier_bands)
        raw_fourier_dim = (fourier_bands * 2) * 2 # *2 for sin/cos, *2 for X/Y
        
        # A. Boundary Embedding
        self.boundary_proj = nn.Linear(raw_fourier_dim, d_model)
        self.boundary_pos_enc = nn.Parameter(torch.randn(1, boundary_seq_len, d_model))
        
        # B. Element Embedding (Disentangled)
        self.emb_cls = nn.Embedding(num_classes, d_model // 4)
        
        # Position gets its own projection after Fourier encoding
        self.emb_pos = nn.Sequential(
            nn.Linear(raw_fourier_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )
        
        # Size (w, h) gets a simple linear projection
        self.emb_size = nn.Linear(2, d_model // 8)
        
        # Rotation (sin, cos) gets a simple linear projection
        self.emb_rot = nn.Linear(2, d_model // 8)
        
        # Fusion Layer: Combines all attributes into d_model
        # Input dim = (d/4) + (d/2) + (d/8) + (d/8) = d_model
        fusion_dim = (d_model // 4) + (d_model // 2) + (d_model // 8) + (d_model // 8)
        self.fusion_proj = nn.Linear(fusion_dim, d_model)

        # --- 2. Core Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- 3. Output Heads (Split) ---
        # Branch 1: Position & Size (x, y, w, h) -> Range [0, 1] via Sigmoid
        self.geo_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4), 
            nn.Sigmoid() 
        )
        
        # Branch 2: Orientation (sin, cos) -> Range [-1, 1] via Tanh
        self.rot_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2), 
            nn.Tanh()
        )

    def forward(self, 
                src_classes, 
                src_geometries, 
                target_boundary_points, 
                src_mask=None):
        
        # src_geometries: (Batch, N, 6) -> [x, y, w, h, sin, cos]
        
        # --- A. Process Boundaries (Sequence) ---
        bx = target_boundary_points[:, :, 0]
        by = target_boundary_points[:, :, 1]
        raw_boundary = torch.cat([self.fourier_enc(bx), self.fourier_enc(by)], dim=-1)
        
        boundary_tokens = self.boundary_proj(raw_boundary)
        boundary_tokens = boundary_tokens + self.boundary_pos_enc # ADD Pos Enc (Ordered)
        
        # --- B. Process Elements (Set) ---
        # 1. Unpack Geometry
        ex = src_geometries[:, :, 0]
        ey = src_geometries[:, :, 1]
        ew_eh = src_geometries[:, :, 2:4]
        esin_ecos = src_geometries[:, :, 4:6]
        
        # 2. Embed Components
        # Class
        feat_cls = self.emb_cls(src_classes) # (B, N, d/4)
        
        # Position (Fourier Encoded just like boundary)
        raw_pos = torch.cat([self.fourier_enc(ex), self.fourier_enc(ey)], dim=-1)
        feat_pos = self.emb_pos(raw_pos) # (B, N, d/2)
        
        # Size & Rot
        feat_size = self.emb_size(ew_eh) # (B, N, d/8)
        feat_rot = self.emb_rot(esin_ecos) # (B, N, d/8)
        
        # 3. Fuse
        concat_feats = torch.cat([feat_cls, feat_pos, feat_size, feat_rot], dim=-1)
        element_tokens = self.fusion_proj(concat_feats)
        
        # NO Pos Enc for elements (Set behavior / Permutation Invariant)
        
        # --- C. Concatenate & Pass ---
        combined_tokens = torch.cat([boundary_tokens, element_tokens], dim=1)
        
        # Handle Masking if provided (pad mask needs to account for boundary length)
        # Assuming src_mask is (B, N), we need to prepend False (valid) for boundary
        if src_mask is not None:
            b_mask = torch.zeros((src_mask.shape[0], self.boundary_seq_len), 
                               device=src_mask.device, dtype=torch.bool)
            full_mask = torch.cat([b_mask, src_mask], dim=1)
        else:
            full_mask = None

        features = self.transformer(combined_tokens, src_key_padding_mask=full_mask)
        
        # --- D. Predict ---
        element_features = features[:, self.boundary_seq_len:, :]
        
        pred_geo = self.geo_head(element_features) # [x, y, w, h]
        pred_rot = self.rot_head(element_features) # [sin, cos]
        
        return torch.cat([pred_geo, pred_rot], dim=-1)

def train_step_detailed(model, sdf_container, optimizer, batch_data, weights=None):
    """
    Detailed training step implementing the full loss formulation.
    """
    model.train()
    optimizer.zero_grad()
    
    if weights is None:
        weights = {'pos': 10.0, 'dim': 10.0, 'rot': 1.0, 'shape': 5.0, 'reg': 1.0}
    
    # Unpack Batch
    # sdf_maps: Pre-computed (Batch, 1, H, W) tensors
    src_cls, src_geo, target_boundary, gt_geo, sdf_maps = batch_data
    
    # Forward Pass
    pred_full = model(src_cls, src_geo, target_boundary)
    
    # Split Output
    pred_xy = pred_full[:, :, :2]
    pred_wh = pred_full[:, :, 2:4]
    pred_rot = pred_full[:, :, 4:]
    
    # Split Ground Truth
    gt_xy = gt_geo[:, :, :2]
    gt_wh = gt_geo[:, :, 2:4]
    gt_rot = gt_geo[:, :, 4:]
    
    # --- Loss Components ---
    
    # 1. Position Loss (MSE)
    loss_pos = F.mse_loss(pred_xy, gt_xy)
    
    # 2. Dimension Loss (L1) - More stable for sizes
    loss_dim = F.l1_loss(pred_wh, gt_wh)
    
    # 3. Rotation Loss (MSE on sin/cos)
    loss_rot = F.mse_loss(pred_rot, gt_rot)
    
    # 4. Containment Loss (SDF)
    # Checks if pred_xy is inside the target shape
    loss_shape = sdf_container(pred_xy, sdf_maps)
    
    # 5. Regularization Loss (Trig Identity)
    # sin^2 + cos^2 should be 1
    norm_sq = (pred_rot ** 2).sum(dim=-1) # sin^2 + cos^2
    loss_reg = torch.abs(norm_sq - 1.0).mean()
    
    # --- Total Loss ---
    total_loss = (weights['pos'] * loss_pos +
                  weights['dim'] * loss_dim +
                  weights['rot'] * loss_rot +
                  weights['shape'] * loss_shape +
                  weights['reg'] * loss_reg)
    
    total_loss.backward()
    optimizer.step()
    
    return {
        "total": total_loss.item(),
        "pos": loss_pos.item(),
        "shape": loss_shape.item()
    }

if __name__ == "__main__":
    # --- Quick Sanity Check ---
    B, N, K = 2, 5, 50
    model = ShapeAwareLayoutTransformer(d_model=256, nhead=4, num_layers=2)
    sdf_mod = SDFContainer()
    opt = torch.optim.Adam(model.parameters())
    
    # Mock Data
    src_cls = torch.randint(0, 5, (B, N))
    src_geo = torch.rand(B, N, 6) # [x,y,w,h,s,c]
    tgt_bnd = torch.rand(B, K, 2)
    gt_geo = torch.rand(B, N, 6)
    
    # Mock SDF Maps (Batch, 1, 64, 64) - Dummy values
    # In reality, load these from your pre-computed dataset
    mock_sdf_maps = torch.randn(B, 1, 64, 64) 
    
    batch = (src_cls, src_geo, tgt_bnd, gt_geo, mock_sdf_maps)
    
    metrics = train_step_detailed(model, sdf_mod, opt, batch)
    print("Training Step Complete. Metrics:", metrics)