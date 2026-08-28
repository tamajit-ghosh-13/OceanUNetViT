"""
================================================================================
OceanEmbed - Hybrid In-Situ ARGO Fine-Tuning Pipeline (train_v5_finetuned.py)
================================================================================
Fine-tunes OceanUNetViT v5 by jointly training on dense reanalysis grids AND
792,481 sparse, real-world physical ARGO float CTD measurements.

MATHEMATICAL HYBRID LOSS FORMULATION:
  L_total = L_grid_physics + lambda_ARGO * L_in_situ

  1. Dense Physics Grid Loss (L_grid_physics):
     Enforces layer-weighted reconstruction, vertical gradient matching, curvature,
     and strict monotonicity across full (101 x 241) ocean grids.

  2. Differentiable Continuous In-Situ Residual Loss (L_in_situ):
     For a float at continuous coordinates (lat_f, lon_f, z_f):
       a. Normalized float coordinates: gx in [0, W-1], gy in [0, H-1]
       b. 4-Corner Grid Vertex Indices: (x0, y0), (x1, y0), (x0, y1), (x1, y1)
       c. Continuous Bilinear Weights:
            w_a = (x1 - gx) * (y1 - gy)
            w_b = (x1 - gx) * (gy - y0)
            w_c = (gx - x0) * (y1 - gy)
            w_d = (gx - x0) * (gy - y0)
       d. Differentiable Predicted Temperature Gather:
            T_pred_float = w_a * I(x0,y0) + w_b * I(x0,y1) + w_c * I(x1,y0) + w_d * I(x1,y1)
       e. Mean Squared Residual Error:
            L_in_situ = (1 / N_batch) * sum_{i=1}^{1024} ( T_pred_float^i - T_argo_norm^i )^2

TRAINING LINEAGE & PARAMETERS:
  • Warm-Start Base: checkpoints/best_ocean_model_v5.pt
  • Unified In-Situ Dataset: 792,481 physical CTD observations across 6 eras
  • Optimizer: AdamW (lr = 1.5e-5, weight_decay = 1e-5) with Cosine Annealing
  • Output Checkpoint: checkpoints/best_ocean_model_v5_finetuned.pt
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    NORMALIZATION_STATS,
    TEMP_TARGET_STATS_PER_DEPTH,
)
from model import create_model
from train import get_compute_device
from data_loader import OceanDataset
from scripts.train_v5 import PhysicsPreservingThermoclineLoss


def build_unified_argo_dataset():
    """Loads and unifies all in-situ ARGO float CSVs."""
    csv_files = glob.glob("./data/argo_*.csv")
    print(f"📦 Found {len(csv_files)} ARGO observation datasets:")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"   - {os.path.basename(f)}: {len(df):,} float records")
    
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["latitude", "longitude", "pres", "temp", "date"])
    df_all = df_all[(df_all["pres"] >= 0) & (df_all["pres"] <= 1050)]
    df_all = df_all[(df_all["temp"] >= 2.0) & (df_all["temp"] <= 35.0)]
    print(f"✅ Total Unified In-Situ ARGO Database: {len(df_all):,} high-quality physical observations.")
    return df_all


class InSituResidualLoss(nn.Module):
    """
    Computes differentiable spatial bilinear interpolation loss for sparse in-situ floats
    using native PyTorch tensor ops (100% MPS backward compatible).
    """
    def __init__(self, lat_min=BBOX["min_lat"], lat_max=BBOX["max_lat"],
                 lon_min=BBOX["min_lon"], lon_max=BBOX["max_lon"]):
        super().__init__()
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    def forward(self, pred_grid: torch.Tensor, float_lats: torch.Tensor, float_lons: torch.Tensor,
                float_depth_idxs: torch.Tensor, float_temps_norm: torch.Tensor) -> torch.Tensor:
        """
        pred_grid: (B, 15, H, W) normalized
        float_lats, float_lons: (N,) coordinates
        float_depth_idxs: (N,) standard depth index (0..14)
        float_temps_norm: (N,) ground truth normalized temperature
        """
        H, W = pred_grid.shape[2], pred_grid.shape[3]

        # Normalized coordinates to floating index in [0, H-1] and [0, W-1]
        gx = (float_lons - self.lon_min) / (self.lon_max - self.lon_min) * (W - 1.0)
        gy = (float_lats - self.lat_min) / (self.lat_max - self.lat_min) * (H - 1.0)

        gx = torch.clamp(gx, 0.0, W - 1.0)
        gy = torch.clamp(gy, 0.0, H - 1.0)

        x0 = torch.floor(gx).long()
        x1 = torch.clamp(x0 + 1, max=W - 1)
        y0 = torch.floor(gy).long()
        y1 = torch.clamp(y0 + 1, max=H - 1)

        wa = ((x1.float() - gx) * (y1.float() - gy))
        wb = ((x1.float() - gx) * (gy - y0.float()))
        wc = ((gx - x0.float()) * (y1.float() - gy))
        wd = ((gx - x0.float()) * (gy - y0.float()))

        # Average prediction across batch dimension (15, H, W)
        pred_mean = pred_grid.mean(dim=0)

        # Vectorized gather from 4 corners: (15, H, W) -> (N,)
        Ia = pred_mean[float_depth_idxs, y0, x0]
        Ib = pred_mean[float_depth_idxs, y1, x0]
        Ic = pred_mean[float_depth_idxs, y0, x1]
        Id = pred_mean[float_depth_idxs, y1, x1]

        pred_at_float = wa * Ia + wb * Ib + wc * Ic + wd * Id
        loss_insitu = torch.mean((pred_at_float - float_temps_norm) ** 2)
        return loss_insitu


def finetune_v5_with_argo():
    device = get_compute_device()
    print("=" * 90)
    print("🌊 OCEANEMBED v5_FINETUNED: HYBRID REANALYSIS + IN-SITU ARGO CALIBRATION PASS")
    print("=" * 90)

    # 1. Load Multi-Season Reanalysis Grid Data (457 days)
    f_existing_in = "data/train_jun25_feb26_surface_inputs_12ch.npy"
    f_existing_tg = "data/train_jun25_feb26_subsurface_targets.npy"
    f_existing_dt = "data/train_jun25_feb26_dates.npy"

    in_25_26 = np.load(f_existing_in)
    tg_25_26 = np.load(f_existing_tg)
    dt_25_26 = np.load(f_existing_dt)

    from fetch_multi_season_dataset import build_full_multiseason_dataset
    in_ms, tg_ms, dt_ms = build_full_multiseason_dataset()

    all_train_inputs = np.concatenate([in_25_26, in_ms], axis=0).astype(np.float32)
    all_train_targets = np.concatenate([tg_25_26, tg_ms], axis=0).astype(np.float32)
    all_train_dates = np.concatenate([dt_25_26, dt_ms], axis=0)

    print(f"\n📊 Grand Multi-Season Training Catalog: {len(all_train_dates)} Total Ocean Days")
    train_ds = OceanDataset(surface_inputs=all_train_inputs, subsurface_targets=all_train_targets, dates=all_train_dates, use_mock_data=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)

    # 2. Build In-Situ ARGO Observation Tensor
    df_argo = build_unified_argo_dataset()
    depth_bins = [0, 2.5, 7.5, 15, 25, 40, 62.5, 87.5, 112.5, 137.5, 175, 250, 400, 600, 850, 1100]
    
    pres_arr = df_argo["pres"].values
    d_indices = np.zeros(len(df_argo), dtype=np.int64)
    for b_idx in range(len(depth_bins) - 1):
        mask = (pres_arr >= depth_bins[b_idx]) & (pres_arr < depth_bins[b_idx + 1])
        d_indices[mask] = b_idx

    temps_raw = df_argo["temp"].values
    temps_norm = np.zeros_like(temps_raw, dtype=np.float32)
    for d_idx in range(15):
        mask = (d_indices == d_idx)
        d_mean = TEMP_TARGET_STATS_PER_DEPTH[d_idx]["mean"]
        d_std  = TEMP_TARGET_STATS_PER_DEPTH[d_idx]["std"]
        temps_norm[mask] = (temps_raw[mask] - d_mean) / d_std

    argo_lats = torch.tensor(df_argo["latitude"].values, dtype=torch.float32, device=device)
    argo_lons = torch.tensor(df_argo["longitude"].values, dtype=torch.float32, device=device)
    argo_d_idxs = torch.tensor(d_indices, dtype=torch.int64, device=device)
    argo_temps_norm = torch.tensor(temps_norm, dtype=torch.float32, device=device)

    # 3. Initialize Model and Load v5 Weights
    print("\n🧠 Initializing OceanUNetViT (12-Channel Backbone)...")
    model = create_model(in_channels=12, out_depth_levels=15).to(device)
    ckpt_v5 = "checkpoints/best_ocean_model_v5.pt"
    if os.path.exists(ckpt_v5):
        print(f"   🔥 Loading baseline weights from {ckpt_v5}...")
        model.load_state_dict(torch.load(ckpt_v5, map_location=device), strict=False)
        print("   ✅ Loaded v5 pre-trained weights!")

    # 4. Loss Functions & Optimizer
    criterion_physics = PhysicsPreservingThermoclineLoss().to(device)
    criterion_insitu = InSituResidualLoss().to(device)

    optimizer = AdamW(model.parameters(), lr=1.5e-5, weight_decay=1e-5)
    epochs = 15
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)

    lambda_argo = 0.15  # In-situ ARGO loss weight
    n_floats = len(argo_lats)

    best_loss = float("inf")
    save_path = "checkpoints/best_ocean_model_v5_finetuned.pt"

    print(f"\n🚀 STARTING HYBRID IN-SITU FINE-TUNING ({epochs} Epochs on {device})...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)

        for b_x, b_y in train_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(b_x)

            # 1. Physics Grid Loss (Dense Reanalysis)
            loss_grid = criterion_physics(preds, b_y)

            # 2. In-Situ ARGO Observation Loss (Random batch of 1,024 real float points)
            float_sample_idxs = torch.randint(0, n_floats, (1024,), device=device)
            f_lats = argo_lats[float_sample_idxs]
            f_lons = argo_lons[float_sample_idxs]
            f_depths = argo_d_idxs[float_sample_idxs]
            f_temps = argo_temps_norm[float_sample_idxs]

            loss_insitu = criterion_insitu(preds, f_lats, f_lons, f_depths, f_temps)

            # Total Hybrid Loss
            loss_total = loss_grid + lambda_argo * loss_insitu
            loss_total.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()

            epoch_loss += loss_total.item()

        mean_epoch_loss = epoch_loss / max(1, n_batches)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs}] | Hybrid Loss: {mean_epoch_loss:.5f} | Grid L: {loss_grid.item():.5f} | ARGO L: {loss_insitu.item():.5f} | LR: {current_lr:.6f}")

        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"   ⭐ Saved best fine-tuned v5 checkpoint: {save_path} (Loss: {best_loss:.5f})")

    print(f"\n🎉 Fine-Tuning Complete! Checkpoint saved to {save_path}")


if __name__ == "__main__":
    finetune_v5_with_argo()


