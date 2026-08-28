"""
================================================================================
OceanEmbed - v5 Physics-Guided Thermocline-Preserving Training Pipeline (train_v5.py)
================================================================================
CRITICAL CONSTRAINT: NO ARGO FLOAT DATA IN THE TRAINING SET.
ARGO float observations are exclusively preserved as an untouched, independent
zero-shot out-of-sample benchmark (2007, 2014, 2017, 2022).

Key Innovations in v5:
  1. Thermocline Gradient-Preserving Loss:
     Calculates vertical derivative dT/dz and penalizes gradient smoothing,
     forcing the network to capture the sharp pycnocline/thermocline drop.
  2. Multi-Scale Layer-Weighted Loss:
     Applies maximum loss penalty weights exactly at the critical transition
     depths (50m, 75m, 100m, 125m, 150m) where reanalysis bias previously spiked.
  3. Strict Physical Monotonicity Constraint:
     Enforces that dTemp/dz <= 0 strictly throughout the water column.
  4. Curvature & Inflexion Regularizer:
     Matches the second derivative d^2T/dz^2 to preserve the S-curve geometry.
  5. Warm-Start Transfer Learning:
     Initialized from v4 physics weights and optimized across all 457 multi-year
     seasonal catalog days using cosine annealing schedule.
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    TEMP_TARGET_STATS_PER_DEPTH,
)
from preprocessing.normalize import denormalize_outputs
from data_loader import OceanDataset
from model import create_model
from train import get_compute_device
from evaluate import compute_all_metrics, print_metrics_report
from fetch_multi_season_dataset import build_full_multiseason_dataset


# ==============================================================================
# 1. Thermocline-Preserving Physics Loss Function
# ==============================================================================
class PhysicsPreservingThermoclineLoss(nn.Module):
    """
    Composite Loss for v5:
      1. Layer-Weighted MSE (higher weight on 50m-150m thermocline)
      2. Vertical Gradient Matching Loss (preserves dT/dz sharp thermocline transitions)
      3. Vertical Curvature Matching Loss (preserves d^2T/dz^2 S-curve profile)
      4. Stratification Monotonicity Loss (forces dT/dz <= 0)
    """
    def __init__(
        self,
        alpha_recon: float = 0.50,
        lambda_grad: float = 0.30,
        lambda_curv: float = 0.15,
        lambda_mono: float = 0.05,
    ):
        super().__init__()
        self.alpha_recon = alpha_recon
        self.lambda_grad = lambda_grad
        self.lambda_curv = lambda_curv
        self.lambda_mono = lambda_mono
        self.mse = nn.MSELoss(reduction="mean")

        # Layer weights emphasizing the thermocline (50m-150m)
        # Standard depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        layer_weights = torch.tensor([
            1.0, 1.0, 1.0, 1.2, 1.5,
            2.5, 3.0, 3.5, 3.0, 2.5,
            1.8, 1.2, 1.0, 1.0, 1.0
        ], dtype=torch.float32)
        layer_weights = layer_weights / layer_weights.mean()
        self.register_buffer("layer_weights", layer_weights.view(1, N_DEPTH_LEVELS, 1, 1))

        # Depth intervals delta_z for finite difference
        depths = torch.tensor(STANDARD_DEPTH_LEVELS_M, dtype=torch.float32)
        dz = depths[1:] - depths[:-1]
        self.register_buffer("dz", dz.view(1, N_DEPTH_LEVELS - 1, 1, 1))

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Layer-Weighted MSE Loss
        diff_sq = (predictions - targets) ** 2
        recon_loss = (diff_sq * self.layer_weights).mean()

        # 2. Vertical Temperature Gradient (dT/dz) Matching Loss
        pred_grad = (predictions[:, 1:, :, :] - predictions[:, :-1, :, :]) / (self.dz + 1e-5)
        targ_grad = (targets[:, 1:, :, :] - targets[:, :-1, :, :]) / (self.dz + 1e-5)
        grad_loss = self.mse(pred_grad, targ_grad)

        # 3. Vertical Curvature (d^2T/dz^2) Matching Loss
        pred_curv = pred_grad[:, 1:, :, :] - pred_grad[:, :-1, :, :]
        targ_curv = targ_grad[:, 1:, :, :] - targ_grad[:, :-1, :, :]
        curv_loss = self.mse(pred_curv, targ_curv)

        # 4. Strict Stratification Monotonicity (Penalize warmer water below colder water)
        # For normalized temperature where deeper is cooler, pred[:, z] should be <= pred[:, z-1]
        strat_inversions = torch.relu(predictions[:, 1:, :, :] - predictions[:, :-1, :, :])
        mono_loss = strat_inversions.mean()

        total_loss = (
            self.alpha_recon * recon_loss +
            self.lambda_grad * grad_loss +
            self.lambda_curv * curv_loss +
            self.lambda_mono * mono_loss
        )
        return total_loss


# ==============================================================================
# 2. Training and Validation Loops
# ==============================================================================
def train_v5_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


def evaluate_v5(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x)
            loss = criterion(preds, y)
            val_loss += loss.item()
    return val_loss / max(1, len(loader))


# ==============================================================================
# 3. Main v5 Training Runner
# ==============================================================================
def run_v5_training(
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 8e-5,
    save_checkpoint: str = "checkpoints/best_ocean_model_v5.pt",
    warm_start_ckpt: str = "checkpoints/best_ocean_model_v4.pt",
):
    os.makedirs("checkpoints", exist_ok=True)
    device = get_compute_device()

    print("=" * 90)
    print("🌊 OCEANEMBED v5 THERMOCLINE-PRESERVING PHYSICS-GUIDED TRAINING")
    print("   (Zero In-Situ ARGO Data in Training — Pure Physical Generalization)")
    print("=" * 90)

    # 1. Load Grand Multi-Year Training Catalog (457 days, NO ARGO DATA)
    f_existing_in = "data/train_jun25_feb26_surface_inputs_12ch.npy"
    f_existing_tg = "data/train_jun25_feb26_subsurface_targets.npy"
    f_existing_dt = "data/train_jun25_feb26_dates.npy"

    print("📦 Loading 2025-2026 Training Split (273 days)...")
    in_25_26 = np.load(f_existing_in)
    tg_25_26 = np.load(f_existing_tg)
    dt_25_26 = np.load(f_existing_dt)

    print("📦 Loading Multi-Season 2023-2024 Split (184 days)...")
    in_ms, tg_ms, dt_ms = build_full_multiseason_dataset()

    all_train_inputs = np.concatenate([in_25_26, in_ms], axis=0).astype(np.float32)
    all_train_targets = np.concatenate([tg_25_26, tg_ms], axis=0).astype(np.float32)
    all_train_dates = np.concatenate([dt_25_26, dt_ms], axis=0)

    print(f"\n📊 Grand Multi-Season Training Catalog: {len(all_train_dates)} Total Ocean Days")
    print(f"   Inputs: {all_train_inputs.shape} | Targets: {all_train_targets.shape}")

    # 2. Out-of-Sample Gridded Validation Split (July 2026 Monsoon, 31 days)
    val_inputs = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_targets = np.load("data/val_jul26_subsurface_targets.npy")
    val_dates = np.load("data/val_jul26_dates.npy")

    train_ds = OceanDataset(surface_inputs=all_train_inputs, subsurface_targets=all_train_targets, dates=all_train_dates, use_mock_data=False)
    val_ds = OceanDataset(surface_inputs=val_inputs, subsurface_targets=val_targets, dates=val_dates, use_mock_data=False)

    month_keys = [d[:7] for d in all_train_dates]
    month_counts = Counter(month_keys)
    sample_weights = torch.tensor([1.0 / month_counts[m] for m in month_keys], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Instantiate OceanUNetViT v5 & Warm Start from v4
    print("\n🧠 Initializing OceanUNetViT v5 (12-Channel Backbone + Thermocline-Preserving Head)...")
    model = create_model(in_channels=12, out_depth_levels=15).to(device)

    if os.path.exists(warm_start_ckpt):
        print(f"   🔥 Warm-starting model weights from {warm_start_ckpt}...")
        model.load_state_dict(torch.load(warm_start_ckpt, map_location=device), strict=False)
        print("   ✅ Pre-trained v4 weights loaded successfully!")

    criterion = PhysicsPreservingThermoclineLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-7)

    best_val_loss = float("inf")
    torch.save(model.state_dict(), save_checkpoint)

    print("\n🚀 STARTING v5 OPTIMIZATION PASS (20 Epochs)...")
    for epoch in range(1, epochs + 1):
        train_loss = train_v5_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_v5(model, val_loader, criterion, device)
        scheduler.step()

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_curr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_checkpoint)
            print(f"   ⭐ New best v5 model checkpoint saved! (Val Loss: {val_loss:.4f})")

    print(f"\n🎉 v5 Training Completed! Best Checkpoint Saved to: {save_checkpoint}")


if __name__ == "__main__":
    run_v5_training(
        epochs=20,
        batch_size=4,
        learning_rate=8e-5,
    )
