"""
================================================================================
OceanEmbed - v4 Physics-Informed Multi-Season Training Pipeline (train_v4.py)
================================================================================
Key Innovations in v4:
  1. Multi-Seasonal Training Catalog: Combines 2023 IOD, 2024 Spring, 2024 Fall,
     and 2025-2026 Monsoon/Winter (457 total days across all ocean regimes).
  2. Physics-Informed Stratification Loss: Penalizes unphysical vertical temperature
     inversions (forces dTemp/dz <= 0).
  3. Warm-Start Transfer Learning: Loads pre-trained v3 weights (12 channels)
     and fine-tunes across multi-year seasonal dynamics.
  4. Per-Depth Learnable Bias Vector: Maintains zero-bias at all 15 depths.
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
from evaluate import compute_all_metrics, print_metrics_report, plot_skill_profiles, plot_prediction_snapshot
from fetch_multi_season_dataset import build_full_multiseason_dataset


# ==============================================================================
# Physics-Informed Stratification & Reconstruction Loss
# ==============================================================================
class PhysicsInformedOceanLoss(nn.Module):
    """
    Composite Loss = Depth-Weighted MSE + Physics-Informed Stratification Regularizer.
    
    L_total = MSE_depth + lambda_strat * ReLU(T(z) - T(z-1))
    
    Penalizes unstable vertical temperature inversions where deeper water is hotter.
    """
    def __init__(self, alpha: float = 0.7, lambda_strat: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.lambda_strat = lambda_strat
        self.mse = nn.MSELoss(reduction="mean")

        raw_depths = torch.tensor(STANDARD_DEPTH_LEVELS_M, dtype=torch.float32)
        depth_weights = 1.0 + raw_depths / raw_depths.max()
        self.register_buffer("depth_weights", depth_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard MSE
        mse_all = self.mse(predictions, targets)

        # Depth-Weighted MSE
        diff_sq = (predictions - targets) ** 2
        w = self.depth_weights.view(1, N_DEPTH_LEVELS, 1, 1)
        weighted_mse = (diff_sq * w).mean()
        recon_loss = self.alpha * mse_all + (1 - self.alpha) * weighted_mse

        # Physics-Informed Stratification Regularizer:
        # T(z) should be <= T(z-1) -> penalty if T(z) > T(z-1)
        # Note: Since targets are normalized per depth or globally, we compute inversion penalty
        temp_diff = predictions[:, 1:, :, :] - predictions[:, :-1, :, :]
        strat_inversion = torch.relu(temp_diff).mean()

        return recon_loss + self.lambda_strat * strat_inversion


def train_v4_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


def evaluate_v4(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            preds = model(x)
            loss = criterion(preds, y)
            val_loss += loss.item()
    return val_loss / max(1, len(loader))


def run_v4_training(
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    save_checkpoint: str = "checkpoints/best_ocean_model_v4.pt",
    warm_start_ckpt: str = "checkpoints/best_ocean_model_v3_unbiased.pt",
):
    os.makedirs("checkpoints", exist_ok=True)
    device = get_compute_device()

    print("=" * 80)
    print("🌊 OCEANEMBED v4 PHYSICS-INFORMED MULTI-SEASON TRAINING")
    print("=" * 80)

    # 1. Load Existing 2025-2026 Dataset (273 days)
    f_existing_in = "data/train_jun25_feb26_surface_inputs_12ch.npy"
    f_existing_tg = "data/train_jun25_feb26_subsurface_targets.npy"
    f_existing_dt = "data/train_jun25_feb26_dates.npy"

    print("📦 Loading Baseline Training Dataset (2025-2026)...")
    in_25_26 = np.load(f_existing_in)
    tg_25_26 = np.load(f_existing_tg)
    dt_25_26 = np.load(f_existing_dt)

    # 2. Build or Load Multi-Season 2023-2024 Dataset (184 days)
    in_ms, tg_ms, dt_ms = build_full_multiseason_dataset()

    # 3. Combine into Grand Multi-Year Catalog (457 total days)
    all_train_inputs = np.concatenate([in_25_26, in_ms], axis=0).astype(np.float32)
    all_train_targets = np.concatenate([tg_25_26, tg_ms], axis=0).astype(np.float32)
    all_train_dates = np.concatenate([dt_25_26, dt_ms], axis=0)

    print("\n" + "=" * 80)
    print(f"📊 COMBINED GRAND TRAINING CATALOG READY:")
    print(f"   Total Multi-Year Days: {len(all_train_dates)} days")
    print(f"   Inputs: {all_train_inputs.shape} | Targets: {all_train_targets.shape}")
    print("=" * 80)

    # 4. Load Out-of-Year Validation Set (July 2026 Monsoon, 31 days)
    val_inputs = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_targets = np.load("data/val_jul26_subsurface_targets.npy")
    val_dates = np.load("data/val_jul26_dates.npy")

    train_ds = OceanDataset(surface_inputs=all_train_inputs, subsurface_targets=all_train_targets, dates=all_train_dates, use_mock_data=False)
    val_ds = OceanDataset(surface_inputs=val_inputs, subsurface_targets=val_targets, dates=val_dates, use_mock_data=False)

    # Stratified Sampling by Month
    month_keys = [d[:7] for d in all_train_dates]
    month_counts = Counter(month_keys)
    sample_weights = torch.tensor([1.0 / month_counts[m] for m in month_keys], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Build Model and Warm-Start from v3
    print("\n🧠 Initializing OceanUNetViT v4 (12 Input Channels + Physics Stratification)...")
    model = create_model(in_channels=12, out_depth_levels=15).to(device)

    if os.path.exists(warm_start_ckpt):
        print(f"   🔥 Warm-starting model weights from {warm_start_ckpt}...")
        model.load_state_dict(torch.load(warm_start_ckpt, map_location=device), strict=False)
        print("   ✅ Pre-trained v3 weights loaded successfully!")

    criterion = PhysicsInformedOceanLoss(alpha=0.7, lambda_strat=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    torch.save(model.state_dict(), save_checkpoint)

    print("\n🚀 STARTING v4 TRAINING PASS...")
    for epoch in range(1, epochs + 1):
        train_loss = train_v4_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_v4(model, val_loader, criterion, device)
        scheduler.step()

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val (July 2026) Loss: {val_loss:.4f} | LR: {lr_curr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_checkpoint)

    print(f"\n🎉 v4 Training Complete! Best Checkpoint Saved to: {save_checkpoint}")


if __name__ == "__main__":
    run_v4_training(
        epochs=15,
        batch_size=4,
        learning_rate=1e-4,
    )
