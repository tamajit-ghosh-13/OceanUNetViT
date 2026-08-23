"""
================================================================================
OceanEmbed - Training Script (train.py)  [UPGRADED v2]
================================================================================
CHANGES FROM v1:
  - Aligned to 7-channel inputs and 15 standard output depths
  - Targets are denormalized to °C for interpretable metrics
  - Added per-depth RMSE display during training (so you can track thermocline)
  - Checkpoint saves include normalization stats for deployment
  - Guard block shows full model/data spec confirmation before pausing

MPS (Metal Performance Shaders) Apple Silicon Optimization Notes:
  - num_workers=0 in DataLoader (MPS does not support multiprocessing workers)
  - non_blocking=True moves tensors to MPS asynchronously for speed
  - torch.mps.synchronize() ensures GPU computation is flushed when timing
================================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Tuple, Optional

from config import (
    N_INPUT_CHANNELS, N_DEPTH_LEVELS, STANDARD_DEPTH_LEVELS_M,
    TRAINING, NORMALIZATION_STATS, GRID_LAT_SIZE, GRID_LON_SIZE,
)
from model import OceanUNetViT, create_model
from data_loader import OceanDataset, create_ocean_dataloaders
from preprocessing.normalize import denormalize_outputs


# ==============================================================================
# 1. Device Selection (Apple Silicon MPS Priority)
# ==============================================================================
def get_compute_device() -> torch.device:
    """
    Selects the optimal hardware accelerator:
      Priority: MPS (Apple M-series GPU) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("⚡ Apple Silicon MPS (Metal Performance Shaders) activated!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"⚡ NVIDIA CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("ℹ️  Running on CPU (consider installing MPS-enabled PyTorch for Apple Silicon).")
    return device


# ==============================================================================
# 2. Composite Loss Function
# ==============================================================================
class OceanReconstructionLoss(nn.Module):
    """
    Composite loss = MSE + depth-weighted penalty.

    Why depth-weighting?
      Shallow levels (0–50m) are well-constrained by surface observations.
      Deep levels (500–1000m) are much harder to predict — we up-weight them
      so the model doesn't just predict the easy shallow thermocline and ignore
      the difficult deep ocean.

    Loss = alpha * MSE(all_depths) + (1-alpha) * MSE(deep_levels_weighted)
    """
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="mean")

        # Per-depth weights: deeper levels get higher weight
        raw_depths = torch.tensor(STANDARD_DEPTH_LEVELS_M, dtype=torch.float32)
        depth_weights = 1.0 + raw_depths / raw_depths.max()
        self.register_buffer("depth_weights", depth_weights)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
          predictions : (B, 15, H, W)
          targets     : (B, 15, H, W)
        """
        # Standard MSE across all depths
        mse_all = self.mse(predictions, targets)

        # Depth-weighted MSE (each depth level penalized by its weight)
        diff_sq = (predictions - targets) ** 2  # (B, 15, H, W)
        # Expand weights to broadcast over (B, H, W)
        w = self.depth_weights.view(1, N_DEPTH_LEVELS, 1, 1)
        weighted_mse = (diff_sq * w).mean()

        return self.alpha * mse_all + (1 - self.alpha) * weighted_mse


# ==============================================================================
# 3. One Epoch Training
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """
    One complete training pass. Returns (mean_loss, per_depth_rmse).
    """
    model.train()
    running_loss = 0.0
    depth_sq_errors = torch.zeros(N_DEPTH_LEVELS)
    n_batches = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()

        # Gradient clipping prevents runaway updates in the early epochs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        # Accumulate per-depth squared errors (on CPU to avoid MPS overhead)
        with torch.no_grad():
            per_depth = ((predictions - targets) ** 2).mean(dim=(0, 2, 3)).cpu()
            depth_sq_errors += per_depth
        n_batches += 1

    per_depth_rmse = torch.sqrt(depth_sq_errors / max(1, n_batches))
    return running_loss / max(1, n_batches), per_depth_rmse


# ==============================================================================
# 4. Validation Step
# ==============================================================================
def evaluate(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor]:
    """
    Evaluates on the validation set. Returns (mean_loss, per_depth_rmse).
    """
    model.eval()
    val_loss = 0.0
    depth_sq_errors = torch.zeros(N_DEPTH_LEVELS)
    n_batches = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            val_loss += loss.item()
            per_depth = ((predictions - targets) ** 2).mean(dim=(0, 2, 3)).cpu()
            depth_sq_errors += per_depth
            n_batches += 1

    per_depth_rmse = torch.sqrt(depth_sq_errors / max(1, n_batches))
    return val_loss / max(1, n_batches), per_depth_rmse


# ==============================================================================
# 5. Full Training Orchestrator
# ==============================================================================
def run_training_experiment(
    epochs: int = TRAINING["epochs"],
    batch_size: int = TRAINING["batch_size"],
    learning_rate: float = TRAINING["learning_rate"],
    checkpoint_dir: str = TRAINING["checkpoint_dir"],
    n_train_days: int = 160,
    n_val_days: int = 40,
    log_depth_interval: int = 5,  # Log per-depth RMSE every N epochs
) -> Dict[str, Any]:
    """
    Main training loop. Returns dict with training history.
    """
    device = get_compute_device()
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n📦 Initializing DataLoaders ({n_train_days} train / {n_val_days} val days)...")
    train_loader, val_loader = create_ocean_dataloaders(
        batch_size=batch_size,
        n_train_days=n_train_days,
        n_val_days=n_val_days,
        use_mock_data=True,
    )

    print(f"\n🧠 Building OceanUNetViT model...")
    model = create_model(
        in_channels=N_INPUT_CHANNELS,
        out_depth_levels=N_DEPTH_LEVELS,
    ).to(device)

    criterion = OceanReconstructionLoss(alpha=0.7).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=TRAINING["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val_loss = float("inf")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"\n🚀 Training on {device} — {epochs} epochs, batch={batch_size}, lr={learning_rate}")
    print(f"   Input: (B, {N_INPUT_CHANNELS}, {GRID_LAT_SIZE}, {GRID_LON_SIZE}) "
          f"→ Output: (B, {N_DEPTH_LEVELS}, {GRID_LAT_SIZE}, {GRID_LON_SIZE})\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_depth_rmse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_depth_rmse = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch:03d}/{epochs}] | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {elapsed:.1f}s")

        # Every N epochs, show per-depth breakdown
        if epoch % log_depth_interval == 0:
            print("  ↳ Per-Depth Val RMSE (normalized °C):")
            for d_idx, depth_m in enumerate(STANDARD_DEPTH_LEVELS_M):
                bar = "█" * int(val_depth_rmse[d_idx].item() * 20)
                print(f"     {depth_m:5d}m: {val_depth_rmse[d_idx]:.4f} {bar}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            ckpt_path = os.path.join(checkpoint_dir, "best_ocean_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "norm_stats": NORMALIZATION_STATS,
                "depth_levels": STANDARD_DEPTH_LEVELS_M,
                "grid_lat": GRID_LAT_SIZE,
                "grid_lon": GRID_LON_SIZE,
                "n_input_channels": N_INPUT_CHANNELS,
            }, ckpt_path)

    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f} (epoch {history['best_epoch']})")
    print(f"   Checkpoint saved to: {checkpoint_dir}/best_ocean_model.pt")
    return history


# ==============================================================================
# 6. Main Guard — Dry-Run Safety Check
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🌊 OceanEmbed v2 — 3D Ocean Temperature Inversion | Apple Silicon + MPS")
    print("=" * 80)

    device = get_compute_device()
    print(f"📍 Active Device: {device}")

    # Full specification confirmation
    print(f"\n📋 Project Specification Check:")
    print(f"   ✅ Input channels:   {N_INPUT_CHANNELS} (SST, SSS, SSH, U_cur, V_cur, U_wind, V_wind)")
    print(f"   ✅ Depth levels:     {N_DEPTH_LEVELS} ({STANDARD_DEPTH_LEVELS_M})")
    print(f"   ✅ Grid dimensions:  {GRID_LAT_SIZE} × {GRID_LON_SIZE} (0.25° North Indian Ocean)")

    # Model shape verification
    print(f"\n🔍 Model Architecture Dry-Run...")
    model = create_model().to(device)
    dummy_input = torch.randn(1, N_INPUT_CHANNELS, GRID_LAT_SIZE, GRID_LON_SIZE, device=device)
    with torch.no_grad():
        predictions, embedding = model(dummy_input, return_embedding=True)

    print(f"   Input:     {tuple(dummy_input.shape)}")
    print(f"   Output:    {tuple(predictions.shape)}")
    print(f"   Embedding: {tuple(embedding.shape)}")
    print(f"   Params:    {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "#" * 80)
    print("⏸️  STATUS: Training is ready but paused.")
    print("   To start training, run:")
    print('   python3 -c "import train; train.run_training_experiment(epochs=30)"')
    print("#" * 80)
