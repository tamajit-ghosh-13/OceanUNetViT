"""
================================================================================
OceanEmbed - Evaluation Framework (evaluate.py)
================================================================================
PURPOSE:
  After training, we need to measure HOW ACCURATE the model is.
  The hackathon authority specifically requires these metrics:
    1. RMSE  — Root Mean Square Error (°C): measures average prediction error
    2. Bias  — Mean signed error (°C): reveals systematic over/under-prediction
    3. Correlation — Pearson r: measures how well spatial patterns match

  These metrics are computed PER DEPTH LEVEL (not just overall) because:
    - The model might be great at the surface (0–50m) but poor at depth (700–1000m)
    - Judges want to see performance across the full water column

VALIDATION DATASETS:
  - Primary: Held-out GLORYS reanalysis (same format as training target)
  - Optional: Independent ARGO profiling float observations from INCOIS

BEGINNER EXPLANATION OF RMSE:
  If your model predicts 28°C but the true temperature is 26°C, the error is 2°C.
  RMSE takes all these individual errors, squares them (to penalize big mistakes more),
  averages them, and takes the square root → gives you a typical error in °C.
================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from config import (
    STANDARD_DEPTH_LEVELS_M, N_DEPTH_LEVELS,
    N_INPUT_CHANNELS, GRID_LAT_SIZE, GRID_LON_SIZE,
    NORMALIZATION_STATS,
)
from model import create_model
from preprocessing.normalize import denormalize_outputs


# ==============================================================================
# 1. Core Metric Computations
# ==============================================================================
def compute_depth_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Computes RMSE (Root Mean Square Error) at each of the 15 depth levels.

    Parameters:
    -----------
    predictions : np.ndarray — shape (N, 15, H, W) in °C
    targets     : np.ndarray — shape (N, 15, H, W) in °C

    Returns:
    --------
    rmse_per_depth : np.ndarray — shape (15,) in °C
    """
    diff_sq = (predictions - targets) ** 2  # (N, 15, H, W)
    return np.sqrt(diff_sq.mean(axis=(0, 2, 3)))  # (15,)


def compute_depth_bias(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Computes BIAS (Mean Signed Error) at each depth level.

    Positive bias = model systematically predicts TOO WARM.
    Negative bias = model systematically predicts TOO COLD.

    Returns shape (15,) in °C.
    """
    return (predictions - targets).mean(axis=(0, 2, 3))  # (15,)


def compute_depth_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Computes Pearson spatial correlation coefficient at each depth level.

    A correlation of 1.0 = perfect spatial pattern match.
    A correlation of 0.0 = random / no skill.

    Returns shape (15,) correlation coefficients.
    """
    N = predictions.shape[0]
    correlations = np.zeros(N_DEPTH_LEVELS)

    for d in range(N_DEPTH_LEVELS):
        pred_d = predictions[:, d].flatten()  # (N × H × W,)
        targ_d = targets[:, d].flatten()

        # Remove mean (center the data)
        pred_centered = pred_d - pred_d.mean()
        targ_centered = targ_d - targ_d.mean()

        numerator = (pred_centered * targ_centered).sum()
        denominator = np.sqrt((pred_centered ** 2).sum() * (targ_centered ** 2).sum())

        correlations[d] = numerator / (denominator + 1e-10)

    return correlations


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper that computes RMSE, Bias, and Correlation together.

    Parameters:
    -----------
    predictions : np.ndarray — shape (N, 15, H, W) in physical units (°C)
    targets     : np.ndarray — shape (N, 15, H, W) in physical units (°C)

    Returns:
    --------
    dict with keys: 'rmse', 'bias', 'correlation', 'depths_m'
    """
    assert predictions.shape == targets.shape, (
        f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
    )
    return {
        "rmse":        compute_depth_rmse(predictions, targets),
        "bias":        compute_depth_bias(predictions, targets),
        "correlation": compute_depth_correlation(predictions, targets),
        "depths_m":    np.array(STANDARD_DEPTH_LEVELS_M),
    }


# ==============================================================================
# 2. Evaluate a Trained Model
# ==============================================================================
def evaluate_model(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    denormalize: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Runs the trained model on a DataLoader and returns aggregated metrics.

    Parameters:
    -----------
    model      : Trained OceanUNetViT
    dataloader : Validation or test DataLoader
    device     : torch.device (mps / cuda / cpu)
    denormalize: If True, converts normalized predictions back to °C

    Returns dict with 'rmse', 'bias', 'correlation', 'depths_m'.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            preds  = model(inputs).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    predictions = np.concatenate(all_preds, axis=0)    # (N, 15, H, W)
    targets_arr = np.concatenate(all_targets, axis=0)  # (N, 15, H, W)

    if denormalize:
        stats = NORMALIZATION_STATS["TEMP_TARGET"]
        predictions = denormalize_outputs(predictions, stats=stats)
        targets_arr = denormalize_outputs(targets_arr, stats=stats)

    return compute_all_metrics(predictions, targets_arr)


# ==============================================================================
# 3. Print Metrics Report to Console
# ==============================================================================
def print_metrics_report(metrics: Dict[str, np.ndarray]) -> None:
    """
    Prints a formatted evaluation report table to the console.
    """
    print("\n" + "=" * 72)
    print("  OceanEmbed Evaluation Report — North Indian Ocean 3D Temperature")
    print("=" * 72)
    print(f"  {'Depth (m)':>10} | {'RMSE (°C)':>10} | {'Bias (°C)':>10} | {'Correlation':>12}")
    print("-" * 72)

    overall_rmse = float(metrics["rmse"].mean())
    overall_bias = float(metrics["bias"].mean())
    overall_corr = float(metrics["correlation"].mean())

    for i, depth in enumerate(metrics["depths_m"]):
        rmse = metrics["rmse"][i]
        bias = metrics["bias"][i]
        corr = metrics["correlation"][i]

        # Visual quality indicator
        if corr >= 0.85:
            flag = "✅"
        elif corr >= 0.60:
            flag = "⚠️ "
        else:
            flag = "❌"

        print(f"  {int(depth):>10} | {rmse:>10.4f} | {bias:>10.4f} | {corr:>12.4f} {flag}")

    print("-" * 72)
    print(f"  {'MEAN':>10} | {overall_rmse:>10.4f} | {overall_bias:>10.4f} | {overall_corr:>12.4f}")
    print("=" * 72)
    print(f"\n  Overall RMSE: {overall_rmse:.4f} °C")
    print(f"  Overall Bias: {overall_bias:+.4f} °C ({'warm bias' if overall_bias > 0 else 'cold bias'})")
    print(f"  Mean Correlation: {overall_corr:.4f}\n")


# ==============================================================================
# 4. Visualization: Depth Profile Plot
# ==============================================================================
def plot_skill_profiles(
    metrics: Dict[str, np.ndarray],
    save_path: str = "evaluation_profiles.png",
) -> None:
    """
    Generates a 3-panel figure showing RMSE, Bias, and Correlation vs Depth.

    This is the standard oceanographic skill assessment visualization that
    reviewers and hackathon judges will recognize immediately.
    """
    if not HAS_MATPLOTLIB:
        print("⚠️  matplotlib not installed. Install with: pip install matplotlib")
        return

    depths = metrics["depths_m"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    fig.suptitle(
        "OceanEmbed — 3D Temperature Reconstruction Skill\nNorth Indian Ocean (5°N–30°N, 45°E–105°E)",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: RMSE ---
    axes[0].plot(metrics["rmse"], depths, "o-", color="#e74c3c", linewidth=2, markersize=6)
    axes[0].axvline(x=0, color="black", linewidth=0.5, linestyle="--")
    axes[0].set_xlabel("RMSE (°C)", fontsize=11)
    axes[0].set_ylabel("Depth (m)", fontsize=11)
    axes[0].set_title("RMSE", fontsize=12)
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)

    # --- Panel 2: Bias ---
    axes[1].plot(metrics["bias"], depths, "s-", color="#3498db", linewidth=2, markersize=6)
    axes[1].axvline(x=0, color="black", linewidth=1.0, linestyle="--", label="Zero Bias")
    axes[1].set_xlabel("Bias (°C)", fontsize=11)
    axes[1].set_title("Bias", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Correlation ---
    axes[2].plot(metrics["correlation"], depths, "^-", color="#2ecc71", linewidth=2, markersize=6)
    axes[2].axvline(x=0.85, color="orange", linewidth=1.0, linestyle="--", label="r=0.85 (Good)")
    axes[2].set_xlabel("Pearson Correlation (r)", fontsize=11)
    axes[2].set_title("Correlation", fontsize=12)
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(-1, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   📊 Skill profile saved to: {save_path}")


# ==============================================================================
# 5. Snapshot Visualization (Predicted vs True at One Time Step)
# ==============================================================================
def plot_prediction_snapshot(
    prediction: np.ndarray,
    target: np.ndarray,
    depth_idx: int = 4,
    save_path: str = "snapshot_comparison.png",
) -> None:
    """
    Side-by-side map comparison: model prediction vs GLORYS ground truth.

    Parameters:
    -----------
    prediction : np.ndarray (15, H, W) — one sample's predicted 3D field
    target     : np.ndarray (15, H, W) — corresponding ground truth
    depth_idx  : int — which of the 15 depth levels to visualize (default: 4 = 30m)
    """
    if not HAS_MATPLOTLIB:
        return

    depth_m = STANDARD_DEPTH_LEVELS_M[depth_idx]
    lat = np.linspace(5.0, 30.0, GRID_LAT_SIZE)
    lon = np.linspace(45.0, 105.0, GRID_LON_SIZE)

    pred_slice = prediction[depth_idx]   # (H, W)
    targ_slice = target[depth_idx]       # (H, W)
    diff_slice = pred_slice - targ_slice # (H, W)

    vmin = min(pred_slice.min(), targ_slice.min())
    vmax = max(pred_slice.max(), targ_slice.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Temperature at {depth_m}m — North Indian Ocean", fontsize=13)

    im0 = axes[0].contourf(lon, lat, targ_slice, levels=20, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("GLORYS (Ground Truth)", fontsize=11)
    plt.colorbar(im0, ax=axes[0], label="°C")

    im1 = axes[1].contourf(lon, lat, pred_slice, levels=20, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("OceanEmbed (Prediction)", fontsize=11)
    plt.colorbar(im1, ax=axes[1], label="°C")

    diff_lim = np.abs(diff_slice).max()
    im2 = axes[2].contourf(lon, lat, diff_slice, levels=20, cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim)
    axes[2].set_title("Difference (Pred − Truth)", fontsize=11)
    plt.colorbar(im2, ax=axes[2], label="°C")

    for ax in axes:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   📊 Snapshot comparison saved to: {save_path}")


# ==============================================================================
# 6. Load Model from Checkpoint
# ==============================================================================
def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Loads a saved model checkpoint for evaluation.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        in_channels=ckpt.get("n_input_channels", N_INPUT_CHANNELS),
        out_depth_levels=len(ckpt.get("depth_levels", STANDARD_DEPTH_LEVELS_M)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"✅ Loaded model from epoch {ckpt.get('epoch', '?')} — val_loss: {ckpt.get('val_loss', '?'):.4f}")
    return model


# ==============================================================================
# 7. Self-Test
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Running evaluate.py self-test (mock data)...")

    # Generate synthetic predictions and targets
    N = 20
    preds_norm = np.random.randn(N, N_DEPTH_LEVELS, GRID_LAT_SIZE, GRID_LON_SIZE).astype(np.float32)
    # Add signal: prediction ≈ target + noise
    noise = 0.2 * np.random.randn(N, N_DEPTH_LEVELS, GRID_LAT_SIZE, GRID_LON_SIZE).astype(np.float32)
    targ_norm = preds_norm + noise

    # Denormalize both to °C
    stats = NORMALIZATION_STATS["TEMP_TARGET"]
    preds_c = denormalize_outputs(preds_norm, stats=stats)
    targs_c = denormalize_outputs(targ_norm, stats=stats)

    metrics = compute_all_metrics(preds_c, targs_c)
    print_metrics_report(metrics)

    plot_skill_profiles(metrics, save_path="evaluation_profiles.png")
    plot_prediction_snapshot(preds_c[0], targs_c[0], depth_idx=4, save_path="snapshot_comparison.png")
    print("✅ evaluate.py verified successfully!")
