"""
================================================================================
OceanEmbed - Generate Visual Heatmap Comparisons for Breeded Ensemble
================================================================================
Generates high-resolution publication-quality 3D ocean heatmaps for the
Optimal Breeded Ensemble (blended with v4 physics-informed model):
  1. Surface Mixed Layer (5m)
  2. Core Thermocline (100m)
  3. Deep Abyssal Ocean (700m)

Comparing:
  - Column 1: GLORYS Ground Truth
  - Column 2: Optimal Breeded Ensemble (Baseline + v4)
  - Column 3: Absolute Error Heatmap (|Ensemble - Ground Truth|)

Saves snapshot charts to:
  - snapshot_ensemble_mixed_5m.png
  - snapshot_ensemble_thermocline_100m.png
  - snapshot_ensemble_deep_700m.png
  - evaluation_profiles_ensemble.png (RMSE / Bias / Correlation Profiles across all 15 depths)
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    TEMP_TARGET_STATS_PER_DEPTH,
    NORMALIZATION_STATS,
)
from model import create_model
from train import get_compute_device
from evaluate_august_december import run_model_inference
from evaluate import compute_all_metrics, plot_skill_profiles
from evaluate_argo import OPTIMAL_WEIGHTS


def plot_ensemble_depth_snapshot(
    preds_ens: np.ndarray,
    targets_c: np.ndarray,
    depth_idx: int = 7,  # Default: 100m (index 7)
    sample_idx: int = 15,
    save_path: str = "snapshot_ensemble_100m.png",
):
    """Generates a 3-panel comparison plot for a specific ocean depth."""
    depth_m = STANDARD_DEPTH_LEVELS_M[depth_idx]
    pred_field = preds_ens[sample_idx, depth_idx]
    targ_field = targets_c[sample_idx, depth_idx]

    # Land masking
    land_mask = np.isnan(targ_field) | (targ_field == 0.0) | (pred_field == 0.0)

    pred_masked = np.ma.masked_array(pred_field, mask=land_mask)
    targ_masked = np.ma.masked_array(targ_field, mask=land_mask)
    err_masked  = np.ma.masked_array(np.abs(pred_field - targ_field), mask=land_mask)

    vmin = np.nanpercentile(targ_masked.compressed(), 2) if len(targ_masked.compressed()) > 0 else 15.0
    vmax = np.nanpercentile(targ_masked.compressed(), 98) if len(targ_masked.compressed()) > 0 else 30.0

    extent = [BBOX["min_lon"], BBOX["max_lon"], BBOX["min_lat"], BBOX["max_lat"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), constrained_layout=True)

    # 1. Ground Truth
    im1 = axes[0].imshow(targ_masked, origin="lower", extent=extent, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"GLORYS Ground Truth ({depth_m}m Depth)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(im1, ax=axes[0], orientation="horizontal", pad=0.08, label="Temperature (°C)")

    # 2. Optimal Breeded Ensemble Prediction
    im2 = axes[1].imshow(pred_masked, origin="lower", extent=extent, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Breeded Ensemble Reconstruction ({depth_m}m Depth)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Longitude (°E)")
    fig.colorbar(im2, ax=axes[1], orientation="horizontal", pad=0.08, label="Temperature (°C)")

    # 3. Absolute Error Map
    im3 = axes[2].imshow(err_masked, origin="lower", extent=extent, cmap="Reds", vmin=0.0, vmax=1.5)
    axes[2].set_title(f"Absolute Error |Ensemble - Truth| (Mean: {err_masked.mean():.3f}°C)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Longitude (°E)")
    fig.colorbar(im3, ax=axes[2], orientation="horizontal", pad=0.08, label="Error (°C)")

    for ax in axes:
        ax.set_facecolor("#e0e0e0")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved ensemble snapshot heatmap: {save_path}")


def generate_ensemble_visuals():
    device = get_compute_device()
    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v4 = "checkpoints/best_ocean_model_v4.pt"

    print("\n" + "=" * 80)
    print("🧬 GENERATING OPTIMAL BREEDING ENSEMBLE (BASELINE + v4) VISUAL HEATMAPS")
    print("=" * 80)

    # 1. Load July 2026 Out-of-Year Validation Set
    val_inputs_12ch = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_inputs_7ch = val_inputs_12ch[:, :7]
    val_targets = np.load("data/val_jul26_subsurface_targets.npy")
    val_dates = np.load("data/val_jul26_dates.npy")

    print(f"📦 Loaded July 2026 Validation Set: {len(val_dates)} days | Shape: {val_inputs_12ch.shape}")

    # 2. Load Models
    model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    model_ft.eval()

    model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v4.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
    model_v4.eval()
    print("🧠 Loaded Models: Finetuned Baseline (7-ch) & OceanUNetViT v4 (12-ch)")

    # 3. Model Inferences
    preds_ft_c = run_model_inference(model_ft, val_inputs_7ch, is_v3=False, device=device)
    preds_v4_c = run_model_inference(model_v4, val_inputs_12ch, is_v3=True, device=device)

    # 4. Construct Optimal Breeded 3D Predictions
    preds_ens = np.zeros_like(preds_ft_c)
    depths = np.array(STANDARD_DEPTH_LEVELS_M)
    for d_idx, depth_val in enumerate(depths):
        w = OPTIMAL_WEIGHTS[depth_val]
        preds_ens[:, d_idx] = w * preds_ft_c[:, d_idx] + (1.0 - w) * preds_v4_c[:, d_idx]

    # 5. Compute Metrics Report
    metrics = compute_all_metrics(preds_ens, val_targets)
    print("\n" + "=" * 75)
    print("📈 OPTIMAL BREEDING ENSEMBLE GENERALIZATION REPORT (JULY 2026 MONSOON)")
    print("=" * 75)
    print(f"{'Depth (m)':>10} | {'RMSE (°C)':>12} | {'Bias (°C)':>12} | {'Correlation':>14}")
    print("-" * 75)
    for d_idx, depth_m in enumerate(STANDARD_DEPTH_LEVELS_M):
        print(f"{depth_m:>10d} | {metrics['rmse'][d_idx]:>12.4f} | {metrics['bias'][d_idx]:>12.4f} | {metrics['correlation'][d_idx]:>14.4f} ✅")
    print("-" * 75)
    print(f"{'OVERALL MEAN':>10} | {metrics['rmse'].mean():>12.4f} | {metrics['bias'].mean():>12.4f} | {metrics['correlation'].mean():>14.4f}")
    print("=" * 75 + "\n")

    plot_skill_profiles(metrics, save_path="evaluation_profiles_ensemble.png")

    # 6. Generate Multi-Depth Snapshots
    # 5m (Mixed Layer)
    plot_ensemble_depth_snapshot(preds_ens, val_targets, depth_idx=1, sample_idx=15, save_path="snapshot_ensemble_mixed_5m.png")
    # 100m (Core Thermocline)
    plot_ensemble_depth_snapshot(preds_ens, val_targets, depth_idx=7, sample_idx=15, save_path="snapshot_ensemble_thermocline_100m.png")
    # 700m (Abyssal Intermediate Water)
    plot_ensemble_depth_snapshot(preds_ens, val_targets, depth_idx=13, sample_idx=15, save_path="snapshot_ensemble_deep_700m.png")

    print("🎉 All Breeded Ensemble visual heatmap comparisons and skill profiles generated successfully!")


if __name__ == "__main__":
    generate_ensemble_visuals()
