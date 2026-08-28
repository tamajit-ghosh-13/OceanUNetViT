"""
================================================================================
OceanEmbed - Generate Visual Heatmap Comparisons for Tri-Breeded Model
================================================================================
Generates high-resolution publication-quality 3D ocean heatmaps for the
Tri-Breeded Model (Baseline + v3 + v4):
  1. Surface Mixed Layer (5m)
  2. Core Thermocline (100m)
  3. Deep Abyssal Ocean (700m)
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
)
from model import create_model
from train import get_compute_device
from evaluate_august_december import run_model_inference
from evaluate import compute_all_metrics, plot_skill_profiles

# Exact Simplex Quadratic Error-Covariance Optimal Weights across all 15 depths (Tri-Breed)
TRI_WEIGHTS = {
    0:    [0.323, 0.144, 0.534],
    5:    [0.223, 0.066, 0.711],
    10:   [0.204, 0.119, 0.677],
    20:   [0.196, 0.000, 0.804],
    30:   [0.210, 0.203, 0.587],
    50:   [0.147, 0.704, 0.150],
    75:   [0.354, 0.435, 0.211],
    100:  [0.136, 0.478, 0.387],
    125:  [0.128, 0.807, 0.065],
    150:  [0.226, 0.671, 0.103],
    200:  [0.188, 0.340, 0.472],
    300:  [0.820, 0.000, 0.180],
    500:  [0.339, 0.330, 0.330],
    700:  [0.228, 0.643, 0.129],
    1000: [0.140, 0.644, 0.216],
}

# 4-Way Optimal Quad-Breed Weights (Baseline + v3 + v4 + v5)
QUAD_BREED_WEIGHTS = {
    0:    [0.249, 0.042, 0.611, 0.098],
    5:    [0.172, 0.000, 0.321, 0.507],
    10:   [0.057, 0.045, 0.500, 0.399],
    20:   [0.106, 0.271, 0.000, 0.623],
    30:   [0.236, 0.323, 0.000, 0.441],
    50:   [0.277, 0.428, 0.000, 0.295],
    75:   [0.318, 0.267, 0.000, 0.415],
    100:  [0.410, 0.267, 0.150, 0.174],
    125:  [0.439, 0.191, 0.209, 0.161],
    150:  [0.448, 0.101, 0.130, 0.321],
    200:  [0.426, 0.151, 0.112, 0.311],
    300:  [0.125, 0.315, 0.000, 0.560],
    500:  [0.099, 0.509, 0.197, 0.195],
    700:  [0.112, 0.504, 0.274, 0.110],
    1000: [0.089, 0.750, 0.034, 0.127],
}


def plot_tri_depth_snapshot(
    preds_tri: np.ndarray,
    targets_c: np.ndarray,
    depth_idx: int = 7,
    sample_idx: int = 15,
    save_path: str = "snapshot_tribreed_100m.png",
):
    depth_m = STANDARD_DEPTH_LEVELS_M[depth_idx]
    pred_field = preds_tri[sample_idx, depth_idx]
    targ_field = targets_c[sample_idx, depth_idx]

    land_mask = np.isnan(targ_field) | (targ_field == 0.0) | (pred_field == 0.0)

    pred_masked = np.ma.masked_array(pred_field, mask=land_mask)
    targ_masked = np.ma.masked_array(targ_field, mask=land_mask)
    err_masked  = np.ma.masked_array(np.abs(pred_field - targ_field), mask=land_mask)

    vmin = np.nanpercentile(targ_masked.compressed(), 2) if len(targ_masked.compressed()) > 0 else 15.0
    vmax = np.nanpercentile(targ_masked.compressed(), 98) if len(targ_masked.compressed()) > 0 else 30.0

    extent = [BBOX["min_lon"], BBOX["max_lon"], BBOX["min_lat"], BBOX["max_lat"]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), constrained_layout=True)

    im1 = axes[0].imshow(targ_masked, origin="lower", extent=extent, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"GLORYS Ground Truth ({depth_m}m Depth)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    fig.colorbar(im1, ax=axes[0], orientation="horizontal", pad=0.08, label="Temperature (°C)")

    im2 = axes[1].imshow(pred_masked, origin="lower", extent=extent, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Tri-Breeded 3-Way Reconstruction ({depth_m}m Depth)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Longitude (°E)")
    fig.colorbar(im2, ax=axes[1], orientation="horizontal", pad=0.08, label="Temperature (°C)")

    im3 = axes[2].imshow(err_masked, origin="lower", extent=extent, cmap="Reds", vmin=0.0, vmax=1.5)
    axes[2].set_title(f"Absolute Error |Tri-Breed - Truth| (Mean: {err_masked.mean():.3f}°C)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Longitude (°E)")
    fig.colorbar(im3, ax=axes[2], orientation="horizontal", pad=0.08, label="Error (°C)")

    for ax in axes:
        ax.set_facecolor("#e0e0e0")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved snapshot heatmap: {save_path}")


def generate_tri_visuals():
    device = get_compute_device()
    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    ckpt_v4 = "checkpoints/best_ocean_model_v4.pt"

    val_inputs_12ch = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_inputs_7ch = val_inputs_12ch[:, :7]
    val_targets = np.load("data/val_jul26_subsurface_targets.npy")
    val_dates = np.load("data/val_jul26_dates.npy")

    model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    model_ft.eval()

    model_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
    model_v3.eval()

    model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v4.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
    model_v4.eval()

    preds_ft_c = run_model_inference(model_ft, val_inputs_7ch, is_v3=False, device=device)
    preds_v3_c = run_model_inference(model_v3, val_inputs_12ch, is_v3=True, device=device)
    preds_v4_c = run_model_inference(model_v4, val_inputs_12ch, is_v3=True, device=device)

    preds_tri = np.zeros_like(preds_ft_c)
    depths = np.array(STANDARD_DEPTH_LEVELS_M)
    for d_idx, depth_val in enumerate(depths):
        w = TRI_WEIGHTS[depth_val]
        preds_tri[:, d_idx] = w[0] * preds_ft_c[:, d_idx] + w[1] * preds_v3_c[:, d_idx] + w[2] * preds_v4_c[:, d_idx]

    metrics = compute_all_metrics(preds_tri, val_targets)
    plot_skill_profiles(metrics, save_path="evaluation_profiles_tribreed.png")

    plot_tri_depth_snapshot(preds_tri, val_targets, depth_idx=1, sample_idx=15, save_path="snapshot_tribreed_mixed_5m.png")
    plot_tri_depth_snapshot(preds_tri, val_targets, depth_idx=7, sample_idx=15, save_path="snapshot_tribreed_thermocline_100m.png")
    plot_tri_depth_snapshot(preds_tri, val_targets, depth_idx=13, sample_idx=15, save_path="snapshot_tribreed_deep_700m.png")

    print(f"🎉 Tri-Breeded Overall Mean Validation RMSE: {metrics['rmse'].mean():.4f}°C | Correlation: {metrics['correlation'].mean():.4f}")


if __name__ == "__main__":
    generate_tri_visuals()
