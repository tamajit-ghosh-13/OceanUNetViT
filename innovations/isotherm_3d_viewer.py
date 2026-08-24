"""
================================================================================
OceanEmbed - 3D Isotherm Surface Reconstruction (isotherm_3d_viewer.py)
================================================================================
Implements Feature #11:
  Extracts and renders the true 3D spatial topography of the 20°C Isotherm Surface
  (D20 - the principal oceanographic indicator of the Thermocline).

Features:
  - 3D Mesh surface plot showing depth (Z in meters) where T = 20°C.
  - Visualizes Somali Upwelling bowl (D20 shoaling to 40m) vs. Bay of Bengal (D20 deepening to 140m).
  - Exports 3D rendering to: 'isotherm_20C_3d_surface.png'
================================================================================
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
from generate_tribreed_snapshots import TRI_WEIGHTS


def extract_d20_thermocline_depth(volume_3d: np.ndarray, depths: np.ndarray, target_temp: float = 20.0) -> np.ndarray:
    """
    Computes depth Z (meters) where temperature crosses target_temp (20°C)
    for every (lat, lon) pixel in the basin.
    """
    _, H, W = volume_3d.shape
    d20_map = np.full((H, W), np.nan)

    for i in range(H):
        for j in range(W):
            profile = volume_3d[:, i, j]
            if np.isnan(profile).any() or profile[0] < target_temp:
                continue
            # Find vertical crossing
            for d in range(len(depths) - 1):
                if profile[d] >= target_temp >= profile[d + 1]:
                    # Linear interpolation of exact depth
                    frac = (profile[d] - target_temp) / (profile[d] - profile[d + 1] + 1e-6)
                    d20_map[i, j] = depths[d] + frac * (depths[d + 1] - depths[d])
                    break
    return d20_map


def generate_3d_isotherm_surface():
    device = get_compute_device()
    print("\n" + "=" * 105)
    print("🏔️ OCEANEMBED - 3D ISOTHERM (20°C THERMOCLINE D20) SURFACE GENERATOR")
    print("=" * 105)

    # 1. Load active validation data & models
    val_inputs_12ch = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_inputs_7ch = val_inputs_12ch[:, :7]
    sample_12ch = val_inputs_12ch[15:16]
    sample_7ch = val_inputs_7ch[15:16]

    model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    model_ft.load_state_dict(torch.load("checkpoints/best_ocean_model_finetuned.pt", map_location=device), strict=False)
    model_ft.eval()

    model_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v3.load_state_dict(torch.load("checkpoints/best_ocean_model_v3_unbiased.pt", map_location=device), strict=False)
    model_v3.eval()

    model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v4.load_state_dict(torch.load("checkpoints/best_ocean_model_v4.pt", map_location=device), strict=False)
    model_v4.eval()

    p1 = run_model_inference(model_ft, sample_7ch, is_v3=False, device=device)[0]
    p2 = run_model_inference(model_v3, sample_12ch, is_v3=True, device=device)[0]
    p3 = run_model_inference(model_v4, sample_12ch, is_v3=True, device=device)[0]

    depths = np.array(STANDARD_DEPTH_LEVELS_M)
    volume_3d = np.zeros_like(p1)
    for d_idx, d_val in enumerate(depths):
        w = TRI_WEIGHTS[d_val]
        volume_3d[d_idx] = w[0] * p1[d_idx] + w[1] * p2[d_idx] + w[2] * p3[d_idx]

    # 2. Extract D20 Depth Field
    d20_depths = extract_d20_thermocline_depth(volume_3d, depths, target_temp=20.0)

    # 3. Create 3D Topography Mesh
    lats = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lons = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)
    LON_grid, LAT_grid = np.meshgrid(lons, lats)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Mask NaNs for surface plotting
    d20_plot = np.where(np.isnan(d20_depths), 0.0, d20_depths)

    surf = ax.plot_surface(
        LON_grid,
        LAT_grid,
        -d20_plot,
        cmap="coolwarm_r",
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )

    ax.set_title("OceanEmbed 3D Reconstructed 20°C Isotherm Thermocline Topography (D20)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Longitude (°E)", fontsize=11, labelpad=10)
    ax.set_ylabel("Latitude (°N)", fontsize=11, labelpad=10)
    ax.set_zlabel("Thermocline Depth (m)", fontsize=11, labelpad=10)

    ax.view_init(elev=32, azim=-55)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Depth of 20°C Isotherm (m)", fontsize=10)

    out_3d = "isotherm_20C_3d_surface.png"
    plt.savefig(out_3d, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Saved 3D Isotherm Surface rendering to: {out_3d}")
    print("✨ Feature #11 (3D Isotherm Surface Reconstruction) Complete!")


if __name__ == "__main__":
    generate_3d_isotherm_surface()
