# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.config import BBOX, STANDARD_DEPTH_LEVELS_M
from backend.model import create_model
from backend.train import get_compute_device
from evaluate_august_december import run_model_inference
from scripts.generate_tribreed_snapshots import TRI_WEIGHTS

def plot_tri_depth_snapshot_custom(preds_tri, targets_c, depth_idx, sample_idx=15, save_path=""):
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
    
    deg = u"\u00b0"

    im1 = axes[0].imshow(targ_masked, origin="lower", extent=extent, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"GLORYS Ground Truth ({depth_m}m Depth)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel(f"Longitude ({deg}E)")
    axes[0].set_ylabel(f"Latitude ({deg}N)")
    fig.colorbar(im1, ax=axes[0], orientation="horizontal", pad=0.08, label=f"Temperature ({deg}C)")

    im2 = axes[1].imshow(pred_masked, origin="lower", extent=extent, cmap="Spectral_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Tri-Breed Engine Reconstruction ({depth_m}m Depth)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel(f"Longitude ({deg}E)")
    fig.colorbar(im2, ax=axes[1], orientation="horizontal", pad=0.08, label=f"Temperature ({deg}C)")

    im3 = axes[2].imshow(err_masked, origin="lower", extent=extent, cmap="Reds", vmin=0.0, vmax=1.5)
    axes[2].set_title(f"Absolute Error |Model - Truth| (Mean: {err_masked.mean():.3f}{deg}C)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel(f"Longitude ({deg}E)")
    fig.colorbar(im3, ax=axes[2], orientation="horizontal", pad=0.08, label=f"Error ({deg}C)")

    for ax in axes:
        ax.set_facecolor("#e0e0e0")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")

def generate_all():
    device = get_compute_device()
    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    ckpt_v4 = "checkpoints/best_ocean_model_v4.pt"

    val_inputs_12ch = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_inputs_7ch = val_inputs_12ch[:, :7]
    val_targets = np.load("data/val_jul26_subsurface_targets.npy")

    model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    model_ft.eval()

    model_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
    model_v3.eval()

    model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v4.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
    model_v4.eval()
    
    print("Running inference...")
    preds_ft_c = run_model_inference(model_ft, val_inputs_7ch, is_v3=False, device=device)
    preds_v3_c = run_model_inference(model_v3, val_inputs_12ch, is_v3=True, device=device)
    preds_v4_c = run_model_inference(model_v4, val_inputs_12ch, is_v3=True, device=device)

    preds_tri = np.zeros_like(preds_ft_c)
    depths = np.array(STANDARD_DEPTH_LEVELS_M)
    for d_idx, depth_val in enumerate(depths):
        w = TRI_WEIGHTS[depth_val]
        preds_tri[:, d_idx] = w[0] * preds_ft_c[:, d_idx] + w[1] * preds_v3_c[:, d_idx] + w[2] * preds_v4_c[:, d_idx]

    print("Generating native plots with distinct colorbars...")
    for d_idx, depth_val in enumerate(depths):
        save_path = f"frontend/public/assets/snapshot_duo_elite_{depth_val}m.png"
        plot_tri_depth_snapshot_custom(preds_tri, val_targets, d_idx, 15, save_path)

if __name__ == "__main__":
    generate_all()
