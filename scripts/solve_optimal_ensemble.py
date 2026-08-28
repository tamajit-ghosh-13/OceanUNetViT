"""
================================================================================
OceanEmbed - Quad-Breed & Optimized Tri-Breed Ensemble Solver
================================================================================
Calculates exact depth-by-depth optimal error-covariance weights across the
available physical models on the validation split (July 2026 Monsoon):
  - Model 1: Baseline (7-channel)
  - Model 2: OceanUNetViT v3 Physical (12-channel)
  - Model 3: OceanUNetViT v4 Physics-Informed (12-channel + Stratification)
  - Model 4: OceanUNetViT v5 Thermocline-Preserving (12-channel + Gradient)

Finds the quadratic simplex optimal weight vector W*(z) = [w1, w2, w3, w4] >= 0, sum(w) = 1
that minimizes variance and RMSE at each individual depth level.
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from scipy.optimize import minimize
from config import STANDARD_DEPTH_LEVELS_M, N_DEPTH_LEVELS
from model import create_model
from train import get_compute_device
from evaluate_august_december import run_model_inference
from evaluate import compute_all_metrics


def solve_optimal_ensemble_weights():
    device = get_compute_device()

    val_inputs_12ch = np.load("data/val_jul26_surface_inputs_12ch.npy")
    val_inputs_7ch = val_inputs_12ch[:, :7]
    val_targets = np.load("data/val_jul26_subsurface_targets.npy")

    # Load 4 models
    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    ckpt_v4 = "checkpoints/best_ocean_model_v4.pt"
    ckpt_v5 = "checkpoints/best_ocean_model_v5.pt"

    print("🧠 Loading models for ensemble optimization...")
    m_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    m_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    m_ft.eval()

    m_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
    m_v3.eval()

    m_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v4.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
    m_v4.eval()

    m_v5 = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v5.load_state_dict(torch.load(ckpt_v5, map_location=device), strict=False)
    m_v5.eval()

    print("🔮 Running validation forward passes...")
    p_ft = run_model_inference(m_ft, val_inputs_7ch, is_v3=False, device=device)
    p_v3 = run_model_inference(m_v3, val_inputs_12ch, is_v3=True, device=device)
    p_v4 = run_model_inference(m_v4, val_inputs_12ch, is_v3=True, device=device)
    p_v5 = run_model_inference(m_v5, val_inputs_12ch, is_v3=True, device=device)

    # Solve optimal weights for each depth
    depths = STANDARD_DEPTH_LEVELS_M
    optimal_weights_dict = {}
    tri_v5_weights_dict = {}

    print("\n" + "=" * 105)
    print(f"{'Depth (m)':>10} | {'Baseline':>10} | {'v3 Phys':>10} | {'v4 Phys':>10} | {'v5 Phys':>10} | {'Ensemble RMSE':>16} | {'Single Best RMSE':>18}")
    print("-" * 105)

    preds_ensemble = np.zeros_like(p_ft)
    preds_tri_v5 = np.zeros_like(p_ft)

    for d_idx, d_val in enumerate(depths):
        y_true = val_targets[:, d_idx].flatten()
        valid = ~np.isnan(y_true) & (y_true > 0)
        y_true_v = y_true[valid]

        y_ft_v = p_ft[:, d_idx].flatten()[valid]
        y_v3_v = p_v3[:, d_idx].flatten()[valid]
        y_v4_v = p_v4[:, d_idx].flatten()[valid]
        y_v5_v = p_v5[:, d_idx].flatten()[valid]

        Y_matrix = np.column_stack([y_ft_v, y_v3_v, y_v4_v, y_v5_v])

        # Loss function: RMSE of weighted sum
        def loss_fn(w):
            pred = Y_matrix @ w
            return np.sqrt(np.mean((pred - y_true_v) ** 2))

        init_w = np.array([0.25, 0.25, 0.25, 0.25])
        bounds = [(0.0, 1.0)] * 4
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        res = minimize(loss_fn, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w_opt = res.x
        w_opt = np.round(w_opt, 3)
        w_opt = w_opt / np.sum(w_opt)  # normalize exactly
        optimal_weights_dict[d_val] = [float(round(x, 3)) for x in w_opt]

        best_single = min(
            np.sqrt(np.mean((y_ft_v - y_true_v) ** 2)),
            np.sqrt(np.mean((y_v3_v - y_true_v) ** 2)),
            np.sqrt(np.mean((y_v4_v - y_true_v) ** 2)),
            np.sqrt(np.mean((y_v5_v - y_true_v) ** 2)),
        )

        ens_rmse = loss_fn(w_opt)
        print(f"{d_val:>10d} | {w_opt[0]:>10.3f} | {w_opt[1]:>10.3f} | {w_opt[2]:>10.3f} | {w_opt[3]:>10.3f} | {ens_rmse:>14.4f}°C | {best_single:>16.4f}°C")

        # Reconstruct 3D ensemble
        preds_ensemble[:, d_idx] = (
            w_opt[0] * p_ft[:, d_idx] +
            w_opt[1] * p_v3[:, d_idx] +
            w_opt[2] * p_v4[:, d_idx] +
            w_opt[3] * p_v5[:, d_idx]
        )

    print("-" * 105)
    overall_metrics = compute_all_metrics(preds_ensemble, val_targets)
    print(f"🎉 New 4-Way Optimal Quad-Breed Validation Overall RMSE: {overall_metrics['rmse'].mean():.4f}°C | Correlation: {overall_metrics['correlation'].mean():.4f}")

    print("\n📦 Python Dictionary Format for Integration:")
    print("QUAD_BREED_WEIGHTS = {")
    for d_val, w_list in optimal_weights_dict.items():
        print(f"    {d_val:>4d}: {w_list},")
    print("}")

    return optimal_weights_dict


if __name__ == "__main__":
    solve_optimal_ensemble_weights()
