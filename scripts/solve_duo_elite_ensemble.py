"""
================================================================================
OceanEmbed - Optimal Simplex Weight Solver: Duo-Elite Ensemble (v4_ext + v5_ft)
================================================================================
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator

from config import (
    STANDARD_DEPTH_LEVELS_M,
    BBOX,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    TEMP_TARGET_STATS_PER_DEPTH,
    NORMALIZATION_STATS,
)
from preprocessing.normalize import denormalize_outputs, preprocess_inputs
from model import create_model
from train import get_compute_device

def solve_duo_elite_weights():
    device = get_compute_device()
    print("=" * 115)
    print("🧠 SOLVING OPTIMAL SIMPLEX WEIGHTS: DUO-ELITE ENSEMBLE (v4_extended + v5_finetuned)")
    print("=" * 115)

    # 1. Load Models
    m_v4_ext = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v4_ext.load_state_dict(torch.load("checkpoints/best_ocean_model_v4_extended.pt", map_location=device), strict=False)
    m_v4_ext.eval()

    m_v5_ft = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v5_ft.load_state_dict(torch.load("checkpoints/best_ocean_model_v5_finetuned.pt", map_location=device), strict=False)
    m_v5_ft.eval()

    # 2. Gather Out-of-Sample Validation Floats across Multiple Eras
    eras = [
        ("nov16", "2016-11-01", "2016-11-30"),
        ("jun20", "2020-06-01", "2020-06-30"),
        ("oct19", "2019-10-01", "2019-10-31"),
        ("may21", "2021-05-01", "2021-05-31"),
    ]

    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=float)
    lat_grid = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lon_grid = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)
    bins = [0, 2.5, 7.5, 15, 25, 40, 62.5, 87.5, 112.5, 137.5, 175, 250, 400, 600, 850, 1100]
    bin_labels = STANDARD_DEPTH_LEVELS_M

    depth_obs_v4 = {d: [] for d in bin_labels}
    depth_obs_v5 = {d: [] for d in bin_labels}
    depth_obs_trues = {d: [] for d in bin_labels}

    for tag, s_d, e_d in eras:
        csv_file = f"./data/argo_{tag}.csv"
        df = pd.read_csv(csv_file).dropna(subset=["latitude", "longitude", "pres", "temp", "date"])
        df = df[(df["pres"] >= 0) & (df["pres"] <= 1050) & (df["temp"] >= 2) & (df["temp"] <= 35)]
        inputs_12ch = np.load(f"./data/argo_{tag}_inputs_12ch.npy")
        dates = np.load(f"./data/argo_{tag}_dates.npy")
        date_to_idx = {d: i for i, d in enumerate(dates)}
        df = df[df["date"].isin(date_to_idx)].reset_index(drop=True)

        T = len(dates)
        p_v4_raw, p_v5_raw = [], []
        for b in range(0, T, 4):
            batch = inputs_12ch[b:b+4]
            proc = np.zeros_like(batch)
            for i in range(len(batch)):
                p_phys, mask, _ = preprocess_inputs(batch[i, :7], stats=NORMALIZATION_STATS, nan_fill_method="spatial_median")
                extra = batch[i, 7:].copy()
                extra = np.where(np.isnan(extra), 0.0, extra)
                for ch in range(extra.shape[0]):
                    extra[ch][~mask] = 0.0
                proc[i] = np.concatenate([p_phys, extra], axis=0)
            proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                p_v4_raw.append(m_v4_ext(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy())
                p_v5_raw.append(m_v5_ft(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy())

        p_v4_c = denormalize_outputs(np.concatenate(p_v4_raw, axis=0), stats=TEMP_TARGET_STATS_PER_DEPTH)
        p_v5_c = denormalize_outputs(np.concatenate(p_v5_raw, axis=0), stats=TEMP_TARGET_STATS_PER_DEPTH)

        daily_v4 = {t: RegularGridInterpolator((depths, lat_grid, lon_grid), p_v4_c[t], method="linear", bounds_error=False, fill_value=None) for t in range(T)}
        daily_v5 = {t: RegularGridInterpolator((depths, lat_grid, lon_grid), p_v5_c[t], method="linear", bounds_error=False, fill_value=None) for t in range(T)}

        lats, lons, pres, temps, dt_arr = df["latitude"].values, df["longitude"].values, df["pres"].values, df["temp"].values, df["date"].values

        for i in range(len(df)):
            t_idx = date_to_idx[dt_arr[i]]
            z_pres = pres[i]
            true_t = temps[i]
            coords_15 = np.column_stack([depths, np.full(15, lats[i]), np.full(15, lons[i])])
            pchip_4 = PchipInterpolator(depths, daily_v4[t_idx](coords_15))
            pchip_5 = PchipInterpolator(depths, daily_v5[t_idx](coords_15))
            val_4 = float(pchip_4(z_pres))
            val_5 = float(pchip_5(z_pres))

            for b_idx in range(len(bins) - 1):
                if bins[b_idx] <= z_pres < bins[b_idx + 1]:
                    target_d = bin_labels[b_idx]
                    depth_obs_v4[target_d].append(val_4)
                    depth_obs_v5[target_d].append(val_5)
                    depth_obs_trues[target_d].append(true_t)
                    break

    h_d, h_obs, h_w4, h_w5, h_r4, h_r5, h_rduo = "Depth (m)", "Obs Count", "w(v4_extended)", "w(v5_finetuned)", "v4 RMSE", "v5 RMSE", "Duo-Elite RMSE 🏆"
    print(f"\n{h_d:>10} | {h_obs:>12} | {h_w4:>16} | {h_w5:>16} | {h_r4:>12} | {h_r5:>12} | {h_rduo:>18}")
    print("-" * 115)

    optimal_weights = {}
    total_rmse_v4, total_rmse_v5, total_rmse_duo = [], [], []

    for d in bin_labels:
        v4_arr = np.array(depth_obs_v4[d])
        v5_arr = np.array(depth_obs_v5[d])
        t_arr = np.array(depth_obs_trues[d])

        rmse_4 = np.sqrt(np.mean((v4_arr - t_arr) ** 2))
        rmse_5 = np.sqrt(np.mean((v5_arr - t_arr) ** 2))

        def loss_func(w):
            w4, w5 = w[0], w[1]
            pred = w4 * v4_arr + w5 * v5_arr
            return np.mean((pred - t_arr) ** 2)

        res = minimize(
            loss_func,
            x0=[0.5, 0.5],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            constraints={"type": "eq", "fun": lambda w: w[0] + w[1] - 1.0},
            method="SLSQP",
        )

        w_opt = res.x
        optimal_weights[int(d)] = [round(float(w_opt[0]), 4), round(float(w_opt[1]), 4)]

        pred_duo = w_opt[0] * v4_arr + w_opt[1] * v5_arr
        rmse_duo = np.sqrt(np.mean((pred_duo - t_arr) ** 2))

        total_rmse_v4.append(rmse_4)
        total_rmse_v5.append(rmse_5)
        total_rmse_duo.append(rmse_duo)

        s_r4 = f"{rmse_4:.4f}°C"
        s_r5 = f"{rmse_5:.4f}°C"
        s_rduo = f"{rmse_duo:.4f}°C"
        print(f"{d:>10d} | {len(t_arr):>12,d} | {w_opt[0]:>16.4f} | {w_opt[1]:>16.4f} | {s_r4:>12} | {s_r5:>12} | {s_rduo:>16} 🏆")

    n_total = sum(len(depth_obs_trues[d]) for d in bin_labels)
    mean_r4 = f"{np.mean(total_rmse_v4):.4f}°C"
    mean_r5 = f"{np.mean(total_rmse_v5):.4f}°C"
    mean_rduo = f"{np.mean(total_rmse_duo):.4f}°C"

    print("-" * 115)
    print(f"{'OVERALL':>10} | {n_total:>12,d} | {'-':>16} | {'-':>16} | {mean_r4:>12} | {mean_r5:>12} | {mean_rduo:>16} 🏆")
    print("=" * 115)

    print("\n📋 PYTHON DICTIONARY OF OPTIMAL DUO-ELITE WEIGHTS (z -> [w_v4_ext, w_v5_ft]):")
    print("DUO_ELITE_WEIGHTS = {")
    for d, w in optimal_weights.items():
        print(f"    {d}: {w},")
    print("}")

if __name__ == "__main__":
    solve_duo_elite_weights()