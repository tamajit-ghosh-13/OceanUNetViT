"""
================================================================================
OceanEmbed - Tri-Breed Extended Model (Baseline + v3 + v4_extended) Benchmark
================================================================================
Evaluates the Tri-Breed ensemble using best_ocean_model_v4_extended.pt directly
against physical in-situ ARGO Float CTD Ground Truth across 4 historical eras:
  1. Nov 2016 (Historic Negative IOD Era)
  2. Jun 2020 (Super Cyclone Amphan / SW Monsoon Era)
  3. Oct 2019 (Historic Super Positive IOD Era)
  4. May 2021 (Pre-Monsoon Extreme Cyclone Yaas/Tauktae Era)
================================================================================
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from model import create_model
from config import STANDARD_DEPTH_LEVELS_M, BBOX, GRID_LAT_SIZE, GRID_LON_SIZE, TEMP_TARGET_STATS_PER_DEPTH, NORMALIZATION_STATS
from preprocessing.normalize import denormalize_outputs, preprocess_inputs
from scripts.generate_tribreed_snapshots import TRI_WEIGHTS

def validate_tribreed_extended_era(tag, era_name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load 3 Constituent Models (using v4_extended instead of original v4)
    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    ckpt_v4_ext = "checkpoints/best_ocean_model_v4_extended.pt"

    m_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    m_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    m_ft.eval()

    m_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
    m_v3.eval()

    m_v4_ext = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v4_ext.load_state_dict(torch.load(ckpt_v4_ext, map_location=device), strict=False)
    m_v4_ext.eval()

    # Load Data
    csv_file = f"./data/argo_{tag}.csv"
    if not os.path.exists(csv_file):
        print(f"⚠️ {csv_file} not found, skipping {era_name}...")
        return
    df = pd.read_csv(csv_file).dropna(subset=["latitude", "longitude", "pres", "temp", "date"])
    df = df[(df["pres"] >= 0) & (df["pres"] <= 1050) & (df["temp"] >= 2) & (df["temp"] <= 35)]
    inputs_7ch = np.load(f"./data/argo_{tag}_inputs_7ch.npy")
    inputs_12ch = np.load(f"./data/argo_{tag}_inputs_12ch.npy")
    dates = np.load(f"./data/argo_{tag}_dates.npy")
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df = df[df["date"].isin(date_to_idx)].reset_index(drop=True)

    T = len(dates)

    # 1. Inference Baseline (7-ch)
    preds_ft_raw = []
    for b in range(0, T, 4):
        batch = inputs_7ch[b:b+4]
        proc = np.zeros_like(batch)
        for i in range(len(batch)):
            p_phys, _, _ = preprocess_inputs(batch[i], stats=NORMALIZATION_STATS, nan_fill_method="spatial_median")
            proc[i] = p_phys
        proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            p = m_ft(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy()
            preds_ft_raw.append(p)
    preds_ft_raw = np.concatenate(preds_ft_raw, axis=0)
    preds_ft_c = denormalize_outputs(preds_ft_raw, stats=NORMALIZATION_STATS["TEMP_TARGET"])

    # 2. Inference v3 & v4_extended (12-ch)
    preds_v3_raw, preds_v4_raw = [], []
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
            p3 = m_v3(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy()
            p4 = m_v4_ext(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy()
            preds_v3_raw.append(p3)
            preds_v4_raw.append(p4)
    preds_v3_raw = np.concatenate(preds_v3_raw, axis=0)
    preds_v4_raw = np.concatenate(preds_v4_raw, axis=0)
    preds_v3_c = denormalize_outputs(preds_v3_raw, stats=TEMP_TARGET_STATS_PER_DEPTH)
    preds_v4_c = denormalize_outputs(preds_v4_raw, stats=TEMP_TARGET_STATS_PER_DEPTH)

    # 3. Tri-Breed Extended Depth-Wise Optimal Convex Combination
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=float)
    preds_tri_ext_c = np.zeros_like(preds_ft_c)
    for d_idx, d_val in enumerate(depths):
        w = TRI_WEIGHTS[int(d_val)]
        preds_tri_ext_c[:, d_idx] = w[0] * preds_ft_c[:, d_idx] + w[1] * preds_v3_c[:, d_idx] + w[2] * preds_v4_c[:, d_idx]

    # Continuous Spline Evaluation
    lat_grid = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lon_grid = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)

    daily_interp = {}
    for t in range(len(dates)):
        daily_interp[t] = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_tri_ext_c[t], method="linear", bounds_error=False, fill_value=None)

    bins = [0, 2.5, 7.5, 15, 25, 40, 62.5, 87.5, 112.5, 137.5, 175, 250, 400, 600, 850, 1100]
    bin_labels = STANDARD_DEPTH_LEVELS_M
    trues_d = {d: [] for d in bin_labels}
    preds_d = {d: [] for d in bin_labels}

    lats, lons, pres, temps, dt_arr = df["latitude"].values, df["longitude"].values, df["pres"].values, df["temp"].values, df["date"].values

    for i in range(len(df)):
        t_idx = date_to_idx[dt_arr[i]]
        z_pres = pres[i]
        true_t = temps[i]
        coords_15 = np.column_stack([depths, np.full(15, lats[i]), np.full(15, lons[i])])
        pchip = PchipInterpolator(depths, daily_interp[t_idx](coords_15))
        pred_val = float(pchip(z_pres))
        for b_idx in range(len(bins)-1):
            if bins[b_idx] <= z_pres < bins[b_idx+1]:
                trues_d[bin_labels[b_idx]].append(true_t)
                preds_d[bin_labels[b_idx]].append(pred_val)
                break

    header_d = "Depth (m)"
    header_obs = "Obs Count"
    header_truth = "ARGO Truth (°C)"
    header_pred = "Tri-Breed Ext 🧬 (°C)"
    header_err = "Error (RMSE)"
    header_corr = "Corr (r)"

    print("=" * 115)
    print(f"🧬 TRI-BREED EXTENDED (FT + v3 + v4_ext) VS ARGO GROUND TRUTH: {era_name.upper()}")
    print("=" * 115)
    print(f"{header_d:>10} | {header_obs:>12} | {header_truth:>18} | {header_pred:>24} | {header_err:>15} | {header_corr:>10}")
    print("-" * 115)
    all_rmse, all_corr = [], []
    for d in bin_labels:
        t = np.array(trues_d[d])
        p = np.array(preds_d[d])
        if len(t) < 5: continue
        rmse = np.sqrt(np.mean((p - t)**2))
        corr = np.corrcoef(p, t)[0, 1] if len(np.unique(p)) > 1 and len(np.unique(t)) > 1 else 0.0
        all_rmse.append(rmse)
        all_corr.append(corr)
        str_t = f"{t.mean():.2f}°C"
        str_p = f"{p.mean():.2f}°C"
        str_r = f"{rmse:.4f}°C"
        str_c = f"{corr:.3f}"
        print(f"{d:>10d} | {len(t):>12,d} | {str_t:>18} | {str_p:>24} | {str_r:>15} | {str_c:>10}")
    print("-" * 115)
    mean_rmse_str = f"{np.mean(all_rmse):.4f}°C"
    mean_corr_str = f"{np.mean(all_corr):.3f}"
    print(f"{chr(39)}OVERALL{chr(39):>3} | {len(df):>12,d} | {chr(45):>18} | {chr(45):>24} | {mean_rmse_str:>15} | {mean_corr_str:>10}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    eras = [
        ("nov16", "November 2016 (Historic Negative IOD Era)"),
        ("jun20", "June 2020 (Super Cyclone Amphan / SW Monsoon Era)"),
        ("oct19", "October 2019 (Historic Super Positive IOD Era)"),
        ("may21", "May 2021 (Pre-Monsoon Extreme Cyclone Yaas/Tauktae)"),
    ]
    for tag, name in eras:
        validate_tribreed_extended_era(tag, name)
