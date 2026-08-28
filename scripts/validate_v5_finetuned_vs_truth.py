import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from model import create_model
from config import STANDARD_DEPTH_LEVELS_M, BBOX, GRID_LAT_SIZE, GRID_LON_SIZE, TEMP_TARGET_STATS_PER_DEPTH, NORMALIZATION_STATS
from preprocessing.normalize import denormalize_outputs, preprocess_inputs

def validate_era(tag, era_name):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load Model
    m_v5 = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v5.load_state_dict(torch.load("checkpoints/best_ocean_model_v5_finetuned.pt", map_location=device), strict=False)
    m_v5.eval()

    # Load Data
    csv_file = f"./data/argo_{tag}.csv"
    df = pd.read_csv(csv_file).dropna(subset=["latitude", "longitude", "pres", "temp", "date"])
    df = df[(df["pres"] >= 0) & (df["pres"] <= 1050) & (df["temp"] >= 2) & (df["temp"] <= 35)]
    inputs_12ch = np.load(f"./data/argo_{tag}_inputs_12ch.npy")
    dates = np.load(f"./data/argo_{tag}_dates.npy")
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df = df[df["date"].isin(date_to_idx)].reset_index(drop=True)

    # Inference
    T = inputs_12ch.shape[0]
    preds_norm = []
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
            p = m_v5(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy()
            preds_norm.append(p)
    preds_norm = np.concatenate(preds_norm, axis=0)
    preds_c = denormalize_outputs(preds_norm, stats=TEMP_TARGET_STATS_PER_DEPTH)

    # Continuous Spline Evaluation
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=float)
    lat_grid = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lon_grid = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)

    daily_interp = {}
    for t in range(len(dates)):
        daily_interp[t] = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_c[t], method="linear", bounds_error=False, fill_value=None)

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
    header_pred = "v5_finetuned (°C)"
    header_err = "Error (RMSE)"
    header_corr = "Corr (r)"

    print("=" * 115)
    print(f"🌊 OCEANUNETVIT v5_FINETUNED VS PHYSICAL ARGO GROUND TRUTH: {era_name.upper()}")
    print("=" * 115)
    print(f"{header_d:>10} | {header_obs:>12} | {header_truth:>18} | {header_pred:>20} | {header_err:>15} | {header_corr:>10}")
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
        print(f"{d:>10d} | {len(t):>12,d} | {str_t:>18} | {str_p:>20} | {str_r:>15} | {str_c:>10}")
    print("-" * 115)
    mean_rmse_str = f"{np.mean(all_rmse):.4f}°C"
    mean_corr_str = f"{np.mean(all_corr):.3f}"
    print(f"{chr(39)}OVERALL{chr(39):>3} | {len(df):>12,d} | {chr(45):>18} | {chr(45):>20} | {mean_rmse_str:>15} | {mean_corr_str:>10}")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    validate_era("nov16", "November 2016 (Historic Negative IOD Era)")
    validate_era("jun20", "June 2020 (Super Cyclone Amphan / SW Monsoon Era)")
