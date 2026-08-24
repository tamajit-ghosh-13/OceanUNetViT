"""
================================================================================
OceanEmbed - Multi-Period Real Argo In-Situ Validator (July 2022 & Dec 2022)
================================================================================
Benchmarks 4 models across 240,064 real physical in-situ Argo observations:
  1. Finetuned Baseline (7 channels)
  2. OceanUNetViT v3 Physical Model (12 channels)
  3. OceanUNetViT v4 Physics-Informed Model (12 channels + Stratification)
  4. Optimal Breeded Ensemble
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import sys
import json
import urllib.request
import urllib.parse
import numpy as np
import xarray as xr
import torch
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta
import copernicusmarine
import gsw

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    NORMALIZATION_STATS,
    TEMP_TARGET_STATS_PER_DEPTH,
)
from preprocessing.regrid import regrid_to_standard_grid, build_standard_grid
from preprocessing.normalize import denormalize_outputs, preprocess_inputs
from model import create_model
from train import get_compute_device
from evaluate_august_december import download_fast_nc_subset, run_model_inference


def download_argo_ifremer(
    start_date: str,
    end_date: str,
    output_csv: str,
) -> pd.DataFrame:
    """Downloads in-situ Argo observations from Ifremer Global GDAC ERDDAP."""
    if os.path.exists(output_csv):
        print(f"📦 Loaded cached Argo observations from {output_csv}")
        return pd.read_csv(output_csv)

    print(f"\n📥 Downloading In-Situ Argo Floats from Ifremer GDAC: {start_date} to {end_date}...")
    base = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json"
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    chunk_days = 10
    total_days = (dt_end - dt_start).days + 1
    n_chunks = int(np.ceil(total_days / chunk_days))

    all_dfs = []
    for c in range(n_chunks):
        c_start = dt_start + timedelta(days=c * chunk_days)
        c_days = min(chunk_days, (dt_end - c_start).days + 1)
        c_end = c_start + timedelta(days=c_days - 1)

        c_start_str = c_start.strftime("%Y-%m-%d") + "T00:00:00Z"
        c_end_str = c_end.strftime("%Y-%m-%d") + "T23:59:59Z"

        query = (
            f"time,latitude,longitude,pres,temp&"
            f"latitude>={BBOX['min_lat']}&latitude<={BBOX['max_lat']}&"
            f"longitude>={BBOX['min_lon']}&longitude<={BBOX['max_lon']}&"
            f"pres>=0.0&pres<=1050.0&"
            f"time>={c_start_str}&time<={c_end_str}"
        )
        url = base + "?" + urllib.parse.quote(query, safe="&,=")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                cols = data["table"]["columnNames"]
                rows = data["table"]["rows"]
                df_c = pd.DataFrame(rows, columns=cols)
                all_dfs.append(df_c)
        except Exception as e:
            print(f"   ⚠️ Warning on chunk {c+1}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    df_full = pd.concat(all_dfs, ignore_index=True)
    df_full = df_full.dropna(subset=["time", "latitude", "longitude", "pres", "temp"])
    df_full["temp"] = pd.to_numeric(df_full["temp"], errors="coerce")
    df_full["pres"] = pd.to_numeric(df_full["pres"], errors="coerce")
    df_full = df_full.dropna(subset=["pres", "temp"])
    df_full = df_full[(df_full["temp"] >= 2.0) & (df_full["temp"] <= 35.0)]
    df_full = df_full[(df_full["pres"] >= 0.0) & (df_full["pres"] <= 1000.0)]
    df_full["date"] = df_full["time"].apply(lambda t: str(t)[:10])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_full.to_csv(output_csv, index=False)
    return df_full


def get_or_download_surface_inputs(
    period_tag: str,
    start_date: str,
    end_date: str,
    save_dir: str = "./data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downloads surface input variables for the target period."""
    os.makedirs(save_dir, exist_ok=True)
    f_7ch = os.path.join(save_dir, f"argo_{period_tag}_inputs_7ch.npy")
    f_12ch = os.path.join(save_dir, f"argo_{period_tag}_inputs_12ch.npy")
    f_dates = os.path.join(save_dir, f"argo_{period_tag}_dates.npy")

    if os.path.exists(f_7ch) and os.path.exists(f_12ch) and os.path.exists(f_dates):
        return np.load(f_7ch), np.load(f_12ch), np.load(f_dates)

    download_dir = os.path.join(save_dir, "nc_downloads")
    target_grid = build_standard_grid()

    temp_nc = download_fast_nc_subset("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ["thetao"], start_date, end_date, f"{period_tag}_sst.nc", depth_range=(0.0, 1.0), download_dir=download_dir)
    so_nc   = download_fast_nc_subset("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ["so"], start_date, end_date, f"{period_tag}_so.nc", depth_range=(0.0, 5.0), download_dir=download_dir)
    zos_nc  = download_fast_nc_subset("cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ["zos"], start_date, end_date, f"{period_tag}_zos.nc", depth_range=(0.0, 1.0), download_dir=download_dir)
    cur_nc  = download_fast_nc_subset("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ["uo", "vo"], start_date, end_date, f"{period_tag}_cur.nc", depth_range=(0.0, 5.0), download_dir=download_dir)

    with xr.open_dataset(temp_nc) as ds_temp:
        regridded_temp = regrid_to_standard_grid(ds_temp["thetao"].isel(depth=0), method="bilinear")
        sst_array = regridded_temp.values.astype(np.float32)
        dates = np.array([str(t)[:10] for t in regridded_temp.time.values])
        T = len(dates)

    with xr.open_dataset(so_nc) as ds_so:
        sss_array = regrid_to_standard_grid(ds_so["so"].isel(depth=0), method="bilinear").values.astype(np.float32)

    with xr.open_dataset(zos_nc) as ds_zos:
        ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values.astype(np.float32)

    with xr.open_dataset(cur_nc) as ds_cur:
        u_cur = regrid_to_standard_grid(ds_cur["uo"].isel(depth=0), method="bilinear").values.astype(np.float32)
        v_cur = regrid_to_standard_grid(ds_cur["vo"].isel(depth=0), method="bilinear").values.astype(np.float32)

    omega = 7.2921e-5
    lat_grid = target_grid["lat"]
    lon_grid = target_grid["lon"]
    lat_rad = np.deg2rad(lat_grid[:, None])
    f = 2 * omega * np.sin(lat_rad)
    f = np.where(np.abs(f) < 1e-5, 1e-5 * np.sign(f), f)
    g = 9.81

    u_wind = np.zeros_like(ssh_array)
    v_wind = np.zeros_like(ssh_array)
    for t in range(T):
        grad_y, grad_x = np.gradient(ssh_array[t], 0.25 * 111000, 0.25 * 111000)
        u_wind[t] = - (g / f) * grad_y * 10.0
        v_wind[t] =   (g / f) * grad_x * 10.0

    inputs_7ch = np.stack([sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind], axis=1).astype(np.float32)

    wind_mag = np.sqrt(u_wind ** 2 + v_wind ** 2)
    doy = np.array([datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates])
    doy_sin = np.sin(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T, GRID_LAT_SIZE, GRID_LON_SIZE), dtype=np.float32)
    doy_cos = np.cos(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T, GRID_LAT_SIZE, GRID_LON_SIZE), dtype=np.float32)

    sst_temporal_mean = np.nanmean(sst_array, axis=0, keepdims=True)
    sst_anomaly = sst_array - sst_temporal_mean

    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    density_sigma0 = np.zeros_like(sst_array)
    for t in range(T):
        sa_t = gsw.SA_from_SP(sss_array[t], 0.0, lon_mesh, lat_mesh)
        ct_t = gsw.CT_from_pt(sa_t, sst_array[t])
        density_sigma0[t] = gsw.sigma0(sa_t, ct_t)

    inputs_12ch = np.stack([
        sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind,
        wind_mag, doy_sin, doy_cos, sst_anomaly, density_sigma0
    ], axis=1).astype(np.float32)

    np.save(f_7ch, inputs_7ch)
    np.save(f_12ch, inputs_12ch)
    np.save(f_dates, dates)

    return inputs_7ch, inputs_12ch, dates


OPTIMAL_WEIGHTS = {
    0:    0.5575,
    5:    0.2678,
    10:   0.2330,
    20:   0.2926,
    30:   0.3745,
    50:   0.4463,
    75:   0.2701,
    100:  0.1408,
    125:  0.0861,
    150:  0.5265,
    200:  0.3977,
    300:  1.0000,
    500:  0.5680,
    700:  0.2339,
    1000: 0.2479,
}


def evaluate_period_on_argo_4way(
    period_tag: str,
    period_name: str,
    start_date: str,
    end_date: str,
    model_ft: torch.nn.Module,
    model_v3: torch.nn.Module,
    model_v4: torch.nn.Module,
    device: torch.device,
):
    """Evaluates the 4 models on a single historical Argo period."""
    csv_file = f"./data/argo_{period_tag}.csv"
    df_argo = download_argo_ifremer(start_date=start_date, end_date=end_date, output_csv=csv_file)

    if df_argo.empty:
        print(f"⚠️ No Argo float data available for {period_name}.")
        return

    inputs_7ch, inputs_12ch, dates = get_or_download_surface_inputs(
        period_tag=period_tag,
        start_date=start_date,
        end_date=end_date,
    )
    date_to_idx = {d: i for i, d in enumerate(dates)}

    df_argo = df_argo[df_argo["date"].isin(date_to_idx)]
    print(f"   ✅ {len(df_argo):,} in-situ measurements align with surface satellite dates.")

    print(f"\n🔮 Running 4-way model inference over {period_name}...")
    preds_ft = run_model_inference(model_ft, inputs_7ch, is_v3=False, device=device)
    preds_v3 = run_model_inference(model_v3, inputs_12ch, is_v3=True, device=device)
    preds_v4 = run_model_inference(model_v4, inputs_12ch, is_v3=True, device=device)

    depths = np.array(STANDARD_DEPTH_LEVELS_M)
    bins = [0, 2.5, 7.5, 15, 25, 40, 62.5, 87.5, 112.5, 137.5, 175, 250, 400, 600, 850, 1100]
    bin_labels = STANDARD_DEPTH_LEVELS_M

    argo_true_temps = {d: [] for d in bin_labels}
    argo_pred_temps_ft = {d: [] for d in bin_labels}
    argo_pred_temps_v3 = {d: [] for d in bin_labels}
    argo_pred_temps_v4 = {d: [] for d in bin_labels}
    argo_pred_temps_opt = {d: [] for d in bin_labels}

    lats = df_argo["latitude"].values
    lons = df_argo["longitude"].values
    pres = df_argo["pres"].values
    temps = df_argo["temp"].values
    dates_arr = df_argo["date"].values

    lat_indices = np.round((lats - BBOX["min_lat"]) / 0.25).astype(int)
    lon_indices = np.round((lons - BBOX["min_lon"]) / 0.25).astype(int)
    lat_indices = np.clip(lat_indices, 0, GRID_LAT_SIZE - 1)
    lon_indices = np.clip(lon_indices, 0, GRID_LON_SIZE - 1)

    for i in range(len(df_argo)):
        d_str = dates_arr[i]
        t_idx = date_to_idx[d_str]
        lat_idx = lat_indices[i]
        lon_idx = lon_indices[i]
        z_pres = pres[i]
        true_t = temps[i]

        profile_ft = preds_ft[t_idx, :, lat_idx, lon_idx]
        profile_v3 = preds_v3[t_idx, :, lat_idx, lon_idx]
        profile_v4 = preds_v4[t_idx, :, lat_idx, lon_idx]

        # Optimal Breeding Profile
        profile_opt = np.zeros_like(profile_ft)
        for d_idx, depth_val in enumerate(depths):
            w = OPTIMAL_WEIGHTS[depth_val]
            profile_opt[d_idx] = w * profile_ft[d_idx] + (1.0 - w) * profile_v4[d_idx]

        pred_t_ft = float(np.interp(z_pres, depths, profile_ft))
        pred_t_v3 = float(np.interp(z_pres, depths, profile_v3))
        pred_t_v4 = float(np.interp(z_pres, depths, profile_v4))
        pred_t_opt = float(np.interp(z_pres, depths, profile_opt))

        for b_idx in range(len(bins) - 1):
            if bins[b_idx] <= z_pres < bins[b_idx + 1]:
                target_bin = bin_labels[b_idx]
                argo_true_temps[target_bin].append(true_t)
                argo_pred_temps_ft[target_bin].append(pred_t_ft)
                argo_pred_temps_v3[target_bin].append(pred_t_v3)
                argo_pred_temps_v4[target_bin].append(pred_t_v4)
                argo_pred_temps_opt[target_bin].append(pred_t_opt)
                break

    print("\n" + "=" * 165)
    print(f"📈 4-WAY IN-SITU ARGO VALIDATION REPORT: {period_name.upper()} ({len(df_argo):,} MEASUREMENTS)")
    print("=" * 165)
    print(f"{'Depth (m)':>10} | {'Observations':>13} | {'Baseline (7-ch)':>18} | {'v3 Phys (12-ch)':>18} | {'v4 Physics-Inf':>18} | {'Optimal Ensemble 🧬':>20} | {'Winner':>16}")
    print(f"{'':>10} | {'':>13} | {'RMSE / Corr':>18} | {'RMSE / Corr':>18} | {'RMSE / Corr':>18} | {'RMSE / Corr':>20} | {'':>16}")
    print("-" * 165)

    all_rmse_ft, all_rmse_v3, all_rmse_v4, all_rmse_opt = [], [], [], []
    all_corr_ft, all_corr_v3, all_corr_v4, all_corr_opt = [], [], [], []

    for depth_m in STANDARD_DEPTH_LEVELS_M:
        trues = np.array(argo_true_temps[depth_m])
        p_ft = np.array(argo_pred_temps_ft[depth_m])
        p_v3 = np.array(argo_pred_temps_v3[depth_m])
        p_v4 = np.array(argo_pred_temps_v4[depth_m])
        p_opt = np.array(argo_pred_temps_opt[depth_m])
        n_obs = len(trues)

        if n_obs < 5:
            continue

        rmse_ft = np.sqrt(np.mean((p_ft - trues) ** 2))
        rmse_v3 = np.sqrt(np.mean((p_v3 - trues) ** 2))
        rmse_v4 = np.sqrt(np.mean((p_v4 - trues) ** 2))
        rmse_opt = np.sqrt(np.mean((p_opt - trues) ** 2))

        all_rmse_ft.append(rmse_ft)
        all_rmse_v3.append(rmse_v3)
        all_rmse_v4.append(rmse_v4)
        all_rmse_opt.append(rmse_opt)

        c_ft = np.corrcoef(p_ft, trues)[0, 1] if len(np.unique(p_ft)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_v3 = np.corrcoef(p_v3, trues)[0, 1] if len(np.unique(p_v3)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_v4 = np.corrcoef(p_v4, trues)[0, 1] if len(np.unique(p_v4)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_opt = np.corrcoef(p_opt, trues)[0, 1] if len(np.unique(p_opt)) > 1 and len(np.unique(trues)) > 1 else 0.0

        all_corr_ft.append(0.0 if np.isnan(c_ft) else c_ft)
        all_corr_v3.append(0.0 if np.isnan(c_v3) else c_v3)
        all_corr_v4.append(0.0 if np.isnan(c_v4) else c_v4)
        all_corr_opt.append(0.0 if np.isnan(c_opt) else c_opt)

        str_ft = f"{rmse_ft:.3f}°C (r={c_ft:.3f})"
        str_v3 = f"{rmse_v3:.3f}°C (r={c_v3:.3f})"
        str_v4 = f"{rmse_v4:.3f}°C (r={c_v4:.3f})"
        str_opt = f"{rmse_opt:.3f}°C (r={c_opt:.3f})"

        rmses = {"Baseline": rmse_ft, "v3 Physical": rmse_v3, "v4 Physics": rmse_v4, "Ensemble 🧬": rmse_opt}
        winner = min(rmses, key=rmses.get)
        print(f"{depth_m:>10d} | {n_obs:>13,d} | {str_ft:>18} | {str_v3:>18} | {str_v4:>18} | {str_opt:>20} | {winner:>16}")

    print("-" * 165)
    mean_rmse_ft = np.mean(all_rmse_ft)
    mean_rmse_v3 = np.mean(all_rmse_v3)
    mean_rmse_v4 = np.mean(all_rmse_v4)
    mean_rmse_opt = np.mean(all_rmse_opt)

    mean_corr_ft = np.mean(all_corr_ft)
    mean_corr_v3 = np.mean(all_corr_v3)
    mean_corr_v4 = np.mean(all_corr_v4)
    mean_corr_opt = np.mean(all_corr_opt)

    print(f"{'OVERALL':>10} | {len(df_argo):>13,d} | {mean_rmse_ft:.4f}°C (r={mean_corr_ft:.4f}) | {mean_rmse_v3:.4f}°C (r={mean_corr_v3:.4f}) | {mean_rmse_v4:.4f}°C (r={mean_corr_v4:.4f}) | {mean_rmse_opt:.4f}°C (r={mean_corr_opt:.4f}) | {'-':>16}")
    print("=" * 165 + "\n")


def run_multi_period_argo_benchmarks_4way():
    device = get_compute_device()

    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    ckpt_v4 = "checkpoints/best_ocean_model_v4.pt"

    model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    model_ft.eval()

    model_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
    model_v3.eval()

    model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v4.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
    model_v4.eval()

    # 1. July 2022 (Southwest Monsoon)
    evaluate_period_on_argo_4way(
        period_tag="jul22",
        period_name="July 2022 (Southwest Monsoon)",
        start_date="2022-07-01",
        end_date="2022-07-31",
        model_ft=model_ft,
        model_v3=model_v3,
        model_v4=model_v4,
        device=device,
    )

    # 2. December 2022 (Northeast Winter Monsoon)
    evaluate_period_on_argo_4way(
        period_tag="dec22",
        period_name="December 2022 (Northeast Winter Monsoon)",
        start_date="2022-12-01",
        end_date="2022-12-31",
        model_ft=model_ft,
        model_v3=model_v3,
        model_v4=model_v4,
        device=device,
    )


if __name__ == "__main__":
    run_multi_period_argo_benchmarks_4way()
