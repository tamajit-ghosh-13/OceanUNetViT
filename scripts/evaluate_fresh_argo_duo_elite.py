import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import pandas as pd
import xarray as xr
import torch
import copernicusmarine
import gsw
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from erddapy import ERDDAP

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    NORMALIZATION_STATS,
    TEMP_TARGET_STATS_PER_DEPTH,
)
from preprocessing.regrid import regrid_to_standard_grid, build_standard_grid
from preprocessing.normalize import denormalize_outputs, preprocess_inputs
from model import create_model
from train import get_compute_device

DUO_ELITE_WEIGHTS = {
    0: [0.5000, 0.5000],
    5: [0.5961, 0.4039],
    10: [0.7996, 0.2004],
    20: [0.8224, 0.1776],
    30: [0.8563, 0.1437],
    50: [0.8605, 0.1395],
    75: [0.8607, 0.1393],
    100: [0.6812, 0.3188],
    125: [0.9199, 0.0801],
    150: [0.9070, 0.0930],
    200: [0.4962, 0.5038],
    300: [0.0441, 0.9559],
    500: [0.2275, 0.7725],
    700: [0.3026, 0.6974],
    1000: [0.2169, 0.7831],
}

def download_fresh_argo(s_date="2018-04-01", e_date="2018-04-30", tag="apr18"):
    csv_file = f"./data/argo_{tag}.csv"
    if os.path.exists(csv_file):
        print(f"📦 Found cached Argo observations: {csv_file}")
        return pd.read_csv(csv_file)

    print(f"🌐 Pulling fresh ARGO float CTD observations from Ifremer GDAC ({s_date} to {e_date})...")
    e = ERDDAP(server="https://erddap.ifremer.fr/erddap", protocol="tabledap")
    e.dataset_id = "ArgoFloats"
    e.constraints = {
        "time>=": f"{s_date}T00:00:00Z",
        "time<=": f"{e_date}T23:59:59Z",
        "latitude>=": BBOX["min_lat"],
        "latitude<=": BBOX["max_lat"],
        "longitude>=": BBOX["min_lon"],
        "longitude<=": BBOX["max_lon"],
        "pres>=": 0,
        "pres<=": 1050,
    }
    e.variables = ["platform_number", "time", "latitude", "longitude", "pres", "temp", "psal"]

    df = e.to_pandas()
    rename_map = {
        "time (UTC)": "time",
        "latitude (degrees_north)": "latitude",
        "longitude (degrees_east)": "longitude",
        "pres (decibar)": "pres",
        "temp (degree_Celsius)": "temp",
    }
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=["latitude", "longitude", "pres", "temp", "time"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
    df = df[(df["pres"] >= 0) & (df["pres"] <= 1050) & (df["temp"] >= 2) & (df["temp"] <= 35)]
    df.to_csv(csv_file, index=False)
    print(f"   ✅ Downloaded {len(df):,} fresh physical ARGO float observations -> {csv_file}")
    return df

def download_fresh_glorys_surface(s_date="2018-04-01", e_date="2018-04-30", tag="apr18"):
    f_in = f"./data/argo_{tag}_inputs_12ch.npy"
    f_dt = f"./data/argo_{tag}_dates.npy"
    if os.path.exists(f_in) and os.path.exists(f_dt):
        print(f"📦 Found cached surface inputs: {f_in}")
        return np.load(f_in), np.load(f_dt)

    print(f"🌐 Pulling GLORYS satellite surface inputs ({s_date} to {e_date})...")
    download_dir = "./data/nc_downloads"
    os.makedirs(download_dir, exist_ok=True)
    reanalysis_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    target_grid = build_standard_grid()

    sst_nc = os.path.join(download_dir, f"{tag}_sst.nc")
    if not os.path.exists(sst_nc):
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["thetao"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1.0,
            start_datetime=f"{s_date}T00:00:00",
            end_datetime=f"{e_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{tag}_sst.nc",
            overwrite=True,
        )

    so_nc = os.path.join(download_dir, f"{tag}_so.nc")
    if not os.path.exists(so_nc):
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["so"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1.0,
            start_datetime=f"{s_date}T00:00:00",
            end_datetime=f"{e_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{tag}_so.nc",
            overwrite=True,
        )

    zos_nc = os.path.join(download_dir, f"{tag}_zos.nc")
    if not os.path.exists(zos_nc):
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["zos"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            start_datetime=f"{s_date}T00:00:00",
            end_datetime=f"{e_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{tag}_zos.nc",
            overwrite=True,
        )

    cur_nc = os.path.join(download_dir, f"{tag}_cur.nc")
    if not os.path.exists(cur_nc):
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["uo", "vo"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1.0,
            start_datetime=f"{s_date}T00:00:00",
            end_datetime=f"{e_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{tag}_cur.nc",
            overwrite=True,
        )

    with xr.open_dataset(sst_nc) as ds_sst:
        r_sst = regrid_to_standard_grid(ds_sst["thetao"].isel(depth=0), method="bilinear")
        sst_array = r_sst.values.astype(np.float32)
        dates = np.array([str(t)[:10] for t in r_sst.time.values])
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

    np.save(f_in, inputs_12ch)
    np.save(f_dt, dates)

    for f_nc in [sst_nc, so_nc, zos_nc, cur_nc]:
        if os.path.exists(f_nc):
            os.remove(f_nc)

    return inputs_12ch, dates

def run_fresh_validation(tag="apr18", era_name="April 2018 (Pre-Monsoon Arabian Sea Warm Pool Peak)", s_date="2018-04-01", e_date="2018-04-30"):
    device = get_compute_device()
    df_argo = download_fresh_argo(s_date, e_date, tag)
    inputs_12ch, dates = download_fresh_glorys_surface(s_date, e_date, tag)

    print("🧠 Loading v4_extended and v5_finetuned backbones...")
    m_v4_ext = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v4_ext.load_state_dict(torch.load("checkpoints/best_ocean_model_v4_extended.pt", map_location=device), strict=False)
    m_v4_ext.eval()

    m_v5_ft = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v5_ft.load_state_dict(torch.load("checkpoints/best_ocean_model_v5_finetuned.pt", map_location=device), strict=False)
    m_v5_ft.eval()

    date_to_idx = {d: i for i, d in enumerate(dates)}
    df_argo = df_argo[df_argo["date"].isin(date_to_idx)].reset_index(drop=True)
    T = len(dates)

    print("🔮 Running 3D deep neural inversion with Duo-Elite convex combination...")
    preds_v4_raw, preds_v5_raw = [], []
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
            preds_v4_raw.append(m_v4_ext(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy())
            preds_v5_raw.append(m_v5_ft(torch.from_numpy(proc.astype(np.float32)).to(device)).cpu().numpy())

    preds_v4_c = denormalize_outputs(np.concatenate(preds_v4_raw, axis=0), stats=TEMP_TARGET_STATS_PER_DEPTH)
    preds_v5_c = denormalize_outputs(np.concatenate(preds_v5_raw, axis=0), stats=TEMP_TARGET_STATS_PER_DEPTH)

    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=float)
    preds_duo_c = np.zeros_like(preds_v4_c)
    for d_idx, d_val in enumerate(depths):
        w = DUO_ELITE_WEIGHTS[int(d_val)]
        preds_duo_c[:, d_idx] = w[0] * preds_v4_c[:, d_idx] + w[1] * preds_v5_c[:, d_idx]

    lat_grid = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lon_grid = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)

    daily_duo = {t: RegularGridInterpolator((depths, lat_grid, lon_grid), preds_duo_c[t], method="linear", bounds_error=False, fill_value=None) for t in range(T)}
    daily_v4 = {t: RegularGridInterpolator((depths, lat_grid, lon_grid), preds_v4_c[t], method="linear", bounds_error=False, fill_value=None) for t in range(T)}

    bins = [0, 2.5, 7.5, 15, 25, 40, 62.5, 87.5, 112.5, 137.5, 175, 250, 400, 600, 850, 1100]
    bin_labels = STANDARD_DEPTH_LEVELS_M
    trues_d = {d: [] for d in bin_labels}
    preds_duo_d = {d: [] for d in bin_labels}
    preds_v4_d = {d: [] for d in bin_labels}

    lats, lons, pres, temps, dt_arr = df_argo["latitude"].values, df_argo["longitude"].values, df_argo["pres"].values, df_argo["temp"].values, df_argo["date"].values

    print("🔬 Interpolating sub-grid continuous 2D bilinear + vertical PCHIP splines...")
    for i in range(len(df_argo)):
        t_idx = date_to_idx[dt_arr[i]]
        z_pres = pres[i]
        true_t = temps[i]
        coords_15 = np.column_stack([depths, np.full(15, lats[i]), np.full(15, lons[i])])
        pchip_duo = PchipInterpolator(depths, daily_duo[t_idx](coords_15))
        pchip_v4 = PchipInterpolator(depths, daily_v4[t_idx](coords_15))
        val_duo = float(pchip_duo(z_pres))
        val_v4 = float(pchip_v4(z_pres))

        for b_idx in range(len(bins)-1):
            if bins[b_idx] <= z_pres < bins[b_idx+1]:
                target_d = bin_labels[b_idx]
                trues_d[target_d].append(true_t)
                preds_duo_d[target_d].append(val_duo)
                preds_v4_d[target_d].append(val_v4)
                break

    h_d, h_obs, h_truth, h_v4, h_duo, h_rduo, h_cduo = "Depth (m)", "Obs Count", "ARGO Truth (°C)", "v4_extended (°C)", "Duo-Elite (°C)", "Duo RMSE", "Corr (r)"
    print("=" * 135)
    print(f"🌊 FRESH IN-SITU VALIDATION: {era_name.upper()} ({len(df_argo):,} MEASUREMENTS)")
    print("=" * 135)
    print(f"{h_d:>10} | {h_obs:>12} | {h_truth:>18} | {h_v4:>20} | {h_duo:>20} | {h_rduo:>14} | {h_cduo:>10}")
    print("-" * 135)

    all_rmse_duo, all_rmse_4, all_corr_duo = [], [], []

    for d in bin_labels:
        t = np.array(trues_d[d])
        p_duo = np.array(preds_duo_d[d])
        p_v4 = np.array(preds_v4_d[d])
        if len(t) < 5: continue
        r_duo = np.sqrt(np.mean((p_duo - t)**2))
        r_v4 = np.sqrt(np.mean((p_v4 - t)**2))
        c_duo = np.corrcoef(p_duo, t)[0, 1] if len(np.unique(p_duo)) > 1 and len(np.unique(t)) > 1 else 0.0

        all_rmse_duo.append(r_duo)
        all_rmse_4.append(r_v4)
        all_corr_duo.append(c_duo)

        str_t = f"{t.mean():.2f}°C"
        str_4 = f"{p_v4.mean():.2f}°C"
        str_duo = f"{p_duo.mean():.2f}°C"
        str_r = f"{r_duo:.4f}°C"
        str_c = f"{c_duo:.3f}"
        print(f"{d:>10d} | {len(t):>12,d} | {str_t:>18} | {str_4:>20} | {str_duo:>20} | {str_r:>14} | {str_c:>10}")

    print("-" * 135)
    mean_rduo_str = f"{np.mean(all_rmse_duo):.4f}°C"
    mean_cduo_str = f"{np.mean(all_corr_duo):.3f}"
    mean_r4_str = f"{np.mean(all_rmse_4):.4f}°C"
    print(f"{'OVERALL':>10} | {len(df_argo):>12,d} | {'-':>18} | {mean_r4_str:>20} | {'-':>20} | {mean_rduo_str:>14} | {mean_cduo_str:>10}")
    print("=" * 135)

if __name__ == "__main__":
    run_fresh_validation(
        tag="apr18",
        era_name="April 2018 (Pre-Monsoon Arabian Sea Warm Pool Peak)",
        s_date="2018-04-01",
        e_date="2018-04-30",
    )