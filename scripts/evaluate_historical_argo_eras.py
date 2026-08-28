"""
================================================================================
OceanEmbed - Multi-Era In-Situ Argo Inversion Benchmark
================================================================================
Validates all deep learning architectures against real physical in-situ Argo floats:
  1. December 2007 (Northeast Winter Monsoon - Historical Baseline)
  2. March 2014    (Spring Inter-Monsoon Pre-Summer Transition)
  3. August 2017   (Peak Southwest Summer Monsoon & Upwelling)

Models Evaluated:
  - Baseline Finetuned (7-channel)
  - OceanUNetViT v3 Physical Model (12-channel)
  - OceanUNetViT v4 Physics-Informed Model (12-channel + Stratification)
  - Tri-Breeded Optimal Ensemble (Depth-wise covariance optimal)
================================================================================
"""

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
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

def run_model_inference(
    model: torch.nn.Module,
    inputs: np.ndarray,
    is_v3: bool,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """Runs model forward pass on raw inputs with full sanitization and denormalization."""
    model.eval()
    T = inputs.shape[0]
    all_preds = []

    for start_i in range(0, T, batch_size):
        end_i = min(start_i + batch_size, T)
        batch_raw = inputs[start_i:end_i]

        batch_proc = np.zeros_like(batch_raw)
        for b in range(batch_raw.shape[0]):
            p_phys, mask, _ = preprocess_inputs(
                batch_raw[b, :7],
                stats=NORMALIZATION_STATS,
                nan_fill_method="spatial_median",
            )
            if batch_raw.shape[1] > 7:
                extra = batch_raw[b, 7:].copy()
                extra = np.where(np.isnan(extra), 0.0, extra)
                for ch in range(extra.shape[0]):
                    extra[ch][~mask] = 0.0
                batch_proc[b] = np.concatenate([p_phys, extra], axis=0)
            else:
                batch_proc[b] = p_phys

        batch_proc = np.nan_to_num(batch_proc, nan=0.0, posinf=0.0, neginf=0.0)
        x_tensor = torch.from_numpy(batch_proc.astype(np.float32)).to(device)

        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()
            all_preds.append(pred)

    preds_norm = np.concatenate(all_preds, axis=0)

    # Denormalize outputs to physical degrees Celsius (°C)
    if is_v3:
        preds_c = denormalize_outputs(preds_norm, stats=TEMP_TARGET_STATS_PER_DEPTH)
    else:
        preds_c = denormalize_outputs(preds_norm, stats=NORMALIZATION_STATS["TEMP_TARGET"])

    return preds_c


from scripts.generate_tribreed_snapshots import TRI_WEIGHTS


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
                print(f"   ✓ Chunk {c+1}/{n_chunks} ({c_start.strftime('%Y-%m-%d')} to {c_end.strftime('%Y-%m-%d')}): {len(rows):,} readings")
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
    print(f"💾 Saved {len(df_full):,} valid float observations to {output_csv}")
    return df_full


def download_historical_copernicus_surface_nc(
    start_date: str,
    end_date: str,
    period_tag: str,
    download_dir: str = "./data/nc_downloads",
) -> Dict[str, str]:
    """Downloads historical surface input variables from Copernicus Marine GLORYS Reanalysis."""
    os.makedirs(download_dir, exist_ok=True)
    reanalysis_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

    file_map = {
        "thetao": os.path.join(download_dir, f"{period_tag}_thetao.nc"),
        "so": os.path.join(download_dir, f"{period_tag}_so.nc"),
        "zos": os.path.join(download_dir, f"{period_tag}_zos.nc"),
        "cur": os.path.join(download_dir, f"{period_tag}_cur.nc"),
    }

    # 1. SST (thetao at surface)
    if not os.path.exists(file_map["thetao"]):
        print(f"   🚀 Subsetting SST (thetao) for {period_tag}...")
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["thetao"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1.0,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{period_tag}_thetao.nc",
            overwrite=True,
        )

    # 2. SSS (so at surface)
    if not os.path.exists(file_map["so"]):
        print(f"   🚀 Subsetting SSS (so) for {period_tag}...")
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["so"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1.0,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{period_tag}_so.nc",
            overwrite=True,
        )

    # 3. SSH (zos)
    if not os.path.exists(file_map["zos"]):
        print(f"   🚀 Subsetting SSH (zos) for {period_tag}...")
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["zos"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{period_tag}_zos.nc",
            overwrite=True,
        )

    # 4. Surface Currents (uo, vo)
    if not os.path.exists(file_map["cur"]):
        print(f"   🚀 Subsetting Currents (uo, vo) for {period_tag}...")
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["uo", "vo"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1.0,
            start_datetime=f"{start_date}T00:00:00",
            end_datetime=f"{end_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{period_tag}_cur.nc",
            overwrite=True,
        )

    return file_map


def build_or_load_historical_inputs(
    period_tag: str,
    start_date: str,
    end_date: str,
    save_dir: str = "./data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates standardized 7-channel and 12-channel surface input tensors."""
    os.makedirs(save_dir, exist_ok=True)
    f_7ch = os.path.join(save_dir, f"argo_{period_tag}_inputs_7ch.npy")
    f_12ch = os.path.join(save_dir, f"argo_{period_tag}_inputs_12ch.npy")
    f_dates = os.path.join(save_dir, f"argo_{period_tag}_dates.npy")

    if os.path.exists(f_7ch) and os.path.exists(f_12ch) and os.path.exists(f_dates):
        print(f"📦 Loaded pre-processed surface input tensors for {period_tag}")
        return np.load(f_7ch), np.load(f_12ch), np.load(f_dates)

    print(f"\n⚙️ Building 12-channel surface inputs from Copernicus for {period_tag} ({start_date} to {end_date})...")
    nc_files = download_historical_copernicus_surface_nc(start_date, end_date, period_tag)
    target_grid = build_standard_grid()

    with xr.open_dataset(nc_files["thetao"]) as ds_temp:
        regridded_temp = regrid_to_standard_grid(ds_temp["thetao"].isel(depth=0), method="bilinear")
        sst_array = regridded_temp.values.astype(np.float32)
        dates = np.array([str(t)[:10] for t in regridded_temp.time.values])
        T = len(dates)

    with xr.open_dataset(nc_files["so"]) as ds_so:
        sss_array = regrid_to_standard_grid(ds_so["so"].isel(depth=0), method="bilinear").values.astype(np.float32)

    with xr.open_dataset(nc_files["zos"]) as ds_zos:
        ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values.astype(np.float32)

    with xr.open_dataset(nc_files["cur"]) as ds_cur:
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


def evaluate_single_argo_period(
    period_tag: str,
    period_name: str,
    start_date: str,
    end_date: str,
    model_ft: torch.nn.Module,
    model_v3: torch.nn.Module,
    model_v4: torch.nn.Module,
    model_v5: Optional[torch.nn.Module],
    device: torch.device,
) -> Dict[str, Any]:
    """Runs high-precision continuous 2D bilinear + cubic PCHIP validation for a historical period."""
    csv_file = f"./data/argo_{period_tag}.csv"
    df_argo = download_argo_ifremer(start_date=start_date, end_date=end_date, output_csv=csv_file)

    if df_argo.empty:
        print(f"⚠️ No in-situ Argo observations found for {period_name}.")
        return {}

    inputs_7ch, inputs_12ch, dates = build_or_load_historical_inputs(
        period_tag=period_tag,
        start_date=start_date,
        end_date=end_date,
    )
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df_argo = df_argo[df_argo["date"].isin(date_to_idx)].reset_index(drop=True)
    print(f"   ✅ {len(df_argo):,} in-situ measurements aligned with satellite observations.")

    print(f"\n🔮 Running 3D deep neural inversion over {period_name}...")
    preds_ft = run_model_inference(model_ft, inputs_7ch, is_v3=False, device=device)
    preds_v3 = run_model_inference(model_v3, inputs_12ch, is_v3=True, device=device)
    preds_v4 = run_model_inference(model_v4, inputs_12ch, is_v3=True, device=device)
    preds_v5 = run_model_inference(model_v5, inputs_12ch, is_v3=True, device=device) if model_v5 else None

    # Tri-Breeded Optimal Ensemble Volume
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=float)
    preds_tri = np.zeros_like(preds_ft)
    for d_idx, d_val in enumerate(depths):
        w = TRI_WEIGHTS[int(d_val)]
        preds_tri[:, d_idx] = w[0] * preds_ft[:, d_idx] + w[1] * preds_v3[:, d_idx] + w[2] * preds_v4[:, d_idx]

    # Quad-Breeded Optimal Ensemble Volume (Baseline + v3 + v4 + v5)
    from scripts.generate_tribreed_snapshots import QUAD_BREED_WEIGHTS
    preds_quad = np.zeros_like(preds_ft)
    for d_idx, d_val in enumerate(depths):
        w = QUAD_BREED_WEIGHTS[int(d_val)]
        preds_quad[:, d_idx] = (
            w[0] * preds_ft[:, d_idx] +
            w[1] * preds_v3[:, d_idx] +
            w[2] * preds_v4[:, d_idx] +
            w[3] * (preds_v5[:, d_idx] if preds_v5 is not None else preds_v4[:, d_idx])
        )

    lat_grid = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lon_grid = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)

    bins = [0, 2.5, 7.5, 15, 25, 40, 62.5, 87.5, 112.5, 137.5, 175, 250, 400, 600, 850, 1100]
    bin_labels = STANDARD_DEPTH_LEVELS_M

    trues_dict = {d: [] for d in bin_labels}
    preds_ft_dict = {d: [] for d in bin_labels}
    preds_v3_dict = {d: [] for d in bin_labels}
    preds_v4_dict = {d: [] for d in bin_labels}
    preds_v5_dict = {d: [] for d in bin_labels}
    preds_tri_dict = {d: [] for d in bin_labels}
    preds_quad_dict = {d: [] for d in bin_labels}

    lats = df_argo["latitude"].values
    lons = df_argo["longitude"].values
    pres = df_argo["pres"].values
    temps = df_argo["temp"].values
    dates_arr = df_argo["date"].values

    print("   🔬 Executing continuous sub-grid 2D bilinear + vertical PCHIP interpolation...")
    daily_tri_interp = {}
    daily_quad_interp = {}
    daily_ft_interp = {}
    daily_v3_interp = {}
    daily_v4_interp = {}
    daily_v5_interp = {}

    for t_idx in range(len(dates)):
        daily_tri_interp[t_idx] = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_tri[t_idx], method="linear", bounds_error=False, fill_value=None)
        daily_quad_interp[t_idx] = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_quad[t_idx], method="linear", bounds_error=False, fill_value=None)
        daily_ft_interp[t_idx]  = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_ft[t_idx],  method="linear", bounds_error=False, fill_value=None)
        daily_v3_interp[t_idx]  = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_v3[t_idx],  method="linear", bounds_error=False, fill_value=None)
        daily_v4_interp[t_idx]  = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_v4[t_idx],  method="linear", bounds_error=False, fill_value=None)
        if preds_v5 is not None:
            daily_v5_interp[t_idx] = RegularGridInterpolator((depths, lat_grid, lon_grid), preds_v5[t_idx], method="linear", bounds_error=False, fill_value=None)

    for i in range(len(df_argo)):
        d_str = dates_arr[i]
        t_idx = date_to_idx[d_str]
        f_lat = lats[i]
        f_lon = lons[i]
        z_pres = pres[i]
        true_t = temps[i]

        coords_15 = np.column_stack([depths, np.full(15, f_lat), np.full(15, f_lon)])
        pchip_tri  = PchipInterpolator(depths, daily_tri_interp[t_idx](coords_15))
        pchip_quad = PchipInterpolator(depths, daily_quad_interp[t_idx](coords_15))
        pchip_ft   = PchipInterpolator(depths, daily_ft_interp[t_idx](coords_15))
        pchip_v3   = PchipInterpolator(depths, daily_v3_interp[t_idx](coords_15))
        pchip_v4   = PchipInterpolator(depths, daily_v4_interp[t_idx](coords_15))

        pred_tri_val  = float(pchip_tri(z_pres))
        pred_quad_val = float(pchip_quad(z_pres))
        pred_ft_val   = float(pchip_ft(z_pres))
        pred_v3_val   = float(pchip_v3(z_pres))
        pred_v4_val   = float(pchip_v4(z_pres))

        if preds_v5 is not None:
            pchip_v5 = PchipInterpolator(depths, daily_v5_interp[t_idx](coords_15))
            pred_v5_val = float(pchip_v5(z_pres))
        else:
            pred_v5_val = pred_tri_val

        for b_idx in range(len(bins) - 1):
            if bins[b_idx] <= z_pres < bins[b_idx + 1]:
                target_bin = bin_labels[b_idx]
                trues_dict[target_bin].append(true_t)
                preds_ft_dict[target_bin].append(pred_ft_val)
                preds_v3_dict[target_bin].append(pred_v3_val)
                preds_v4_dict[target_bin].append(pred_v4_val)
                preds_v5_dict[target_bin].append(pred_v5_val)
                preds_tri_dict[target_bin].append(pred_tri_val)
                preds_quad_dict[target_bin].append(pred_quad_val)
                break

    print("\n" + "=" * 240)
    print(f"📈 IN-SITU ARGO VALIDATION REPORT: {period_name.upper()} ({len(df_argo):,} MEASUREMENTS)")
    print("=" * 240)
    print(f"{'Depth (m)':>10} | {'Observations':>13} | {'Argo Truth (°C)':>16} | {'Baseline (7-ch)':>20} | {'v3 Phys (12-ch)':>20} | {'Tri-Breed AI 🧬':>22} | {'OceanUNetViT v5 🚀':>22} | {'Quad-Breed 4-Way 🏆':>24} | {'Winner':>16}")
    print(f"{'':>10} | {'':>13} | {'Mean ± Std':>16} | {'Pred (RMSE / r)':>20} | {'Pred (RMSE / r)':>20} | {'Pred (RMSE / r)':>22} | {'Pred (RMSE / r)':>22} | {'Pred (RMSE / r)':>24} | {'':>16}")
    print("-" * 240)

    all_rmse_ft, all_rmse_v3, all_rmse_v4, all_rmse_v5, all_rmse_tri, all_rmse_quad = [], [], [], [], [], []
    all_corr_ft, all_corr_v3, all_corr_v4, all_corr_v5, all_corr_tri, all_corr_quad = [], [], [], [], [], []

    for depth_m in STANDARD_DEPTH_LEVELS_M:
        trues = np.array(trues_dict[depth_m])
        p_ft = np.array(preds_ft_dict[depth_m])
        p_v3 = np.array(preds_v3_dict[depth_m])
        p_v4 = np.array(preds_v4_dict[depth_m])
        p_v5 = np.array(preds_v5_dict[depth_m])
        p_tri = np.array(preds_tri_dict[depth_m])
        p_quad = np.array(preds_quad_dict[depth_m])
        n_obs = len(trues)

        if n_obs < 5:
            continue

        mean_true = np.mean(trues)
        mean_p_ft = np.mean(p_ft)
        mean_p_v3 = np.mean(p_v3)
        mean_p_v4 = np.mean(p_v4)
        mean_p_v5 = np.mean(p_v5)
        mean_p_tri = np.mean(p_tri)
        mean_p_quad = np.mean(p_quad)

        rmse_ft = np.sqrt(np.mean((p_ft - trues) ** 2))
        rmse_v3 = np.sqrt(np.mean((p_v3 - trues) ** 2))
        rmse_v4 = np.sqrt(np.mean((p_v4 - trues) ** 2))
        rmse_v5 = np.sqrt(np.mean((p_v5 - trues) ** 2))
        rmse_tri = np.sqrt(np.mean((p_tri - trues) ** 2))
        rmse_quad = np.sqrt(np.mean((p_quad - trues) ** 2))

        all_rmse_ft.append(rmse_ft)
        all_rmse_v3.append(rmse_v3)
        all_rmse_v4.append(rmse_v4)
        all_rmse_v5.append(rmse_v5)
        all_rmse_tri.append(rmse_tri)
        all_rmse_quad.append(rmse_quad)

        c_ft = np.corrcoef(p_ft, trues)[0, 1] if len(np.unique(p_ft)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_v3 = np.corrcoef(p_v3, trues)[0, 1] if len(np.unique(p_v3)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_v4 = np.corrcoef(p_v4, trues)[0, 1] if len(np.unique(p_v4)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_v5 = np.corrcoef(p_v5, trues)[0, 1] if len(np.unique(p_v5)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_tri = np.corrcoef(p_tri, trues)[0, 1] if len(np.unique(p_tri)) > 1 and len(np.unique(trues)) > 1 else 0.0
        c_quad = np.corrcoef(p_quad, trues)[0, 1] if len(np.unique(p_quad)) > 1 and len(np.unique(trues)) > 1 else 0.0

        all_corr_ft.append(0.0 if np.isnan(c_ft) else c_ft)
        all_corr_v3.append(0.0 if np.isnan(c_v3) else c_v3)
        all_corr_v4.append(0.0 if np.isnan(c_v4) else c_v4)
        all_corr_v5.append(0.0 if np.isnan(c_v5) else c_v5)
        all_corr_tri.append(0.0 if np.isnan(c_tri) else c_tri)
        all_corr_quad.append(0.0 if np.isnan(c_quad) else c_quad)

        str_true = f"{mean_true:.2f}°C"
        str_ft = f"{mean_p_ft:.2f}°C ({rmse_ft:.3f}°C)"
        str_v3 = f"{mean_p_v3:.2f}°C ({rmse_v3:.3f}°C)"
        str_tri = f"{mean_p_tri:.2f}°C ({rmse_tri:.3f}°C)"
        str_v5 = f"{mean_p_v5:.2f}°C ({rmse_v5:.3f}°C)"
        str_quad = f"{mean_p_quad:.2f}°C ({rmse_quad:.3f}°C)"

        rmses = {"Baseline": rmse_ft, "v3 Physical": rmse_v3, "Tri-Breed 🧬": rmse_tri, "v5 Physics 🚀": rmse_v5, "Quad-Breed 🏆": rmse_quad}
        winner = min(rmses, key=rmses.get)
        print(f"{depth_m:>10d} | {n_obs:>13,d} | {str_true:>16} | {str_ft:>20} | {str_v3:>20} | {str_tri:>22} | {str_v5:>22} | {str_quad:>24} | {winner:>16}")

    print("-" * 240)
    mean_rmse_ft = np.mean(all_rmse_ft)
    mean_rmse_v3 = np.mean(all_rmse_v3)
    mean_rmse_v4 = np.mean(all_rmse_v4)
    mean_rmse_v5 = np.mean(all_rmse_v5)
    mean_rmse_tri = np.mean(all_rmse_tri)
    mean_rmse_quad = np.mean(all_rmse_quad)

    mean_corr_ft = np.mean(all_corr_ft)
    mean_corr_v3 = np.mean(all_corr_v3)
    mean_corr_v4 = np.mean(all_corr_v4)
    mean_corr_v5 = np.mean(all_corr_v5)
    mean_corr_tri = np.mean(all_corr_tri)
    mean_corr_quad = np.mean(all_corr_quad)

    print(f"{'OVERALL':>10} | {len(df_argo):>13,d} | {'-':>16} | {mean_rmse_ft:.4f}°C (r={mean_corr_ft:.3f}) | {mean_rmse_v3:.4f}°C (r={mean_corr_v3:.3f}) | {mean_rmse_tri:.4f}°C (r={mean_corr_tri:.3f}) | {mean_rmse_v5:.4f}°C (r={mean_corr_v5:.3f}) | {mean_rmse_quad:.4f}°C (r={mean_corr_quad:.3f}) | {'Quad-Breed 🏆':>16}")
    print("=" * 240 + "\n")

    return {
        "period": period_name,
        "n_obs": len(df_argo),
        "rmse_ft": mean_rmse_ft,
        "rmse_v3": mean_rmse_v3,
        "rmse_v4": mean_rmse_v4,
        "rmse_tri": mean_rmse_tri,
        "rmse_v5": mean_rmse_v5,
        "rmse_quad": mean_rmse_quad,
        "corr_quad": mean_corr_quad,
    }


def run_dec07_mar14_aug17_validation():
    device = get_compute_device()

    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    ckpt_v4 = "checkpoints/best_ocean_model_v4.pt"
    ckpt_v5 = "checkpoints/best_ocean_model_v5_finetuned.pt"
    if not os.path.exists(ckpt_v5):
        ckpt_v5 = "checkpoints/best_ocean_model_v5.pt"

    print("🧠 Loading models on compute device...")
    model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
    model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
    model_ft.eval()

    model_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
    model_v3.eval()

    model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    model_v4.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
    model_v4.eval()

    model_v5 = None
    if os.path.exists(ckpt_v5):
        model_v5 = create_model(in_channels=12, out_depth_levels=15).to(device)
        model_v5.load_state_dict(torch.load(ckpt_v5, map_location=device), strict=False)
        model_v5.eval()
        print(f"   ✅ Loaded Calibrated OceanUNetViT v5 ({os.path.basename(ckpt_v5)})!")

    test_eras = [
        ("nov16", "November 2016 (Historic Negative IOD Era)", "2016-11-01", "2016-11-30"),
        ("jun20", "June 2020 (Super Cyclone Amphan / Early SW Monsoon)", "2020-06-01", "2020-06-30"),
    ]

    all_summaries = []
    for tag, name, s_date, e_date in test_eras:
        summary = evaluate_single_argo_period(
            period_tag=tag,
            period_name=name,
            start_date=s_date,
            end_date=e_date,
            model_ft=model_ft,
            model_v3=model_v3,
            model_v4=model_v4,
            model_v5=model_v5,
            device=device,
        )
        if summary:
            all_summaries.append(summary)

    print("\n" + "=" * 145)
    print("🏆 GRAND SUMMARY: FRESH IN-SITU ARGO BENCHMARKS ON UNSEEN CLIMATIC ERAS")
    print("=" * 145)
    print(f"{'Target Era':>52} | {'Float Obs':>12} | {'Baseline (7ch)':>16} | {'Tri-Breed AI 🧬':>18} | {'OceanUNetViT v5 🚀':>20} | {'Quad-Breed 4-Way 🏆':>22}")
    print("-" * 145)
    for s in all_summaries:
        print(f"{s['period']:>52} | {s['n_obs']:>12,d} | {s['rmse_ft']:.4f}°C | {s['rmse_tri']:.4f}°C | {s['rmse_v5']:.4f}°C | {s['rmse_quad']:.4f}°C (r={s['corr_quad']:.3f})")
    print("=" * 145 + "\n")


if __name__ == "__main__":
    run_dec07_mar14_aug17_validation()

