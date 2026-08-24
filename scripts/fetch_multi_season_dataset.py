"""
================================================================================
OceanEmbed - Multi-Season Balanced Dataset Builder (fetch_multi_season_dataset.py)
================================================================================
Downloads and harmonizes 3 distinct, previously untouched oceanographic seasons
in fast 30-day chunks with live progress tracking:
  1. Season A (Spring Pre-Monsoon Peak Heating): March 1 - April 30, 2024 (61 days)
  2. Season B (Fall Post-Monsoon Cyclone Season): October 1 - November 30, 2024 (61 days)
  3. Season C (Extreme Positive Indian Ocean Dipole): July 1 - August 31, 2023 (62 days)

Saved as a combined 12-channel physical cube:
  - data/train_multiseason_surface_inputs_12ch.npy
  - data/train_multiseason_subsurface_targets.npy
  - data/train_multiseason_dates.npy
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import sys
import numpy as np
import xarray as xr
import torch
from datetime import datetime, timedelta
import copernicusmarine
import gsw

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
)
from preprocessing.regrid import regrid_to_standard_grid, build_standard_grid
from preprocessing.harmonize import select_standard_depths


def download_fast_nc_subset(
    dataset_id: str,
    variables: list,
    start_date: str,
    end_date: str,
    output_filename: str,
    download_dir: str = "./data/nc_downloads",
    depth_range: tuple = (0.0, 1100.0),
) -> str:
    os.makedirs(download_dir, exist_ok=True)
    out_path = os.path.join(download_dir, output_filename)

    if os.path.exists(out_path):
        return out_path

    print(f"   🚀 Subsetting & downloading {output_filename} ({start_date} to {end_date})...")
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_latitude=BBOX["min_lat"] - 0.5,
        maximum_latitude=BBOX["max_lat"] + 0.5,
        minimum_longitude=BBOX["min_lon"] - 0.5,
        maximum_longitude=BBOX["max_lon"] + 0.5,
        minimum_depth=depth_range[0],
        maximum_depth=depth_range[1],
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
        output_directory=download_dir,
        output_filename=output_filename,
        overwrite=True,
    )
    return out_path


SEASONAL_PERIODS = [
    {
        "tag": "spring24",
        "name": "Spring Pre-Monsoon Peak Heating (March - April 2024)",
        "start": "2024-03-01",
        "end": "2024-04-30",
    },
    {
        "tag": "fall24",
        "name": "Fall Post-Monsoon Transition (October - November 2024)",
        "start": "2024-10-01",
        "end": "2024-11-30",
    },
    {
        "tag": "iod23",
        "name": "Extreme Positive Indian Ocean Dipole (July - August 2023)",
        "start": "2023-07-01",
        "end": "2023-08-31",
    },
]


def fetch_period_chunked(
    period_info: dict,
    chunk_days: int = 30,
    download_dir: str = "./data/nc_downloads",
    target_grid: dict = None,
):
    tag = period_info["tag"]
    name = period_info["name"]
    start_date = period_info["start"]
    end_date = period_info["end"]

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1
    n_chunks = int(np.ceil(total_days / chunk_days))

    print("\n" + "=" * 80)
    print(f"🌊 FETCHING {name.upper()}: {start_date} to {end_date} ({total_days} Days in {n_chunks} Chunks)")
    print("=" * 80)

    all_inputs, all_targets, all_dates = [], [], []

    omega = 7.2921e-5
    lat_grid = target_grid["lat"]
    lon_grid = target_grid["lon"]
    lat_rad = np.deg2rad(lat_grid[:, None])
    f = 2 * omega * np.sin(lat_rad)
    f = np.where(np.abs(f) < 1e-5, 1e-5 * np.sign(f), f)
    g = 9.81
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    for c in range(n_chunks):
        c_start_dt = start_dt + timedelta(days=c * chunk_days)
        c_days = min(chunk_days, (end_dt - c_start_dt).days + 1)
        c_end_dt = c_start_dt + timedelta(days=c_days - 1)

        c_start_str = c_start_dt.strftime("%Y-%m-%d")
        c_end_str = c_end_dt.strftime("%Y-%m-%d")

        print(f"\n📥 [{tag.upper()} - Chunk {c+1}/{n_chunks}] ({c_start_str} to {c_end_str}, {c_days} days)...")

        temp_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"ms_{tag}_thetao_c{c+1}.nc",
            depth_range=(0.0, 1100.0),
            download_dir=download_dir,
        )
        so_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"ms_{tag}_so_c{c+1}.nc",
            depth_range=(0.0, 5.0),
            download_dir=download_dir,
        )
        zos_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"ms_{tag}_zos_c{c+1}.nc",
            depth_range=(0.0, 1.0),
            download_dir=download_dir,
        )
        cur_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"ms_{tag}_cur_c{c+1}.nc",
            depth_range=(0.0, 5.0),
            download_dir=download_dir,
        )

        with xr.open_dataset(temp_nc) as ds_temp:
            ds_temp_15 = select_standard_depths(ds_temp, depth_variable="depth")
            regridded_temp = regrid_to_standard_grid(ds_temp_15["thetao"], method="bilinear")
            sst_array = regridded_temp.isel(depth=0).values.astype(np.float32)
            target_3d = regridded_temp.values.astype(np.float32)
            dates_c = [str(t)[:10] for t in regridded_temp.time.values]
            T_chunk = len(dates_c)

        with xr.open_dataset(so_nc) as ds_so:
            sss_array = regrid_to_standard_grid(ds_so["so"].isel(depth=0), method="bilinear").values.astype(np.float32)

        with xr.open_dataset(zos_nc) as ds_zos:
            ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values.astype(np.float32)

        with xr.open_dataset(cur_nc) as ds_cur:
            u_cur = regrid_to_standard_grid(ds_cur["uo"].isel(depth=0), method="bilinear").values.astype(np.float32)
            v_cur = regrid_to_standard_grid(ds_cur["vo"].isel(depth=0), method="bilinear").values.astype(np.float32)

        u_wind = np.zeros_like(ssh_array)
        v_wind = np.zeros_like(ssh_array)
        for t in range(T_chunk):
            grad_y, grad_x = np.gradient(ssh_array[t], 0.25 * 111000, 0.25 * 111000)
            u_wind[t] = - (g / f) * grad_y * 10.0
            v_wind[t] =   (g / f) * grad_x * 10.0

        wind_mag = np.sqrt(u_wind ** 2 + v_wind ** 2)
        doy = np.array([datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates_c])
        doy_sin = np.sin(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T_chunk, GRID_LAT_SIZE, GRID_LON_SIZE), dtype=np.float32)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T_chunk, GRID_LAT_SIZE, GRID_LON_SIZE), dtype=np.float32)

        sst_temporal_mean = np.nanmean(sst_array, axis=0, keepdims=True)
        sst_anomaly = sst_array - sst_temporal_mean

        density_sigma0 = np.zeros_like(sst_array)
        for t in range(T_chunk):
            sa_t = gsw.SA_from_SP(sss_array[t], 0.0, lon_mesh, lat_mesh)
            ct_t = gsw.CT_from_pt(sa_t, sst_array[t])
            density_sigma0[t] = gsw.sigma0(sa_t, ct_t)

        inputs_chunk = np.stack([
            sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind,
            wind_mag, doy_sin, doy_cos, sst_anomaly, density_sigma0
        ], axis=1).astype(np.float32)

        all_inputs.append(inputs_chunk)
        all_targets.append(target_3d)
        all_dates.extend(dates_c)
        print(f"   ✨ Chunk {c+1} processed: {T_chunk} days ready.")

    inputs_full = np.concatenate(all_inputs, axis=0).astype(np.float32)
    targets_full = np.concatenate(all_targets, axis=0).astype(np.float32)
    return inputs_full, targets_full, all_dates


def build_full_multiseason_dataset(save_dir: str = "./data"):
    os.makedirs(save_dir, exist_ok=True)
    out_inputs = os.path.join(save_dir, "train_multiseason_surface_inputs_12ch.npy")
    out_targets = os.path.join(save_dir, "train_multiseason_subsurface_targets.npy")
    out_dates = os.path.join(save_dir, "train_multiseason_dates.npy")

    if os.path.exists(out_inputs) and os.path.exists(out_targets) and os.path.exists(out_dates):
        print(f"📦 Found cached multi-season dataset on disk: {out_inputs}")
        return np.load(out_inputs), np.load(out_targets), np.load(out_dates)

    target_grid = build_standard_grid()
    download_dir = os.path.join(save_dir, "nc_downloads")

    all_inputs, all_targets, all_dates = [], [], []

    for period in SEASONAL_PERIODS:
        inputs_p, targets_p, dates_p = fetch_period_chunked(
            period_info=period,
            chunk_days=30,
            download_dir=download_dir,
            target_grid=target_grid,
        )
        all_inputs.append(inputs_p)
        all_targets.append(targets_p)
        all_dates.extend(dates_p)

    inputs_full = np.concatenate(all_inputs, axis=0).astype(np.float32)
    targets_full = np.concatenate(all_targets, axis=0).astype(np.float32)
    dates_full = np.array(all_dates)

    np.save(out_inputs, inputs_full)
    np.save(out_targets, targets_full)
    np.save(out_dates, dates_full)

    print("\n" + "=" * 80)
    print(f"🎉 MULTI-SEASON DATASET COMPLETE!")
    print(f"   Total Days: {len(dates_full)} days")
    print(f"   Input Shape: {inputs_full.shape} -> {out_inputs}")
    print(f"   Target Shape: {targets_full.shape} -> {out_targets}")
    print("=" * 80)
    return inputs_full, targets_full, dates_full


if __name__ == "__main__":
    build_full_multiseason_dataset()
