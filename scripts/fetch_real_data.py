"""
================================================================================
OceanEmbed - Multi-Month Batch Data Ingestion (fetch_real_data.py)
================================================================================
Fetches real, high-resolution daily ocean data from Copernicus Marine in chunks:
  - Total duration: 12 weeks (84 days)
  - Train split: 10 weeks (70 days)
  - Val / Eval split: 2 weeks (14 days)
  - Period: 2025-03-01 to 2025-05-23 (84 continuous days)
  - Regridded to standard 0.25° grid (101 lat × 241 lon)
  - Subsets 15 standard depths (0m to 1000m)
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import sys
import numpy as np
import xarray as xr
import copernicusmarine
from datetime import datetime, timedelta

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    N_INPUT_CHANNELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
)
from preprocessing.regrid import regrid_to_standard_grid, build_standard_grid
from preprocessing.harmonize import select_standard_depths


def fetch_copernicus_3month_dataset(
    start_date: str = "2025-03-01",
    total_days: int = 84,  # 12 weeks = 84 days (70 train + 14 eval)
    chunk_days: int = 21,  # 3 weeks per API fetch chunk to keep memory & connection resilient
    save_dir: str = "./data",
):
    """
    Streams 12 weeks of daily ocean data in robust chunks and stacks them into a single dataset.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_grid = build_standard_grid()
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=total_days - 1)
    end_date_str = end_dt.strftime("%Y-%m-%d")
    
    print("=" * 80)
    print(f"🌊 Copernicus 12-Week Dataset Fetch: {start_date} to {end_date_str}")
    print(f"   Total days: {total_days} (10 weeks / 70 days train, 2 weeks / 14 days eval)")
    print(f"   Domain: Lat {BBOX['min_lat']}°N–{BBOX['max_lat']}°N | Lon {BBOX['min_lon']}°E–{BBOX['max_lon']}°E")
    print(f"   Target Grid: {GRID_LAT_SIZE} × {GRID_LON_SIZE} (0.25° resolution)")
    print("=" * 80)

    # Calculate chunks
    n_chunks = int(np.ceil(total_days / chunk_days))
    all_inputs = []
    all_targets = []
    all_dates = []

    omega = 7.2921e-5
    lat_rad = np.deg2rad(target_grid["lat"][:, None])
    f = 2 * omega * np.sin(lat_rad)
    f = np.where(np.abs(f) < 1e-5, 1e-5 * np.sign(f), f)
    g = 9.81

    for c in range(n_chunks):
        c_start_dt = start_dt + timedelta(days=c * chunk_days)
        c_days = min(chunk_days, (end_dt - c_start_dt).days + 1)
        c_end_dt = c_start_dt + timedelta(days=c_days - 1)
        
        c_start_str = c_start_dt.strftime("%Y-%m-%d")
        c_end_str = c_end_dt.strftime("%Y-%m-%d")
        
        print(f"\n📦 [Chunk {c+1}/{n_chunks}] Fetching {c_days} days: {c_start_str} to {c_end_str}...")
        
        common_kwargs = dict(
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            start_datetime=f"{c_start_str}T00:00:00",
            end_datetime=f"{c_end_str}T23:59:59",
        )

        # 1. 3D Temperature
        print("   ↳ [1/4] 3D Ocean Temperature (thetao)...")
        ds_temp = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            **common_kwargs,
        )
        ds_temp_15 = select_standard_depths(ds_temp, depth_variable="depth")
        regridded_temp = regrid_to_standard_grid(ds_temp_15["thetao"], method="bilinear")
        sst_array = regridded_temp.isel(depth=0).values  # (T, H, W)
        target_3d = regridded_temp.values                # (T, 15, H, W)
        dates_chunk = [str(t)[:10] for t in regridded_temp.time.values]
        T_chunk = len(dates_chunk)

        # 2. Salinity
        print("   ↳ [2/4] Sea Surface Salinity (so)...")
        ds_so = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            **common_kwargs,
        )
        ds_sss = ds_so["so"].isel(depth=0)
        sss_array = regrid_to_standard_grid(ds_sss, method="bilinear").values

        # 3. Sea Surface Height
        print("   ↳ [3/4] Sea Surface Height (zos)...")
        ds_zos = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"],
            **common_kwargs,
        )
        ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values

        # 4. Currents
        print("   ↳ [4/4] Surface Currents (uo, vo)...")
        ds_cur = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            **common_kwargs,
        )
        u_cur = regrid_to_standard_grid(ds_cur["uo"].isel(depth=0), method="bilinear").values
        v_cur = regrid_to_standard_grid(ds_cur["vo"].isel(depth=0), method="bilinear").values

        # Compute Geostrophic Wind
        u_wind = np.zeros_like(ssh_array)
        v_wind = np.zeros_like(ssh_array)
        for t in range(T_chunk):
            grad_y, grad_x = np.gradient(ssh_array[t], 0.25 * 111000, 0.25 * 111000)
            u_wind[t] = - (g / f) * grad_y * 10.0
            v_wind[t] =   (g / f) * grad_x * 10.0

        # Stack into 7-channel input
        inputs_chunk = np.stack([sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind], axis=1)

        all_inputs.append(inputs_chunk)
        all_targets.append(target_3d)
        all_dates.extend(dates_chunk)
        print(f"   ✅ Chunk {c+1} loaded: {T_chunk} days")

    # Combine all chunks
    inputs_full = np.concatenate(all_inputs, axis=0).astype(np.float32)
    targets_full = np.concatenate(all_targets, axis=0).astype(np.float32)
    dates_full = np.array(all_dates)

    input_file = os.path.join(save_dir, "real_surface_inputs.npy")
    target_file = os.path.join(save_dir, "real_subsurface_targets.npy")
    dates_file = os.path.join(save_dir, "dates.npy")

    np.save(input_file, inputs_full)
    np.save(target_file, targets_full)
    np.save(dates_file, dates_full)

    print("\n" + "=" * 80)
    print(f"🎉 3-MONTH (12-WEEK) DATASET COMPLETE & SAVED TO DISK!")
    print(f"   Total Days:     {len(dates_full)} days ({dates_full[0]} to {dates_full[-1]})")
    print(f"   Train Days:     70 days (10 weeks) -> days 0 to 69")
    print(f"   Eval Days:      14 days (2 weeks)  -> days 70 to 83")
    print(f"   Inputs Shape:   {inputs_full.shape}  -> {input_file}")
    print(f"   Targets Shape:  {targets_full.shape} -> {target_file}")
    print("=" * 80)

    return inputs_full, targets_full, dates_full


if __name__ == "__main__":
    fetch_copernicus_3month_dataset(
        start_date="2025-03-01",
        total_days=84,   # 12 weeks = 84 days
        chunk_days=21,   # 4 chunks of 21 days
    )
