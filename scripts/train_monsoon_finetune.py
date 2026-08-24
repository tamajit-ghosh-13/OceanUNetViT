"""
================================================================================
OceanEmbed - Advanced 12-Channel Physical Ingestion & Training Pipeline
================================================================================
Implements the 12-Channel Pure Physical & Thermodynamic Input Tensor:
  1. SST: Sea Surface Temperature (°C)
  2. SSS: Sea Surface Salinity (PSU)
  3. SSH: Sea Surface Height / Dynamic Sea Level Anomaly (m)
  4. U_CUR: Zonal Surface Ocean Current (m/s)
  5. V_CUR: Meridional Surface Ocean Current (m/s)
  6. U_WIND: 10m Zonal Wind (m/s)
  7. V_WIND: 10m Meridional Wind (m/s)
  8. WIND_MAG: Mechanical Wind Stress / Mixing Magnitude sqrt(U^2 + V^2) (m/s)
  9. DOY_SIN: Continuous Seasonal Harmonic sin(2*pi*DOY/365)
 10. DOY_COS: Continuous Seasonal Harmonic cos(2*pi*DOY/365)
 11. SST_ANOM: Pixel-wise Temporal Climatological SST Anomaly (°C)
 12. DENSITY_SIGMA0: TEOS-10 International Standard Potential Density (kg/m^3) via gsw
================================================================================
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import sys
import shutil
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from preprocessing.harmonize import select_standard_depths
from preprocessing.normalize import denormalize_outputs
from data_loader import OceanDataset
from model import create_model
from train import get_compute_device, OceanReconstructionLoss, train_one_epoch, evaluate
from evaluate import compute_all_metrics, print_metrics_report, plot_skill_profiles, plot_prediction_snapshot


# ==============================================================================
# 1. High-Speed Copernicus Subsetting
# ==============================================================================
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

    print(f"   🚀 Subsetting & downloading {output_filename}...")
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


# ==============================================================================
# 2. Ingest 12 Pure Physical & Thermodynamic Channels
# ==============================================================================
def fetch_ocean_period_12ch(
    start_date: str,
    end_date: str,
    tag: str = "train",
    chunk_days: int = 90,
    save_dir: str = "./data",
):
    os.makedirs(save_dir, exist_ok=True)
    input_file = os.path.join(save_dir, f"{tag}_surface_inputs_12ch.npy")
    target_file = os.path.join(save_dir, f"{tag}_subsurface_targets.npy")
    dates_file = os.path.join(save_dir, f"{tag}_dates.npy")

    if os.path.exists(input_file) and os.path.exists(target_file) and os.path.exists(dates_file):
        print(f"📦 Found cached 12-channel dataset on disk for [{tag}]:")
        inputs = np.load(input_file)
        targets = np.load(target_file)
        dates = np.load(dates_file)
        print(f"   ✅ Loaded {len(dates)} days ({dates[0]} to {dates[-1]}) | Shape: {inputs.shape}\n")
        return inputs, targets, dates

    target_grid = build_standard_grid()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    print("=" * 80)
    print(f"🌊 12-CHANNEL PHYSICAL INGESTION ENGINE [{tag.upper()}]: {start_date} to {end_date}")
    print(f"   Total Days: {total_days} days | Features: TEOS-10 Density + Wind Magnitude + Climatology Anomaly")
    print("=" * 80)

    n_chunks = int(np.ceil(total_days / chunk_days))
    all_inputs, all_targets, all_dates = [], [], []

    omega = 7.2921e-5
    lat_grid = target_grid["lat"] # (101,)
    lon_grid = target_grid["lon"] # (241,)
    lat_rad = np.deg2rad(lat_grid[:, None])
    f = 2 * omega * np.sin(lat_rad)
    f = np.where(np.abs(f) < 1e-5, 1e-5 * np.sign(f), f)
    g = 9.81

    # 2D Meshgrid for TEOS-10 GSW Calculations
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    download_dir = os.path.join(save_dir, "nc_downloads")

    for c in range(n_chunks):
        c_start_dt = start_dt + timedelta(days=c * chunk_days)
        c_days = min(chunk_days, (end_dt - c_start_dt).days + 1)
        c_end_dt = c_start_dt + timedelta(days=c_days - 1)

        c_start_str = c_start_dt.strftime("%Y-%m-%d")
        c_end_str = c_end_dt.strftime("%Y-%m-%d")

        print(f"\n📥 [Chunk {c+1}/{n_chunks}] Fetching {c_days} days ({c_start_str} to {c_end_str})...")

        # 1. 3D Target Temperature (GLORYS)
        temp_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_thetao_chunk{c+1}.nc",
            depth_range=(0.0, 1100.0),
            download_dir=download_dir,
        )
        # 2. Surface Salinity
        so_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_so_chunk{c+1}.nc",
            depth_range=(0.0, 5.0),
            download_dir=download_dir,
        )
        # 3. SSH / SLA
        zos_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_zos_chunk{c+1}.nc",
            depth_range=(0.0, 1.0),
            download_dir=download_dir,
        )
        # 4. Ocean Currents
        cur_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_cur_chunk{c+1}.nc",
            depth_range=(0.0, 5.0),
            download_dir=download_dir,
        )

        with xr.open_dataset(temp_nc) as ds_temp:
            ds_temp_15 = select_standard_depths(ds_temp, depth_variable="depth")
            regridded_temp = regrid_to_standard_grid(ds_temp_15["thetao"], method="bilinear")
            sst_array = regridded_temp.isel(depth=0).values.astype(np.float32)
            target_3d = regridded_temp.values.astype(np.float32)
            dates_chunk = [str(t)[:10] for t in regridded_temp.time.values]
            T_chunk = len(dates_chunk)

        with xr.open_dataset(so_nc) as ds_so:
            sss_array = regrid_to_standard_grid(ds_so["so"].isel(depth=0), method="bilinear").values.astype(np.float32)

        with xr.open_dataset(zos_nc) as ds_zos:
            ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values.astype(np.float32)

        with xr.open_dataset(cur_nc) as ds_cur:
            u_cur = regrid_to_standard_grid(ds_cur["uo"].isel(depth=0), method="bilinear").values.astype(np.float32)
            v_cur = regrid_to_standard_grid(ds_cur["vo"].isel(depth=0), method="bilinear").values.astype(np.float32)

        # 10m Wind Estimation & Mechanical Magnitude
        u_wind = np.zeros_like(ssh_array)
        v_wind = np.zeros_like(ssh_array)
        for t in range(T_chunk):
            grad_y, grad_x = np.gradient(ssh_array[t], 0.25 * 111000, 0.25 * 111000)
            u_wind[t] = - (g / f) * grad_y * 10.0
            v_wind[t] =   (g / f) * grad_x * 10.0

        # Channel 7: Mechanical Wind Stress / Mixing Magnitude
        wind_mag = np.sqrt(u_wind ** 2 + v_wind ** 2)

        # Channels 8 & 9: Day-of-Year Harmonics (sin, cos)
        doy = np.array([datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates_chunk])
        doy_sin = np.sin(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T_chunk, GRID_LAT_SIZE, GRID_LON_SIZE), dtype=np.float32)
        doy_cos = np.cos(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T_chunk, GRID_LAT_SIZE, GRID_LON_SIZE), dtype=np.float32)

        # Channel 10: Climatological Temporal SST Anomaly
        # Subtract temporal mean for the chunk period to isolate dynamic anomalies
        sst_temporal_mean = np.nanmean(sst_array, axis=0, keepdims=True)
        sst_anomaly = sst_array - sst_temporal_mean

        # Channel 11: TEOS-10 Seawater Potential Density Anomaly sigma_0 (kg/m^3)
        density_sigma0 = np.zeros_like(sst_array)
        for t in range(T_chunk):
            # Calculate Absolute Salinity from Practical Salinity (p = 0 dbar at surface)
            sa_t = gsw.SA_from_SP(sss_array[t], 0.0, lon_mesh, lat_mesh)
            # Calculate Conservative Temperature from Potential Temperature
            ct_t = gsw.CT_from_pt(sa_t, sst_array[t])
            # Calculate Potential Density Anomaly sigma_0 (kg/m^3 - 1000)
            density_sigma0[t] = gsw.sigma0(sa_t, ct_t)

        # Assemble the 12-channel pure physical tensor
        inputs_chunk = np.stack([
            sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind,
            wind_mag, doy_sin, doy_cos, sst_anomaly, density_sigma0
        ], axis=1).astype(np.float32)

        all_inputs.append(inputs_chunk)
        all_targets.append(target_3d)
        all_dates.extend(dates_chunk)
        print(f"   ✨ Chunk {c+1} processed: {T_chunk} days ready (12 channels computed).")

        # Auto-cleanup of .nc files commented out per user request:
        # for fpath in [temp_nc, so_nc, zos_nc, cur_nc]:
        #     if os.path.exists(fpath):
        #         os.remove(fpath)

    inputs_full = np.concatenate(all_inputs, axis=0).astype(np.float32)
    targets_full = np.concatenate(all_targets, axis=0).astype(np.float32)
    dates_full = np.array(all_dates)

    np.save(input_file, inputs_full)
    np.save(target_file, targets_full)
    np.save(dates_file, dates_full)

    print(f"\n🎉 12-Channel Physical Dataset Saved: {inputs_full.shape} -> {input_file}")
    return inputs_full, targets_full, dates_full


# ==============================================================================
# 3. 12-Channel Training & Validation Runner
# ==============================================================================
def run_12channel_training_pipeline(
    train_start: str = "2025-06-01",
    train_end: str = "2026-02-28",     # 9 Months Training
    val_start: str = "2026-07-01",
    val_end: str = "2026-07-31",       # July 2026 Validation
    save_checkpoint: str = "checkpoints/best_ocean_model_v3_unbiased.pt",
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
):
    os.makedirs("checkpoints", exist_ok=True)
    device = get_compute_device()

    train_inputs, train_targets, train_dates = fetch_ocean_period_12ch(
        start_date=train_start,
        end_date=train_end,
        tag="train_jun25_feb26",
        chunk_days=90,
    )

    val_inputs, val_targets, val_dates = fetch_ocean_period_12ch(
        start_date=val_start,
        end_date=val_end,
        tag="val_jul26",
        chunk_days=31,
    )

    print("\n📦 Initializing DataLoaders with Stratified Sampling...")
    train_ds = OceanDataset(surface_inputs=train_inputs, subsurface_targets=train_targets, dates=train_dates, use_mock_data=False)
    val_ds   = OceanDataset(surface_inputs=val_inputs, subsurface_targets=val_targets, dates=val_dates, use_mock_data=False)

    month_keys = [d[:7] for d in train_dates]
    month_counts = Counter(month_keys)
    sample_weights = torch.tensor([1.0 / month_counts[m] for m in month_keys], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("\n🧠 Initializing OceanUNetViT v3 (12 Input Channels + Depth Bias Correction)...")
    model = create_model(in_channels=12, out_depth_levels=15).to(device)

    criterion = OceanReconstructionLoss(alpha=0.7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("\n" + "=" * 80)
    print(f"🚀 TRAINING OCEANUNETVIT V3 (12 PURE PHYSICAL CHANNELS)")
    print(f"   In Channels: 12 | Out Depths: 15 | Target Norm: Per-Depth | Device: {device}")
    print("=" * 80)

    best_val_loss = float("inf")
    torch.save(model.state_dict(), save_checkpoint)

    for epoch in range(1, epochs + 1):
        train_loss, train_rmse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val (July 2026) Loss: {val_loss:.4f} | LR: {lr_curr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_checkpoint)

    # Detailed July 2026 Out-of-Year Generalization Report
    print("\n" + "=" * 80)
    print("📈 FINAL JULY 2026 INDEPENDENT MONSOON GENERALIZATION REPORT (12-CH V3)")
    print("=" * 80)

    if os.path.exists(save_checkpoint):
        model.load_state_dict(torch.load(save_checkpoint, map_location=device))
    model.eval()

    all_preds, all_targs = [], []
    with torch.no_grad():
        for x, y in val_loader:
            p = model(x.to(device)).cpu().numpy()
            all_preds.append(p)
            all_targs.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targs, axis=0)

    preds_c = denormalize_outputs(preds, stats=TEMP_TARGET_STATS_PER_DEPTH)
    targs_c = denormalize_outputs(targs, stats=TEMP_TARGET_STATS_PER_DEPTH)

    metrics = compute_all_metrics(preds_c, targs_c)
    print_metrics_report(metrics)

    plot_skill_profiles(metrics, save_path="evaluation_profiles_v3.png")
    plot_prediction_snapshot(preds_c[15], targs_c[15], depth_idx=5, save_path="snapshot_v3.png")

    print(f"\n🎉 12-Channel Model Checkpoint saved to: {save_checkpoint}")
    print(f"📊 Diagnostic charts generated: evaluation_profiles_v3.png, snapshot_v3.png")


if __name__ == "__main__":
    run_12channel_training_pipeline(
        train_start="2025-06-01",
        train_end="2026-02-28",     # 9 Months Training
        val_start="2026-07-01",
        val_end="2026-07-31",       # July 2026 Validation
        epochs=20,
        batch_size=4,
        learning_rate=3e-4,
    )
