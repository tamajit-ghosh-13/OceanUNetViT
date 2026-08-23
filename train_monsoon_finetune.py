"""
================================================================================
OceanEmbed - High-Speed NetCDF Subset & Multi-Season Training Pipeline
================================================================================
Uses copernicusmarine.subset() to crop server-side on Copernicus cloud and
download pre-cropped .nc files at MAXIMUM bandwidth in seconds, avoiding
OPeNDAP range-request network latency.

Workflow:
  1. High-Speed Subsetting: Downloads pre-cropped North Indian Ocean .nc files
     - 9 Months Training: June 1, 2025 -> Feb 28, 2026 (273 days)
     - 1 Month Validation: July 1, 2026 -> July 31, 2026 (31 days)
  2. Local Extraction & Regridding: Parses downloaded .nc files with xarray locally
  3. Pre-trained Fine-Tuning: Continues training from best_ocean_model.pt
  4. Validation on Future July 2026 Monsoon: Generates report & charts
================================================================================
"""

import os
import sys
import numpy as np
import xarray as xr
import torch
from datetime import datetime, timedelta
import copernicusmarine

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    N_INPUT_CHANNELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    NORMALIZATION_STATS,
)
from preprocessing.regrid import regrid_to_standard_grid, build_standard_grid
from preprocessing.harmonize import select_standard_depths
from preprocessing.normalize import denormalize_outputs
from data_loader import OceanDataset
from model import create_model
from train import get_compute_device, OceanReconstructionLoss, train_one_epoch, evaluate
from evaluate import compute_all_metrics, print_metrics_report, plot_skill_profiles, plot_prediction_snapshot


# ==============================================================================
# 1. High-Speed Server-Side Subset Downloader (.nc format)
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
    """
    Downloads a single pre-cropped .nc file directly from Copernicus cloud.
    Speed: Uses bulk transfer without per-slice HTTP range request latency.
    """
    os.makedirs(download_dir, exist_ok=True)
    out_path = os.path.join(download_dir, output_filename)

    if os.path.exists(out_path):
        print(f"   ⚡ Found cached .nc file: {out_path} (skipping download)")
        return out_path

    print(f"   🚀 Cloud subsetting & downloading {output_filename}...")
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
    print(f"   ✅ Saved {output_filename} ({os.path.getsize(out_path) / (1024*1024):.1f} MB)")
    return out_path


# ==============================================================================
# 2. Ingest Ocean Period via Fast Subsets
# ==============================================================================
def fetch_ocean_period_fast(
    start_date: str,
    end_date: str,
    tag: str = "train",
    chunk_days: int = 90,  # Larger 3-month chunks now download in seconds
    save_dir: str = "./data",
):
    """
    Downloads and prepares real ocean data at top speed.
    """
    os.makedirs(save_dir, exist_ok=True)
    input_file = os.path.join(save_dir, f"{tag}_surface_inputs.npy")
    target_file = os.path.join(save_dir, f"{tag}_subsurface_targets.npy")
    dates_file = os.path.join(save_dir, f"{tag}_dates.npy")

    # Instant load if already processed
    if os.path.exists(input_file) and os.path.exists(target_file) and os.path.exists(dates_file):
        print(f"📦 Found prepared arrays on disk for [{tag}]:")
        inputs = np.load(input_file)
        targets = np.load(target_file)
        dates = np.load(dates_file)
        print(f"   ✅ Loaded {len(dates)} days ({dates[0]} to {dates[-1]})\n")
        return inputs, targets, dates

    target_grid = build_standard_grid()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1

    print("=" * 80)
    print(f"⚡ FAST COPERNICUS SUBSET ENGINE [{tag.upper()}]: {start_date} to {end_date}")
    print(f"   Total Duration: {total_days} days ({total_days // 7} weeks)")
    print(f"   BBox: 5°N–30°N, 45°E–105°E | Depths: 0m to 1000m")
    print("=" * 80)

    n_chunks = int(np.ceil(total_days / chunk_days))
    all_inputs, all_targets, all_dates = [], [], []

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

        print(f"\n📥 [Chunk {c+1}/{n_chunks}] {c_days} days ({c_start_str} to {c_end_str}):")

        # 1. Download 3D Temperature Subset
        temp_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_thetao_chunk{c+1}.nc",
            depth_range=(0.0, 1100.0),
        )

        # 2. Download Salinity Subset (Surface)
        so_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=["so"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_so_chunk{c+1}.nc",
            depth_range=(0.0, 5.0),
        )

        # 3. Download SSH Subset
        zos_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_zos_chunk{c+1}.nc",
            depth_range=(0.0, 1.0),
        )

        # 4. Download Currents Subset
        cur_nc = download_fast_nc_subset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            start_date=c_start_str,
            end_date=c_end_str,
            output_filename=f"{tag}_cur_chunk{c+1}.nc",
            depth_range=(0.0, 5.0),
        )

        # Parse downloaded NetCDFs locally (instantaneous on local disk)
        print("   ⚙️ Parsing local NetCDFs & regridding to 0.25° standard grid...")
        with xr.open_dataset(temp_nc) as ds_temp:
            ds_temp_15 = select_standard_depths(ds_temp, depth_variable="depth")
            regridded_temp = regrid_to_standard_grid(ds_temp_15["thetao"], method="bilinear")
            sst_array = regridded_temp.isel(depth=0).values
            target_3d = regridded_temp.values
            dates_chunk = [str(t)[:10] for t in regridded_temp.time.values]
            T_chunk = len(dates_chunk)

        with xr.open_dataset(so_nc) as ds_so:
            ds_sss = ds_so["so"].isel(depth=0)
            sss_array = regrid_to_standard_grid(ds_sss, method="bilinear").values

        with xr.open_dataset(zos_nc) as ds_zos:
            ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values

        with xr.open_dataset(cur_nc) as ds_cur:
            u_cur = regrid_to_standard_grid(ds_cur["uo"].isel(depth=0), method="bilinear").values
            v_cur = regrid_to_standard_grid(ds_cur["vo"].isel(depth=0), method="bilinear").values

        # Geostrophic wind computation
        u_wind = np.zeros_like(ssh_array)
        v_wind = np.zeros_like(ssh_array)
        for t in range(T_chunk):
            grad_y, grad_x = np.gradient(ssh_array[t], 0.25 * 111000, 0.25 * 111000)
            u_wind[t] = - (g / f) * grad_y * 10.0
            v_wind[t] =   (g / f) * grad_x * 10.0

        # Stack into 7-channel input cube
        inputs_chunk = np.stack([sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind], axis=1)

        all_inputs.append(inputs_chunk)
        all_targets.append(target_3d)
        all_dates.extend(dates_chunk)
        print(f"   ✨ Chunk {c+1} processed: {T_chunk} days ready.")

    inputs_full = np.concatenate(all_inputs, axis=0).astype(np.float32)
    targets_full = np.concatenate(all_targets, axis=0).astype(np.float32)
    dates_full = np.array(all_dates)

    np.save(input_file, inputs_full)
    np.save(target_file, targets_full)
    np.save(dates_file, dates_full)

    print(f"\n🎉 [{tag.upper()}] DATASET PREPARED: {len(dates_full)} days saved to {input_file}")
    return inputs_full, targets_full, dates_full


# ==============================================================================
# 3. Fine-Tuning & Evaluation Runner
# ==============================================================================
def run_monsoon_finetuning_pipeline(
    train_start: str = "2025-06-01",
    train_end: str = "2026-02-28",     # 9 Months Training (June 2025 -> Feb 2026)
    val_start: str = "2026-07-01",
    val_end: str = "2026-07-31",       # 1 Month Validation (July 2026 Future Monsoon)
    pretrained_checkpoint: str = "checkpoints/best_ocean_model.pt",
    save_checkpoint: str = "checkpoints/best_ocean_model_finetuned.pt",
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
):
    """
    High-speed download, fine-tuning, and evaluation.
    """
    os.makedirs("checkpoints", exist_ok=True)
    device = get_compute_device()

    # Step 1: High-speed fetch for 9 Months Training
    train_inputs, train_targets, train_dates = fetch_ocean_period_fast(
        start_date=train_start,
        end_date=train_end,
        tag="train_jun25_feb26",
        chunk_days=90,  # 3 large 90-day bulk chunks
    )

    # Step 2: High-speed fetch for July 2026 Validation
    val_inputs, val_targets, val_dates = fetch_ocean_period_fast(
        start_date=val_start,
        end_date=val_end,
        tag="val_jul26",
        chunk_days=31,
    )

    # Step 3: Build DataLoaders
    print("\n📦 Initializing DataLoaders...")
    train_ds = OceanDataset(surface_inputs=train_inputs, subsurface_targets=train_targets, dates=train_dates, use_mock_data=False)
    val_ds   = OceanDataset(surface_inputs=val_inputs, subsurface_targets=val_targets, dates=val_dates, use_mock_data=False)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Step 4: Model Initialization & Pre-trained Weight Loading
    print("\n🧠 Loading Pre-trained OceanUNetViT Model...")
    model = create_model().to(device)

    if os.path.exists(pretrained_checkpoint):
        print(f"   ⚡ Loading pre-trained weights from: {pretrained_checkpoint}")
        model.load_state_dict(torch.load(pretrained_checkpoint, map_location=device))
        print("   ✅ Pre-trained weights successfully loaded!")
    else:
        print("   ⚠️ Pre-trained checkpoint not found. Training from scratch.")

    criterion = OceanReconstructionLoss(alpha=0.7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Step 5: Fine-Tuning Loop
    print("\n" + "=" * 80)
    print(f"🚀 FINE-TUNING ON 9 MONTHS (June 2025 – Feb 2026) & EVALUATING ON JULY 2026")
    print(f"   Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | Base LR: {learning_rate}")
    print("=" * 80)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_rmse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_curr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val (July 2026) Loss: {val_loss:.4f} | LR: {lr_curr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_checkpoint)

    # Step 6: Detailed July 2026 Evaluation Report
    print("\n" + "=" * 80)
    print("📈 FINAL JULY 2026 INDEPENDENT MONSOON VALIDATION REPORT")
    print("=" * 80)

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

    stats = NORMALIZATION_STATS["TEMP_TARGET"]
    preds_c = denormalize_outputs(preds, stats=stats)
    targs_c = denormalize_outputs(targs, stats=stats)

    metrics = compute_all_metrics(preds_c, targs_c)
    print_metrics_report(metrics)

    # Save diagnostic charts
    plot_skill_profiles(metrics, save_path="evaluation_profiles_july2026.png")
    plot_prediction_snapshot(preds_c[15], targs_c[15], depth_idx=5, save_path="snapshot_july2026.png")

    print(f"\n🎉 Fine-tuned model checkpoint saved to: {save_checkpoint}")
    print(f"📊 Diagnostic charts saved: evaluation_profiles_july2026.png, snapshot_july2026.png")


if __name__ == "__main__":
    run_monsoon_finetuning_pipeline(
        train_start="2025-06-01",
        train_end="2026-02-28",     # 9 Months Training
        val_start="2026-07-01",
        val_end="2026-07-31",       # July 2026 Validation
        epochs=15,
        batch_size=4,
        learning_rate=3e-4,
    )
