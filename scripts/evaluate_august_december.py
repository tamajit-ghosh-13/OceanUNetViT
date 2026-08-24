"""
================================================================================
OceanEmbed - Multi-Period Cross-Model Comparative Evaluator
================================================================================
Evaluates and compares both:
  1. Fine-tuned Baseline Model (7 channels): checkpoints/best_ocean_model_finetuned.pt
  2. OceanUNetViT v3 Unbiased Model (11 channels): checkpoints/best_ocean_model_v3_unbiased.pt

Across 2 Independent Target Periods pulled directly from Copernicus:
  - Period A: August 2026 (Peak Southwest Monsoon - Future Test)
  - Period B: December 2024 (Northeast Winter Monsoon - Historical Out-of-Sample)
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
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime, timedelta
import copernicusmarine

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
from preprocessing.normalize import denormalize_outputs, preprocess_inputs
from model import create_model
from train import get_compute_device
from evaluate import compute_all_metrics, print_metrics_report


# ==============================================================================
# 1. Fast Subsetting & Dataset Builder (Builds both 7-ch and 11-ch inputs)
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


def get_or_download_eval_data(
    period_tag: str,
    start_date: str,
    end_date: str,
    save_dir: str = "./data",
):
    """
    Downloads and caches:
      - 7-channel input array
      - 11-channel input array
      - 15-depth ground truth targets
    """
    os.makedirs(save_dir, exist_ok=True)
    f_7ch = os.path.join(save_dir, f"eval_{period_tag}_inputs_7ch.npy")
    f_11ch = os.path.join(save_dir, f"eval_{period_tag}_inputs_11ch.npy")
    f_targs = os.path.join(save_dir, f"eval_{period_tag}_targets.npy")
    f_dates = os.path.join(save_dir, f"eval_{period_tag}_dates.npy")

    if os.path.exists(f_7ch) and os.path.exists(f_11ch) and os.path.exists(f_targs) and os.path.exists(f_dates):
        print(f"📦 Found cached dataset for [{period_tag.upper()}]:")
        inputs_7ch = np.load(f_7ch)
        inputs_11ch = np.load(f_11ch)
        targets = np.load(f_targs)
        dates = np.load(f_dates)
        print(f"   ✅ Loaded {len(dates)} days ({dates[0]} to {dates[-1]})\n")
        return inputs_7ch, inputs_11ch, targets, dates

    print("=" * 80)
    print(f"📥 FETCHING FROM COPERNICUS FOR [{period_tag.upper()}]: {start_date} to {end_date}")
    print("=" * 80)

    download_dir = os.path.join(save_dir, "nc_downloads")
    target_grid = build_standard_grid()

    # 1. Download NetCDF subsets
    temp_nc = download_fast_nc_subset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        start_date=start_date,
        end_date=end_date,
        output_filename=f"{period_tag}_thetao.nc",
        depth_range=(0.0, 1100.0),
        download_dir=download_dir,
    )
    so_nc = download_fast_nc_subset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=["so"],
        start_date=start_date,
        end_date=end_date,
        output_filename=f"{period_tag}_so.nc",
        depth_range=(0.0, 5.0),
        download_dir=download_dir,
    )
    zos_nc = download_fast_nc_subset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["zos"],
        start_date=start_date,
        end_date=end_date,
        output_filename=f"{period_tag}_zos.nc",
        depth_range=(0.0, 1.0),
        download_dir=download_dir,
    )
    cur_nc = download_fast_nc_subset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        start_date=start_date,
        end_date=end_date,
        output_filename=f"{period_tag}_cur.nc",
        depth_range=(0.0, 5.0),
        download_dir=download_dir,
    )

    print("   ⚙️ Parsing and regridding to 0.25° grid...")
    with xr.open_dataset(temp_nc) as ds_temp:
        ds_temp_15 = select_standard_depths(ds_temp, depth_variable="depth")
        regridded_temp = regrid_to_standard_grid(ds_temp_15["thetao"], method="bilinear")
        sst_array = regridded_temp.isel(depth=0).values
        targets = regridded_temp.values.astype(np.float32)
        dates = np.array([str(t)[:10] for t in regridded_temp.time.values])
        T = len(dates)

    with xr.open_dataset(so_nc) as ds_so:
        sss_array = regrid_to_standard_grid(ds_so["so"].isel(depth=0), method="bilinear").values

    with xr.open_dataset(zos_nc) as ds_zos:
        ssh_array = regrid_to_standard_grid(ds_zos["zos"], method="bilinear").values

    with xr.open_dataset(cur_nc) as ds_cur:
        u_cur = regrid_to_standard_grid(ds_cur["uo"].isel(depth=0), method="bilinear").values
        v_cur = regrid_to_standard_grid(ds_cur["vo"].isel(depth=0), method="bilinear").values

    omega = 7.2921e-5
    lat_rad = np.deg2rad(target_grid["lat"][:, None])
    f = 2 * omega * np.sin(lat_rad)
    f = np.where(np.abs(f) < 1e-5, 1e-5 * np.sign(f), f)
    g = 9.81

    u_wind = np.zeros_like(ssh_array)
    v_wind = np.zeros_like(ssh_array)
    for t in range(T):
        grad_y, grad_x = np.gradient(ssh_array[t], 0.25 * 111000, 0.25 * 111000)
        u_wind[t] = - (g / f) * grad_y * 10.0
        v_wind[t] =   (g / f) * grad_x * 10.0

    # 7-channel cube
    inputs_7ch = np.stack([sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind], axis=1).astype(np.float32)

    # 11-channel features
    doy = np.array([datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday for d in dates])
    doy_sin = np.sin(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T, GRID_LAT_SIZE, GRID_LON_SIZE))
    doy_cos = np.cos(2 * np.pi * doy / 365.0)[:, None, None] * np.ones((T, GRID_LAT_SIZE, GRID_LON_SIZE))
    sst_anomaly = sst_array - 28.5
    mld_proxy = np.abs(ssh_array) * (np.maximum(sst_array, 20.0) - 20.0) / (np.maximum(sss_array, 30.0) - 29.0)
    mld_proxy = np.clip(mld_proxy, 0.0, 5.0)

    inputs_11ch = np.stack([
        sst_array, sss_array, ssh_array, u_cur, v_cur, u_wind, v_wind,
        doy_sin, doy_cos, sst_anomaly, mld_proxy
    ], axis=1).astype(np.float32)

    np.save(f_7ch, inputs_7ch)
    np.save(f_11ch, inputs_11ch)
    np.save(f_targs, targets)
    np.save(f_dates, dates)

    print(f"🎉 Saved {period_tag.upper()} datasets ({T} days) to disk.")
    return inputs_7ch, inputs_11ch, targets, dates


# ==============================================================================
# 2. Model Inference & Evaluation Engine
# ==============================================================================
def run_model_inference(
    model: torch.nn.Module,
    inputs: np.ndarray,
    is_v3: bool,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Runs model forward pass on raw inputs with full sanitization and denormalization.
    """
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

    preds_norm = np.concatenate(all_preds, axis=0)  # (T, 15, 101, 241)

    # Denormalize to °C
    if is_v3:
        preds_c = denormalize_outputs(preds_norm, stats=TEMP_TARGET_STATS_PER_DEPTH)
    else:
        preds_c = denormalize_outputs(preds_norm, stats=NORMALIZATION_STATS["TEMP_TARGET"])

    return preds_c


def print_side_by_side_comparison(
    period_name: str,
    dates: np.ndarray,
    targets: np.ndarray,
    preds_finetuned: Optional[np.ndarray],
    preds_v3: Optional[np.ndarray],
):
    """
    Prints a clear, side-by-side comparative table for Hackathon review.
    """
    print("\n" + "=" * 115)
    print(f"📊 COMPARATIVE VALIDATION REPORT: {period_name.upper()} ({dates[0]} to {dates[-1]}, {len(dates)} Days)")
    print("=" * 115)
    print(f"{'Depth (m)':>10} | {'True Avg (°C)':>14} | {'Finetuned (7-ch)':>20} | {'v3 Unbiased (11-ch)':>20} | {'Winner':>14}")
    print(f"{'':>10} | {'':>14} | {'RMSE / Corr':>20} | {'RMSE / Corr':>20} | {'':>14}")
    print("-" * 115)

    metrics_ft = compute_all_metrics(preds_finetuned, targets) if preds_finetuned is not None else None
    metrics_v3 = compute_all_metrics(preds_v3, targets) if preds_v3 is not None else None

    for d_idx, depth_m in enumerate(STANDARD_DEPTH_LEVELS_M):
        t_slice = targets[:, d_idx]
        valid = ~np.isnan(t_slice)
        true_mean = float(np.mean(t_slice[valid])) if valid.any() else 0.0

        str_ft = f"{metrics_ft['rmse'][d_idx]:.3f}°C (r={metrics_ft['correlation'][d_idx]:.3f})" if metrics_ft else "N/A"
        str_v3 = f"{metrics_v3['rmse'][d_idx]:.3f}°C (r={metrics_v3['correlation'][d_idx]:.3f})" if metrics_v3 else "N/A"

        winner = "v3 Unbiased 🏆" if (metrics_v3 and metrics_ft and metrics_v3['rmse'][d_idx] < metrics_ft['rmse'][d_idx]) else "Finetuned"
        print(f"{depth_m:>10d} | {true_mean:>12.2f}°C | {str_ft:>20} | {str_v3:>20} | {winner:>14}")

    print("-" * 115)
    mean_rmse_ft = f"{metrics_ft['rmse'].mean():.4f}°C" if metrics_ft else "N/A"
    mean_rmse_v3 = f"{metrics_v3['rmse'].mean():.4f}°C" if metrics_v3 else "N/A"
    mean_corr_ft = f"r={metrics_ft['correlation'].mean():.4f}" if metrics_ft else "N/A"
    mean_corr_v3 = f"r={metrics_v3['correlation'].mean():.4f}" if metrics_v3 else "N/A"

    print(f"{'OVERALL':>10} | {'-':>14} | {mean_rmse_ft:>9} ({mean_corr_ft}) | {mean_rmse_v3:>9} ({mean_corr_v3}) | {'-':>14}")
    print("=" * 115 + "\n")


# ==============================================================================
# 3. Main Evaluation Runner
# ==============================================================================
def run_full_cross_evaluation():
    device = get_compute_device()

    # Load Model 1: Finetuned (7 channels)
    ckpt_ft = "checkpoints/best_ocean_model_finetuned.pt"
    model_ft = None
    if os.path.exists(ckpt_ft):
        print(f"🧠 Loading Finetuned Model (7-ch) from {ckpt_ft}...")
        model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
        # Use strict=False so older checkpoints without depth_bias load cleanly
        model_ft.load_state_dict(torch.load(ckpt_ft, map_location=device), strict=False)
        model_ft.eval()
        print("   ✅ Loaded Finetuned Model (7-ch)")
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_ft}")

    # Load Model 2: v3 Unbiased (11 channels)
    ckpt_v3 = "checkpoints/best_ocean_model_v3_unbiased.pt"
    model_v3 = None
    if os.path.exists(ckpt_v3):
        print(f"🧠 Loading OceanUNetViT v3 (11-ch) from {ckpt_v3}...")
        model_v3 = create_model(in_channels=11, out_depth_levels=15).to(device)
        model_v3.load_state_dict(torch.load(ckpt_v3, map_location=device), strict=False)
        model_v3.eval()
        print("   ✅ Loaded OceanUNetViT v3 (11-ch)")
    else:
        print(f"⚠️ Checkpoint not found: {ckpt_v3} (Run train_monsoon_finetune.py first if you want to compare v3)")

    # Target Evaluation Periods
    periods = [
        ("aug26", "2026-08-01", "2026-08-31", "August 2026 (Peak Southwest Monsoon)"),
        ("dec24", "2024-12-01", "2024-12-31", "December 2024 (Northeast Winter Monsoon)"),
    ]

    for p_tag, start_d, end_d, p_name in periods:
        inputs_7ch, inputs_11ch, targets, dates = get_or_download_eval_data(
            period_tag=p_tag,
            start_date=start_d,
            end_date=end_d,
        )

        preds_ft = run_model_inference(model_ft, inputs_7ch, is_v3=False, device=device) if model_ft else None
        preds_v3 = run_model_inference(model_v3, inputs_11ch, is_v3=True, device=device) if model_v3 else None

        print_side_by_side_comparison(
            period_name=p_name,
            dates=dates,
            targets=targets,
            preds_finetuned=preds_ft,
            preds_v3=preds_v3,
        )


if __name__ == "__main__":
    run_full_cross_evaluation()
