"""""
================================================================================
OceanEmbed - 12-Month Extreme Climatic Anomaly Training Pipeline (train_v4_extended.py)
================================================================================
Trains on top of best_ocean_model_v4.pt using 12 distinct historical anomaly months:
  1.  Nov 2016 - Record Negative IOD (-IOD)
  2.  Jun 2020 - Super Cyclone Amphan & SW Monsoon onset
  3.  Oct 2019 - Record Super Positive IOD (+IOD)
  4.  May 2021 - Pre-Monsoon Extreme Cyclones Tauktae & Yaas
  5.  Dec 2015 - Super El Niño Peak / Basin-Wide Warming
  6.  May 2010 - Pre-Monsoon Super Heatwave & Warm Pool Expansion
  7.  Aug 2018 - Kerala Extreme Monsoon & Strong Somali Jet Upwelling
  8.  May 2019 - Extremely Severe Cyclonic Storm Fani
  9.  Nov 2017 - Severe Cyclone Ockhi Rapid Intensification
  10. Aug 2008 - Strong Positive IOD Summer Upwelling
  11. Jul 2013 - Intense Southwest Monsoon Deep Active Phase
  12. Jan 2012 - Strong La Niña Winter Northeast Monsoon Convection
================================================================================
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from datetime import datetime
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
from model import create_model
from train import get_compute_device
from data_loader import OceanDataset

ANOMALY_PERIODS = [
    ("nov16", "2016-11-01", "2016-11-30", "Nov 2016: Record Negative IOD"),
    ("jun20", "2020-06-01", "2020-06-30", "Jun 2020: Super Cyclone Amphan"),
    ("oct19", "2019-10-01", "2019-10-31", "Oct 2019: Record Positive IOD"),
    ("may21", "2021-05-01", "2021-05-31", "May 2021: Extreme Cyclones Yaas/Tauktae"),
    ("dec15", "2015-12-01", "2015-12-31", "Dec 2015: Super El Niño Peak"),
    ("may10", "2010-05-01", "2010-05-31", "May 2010: Pre-Monsoon Warm Pool Peak"),
    ("aug18", "2018-08-01", "2018-08-31", "Aug 2018: Kerala Monsoon & Somali Jet"),
    ("may19", "2019-05-01", "2019-05-31", "May 2019: Super Cyclone Fani"),
    ("nov17", "2017-11-01", "2017-11-30", "Nov 2017: Cyclone Ockhi"),
    ("aug08", "2008-08-01", "2008-08-31", "Aug 2008: Strong Positive IOD"),
    ("jul13", "2013-07-01", "2013-07-31", "Jul 2013: Active Monsoon Break Cycle"),
    ("jan12", "2012-01-01", "2012-01-31", "Jan 2012: La Niña Winter Convection"),
]

def download_and_process_month(tag, s_date, e_date, name, save_dir="./data"):
    os.makedirs(save_dir, exist_ok=True)
    download_dir = os.path.join(save_dir, "nc_downloads")
    os.makedirs(download_dir, exist_ok=True)

    f_in = os.path.join(save_dir, f"anomaly_{tag}_inputs_12ch.npy")
    f_tg = os.path.join(save_dir, f"anomaly_{tag}_targets_15d.npy")
    f_dt = os.path.join(save_dir, f"anomaly_{tag}_dates.npy")

    if os.path.exists(f_in) and os.path.exists(f_tg) and os.path.exists(f_dt):
        print(f"📦 [{tag.upper()}] Loaded cached anomaly tensors: {name}")
        return np.load(f_in), np.load(f_tg), np.load(f_dt)

    print(f"\n🌐 [{tag.upper()}] Fetching GLORYS reanalysis for {name} ({s_date} to {e_date})...")
    reanalysis_id = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    target_grid = build_standard_grid()

    temp_nc = os.path.join(download_dir, f"{tag}_thetao_3d.nc")
    if not os.path.exists(temp_nc):
        copernicusmarine.subset(
            dataset_id=reanalysis_id,
            variables=["thetao"],
            minimum_latitude=BBOX["min_lat"] - 0.5,
            maximum_latitude=BBOX["max_lat"] + 0.5,
            minimum_longitude=BBOX["min_lon"] - 0.5,
            maximum_longitude=BBOX["max_lon"] + 0.5,
            minimum_depth=0.0,
            maximum_depth=1050.0,
            start_datetime=f"{s_date}T00:00:00",
            end_datetime=f"{e_date}T23:59:59",
            output_directory=download_dir,
            output_filename=f"{tag}_thetao_3d.nc",
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

    print(f"   ⚙️ Regridding & extracting physical feature cubes...")
    with xr.open_dataset(temp_nc) as ds_temp:
        ds_t_15 = ds_temp["thetao"].sel(depth=STANDARD_DEPTH_LEVELS_M, method="nearest")
        regridded_targets = regrid_to_standard_grid(ds_t_15, method="bilinear")
        targets = regridded_targets.values.astype(np.float32)
        sst_array = targets[:, 0, :, :]
        dates = np.array([str(t)[:10] for t in regridded_targets.time.values])
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
    np.save(f_tg, targets)
    np.save(f_dt, dates)
    print(f"   ✅ Saved {T} anomaly days for [{tag.upper()}]")
    return inputs_12ch, targets, dates


def build_grand_anomaly_catalog():
    all_inputs, all_targets, all_dates = [], [], []
    print("=" * 95)
    print("📦 BUILDING GRAND 12-MONTH HISTORICAL EXTREME CLIMATIC ANOMALY DATASET")
    print("=" * 95)
    for tag, s_d, e_d, name in ANOMALY_PERIODS:
        in_t, tg_t, dt_t = download_and_process_month(tag, s_d, e_d, name)
        all_inputs.append(in_t)
        all_targets.append(tg_t)
        all_dates.append(dt_t)

    grand_inputs = np.concatenate(all_inputs, axis=0).astype(np.float32)
    grand_targets = np.concatenate(all_targets, axis=0).astype(np.float32)
    grand_dates = np.concatenate(all_dates, axis=0)

    print("\n" + "=" * 95)
    print(f"🌊 Grand Anomaly Training Catalog: {len(grand_dates)} Total Anomaly Days")
    print(f"   Inputs:  {grand_inputs.shape}  (12 Surface Channels)")
    print(f"   Targets: {grand_targets.shape} (15 Subsurface Depths)")
    print("=" * 95)
    return grand_inputs, grand_targets, grand_dates


class PhysicsStratificationLoss(nn.Module):
    def __init__(self, lambda_mono=0.08, lambda_grad=0.25):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_grad = lambda_grad
        self.mse = nn.MSELoss()
        
        layer_weights = torch.tensor([
            1.0, 1.0, 1.0, 1.2, 1.5,
            2.5, 3.0, 3.5, 3.0, 2.5,
            1.8, 1.2, 1.0, 1.0, 1.0
        ], dtype=torch.float32)
        self.register_buffer("layer_weights", (layer_weights / layer_weights.mean()).view(1, 15, 1, 1))

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        recon_loss = (diff_sq * self.layer_weights).mean()
        pred_grad = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        targ_grad = target[:, 1:, :, :] - target[:, :-1, :, :]
        grad_loss = self.mse(pred_grad, targ_grad)
        mono_loss = torch.relu(pred[:, 1:, :, :] - pred[:, :-1, :, :]).mean()
        return recon_loss + self.lambda_grad * grad_loss + self.lambda_mono * mono_loss


def train_v4_extended(
    epochs: int = 25,
    batch_size: int = 4,
    learning_rate: float = 4e-5,
    warm_start_ckpt: str = "checkpoints/best_ocean_model_v4.pt",
    save_ckpt: str = "checkpoints/best_ocean_model_v4_extended.pt",
):
    device = get_compute_device()
    grand_inputs, grand_targets, grand_dates = build_grand_anomaly_catalog()

    train_ds = OceanDataset(surface_inputs=grand_inputs, subsurface_targets=grand_targets, dates=grand_dates, use_mock_data=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print("\n🧠 Initializing OceanUNetViT (12 Channels -> 15 Depths)... ")
    model = create_model(in_channels=12, out_depth_levels=15).to(device)

    if os.path.exists(warm_start_ckpt):
        print(f"   🔥 Warm-starting model weights from {warm_start_ckpt}...")
        model.load_state_dict(torch.load(warm_start_ckpt, map_location=device), strict=False)
        print("   ✅ Successfully loaded pre-trained v4 weights!")

    criterion = PhysicsStratificationLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float("inf")
    print(f"\n🚀 STARTING v4_EXTENDED TRAINING PASS ({epochs} Epochs on {device})...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = len(train_loader)

        for b_x, b_y in train_loader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(b_x)
            loss = criterion(preds, b_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / max(1, n_batches)
        scheduler.step()
        lr_curr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Extended Anomaly Loss: {epoch_loss:.5f} | LR: {lr_curr:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_ckpt)
            print(f"   ⭐ Saved best v4_extended checkpoint: {save_ckpt} (Loss: {best_loss:.5f})")

    print(f"\n🎉 Training Complete! Final checkpoint saved to: {save_ckpt}")


if __name__ == "__main__":
    train_v4_extended()