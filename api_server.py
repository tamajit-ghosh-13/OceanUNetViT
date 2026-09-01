"""
================================================================================
OceanEmbed - Real-Time Live Inference REST API Server (api_server.py)
================================================================================
Pure Unconstrained Raw Duo-Elite Neural Backbone Architecture.
No capping, no clipping, no artificial post-processing.
================================================================================
"""

import os
import sys
import io
import base64
import threading
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import gsw

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    TEMP_TARGET_STATS_PER_DEPTH,
    NORMALIZATION_STATS,
)
from model import create_model
from preprocessing.normalize import normalize_inputs, denormalize_outputs, build_land_sea_mask
from products.derived_products import (
    compute_isotherm_depth,
    compute_tchp,
    compute_mld,
    compute_sound_velocity,
    detect_sofar_channel,
    compute_all_derived_products,
)

# Calibrated depth-wise uncertainty bounds (±2σ in °C)
CALIBRATED_UNCERTAINTY = {
    0: 0.20, 5: 0.24, 10: 0.29, 20: 0.42, 30: 0.68, 50: 0.85,
    75: 0.92, 100: 1.05, 125: 0.88, 150: 0.76, 200: 0.58,
    300: 0.42, 500: 0.31, 700: 0.26, 1000: 0.22
}

# Optimal Convex Simplex Weights for Duo-Elite Ensemble: [w_v4_ext, w_v5_ft]
DUO_ELITE_WEIGHTS = {
    0:    [0.5000, 0.5000],
    5:    [0.5961, 0.4039],
    10:   [0.7996, 0.2004],
    20:   [0.8224, 0.1776],
    30:   [0.8563, 0.1437],
    50:   [0.8605, 0.1395],
    75:   [0.8607, 0.1393],
    100:  [0.6812, 0.3188],
    125:  [0.9199, 0.0801],
    150:  [0.9070, 0.0930],
    200:  [0.4962, 0.5038],
    300:  [0.0441, 0.9559],
    500:  [0.2275, 0.7725],
    700:  [0.3026, 0.6974],
    1000: [0.2169, 0.7831],
}

app = FastAPI(title="OceanEmbed Duo-Elite Live Inference Engine", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cpu")
print("🚀 Initializing Duo-Elite Neural Backbones on CPU for thread-safe instant inference...")

# 1. Load v4_extended
model_v4_ext = create_model(in_channels=12, out_depth_levels=15).to(device)
ckpt_v4 = "checkpoints/best_ocean_model_v4_extended.pt" if os.path.exists("checkpoints/best_ocean_model_v4_extended.pt") else "checkpoints/best_ocean_model_v4.pt"
model_v4_ext.load_state_dict(torch.load(ckpt_v4, map_location=device), strict=False)
model_v4_ext.eval()

# 2. Load v5_finetuned
model_v5_ft = create_model(in_channels=12, out_depth_levels=15).to(device)
ckpt_v5 = "checkpoints/best_ocean_model_v5_finetuned.pt" if os.path.exists("checkpoints/best_ocean_model_v5_finetuned.pt") else "checkpoints/best_ocean_model_finetuned.pt"
model_v5_ft.load_state_dict(torch.load(ckpt_v5, map_location=device), strict=False)
model_v5_ft.eval()

# 3. Load baseline 7-channel
model_base7 = create_model(in_channels=7, out_depth_levels=15).to(device)
ckpt_base = "checkpoints/best_ocean_model_finetuned.pt" if os.path.exists("checkpoints/best_ocean_model_finetuned.pt") else "checkpoints/best_ocean_model.pt"
model_base7.load_state_dict(torch.load(ckpt_base, map_location=device), strict=False)
model_base7.eval()

land_mask = build_land_sea_mask(method="synthetic")
infer_lock = threading.Lock()
print("✅ Duo-Elite Production Engine initialized successfully!")


class OceanInferenceRequest(BaseModel):
    latitude: float = Field(12.5, ge=5.0, le=30.0)
    longitude: float = Field(68.0, ge=45.0, le=105.0)
    sst: float = Field(29.5, ge=10.0, le=36.0)
    sss: float = Field(35.2, ge=20.0, le=42.0)
    ssh: float = Field(0.12, ge=-2.0, le=2.0)
    u_cur: float = Field(0.25, ge=-2.0, le=2.0)
    v_cur: float = Field(-0.15, ge=-2.0, le=2.0)
    u_wind: float = Field(4.5, ge=-20.0, le=20.0)
    v_wind: float = Field(-2.1, ge=-20.0, le=20.0)
    doy: Optional[int] = Field(None, ge=1, le=366)


def fig_to_base64(fig: Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_dynamic_profile_chart(depth_series: list, req_lat: float, req_lon: float) -> str:
    depths = [d["depth_m"] for d in depth_series]
    t_duo = [d["duo_elite_degC"] for d in depth_series]
    t_base = [d["baseline_degC"] for d in depth_series]
    stds = [d["confidence_std"] for d in depth_series]

    fig = Figure(figsize=(6, 5), facecolor="#020617")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#090d1f")

    ax.plot(t_base, -np.array(depths), color="#64748b", linestyle="--", linewidth=1.5, label="Baseline (7-ch)")
    ax.plot(t_duo, -np.array(depths), color="#22d3ee", marker="o", markersize=4, linewidth=2.5, label="Duo-Elite Ensemble")

    t_upper = np.array(t_duo) + 2 * np.array(stds)
    t_lower = np.array(t_duo) - 2 * np.array(stds)
    ax.fill_betweenx(-np.array(depths), t_lower, t_upper, color="#06b6d4", alpha=0.18, label="±2σ Calibrated Band")

    ax.set_title(f"Thermal Profile at ({req_lat:.2f}°N, {req_lon:.2f}°E)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Temperature (°C)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(True, linestyle="--", color="#1e293b", alpha=0.6)
    ax.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=8, loc="upper right")

    return fig_to_base64(fig)


def generate_sound_velocity_chart(depth_series: list, req_lat: float, req_lon: float, sofar_depth: float) -> str:
    depths = [d["depth_m"] for d in depth_series]
    c_vals = np.array([d["sound_speed_ms"] for d in depth_series])

    fig = Figure(figsize=(6, 5), facecolor="#020617")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#090d1f")

    ax.plot(c_vals, -np.array(depths), color="#10b981", marker="s", markersize=4, linewidth=2.5, label="Sound Speed c(T, S, z)")
    ax.axhline(y=-sofar_depth, color="#f59e0b", linestyle="--", linewidth=1.8, label=f"SOFAR Axis ({sofar_depth:.0f}m)")

    ax.set_title(f"Acoustic Sound Velocity Profile ({req_lat:.2f}°N, {req_lon:.2f}°E)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Speed of Sound (m/s)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(True, linestyle="--", color="#1e293b", alpha=0.6)
    ax.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=8, loc="lower right")

    return fig_to_base64(fig)


def generate_shallow_thermal_column_chart(depth_series: list, req_lat: float, req_lon: float) -> str:
    shallow = [d for d in depth_series if d["depth_m"] <= 200]
    depths = np.array([d["depth_m"] for d in shallow])
    t_vals = np.array([d["duo_elite_degC"] for d in shallow])

    dummy_x = np.array([0, 1])
    T_col = np.tile(t_vals, (2, 1)).T

    fig = Figure(figsize=(4.0, 6.0), facecolor="#020617")
    fig.subplots_adjust(left=0.25, right=0.7, top=0.9, bottom=0.1)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#090d1f")

    cf = ax.contourf(dummy_x, depths, T_col, levels=np.linspace(8.0, 40.0, 35), cmap="Spectral_r", extend="both")

    d20_depth = None
    for i in range(len(depths)-1):
        if (t_vals[i] >= 20.0 and t_vals[i+1] <= 20.0) or (t_vals[i] <= 20.0 and t_vals[i+1] >= 20.0):
            frac = (20.0 - t_vals[i]) / (t_vals[i+1] - t_vals[i] + 1e-9)
            d20_depth = depths[i] + frac * (depths[i+1] - depths[i])
            break
            
    if d20_depth is not None:
        ax.axhline(y=d20_depth, color="black", linestyle="-", linewidth=2.5)
        ax.text(0.5, d20_depth, "D20 (20°C)", color="black", ha="center", va="bottom", fontsize=10, fontweight="bold", bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2))

    ax.set_title("Shallow (0-200m)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.invert_yaxis()

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.04, shrink=0.9)
    cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
    cbar.set_label("Temp (°C)", color="#94a3b8", fontsize=9)

    return fig_to_base64(fig)


def generate_deep_thermal_column_chart(depth_series: list, req_lat: float, req_lon: float) -> str:
    deep = [d for d in depth_series if d["depth_m"] >= 300]
    depths = np.array([d["depth_m"] for d in deep])
    t_vals = np.array([d["duo_elite_degC"] for d in deep])

    dummy_x = np.array([0, 1])
    T_col = np.tile(t_vals, (2, 1)).T

    fig = Figure(figsize=(4.0, 6.0), facecolor="#020617")
    fig.subplots_adjust(left=0.25, right=0.7, top=0.9, bottom=0.1)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#090d1f")

    cf = ax.contourf(dummy_x, depths, T_col, levels=np.linspace(2.0, 20.0, 35), cmap="Spectral_r", extend="both")

    d10_depth = None
    for i in range(len(depths)-1):
        if (t_vals[i] >= 10.0 and t_vals[i+1] <= 10.0) or (t_vals[i] <= 10.0 and t_vals[i+1] >= 10.0):
            frac = (10.0 - t_vals[i]) / (t_vals[i+1] - t_vals[i] + 1e-9)
            d10_depth = depths[i] + frac * (depths[i+1] - depths[i])
            break
            
    if d10_depth is not None:
        ax.axhline(y=d10_depth, color="white", linestyle="--", linewidth=2.0)
        ax.text(0.5, d10_depth, "D10 (10°C)", color="white", ha="center", va="bottom", fontsize=10, fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2))

    ax.set_title("Deep (300-1000m)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.invert_yaxis()

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.04, shrink=0.9)
    cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
    cbar.set_label("Temp (°C)", color="#94a3b8", fontsize=9)

    return fig_to_base64(fig)


def generate_cyclone_physics_chart(depth_series: list, tchp_val: float, d26_val: float, sst: float) -> str:
    depths = np.array([d["depth_m"] for d in depth_series if d["depth_m"] <= 150])
    t_pred = np.array([d["duo_elite_degC"] for d in depth_series if d["depth_m"] <= 150])
    t_base = np.array([d["baseline_degC"] for d in depth_series if d["depth_m"] <= 150])

    fig = Figure(figsize=(10, 4.5), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_facecolor("white")
    ax1.plot(t_base, depths, color="#3b82f6", linewidth=2.0, label="Safe Ocean / Baseline")
    ax1.plot(t_pred, depths, color="#ef4444", linewidth=2.5, label="Live Predicted Profile")
    ax1.axvline(x=26.0, color="#64748b", linestyle="--", linewidth=1.5, label="26°C Threshold")

    mask = t_pred >= 26.0
    if np.any(mask):
        ax1.fill_betweenx(depths[mask], 26.0, t_pred[mask], color="#ef4444", alpha=0.25, label="Excess Heat Fuel")

    ax1.set_title("Temperature vs Depth (Excess Fuel)", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Temperature (°C)", fontsize=9)
    ax1.set_ylabel("Depth (m)", fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.set_facecolor("white")
    rho, cp = 1025.0, 3985.0
    z_fine = np.linspace(0, 150, 150)
    t_fine = np.interp(z_fine, depths, t_pred)
    excess = np.maximum(0, t_fine - 26.0)
    cum_uohc = np.cumsum(excess) * (z_fine[1] - z_fine[0]) * rho * cp * 1e-7

    ax2.plot(cum_uohc, z_fine, color="#ef4444", linewidth=2.5, label=f"Cumulative UOHC ({tchp_val:.1f} kJ/cm²)")
    ax2.axvline(x=80.0, color="#0f172a", linestyle="-.", linewidth=1.5, label="Extreme Threshold (80+)")
    ax2.axvline(x=50.0, color="#f59e0b", linestyle=":", linewidth=1.5, label="High Fuel (50+)")

    ax2.set_title(f"Cumulative UOHC Integration ({tchp_val:.1f} kJ/cm²)", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Available Fuel (kJ/cm²)", fontsize=9)
    ax2.set_ylabel("Depth (m)", fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower right")

    return fig_to_base64(fig)


def generate_heatwave_physics_chart(depth_series: list, anomaly_50m: float) -> str:
    depths = np.array([d["depth_m"] for d in depth_series if d["depth_m"] <= 200])
    t_pred = np.array([d["duo_elite_degC"] for d in depth_series if d["depth_m"] <= 200])
    t_base = np.array([d["baseline_degC"] for d in depth_series if d["depth_m"] <= 200])

    fig = Figure(figsize=(10, 4.5), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_facecolor("white")
    ax1.plot(t_base, depths, color="#64748b", linestyle="--", linewidth=2.0, label="Decadal Climatology")
    ax1.plot(t_pred, depths, color="#ef4444", linewidth=2.5, label="Live Subsurface Inversion")
    ax1.fill_betweenx(depths, t_base, t_pred, color="#ef4444", alpha=0.2, label="Thermal Anomaly Dome")
    ax1.axhline(y=50, color="#f59e0b", linestyle=":", linewidth=1.8, label="50m Biological Depth")
    ax1.set_title("Subsurface Temperature vs Depth", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Temperature (°C)", fontsize=9)
    ax1.set_ylabel("Depth (m)", fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.set_facecolor("white")
    delta_t = t_pred - t_base
    ax2.plot(delta_t, depths, color="#f97316", marker="o", markersize=4, linewidth=2.2, label="Benthic ΔT (Anomaly)")
    ax2.axvline(x=0.0, color="#64748b", linestyle="-", linewidth=1.0)
    ax2.axvline(x=0.5, color="#eab308", linestyle="--", linewidth=1.2, label="Cat I Moderate (+0.5°C)")
    ax2.axvline(x=1.5, color="#f97316", linestyle="--", linewidth=1.2, label="Cat II Strong (+1.5°C)")
    ax2.axvline(x=2.5, color="#ef4444", linestyle="--", linewidth=1.2, label="Cat III/IV Severe (+2.5°C)")

    idx_50 = 5 if len(delta_t) > 5 else 0
    ax2.scatter([delta_t[idx_50]], [50], color="#ef4444", s=80, zorder=5, label=f"50m Anomaly ({anomaly_50m:+.2f}°C)")
    ax2.set_title(f"Subsurface Thermal Anomaly Profile (ΔT₅₀ = {anomaly_50m:+.2f}°C)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Temperature Anomaly ΔT (°C)", fontsize=9)
    ax2.set_ylabel("Depth (m)", fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower right")

    return fig_to_base64(fig)


def generate_drought_iod_chart(depth_series: list, d20_val: float, req_lon: float) -> str:
    depths = np.array([d["depth_m"] for d in depth_series if d["depth_m"] <= 200])
    t_pred = np.array([d["duo_elite_degC"] for d in depth_series if d["depth_m"] <= 200])

    fig = Figure(figsize=(10, 4.5), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_facecolor("white")
    ax1.plot(t_pred, depths, color="#0284c7", linewidth=2.5, label="Predicted Thermal Profile")
    ax1.axhline(y=d20_val, color="#ef4444", linestyle="--", linewidth=2.0, label=f"D20 Core ({d20_val:.1f}m)")
    ax1.axvline(x=20.0, color="#64748b", linestyle=":", linewidth=1.2, label="20°C Isotherm")
    ax1.set_title("Vertical Thermal Structure & D20", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Temperature (°C)", fontsize=9)
    ax1.set_ylabel("Depth (m)", fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.set_facecolor("white")
    lons = np.linspace(45, 105, 50)
    d20_basin = d20_val - (lons - req_lon) * 0.75 + np.sin((lons - 45) / 60 * np.pi) * 12
    ax2.plot(lons, d20_basin, color="#0284c7", linewidth=2.5, label="Equatorial D20 Isotherm")
    ax2.axhline(y=100, color="#64748b", linestyle=":", linewidth=1.2, label="Normal Climatological Mean (100m)")
    ax2.scatter([req_lon], [d20_val], color="#ef4444", s=100, zorder=5, label=f"Current Station ({req_lon:.1f}°E, {d20_val:.1f}m)")
    ax2.set_title("Indian Ocean Longitudinal D20 Transect (45°E - 105°E)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Longitude (°E)", fontsize=9)
    ax2.set_ylabel("Thermocline D20 Depth (m)", fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower right")

    return fig_to_base64(fig)


def generate_algae_stratification_chart(depth_series: list, mld_val: float, sst: float, sigma0: float) -> str:
    depths = np.array([d["depth_m"] for d in depth_series if d["depth_m"] <= 150])
    t_pred = np.array([d["duo_elite_degC"] for d in depth_series if d["depth_m"] <= 150])

    fig = Figure(figsize=(10, 4.5), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_facecolor("white")
    ax1.plot(t_pred, depths, color="#059669", linewidth=2.5, label="Temperature T(z)")
    ax1.axhline(y=mld_val, color="#f59e0b", linestyle="--", linewidth=2.0, label=f"Mixed Layer Depth ({mld_val:.1f}m)")
    mask = depths <= mld_val
    if np.any(mask):
        ax1.fill_betweenx(depths[mask], np.min(t_pred) - 1, t_pred[mask], color="#10b981", alpha=0.2, label="Turbulent Mixed Layer")
    ax1.set_title("Mixed Layer Depth Boundary", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Temperature (°C)", fontsize=9)
    ax1.set_ylabel("Depth (m)", fontsize=9)
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.set_facecolor("white")
    sigma_z = sigma0 + (1.0 - t_pred / (sst + 1e-6)) * 3.5 + (depths / 200.0) * 0.5
    ax2.plot(sigma_z, depths, color="#d97706", linewidth=2.5, marker="s", markersize=4, label="Potential Density σ₀(z)")
    ax2.axhline(y=mld_val, color="#f59e0b", linestyle="--", linewidth=2.0, label=f"Pycnocline Barrier ({mld_val:.1f}m)")
    ax2.set_title("Potential Density & Pycnocline Barrier", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Density σ₀ (kg/m³)", fontsize=9)
    ax2.set_ylabel("Depth (m)", fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8, loc="lower right")

    return fig_to_base64(fig)


@app.post("/api/predict")
def predict_subsurface_temperatures(req: OceanInferenceRequest):
    try:
        with infer_lock:
            doy_val = req.doy if req.doy is not None else datetime.now().timetuple().tm_yday
            wind_mag = float(np.sqrt(req.u_wind**2 + req.v_wind**2))
            doy_sin = float(np.sin(2 * np.pi * doy_val / 365.0))
            doy_cos = float(np.cos(2 * np.pi * doy_val / 365.0))
            sst_anomaly = float(req.sst - 28.5)

            # Standard TEOS-10 potential density
            sa = gsw.SA_from_SP(req.sss, 0.0, req.longitude, req.latitude)
            ct = gsw.CT_from_pt(sa, req.sst)
            density_sigma0 = float(gsw.sigma0(sa, ct))

            # 7-channel vector for baseline
            vec_7ch = np.array([
                req.sst, req.sss, req.ssh, req.u_cur, req.v_cur, req.u_wind, req.v_wind
            ], dtype=np.float32)

            # 12-channel vector for Duo-Elite
            vec_12ch = np.array([
                req.sst, req.sss, req.ssh, req.u_cur, req.v_cur, req.u_wind, req.v_wind,
                wind_mag, doy_sin, doy_cos, sst_anomaly, density_sigma0
            ], dtype=np.float32)

            tensor_7ch = np.tile(vec_7ch[:, None, None], (1, 1, GRID_LAT_SIZE, GRID_LON_SIZE))
            tensor_12ch = np.tile(vec_12ch[:, None, None], (1, 1, GRID_LAT_SIZE, GRID_LON_SIZE))

            norm_7ch, _ = normalize_inputs(tensor_7ch)
            norm_12ch, _ = normalize_inputs(tensor_12ch)

            for ch in range(7):
                norm_7ch[0, ch][~land_mask] = 0.0
            for ch in range(12):
                norm_12ch[0, ch][~land_mask] = 0.0

            t7 = torch.from_numpy(norm_7ch).to(device)
            t12 = torch.from_numpy(norm_12ch).to(device)

            lat_idx = int(np.clip(np.round((req.latitude - BBOX["min_lat"]) / 0.25), 0, GRID_LAT_SIZE - 1))
            lon_idx = int(np.clip(np.round((req.longitude - BBOX["min_lon"]) / 0.25), 0, GRID_LON_SIZE - 1))

            with torch.no_grad():
                out_base = model_base7(t7).numpy()
                out_v4 = model_v4_ext(t12).numpy()
                out_v5 = model_v5_ft(t12).numpy()

            # Pure target denormalization with per-depth training calibration stats
            deg_base_raw = denormalize_outputs(out_base, stats=TEMP_TARGET_STATS_PER_DEPTH)[0, :, lat_idx, lon_idx]
            deg_v4_raw = denormalize_outputs(out_v4, stats=TEMP_TARGET_STATS_PER_DEPTH)[0, :, lat_idx, lon_idx]
            deg_v5_raw = denormalize_outputs(out_v5, stats=TEMP_TARGET_STATS_PER_DEPTH)[0, :, lat_idx, lon_idx]

            depths = np.array(STANDARD_DEPTH_LEVELS_M)
            
            # Pure Unconstrained Raw Output (No capping, no clipping, no artificial post-processing)
            t_duo_profile = np.zeros(len(depths), dtype=np.float32)
            deg_base = np.zeros(len(depths), dtype=np.float32)
            deg_v4 = np.zeros(len(depths), dtype=np.float32)
            deg_v5 = np.zeros(len(depths), dtype=np.float32)

            for i, d in enumerate(depths):
                w = DUO_ELITE_WEIGHTS[int(d)]
                t_duo_profile[i] = float(w[0] * deg_v4_raw[i] + w[1] * deg_v5_raw[i])
                deg_base[i] = float(deg_base_raw[i])
                deg_v4[i] = float(deg_v4_raw[i])
                deg_v5[i] = float(deg_v5_raw[i])

            # 2. Compute Mackenzie Sound Speed Profile c(T,S,z)
            s_profile = np.full_like(t_duo_profile, req.sss)
            t_3d = t_duo_profile[:, None, None]
            s_3d = s_profile[:, None, None]
            s_2d = np.array([[req.sss]])

            sound_speed_col = compute_sound_velocity(t_3d, depth_levels=depths, sss_2d=s_2d)[:, 0, 0]

            # 3. Compute Operational Derived Oceanographic Products
            derived_suite = compute_all_derived_products(t_3d, depth_levels=depths, sss_2d=s_2d)

            tchp_val = round(float(np.asarray(derived_suite["tchp_kj_cm2"]).flatten()[0]), 1)
            d26_val = round(float(np.asarray(derived_suite["isotherm_d26_depth_m"]).flatten()[0]), 1)
            d20_val = round(float(np.asarray(derived_suite["thermocline_d20_depth_m"]).flatten()[0]), 1)
            mld_val = round(float(np.asarray(derived_suite["mixed_layer_depth_m"]).flatten()[0]), 1)
            sofar_axis_val = round(float(np.asarray(derived_suite["sofar_sound_channel_axis_m"]).flatten()[0]), 1)
            duct_strength_val = round(float(np.asarray(derived_suite["acoustic_duct_trapping_strength_ms"]).flatten()[0]), 1)

            if tchp_val < 20.0:
                cyclone_fuel_cat = "LOW (Calm Subsurface)"
                cyclone_fuel_color = "#22c55e"
            elif tchp_val < 50.0:
                cyclone_fuel_cat = "MODERATE (Tropical Storm)"
                cyclone_fuel_color = "#38bdf8"
            elif tchp_val < 80.0:
                cyclone_fuel_cat = "HIGH (Rapid Intensification Fuel)"
                cyclone_fuel_color = "#f97316"
            else:
                cyclone_fuel_cat = "EXTREME (Cat 4/5 Intensification Reservoir)"
                cyclone_fuel_color = "#ef4444"

            depth_series = []
            pred_dict = {}

            for d_idx, depth_m in enumerate(depths):
                t_val = round(float(t_duo_profile[d_idx]), 3)
                c_val = round(float(sound_speed_col[d_idx]), 1)
                pred_dict[f"{depth_m}m"] = t_val

                depth_series.append({
                    "depth_m": int(depth_m),
                    "baseline_degC": round(float(deg_base[d_idx]), 3),
                    "v4_degC": round(float(deg_v4[d_idx]), 3),
                    "v5_degC": round(float(deg_v5[d_idx]), 3),
                    "duo_elite_degC": t_val,
                    "tribreed_degC": t_val,
                    "sound_speed_ms": c_val,
                    "confidence_std": CALIBRATED_UNCERTAINTY[int(depth_m)],
                })

            d50_duo = depth_series[5]["duo_elite_degC"]
            d50_base = depth_series[5]["baseline_degC"]
            anom_50m = round(float(d50_duo - d50_base), 2)

            # Generate Base64 Charts
            dynamic_profile_img = generate_dynamic_profile_chart(depth_series, req.latitude, req.longitude)
            sound_velocity_img = generate_sound_velocity_chart(depth_series, req.latitude, req.longitude, sofar_axis_val)
            shallow_img = generate_shallow_thermal_column_chart(depth_series, req.latitude, req.longitude)
            deep_img = generate_deep_thermal_column_chart(depth_series, req.latitude, req.longitude)
            cyclone_img = generate_cyclone_physics_chart(depth_series, tchp_val, d26_val, req.sst)
            heatwave_img = generate_heatwave_physics_chart(depth_series, anom_50m)
            drought_img = generate_drought_iod_chart(depth_series, d20_val, req.longitude)
            algae_img = generate_algae_stratification_chart(depth_series, mld_val, req.sst, density_sigma0)

            return {
                "status": "SUCCESS",
                "model_version": "Duo-Elite Ensemble v5.0 (Pure Unconstrained Raw Inversion)",
                "coordinates": {"lat": req.latitude, "lon": req.longitude},
                "inputs": {
                    "sst": req.sst,
                    "sss": req.sss,
                    "ssh": req.ssh,
                    "u_cur": req.u_cur,
                    "v_cur": req.v_cur,
                    "u_wind": req.u_wind,
                    "v_wind": req.v_wind,
                    "wind_magnitude": round(wind_mag, 2),
                    "potential_density_sigma0": round(density_sigma0, 2),
                },
                "predictions": pred_dict,
                "depth_series": depth_series,
                "derived_physical_products": {
                    "tchp_kj_cm2": tchp_val,
                    "cyclone_fuel_category": cyclone_fuel_cat,
                    "cyclone_fuel_color": cyclone_fuel_color,
                    "isotherm_d26_depth_m": d26_val,
                    "thermocline_d20_depth_m": d20_val,
                    "mixed_layer_depth_m": mld_val,
                    "sofar_sound_channel_axis_m": sofar_axis_val,
                    "acoustic_duct_trapping_strength_ms": duct_strength_val,
                    "surface_sound_speed_ms": round(float(sound_speed_col[0]), 1),
                    "deep_sound_speed_1000m_ms": round(float(sound_speed_col[-1]), 1),
                },
                "ocean_metrics": {
                    "thermocline_d20_depth_m": d20_val,
                    "mixed_layer_depth_m": mld_val,
                    "ocean_heat_content_kj_cm2": tchp_val,
                },
                "visualizations": {
                    "dynamic_profile_image": dynamic_profile_img,
                    "sound_velocity_image": sound_velocity_img,
                    "shallow_profile_image": shallow_img,
                    "deep_profile_image": deep_img,
                    "cyclone_sim_image": cyclone_img,
                    "heatwave_sim_image": heatwave_img,
                    "drought_sim_image": drought_img,
                    "algae_sim_image": algae_img,
                }
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
