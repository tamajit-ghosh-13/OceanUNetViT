"""
================================================================================
OceanEmbed - Real-Time Live Inference REST API Server (api_server.py)
================================================================================
Exposes FastAPI endpoint:
  POST /api/predict

Returns:
  - 15-Depth Table (Baseline, v3, v4, Tri-Breeded, ±2σ band)
  - Real-time dynamic Base64-encoded SVG/PNG Temperature-vs-Depth Vertical Curve
  - Real-time dynamic Base64-encoded Regional Zonal Cross-Section Slice
  - Derived Ocean Diagnostics (D20, MLD, OHC, Buoyancy Density)
================================================================================
"""

import os
import sys
import io
import base64
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import gsw

from config import (
    BBOX,
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
)
from model import create_model
from train import get_compute_device
from preprocessing.normalize import normalize_inputs, denormalize_outputs

TRI_WEIGHTS = {
    0:    [0.323, 0.144, 0.534],
    5:    [0.223, 0.066, 0.711],
    10:   [0.204, 0.119, 0.677],
    20:   [0.196, 0.000, 0.804],
    30:   [0.210, 0.203, 0.587],
    50:   [0.147, 0.704, 0.150],
    75:   [0.354, 0.435, 0.211],
    100:  [0.136, 0.478, 0.387],
    125:  [0.128, 0.807, 0.065],
    150:  [0.226, 0.671, 0.103],
    200:  [0.188, 0.340, 0.472],
    300:  [0.820, 0.000, 0.180],
    500:  [0.339, 0.330, 0.330],
    700:  [0.228, 0.643, 0.129],
    1000: [0.140, 0.644, 0.216],
}

CALIBRATED_UNCERTAINTY = {
    0: 0.25, 5: 0.28, 10: 0.35, 20: 0.52, 30: 0.85, 50: 1.15,
    75: 1.25, 100: 1.35, 125: 1.10, 150: 0.95, 200: 0.80,
    300: 0.55, 500: 0.38, 700: 0.32, 1000: 0.28
}

app = FastAPI(title="OceanEmbed Live Dynamic Inference Server", version="4.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_compute_device()
print(f"🚀 Initializing OceanEmbed Live Inference Models on device: {device}...")

model_ft = create_model(in_channels=7, out_depth_levels=15).to(device)
model_ft.load_state_dict(torch.load("checkpoints/best_ocean_model_finetuned.pt", map_location=device), strict=False)
model_ft.eval()

model_v3 = create_model(in_channels=12, out_depth_levels=15).to(device)
model_v3.load_state_dict(torch.load("checkpoints/best_ocean_model_v3_unbiased.pt", map_location=device), strict=False)
model_v3.eval()

model_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
model_v4.load_state_dict(torch.load("checkpoints/best_ocean_model_v4.pt", map_location=device), strict=False)
model_v4.eval()

print("✅ Live Dynamic Inference Server initialized successfully!")


class OceanInferenceRequest(BaseModel):
    latitude: float = Field(12.5, ge=5.0, le=30.0)
    longitude: float = Field(68.0, ge=45.0, le=105.0)
    sst: float = Field(29.5, ge=10.0, le=36.0)
    sss: float = Field(35.2, ge=20.0, le=42.0)
    ssh: float = Field(0.12, ge=-2.0, le=2.0)
    u_cur: float = Field(0.25, ge=-3.0, le=3.0)
    v_cur: float = Field(-0.15, ge=-3.0, le=3.0)
    u_wind: float = Field(4.5, ge=-30.0, le=30.0)
    v_wind: float = Field(-2.1, ge=-30.0, le=30.0)
    doy: Optional[int] = Field(None, ge=1, le=366)


def generate_dynamic_profile_chart(depth_series: list, req_lat: float, req_lon: float) -> str:
    """Renders a dynamic Temperature vs Depth Profile Curve with Uncertainty Bounds."""
    depths = [d["depth_m"] for d in depth_series]
    t_tri = np.array([d["tribreed_degC"] for d in depth_series])
    t_base = np.array([d["baseline_degC"] for d in depth_series])
    t_v4 = np.array([d["v4_degC"] for d in depth_series])
    stds = np.array([d["confidence_std"] for d in depth_series])

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#020617", constrained_layout=True)
    ax.set_facecolor("#090d1f")

    # Plot confidence band
    ax.fill_betweenx(
        -np.array(depths),
        t_tri - 2 * stds,
        t_tri + 2 * stds,
        color="#06b6d4",
        alpha=0.18,
        label="95% Confidence (±2σ)"
    )

    ax.plot(t_base, -np.array(depths), color="#64748b", linestyle="--", linewidth=1.5, label="Baseline (7-ch)")
    ax.plot(t_v4, -np.array(depths), color="#3b82f6", linestyle=":", linewidth=1.5, label="v4 Physics-Inf")
    ax.plot(t_tri, -np.array(depths), color="#22d3ee", marker="o", markersize=4, linewidth=2.5, label="Tri-Breed AI 🧬")

    ax.set_title(f"Dynamic Profile at ({req_lat:.2f}°N, {req_lon:.2f}°E)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax.set_xlabel("Temperature (°C)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.grid(True, linestyle="--", color="#1e293b", alpha=0.6)
    ax.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc", fontsize=7.5, loc="upper right")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")




def generate_shallow_thermal_column_chart(depth_series: list, req_lat: float, req_lon: float) -> str:
    """Generates a vertical 1D thermal gradient column for Shallow Ocean (0-200m)."""
    shallow = [d for d in depth_series if d["depth_m"] <= 200]
    depths = np.array([d["depth_m"] for d in shallow])
    t_vals = np.array([d["tribreed_degC"] for d in shallow])

    dummy_x = np.array([0, 1])
    T_col = np.tile(t_vals, (2, 1)).T

    fig, ax = plt.subplots(figsize=(4.0, 6.0), facecolor="#020617")
    fig.subplots_adjust(left=0.25, right=0.7, top=0.9, bottom=0.1)
    ax.set_facecolor("#090d1f")

    cf = ax.contourf(dummy_x, depths, T_col, levels=np.linspace(8.0, 32.0, 35), cmap="Spectral_r", extend="both")

    # Highlight the D20 (20°C) isotherm
    d20_depth = None
    for i in range(len(depths)-1):
        if (t_vals[i] >= 20.0 and t_vals[i+1] <= 20.0) or (t_vals[i] <= 20.0 and t_vals[i+1] >= 20.0):
            frac = (20.0 - t_vals[i]) / (t_vals[i+1] - t_vals[i] + 1e-9)
            d20_depth = depths[i] + frac * (depths[i+1] - depths[i])
            break
            
    if d20_depth is not None:
        ax.axhline(y=d20_depth, color="black", linestyle="-", linewidth=2.5)
        ax.text(0.5, d20_depth, "D20 (20°C)", color="black", ha="center", va="bottom", fontsize=10, fontweight="bold", bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2))

    ax.set_title(f"Shallow (0-200m)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.invert_yaxis()

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.04, shrink=0.9)
    cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
    cbar.set_label("Temp (°C)", color="#94a3b8", fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def generate_deep_thermal_column_chart(depth_series: list, req_lat: float, req_lon: float) -> str:
    """Generates a vertical 1D thermal gradient column for Deep Ocean (300-1000m)."""
    deep = [d for d in depth_series if d["depth_m"] >= 300]
    depths = np.array([d["depth_m"] for d in deep])
    t_vals = np.array([d["tribreed_degC"] for d in deep])

    dummy_x = np.array([0, 1])
    T_col = np.tile(t_vals, (2, 1)).T

    fig, ax = plt.subplots(figsize=(4.0, 6.0), facecolor="#020617")
    fig.subplots_adjust(left=0.25, right=0.7, top=0.9, bottom=0.1)
    ax.set_facecolor("#090d1f")

    cf = ax.contourf(dummy_x, depths, T_col, levels=np.linspace(2.0, 15.0, 35), cmap="Spectral_r", extend="both")

    # Highlight the D10 (10°C) isotherm
    d10_depth = None
    for i in range(len(depths)-1):
        if (t_vals[i] >= 10.0 and t_vals[i+1] <= 10.0) or (t_vals[i] <= 10.0 and t_vals[i+1] >= 10.0):
            frac = (10.0 - t_vals[i]) / (t_vals[i+1] - t_vals[i] + 1e-9)
            d10_depth = depths[i] + frac * (depths[i+1] - depths[i])
            break
            
    if d10_depth is not None:
        ax.axhline(y=d10_depth, color="white", linestyle="--", linewidth=2.0)
        ax.text(0.5, d10_depth, "D10 (10°C)", color="white", ha="center", va="bottom", fontsize=10, fontweight="bold", bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

    ax.set_title(f"Deep (300-1000m)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Depth (m)", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.invert_yaxis()

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.04, shrink=0.9)
    cbar.ax.tick_params(labelsize=8, colors="#94a3b8")
    cbar.set_label("Temp (°C)", color="#94a3b8", fontsize=9)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/api/predict")
def predict_subsurface_temperatures(req: OceanInferenceRequest):
    try:
        doy_val = req.doy if req.doy is not None else datetime.now().timetuple().tm_yday
        wind_mag = float(np.sqrt(req.u_wind**2 + req.v_wind**2))
        doy_sin = float(np.sin(2 * np.pi * doy_val / 365.0))
        doy_cos = float(np.cos(2 * np.pi * doy_val / 365.0))
        sst_anomaly = float(req.sst - 28.5)

        sa = gsw.SA_from_SP(req.sss, 0.0, req.longitude, req.latitude)
        ct = gsw.CT_from_pt(sa, req.sst)
        density_sigma0 = float(gsw.sigma0(sa, ct))

        vec_7ch = np.array([
            req.sst, req.sss, req.ssh, req.u_cur, req.v_cur, req.u_wind, req.v_wind
        ], dtype=np.float32)

        vec_12ch = np.array([
            req.sst, req.sss, req.ssh, req.u_cur, req.v_cur, req.u_wind, req.v_wind,
            wind_mag, doy_sin, doy_cos, sst_anomaly, density_sigma0
        ], dtype=np.float32)

        tensor_7ch = np.tile(vec_7ch[:, None, None], (1, 1, GRID_LAT_SIZE, GRID_LON_SIZE))
        tensor_12ch = np.tile(vec_12ch[:, None, None], (1, 1, GRID_LAT_SIZE, GRID_LON_SIZE))

        norm_7ch, _ = normalize_inputs(tensor_7ch)
        norm_12ch, _ = normalize_inputs(tensor_12ch)

        t7 = torch.from_numpy(norm_7ch).to(device)
        t12 = torch.from_numpy(norm_12ch).to(device)

        lat_idx = int(np.clip(np.round((req.latitude - BBOX["min_lat"]) / 0.25), 0, GRID_LAT_SIZE - 1))
        lon_idx = int(np.clip(np.round((req.longitude - BBOX["min_lon"]) / 0.25), 0, GRID_LON_SIZE - 1))

        with torch.no_grad():
            out_7 = model_ft(t7).cpu().numpy()
            out_v3 = model_v3(t12).cpu().numpy()
            out_v4 = model_v4(t12).cpu().numpy()

        deg_7 = denormalize_outputs(out_7)[0, :, lat_idx, lon_idx]
        deg_v3 = denormalize_outputs(out_v3)[0, :, lat_idx, lon_idx]
        deg_v4 = denormalize_outputs(out_v4)[0, :, lat_idx, lon_idx]

        depths = STANDARD_DEPTH_LEVELS_M
        depth_series = []
        pred_dict = {}

        for d_idx, depth_m in enumerate(depths):
            w = TRI_WEIGHTS[depth_m]
            t_blend = float(w[0] * deg_7[d_idx] + w[1] * deg_v3[d_idx] + w[2] * deg_v4[d_idx])
            pred_dict[f"{depth_m}m"] = round(t_blend, 3)

            depth_series.append({
                "depth_m": depth_m,
                "baseline_degC": round(float(deg_7[d_idx]), 3),
                "v3_degC": round(float(deg_v3[d_idx]), 3),
                "v4_degC": round(float(deg_v4[d_idx]), 3),
                "tribreed_degC": round(t_blend, 3),
                "confidence_std": CALIBRATED_UNCERTAINTY[depth_m],
            })

        # Calculate D20
        d20_val = 100.0
        for d_idx in range(len(depths) - 1):
            t1 = depth_series[d_idx]["tribreed_degC"]
            t2 = depth_series[d_idx + 1]["tribreed_degC"]
            if t1 >= 20.0 >= t2:
                d1 = depths[d_idx]
                d2 = depths[d_idx + 1]
                frac = (t1 - 20.0) / (t1 - t2 + 1e-6)
                d20_val = round(float(d1 + frac * (d2 - d1)), 1)
                break

        # Calculate MLD
        mld_thresh = req.sst - 0.2
        mld_val = 25.0
        for d_idx in range(len(depths) - 1):
            t1 = depth_series[d_idx]["tribreed_degC"]
            t2 = depth_series[d_idx + 1]["tribreed_degC"]
            if t1 >= mld_thresh >= t2:
                d1 = depths[d_idx]
                d2 = depths[d_idx + 1]
                frac = (t1 - mld_thresh) / (t1 - t2 + 1e-6)
                mld_val = round(float(d1 + frac * (d2 - d1)), 1)
                break

        temps_300 = [ds["tribreed_degC"] for ds in depth_series if ds["depth_m"] <= 300]
        ohc_approx = round(float(np.mean(temps_300) * 3.85), 1)

        # Real-time Dynamic Charts
        dynamic_profile_img = generate_dynamic_profile_chart(depth_series, req.latitude, req.longitude)
        shallow_img = generate_shallow_thermal_column_chart(depth_series, req.latitude, req.longitude)
        deep_img = generate_deep_thermal_column_chart(depth_series, req.latitude, req.longitude)

        return {
            "status": "SUCCESS",
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
            "ocean_metrics": {
                "thermocline_d20_depth_m": d20_val,
                "mixed_layer_depth_m": mld_val,
                "ocean_heat_content_kj_cm2": ohc_approx,
            },
            "visualizations": {
                "dynamic_profile_image": dynamic_profile_img,
                "shallow_profile_image": shallow_img,
                "deep_profile_image": deep_img,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
