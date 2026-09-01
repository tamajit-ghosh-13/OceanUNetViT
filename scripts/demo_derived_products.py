"""
================================================================================
OceanEmbed - Real-Time Physical Products Demo Pipeline (scripts/demo_derived_products.py)
================================================================================
Demonstrates the computation of operational physical oceanographic products
from 3D temperature volumes predicted by the Duo-Elite Ensemble:
  • Tropical Cyclone Heat Potential (TCHP in kJ/cm²)
  • Mixed Layer Depth (MLD in meters)
  • 20°C Isotherm Depth (D20 in meters)
  • 26°C Isotherm Depth (D26 in meters)
  • 3D Sound Velocity Volume c(T, S, z) in m/s
  • SOFAR Sound Channel Axis Depth in meters
================================================================================
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from model import create_model
from config import (
    STANDARD_DEPTH_LEVELS_M,
    TEMP_TARGET_STATS_PER_DEPTH,
    NORMALIZATION_STATS,
)
from preprocessing.normalize import denormalize_outputs, preprocess_inputs
from products.derived_products import compute_all_derived_products

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


def run_physical_products_demo(input_file="./data/argo_jun20_inputs_12ch.npy", event_name="June 2020 (Super Cyclone Amphan)"):
    print("=" * 100)
    print(f"🌊 OCEANEMBED DERIVED PHYSICAL PRODUCTS ENGINE: {event_name.upper()}")
    print("=" * 100)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Running on compute device: {device}")

    # 1. Load Duo-Elite Ensemble Backbones
    m_v4 = create_model(in_channels=12, out_depth_levels=15).to(device)
    m_v5 = create_model(in_channels=12, out_depth_levels=15).to(device)

    m_v4.load_state_dict(torch.load("checkpoints/best_ocean_model_v4_extended.pt", map_location=device), strict=False)
    m_v5.load_state_dict(torch.load("checkpoints/best_ocean_model_v5_finetuned.pt", map_location=device), strict=False)
    m_v4.eval()
    m_v5.eval()
    print("🧠 Loaded v4_extended and v5_finetuned neural backbones.")

    # 2. Load Satellite Surface Inputs
    inputs_12ch = np.load(input_file)
    T = len(inputs_12ch)
    
    # Preprocess 12-channel inputs
    proc = np.zeros_like(inputs_12ch)
    for i in range(T):
        p_phys, mask, _ = preprocess_inputs(inputs_12ch[i, :7], stats=NORMALIZATION_STATS, nan_fill_method="spatial_median")
        extra = inputs_12ch[i, 7:].copy()
        extra = np.where(np.isnan(extra), 0.0, extra)
        for ch in range(extra.shape[0]):
            extra[ch][~mask] = 0.0
        proc[i] = np.concatenate([p_phys, extra], axis=0)
    proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. 3D Neural Inversion
    x_tensor = torch.from_numpy(proc.astype(np.float32)).to(device)
    with torch.no_grad():
        p4 = m_v4(x_tensor).cpu().numpy()
        p5 = m_v5(x_tensor).cpu().numpy()

    p4_deg = denormalize_outputs(p4, stats=TEMP_TARGET_STATS_PER_DEPTH)
    p5_deg = denormalize_outputs(p5, stats=TEMP_TARGET_STATS_PER_DEPTH)

    # Blend with Duo-Elite simplex weights
    T_duo = np.zeros_like(p4_deg)
    for d_idx, d_m in enumerate(STANDARD_DEPTH_LEVELS_M):
        w4, w5 = DUO_ELITE_WEIGHTS[d_m]
        T_duo[:, d_idx] = w4 * p4_deg[:, d_idx] + w5 * p5_deg[:, d_idx]

    # Re-apply land mask from raw SST
    land_mask = np.isnan(inputs_12ch[:, 0, :, :])
    for d_idx in range(15):
        T_duo[:, d_idx][land_mask] = np.nan

    # Take mid-month snapshot (e.g. Day 15 during Cyclone Amphan peak)
    snap_idx = min(15, T_duo.shape[0] - 1)
    temp_snap = T_duo[snap_idx]  # (15, 101, 241)
    sss_snap = inputs_12ch[snap_idx, 1]  # (101, 241)

    print(f"🔮 Inverted 3D Subsurface Temperature Field: shape {temp_snap.shape}")

    # 4. Compute Derived Physical Oceanographic Products
    products = compute_all_derived_products(temp_snap, sss_2d=sss_snap)

    tchp = products["TCHP_kJ_cm2"]
    d26 = products["D26_m"]
    d20 = products["D20_m"]
    mld = products["MLD_m"]
    svp = products["Sound_Velocity_ms"]
    sofar_depth = products["SOFAR_Axis_m"]
    duct_strength = products["Duct_Strength_ms"]

    valid_ocean = ~np.isnan(tchp)
    n_ocean_cells = np.sum(valid_ocean)

    # 5. Display Physical Summary Report
    print("\n" + "=" * 100)
    print("📊 OPERATIONAL OCEANOGRAPHIC DIAGNOSTICS & METRICS:")
    print("=" * 100)

    # TCHP
    mean_tchp = np.nanmean(tchp)
    max_tchp = np.nanmax(tchp)
    high_cyclone_fuel_pct = np.sum(tchp[valid_ocean] >= 50.0) / n_ocean_cells * 100.0
    super_fuel_pct = np.sum(tchp[valid_ocean] >= 80.0) / n_ocean_cells * 100.0

    print(f"🌪️ Tropical Cyclone Heat Potential (TCHP):")
    print(f"   • Basin-Wide Mean TCHP:           {mean_tchp:.2f} kJ/cm²")
    print(f"   • Peak Hotspot TCHP:              {max_tchp:.2f} kJ/cm² (Extreme Intensification Reservoir)")
    print(f"   • Active Cyclone Fuel (>50 kJ):   {high_cyclone_fuel_pct:.1f}% of North Indian Ocean")
    print(f"   • High-Intensity Fuel (>80 kJ):   {super_fuel_pct:.1f}% of North Indian Ocean")

    # Isotherms & MLD
    print(f"\n🌊 Thermocline & Mixed Layer Metrics:")
    print(f"   • Mean Mixed Layer Depth (MLD):   {np.nanmean(mld):.1f} meters (Min: {np.nanmin(mld):.1f}m, Max: {np.nanmax(mld):.1f}m)")
    print(f"   • Mean 26°C Isotherm Depth (D26): {np.nanmean(d26):.1f} meters (Peak Reservoir: {np.nanmax(d26):.1f}m)")
    print(f"   • Mean 20°C Isotherm Depth (D20): {np.nanmean(d20):.1f} meters (Upwelling Front Proxy)")

    # Acoustics & Sound Velocity
    print(f"\n🔊 Underwater Acoustics & Sonar Velocity:")
    print(f"   • Surface Sound Speed c(0m):      {np.nanmean(svp[0]):.1f} m/s")
    print(f"   • Deep Ocean Sound Speed c(1000m):{np.nanmean(svp[-1]):.1f} m/s")
    print(f"   • Mean SOFAR Channel Axis Depth:  {np.nanmean(sofar_depth):.1f} meters")
    print(f"   • Mean Acoustic Duct Strength:    {np.nanmean(duct_strength):.1f} m/s")
    print("=" * 100)

    # 6. Save products for downstream visualization / API delivery
    os.makedirs("data/derived_products", exist_ok=True)
    out_path = "data/derived_products/amphan_jun20_physical_products.npz"
    np.savez_compressed(
        out_path,
        TCHP_kJ_cm2=tchp,
        D26_m=d26,
        D20_m=d20,
        MLD_m=mld,
        Sound_Velocity_ms=svp,
        SOFAR_Axis_m=sofar_depth,
        Duct_Strength_ms=duct_strength,
        Temperature_3D=temp_snap,
    )
    print(f"\n💾 Saved compressed physical products archive -> {out_path}")
    print("🎉 Derived Products Engine Demo Completed Successfully!")


if __name__ == "__main__":
    run_physical_products_demo()
