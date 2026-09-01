import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
Unit Tests for OceanEmbed Derived Physical Products Engine
"""

import numpy as np
import torch

from products.derived_products import (
    compute_isotherm_depth,
    compute_tchp,
    compute_mld,
    compute_sound_velocity,
    detect_sofar_channel,
    compute_all_derived_products,
)
from config import STANDARD_DEPTH_LEVELS_M


def test_isotherm_depth_analytical():
    # Depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    # Linear temperature profile from 30°C at 0m to 10°C at 1000m
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=np.float32)
    
    # Profile 1: Linear drop from 30°C to 10°C (rate = -0.02°C/m)
    # T(z) = 30 - 0.02 * z
    # T(z) = 26°C -> 0.02 * z = 4 -> z = 200m
    # T(z) = 20°C -> 0.02 * z = 10 -> z = 500m
    prof1 = 30.0 - 0.02 * depths
    
    # Profile 2: Cold surface (24°C at 0m) -> D26 should be 0.0m
    prof2 = 24.0 - 0.015 * depths
    
    # Profile 3: Land pixel (all NaNs)
    prof3 = np.full_like(depths, np.nan)
    
    # Grid shape: (15, 3, 1)
    grid = np.zeros((15, 3, 1), dtype=np.float32)
    grid[:, 0, 0] = prof1
    grid[:, 1, 0] = prof2
    grid[:, 2, 0] = prof3
    
    d26 = compute_isotherm_depth(grid, target_temp=26.0)
    d20 = compute_isotherm_depth(grid, target_temp=20.0)
    
    assert np.isclose(d26[0, 0], 200.0, atol=1e-3)
    assert d26[1, 0] == 0.0
    assert np.isnan(d26[2, 0])
    
    assert np.isclose(d20[0, 0], 500.0, atol=1e-3)
    assert np.isnan(d20[2, 0])
    print("✅ test_isotherm_depth_analytical passed!")


def test_tchp_analytical():
    # Depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=np.float32)
    
    # Construct profile: Constant 30°C in upper 50m, drops linearly to 26°C at 100m, then 15°C at 200m
    # T(0..50) = 30°C
    # T(100) = 26°C
    # T(125..1000) = 20°C..5°C
    prof = np.zeros_like(depths)
    prof[depths <= 50] = 30.0
    prof[depths == 75] = 28.0
    prof[depths == 100] = 26.0
    prof[depths > 100] = 20.0
    
    # Hand calculation of Heat Content integral:
    # 0 to 50m: (30 - 26) * 50 = 4 * 50 = 200 °C*m
    # 50 to 75m: 0.5 * (4 + 2) * 25 = 3 * 25 = 75 °C*m
    # 75 to 100m: 0.5 * (2 + 0) * 25 = 1 * 25 = 25 °C*m
    # Total integral = 200 + 75 + 25 = 300 °C*m
    # Total Joules = 1025 * 3985 * 300 = 1,225,387,500 J/m²
    # In kJ/cm²: 1.2253875e9 * 1e-7 = 122.53875 kJ/cm²
    
    grid = prof[:, np.newaxis, np.newaxis]  # (15, 1, 1)
    tchp = compute_tchp(grid, output_units="kJ_cm2")
    
    expected_tchp = 1025.0 * 3985.0 * 300.0 * 1e-7
    assert np.isclose(tchp[0, 0], expected_tchp, atol=1e-2)
    print(f"✅ test_tchp_analytical passed! (Computed: {tchp[0, 0]:.3f} kJ/cm², Expected: {expected_tchp:.3f} kJ/cm²)")


def test_mld_de_boyer():
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=np.float32)
    # Profile: Uniform 29.0°C from 0 to 30m, then drops to 28.6°C at 50m
    # Reference at 10m is 29.0°C.
    # Threshold = 29.0 - 0.2 = 28.8°C.
    # Between 30m (29.0°C) and 50m (28.6°C), T = 28.8°C occurs at:
    # 30 + (29.0 - 28.8)/(29.0 - 28.6) * (50 - 30) = 30 + 0.5 * 20 = 40.0m
    prof = np.array([29.0, 29.0, 29.0, 29.0, 29.0, 28.6, 26.0, 23.0, 20.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0], dtype=np.float32)
    grid = prof[:, np.newaxis, np.newaxis]
    
    mld = compute_mld(grid, delta_t=0.2, ref_depth=10.0)
    assert np.isclose(mld[0, 0], 40.0, atol=1e-3)
    print(f"✅ test_mld_de_boyer passed! (Computed MLD: {mld[0, 0]:.1f}m, Expected: 40.0m)")


def test_sound_velocity_and_sofar():
    depths = np.array(STANDARD_DEPTH_LEVELS_M, dtype=np.float32)
    # Typical tropical profile:
    # Surface (0m, 29°C, 35 PSU) -> sound speed ~ 1543 m/s
    # 700m (10°C, 35 PSU) -> sound speed ~ 1495 m/s (min SOFAR channel)
    # 1000m (7°C, 35 PSU) -> pressure effect increases speed slightly ~ 1500 m/s
    prof = np.array([29.0, 28.9, 28.8, 28.5, 27.5, 25.0, 22.0, 18.0, 15.0, 13.0, 11.0, 9.0, 8.0, 7.0, 6.5], dtype=np.float32)
    grid = prof[:, np.newaxis, np.newaxis]
    
    c = compute_sound_velocity(grid)
    sofar_depth, duct_strength = detect_sofar_channel(c)
    
    # Sound speed should be physically reasonable (between 1450 and 1560 m/s)
    assert np.all(c >= 1450.0) and np.all(c <= 1560.0)
    assert duct_strength[0, 0] > 0.0
    print(f"✅ test_sound_velocity_and_sofar passed! (Surface Sound Speed: {c[0, 0, 0]:.1f} m/s, SOFAR Axis: {sofar_depth[0, 0]:.0f}m)")


def test_all_products_tensor_support():
    # Verify PyTorch Tensor input support
    t_tensor = torch.full((15, 10, 10), 28.0)
    products = compute_all_derived_products(t_tensor)
    
    assert "TCHP_kJ_cm2" in products
    assert "D26_m" in products
    assert "D20_m" in products
    assert "MLD_m" in products
    assert "Sound_Velocity_ms" in products
    assert "SOFAR_Axis_m" in products
    assert "Duct_Strength_ms" in products
    print("✅ test_all_products_tensor_support passed!")


if __name__ == "__main__":
    test_isotherm_depth_analytical()
    test_tchp_analytical()
    test_mld_de_boyer()
    test_sound_velocity_and_sofar()
    test_all_products_tensor_support()
    print("\n🎉 ALL 5 DERIVED PRODUCT UNIT TESTS PASSED PERFECTLY!")
