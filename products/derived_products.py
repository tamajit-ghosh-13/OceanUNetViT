"""
================================================================================
OceanEmbed - Derived Physical Products Engine (products/derived_products.py)
================================================================================
Computes operational, real-time oceanographic indicators directly from 3D subsurface
thermal (and haline) prediction volumes:

1. Tropical Cyclone Heat Potential (TCHP):
     TCHP = rho * c_p * integral_0^{D_26} (T(z) - 26.0°C) dz
   Standard meteorological metric for tropical cyclone intensification fuel.

2. Isotherm Depths:
     • D_26: Depth of the 26.0°C isotherm (cyclone heat reservoir boundary)
     • D_20: Depth of the 20.0°C isotherm (tropical thermocline core proxy)
     • D_12: Depth of the 12.0°C isotherm (intermediate water boundary)

3. Mixed Layer Depth (MLD):
     Depth where T(z) = T(10m) - 0.2°C (de Boyer Montégut et al. 2004 criterion).

4. Sound Velocity Profiles (SVP) & SOFAR Channel Minimum:
     c(T, S, z) via Mackenzie (1981) 9-term UNESCO empirical formulation.
     Acoustic duct mapping for sonar, marine navigation, and tomography.
================================================================================
"""

import numpy as np
import torch
from typing import Dict, Any, Union, Tuple, Optional
from config import STANDARD_DEPTH_LEVELS_M


# ==============================================================================
# 1. Continuous Isotherm Depth Calculation (D_T)
# ==============================================================================
def compute_isotherm_depth(
    temp_3d: Union[np.ndarray, torch.Tensor],
    depth_levels: list = STANDARD_DEPTH_LEVELS_M,
    target_temp: float = 26.0,
) -> np.ndarray:
    """
    Computes the continuous vertical depth (in meters) of a target isotherm T_target.
    
    Parameters:
    -----------
    temp_3d : (15, H, W) or (B, 15, H, W) array of temperatures in °C.
    depth_levels : List of depth values in meters (length 15).
    target_temp : Target isotherm temperature in °C (default 26.0°C).
    
    Returns:
    --------
    isotherm_depth : (H, W) or (B, H, W) array of continuous depths in meters.
                     Returns 0.0 if surface temperature < target_temp.
                     Returns max_depth if whole column >= target_temp.
                     Returns NaN for land pixels.
    """
    is_torch = isinstance(temp_3d, torch.Tensor)
    if is_torch:
        t_arr = temp_3d.detach().cpu().numpy()
    else:
        t_arr = np.array(temp_3d, copy=True)

    squeeze_output = False
    if t_arr.ndim == 3:
        t_arr = t_arr[np.newaxis, ...]  # (1, 15, H, W)
        squeeze_output = True

    B, Nz, H, W = t_arr.shape
    depths = np.asarray(depth_levels, dtype=np.float32).flatten()
    max_depth = float(depths[-1])

    isotherm_map = np.full((B, H, W), np.nan, dtype=np.float32)

    for b in range(B):
        # Identify valid ocean cells (where surface is non-NaN)
        valid_mask = ~np.isnan(t_arr[b, 0, :, :])
        if not np.any(valid_mask):
            continue

        b_temp = t_arr[b]  # (15, H, W)

        # 1. If surface temp < target_temp, isotherm depth is 0.0m
        sub_target_mask = (b_temp[0] < target_temp) & valid_mask
        isotherm_map[b, sub_target_mask] = 0.0

        # 2. If entire water column >= target_temp, isotherm depth is max_depth (1000m)
        deep_warm_mask = (b_temp[-1] >= target_temp) & valid_mask
        isotherm_map[b, deep_warm_mask] = max_depth

        # 3. Intermediate crossing pixels: find exact vertical linear interpolation
        crossing_mask = valid_mask & (~sub_target_mask) & (~deep_warm_mask)
        if not np.any(crossing_mask):
            continue

        y_coords, x_coords = np.where(crossing_mask)
        for y, x in zip(y_coords, x_coords):
            profile = b_temp[:, y, x]
            # Find first level where temperature drops below target_temp
            idx_below = np.where(profile < target_temp)[0]
            if len(idx_below) == 0:
                isotherm_map[b, y, x] = max_depth
                continue
            
            k = idx_below[0]  # First depth where T < target_temp
            if k == 0:
                isotherm_map[b, y, x] = 0.0
                continue
            
            z_above, z_below = depths[k - 1], depths[k]
            t_above, t_below = profile[k - 1], profile[k]

            if abs(t_above - t_below) < 1e-6:
                isotherm_map[b, y, x] = z_above
            else:
                # Linear interpolation for continuous crossing depth
                alpha = (t_above - target_temp) / (t_above - t_below)
                alpha = np.clip(alpha, 0.0, 1.0)
                isotherm_map[b, y, x] = z_above + alpha * (z_below - z_above)

    if squeeze_output:
        return isotherm_map[0]
    return isotherm_map


# ==============================================================================
# 2. Tropical Cyclone Heat Potential (TCHP)
# ==============================================================================
def compute_tchp(
    temp_3d: Union[np.ndarray, torch.Tensor],
    depth_levels: list = STANDARD_DEPTH_LEVELS_M,
    rho: float = 1025.0,
    cp: float = 3985.0,
    output_units: str = "kJ_cm2",
) -> np.ndarray:
    """
    Computes the Tropical Cyclone Heat Potential (TCHP) in kJ/cm² or MJ/m².
    
    Formula:
        TCHP = rho * c_p * integral_0^{D_26} max(0, T(z) - 26.0°C) dz
        
    Parameters:
    -----------
    temp_3d : (15, H, W) or (B, 15, H, W) 3D temperature field (°C).
    depth_levels : Standard vertical depth levels (m).
    rho : Seawater reference density (kg/m³, default 1025.0).
    cp : Specific heat capacity of seawater (J / (kg * °C), default 3985.0).
    output_units : kJ_cm2 (default) or MJ_m2.
                   1 kJ/cm² = 10^7 J/m² = 10 MJ/m².
                   
    Returns:
    --------
    tchp_map : (H, W) or (B, H, W) array of TCHP values.
               Typical active tropical values: 20 - 120 kJ/cm².
               Threshold for rapid cyclone intensification: > 50 kJ/cm².
    """
    is_torch = isinstance(temp_3d, torch.Tensor)
    if is_torch:
        t_arr = temp_3d.detach().cpu().numpy()
    else:
        t_arr = np.array(temp_3d, copy=True)

    squeeze_output = False
    if t_arr.ndim == 3:
        t_arr = t_arr[np.newaxis, ...]  # (1, 15, H, W)
        squeeze_output = True

    B, Nz, H, W = t_arr.shape
    depths = np.asarray(depth_levels, dtype=np.float32).flatten()

    # Compute D26 isotherm depth
    d26_map = compute_isotherm_depth(t_arr, depth_levels=depth_levels, target_temp=26.0)

    tchp_map = np.full((B, H, W), np.nan, dtype=np.float32)

    for b in range(B):
        valid_mask = ~np.isnan(t_arr[b, 0, :, :])
        if not np.any(valid_mask):
            continue

        b_temp = t_arr[b]
        b_d26 = d26_map[b]

        y_coords, x_coords = np.where(valid_mask)
        for y, x in zip(y_coords, x_coords):
            d26 = b_d26[y, x]
            if np.isnan(d26) or d26 <= 0.0:
                tchp_map[b, y, x] = 0.0
                continue

            profile = b_temp[:, y, x]
            integral_val = 0.0

            # Integrate layer by layer down to D26
            for k in range(len(depths) - 1):
                z_top = depths[k]
                z_bot = depths[k + 1]
                t_top = profile[k]
                t_bot = profile[k + 1]

                if z_top >= d26:
                    break

                if z_bot <= d26:
                    # Full interval is above D26
                    t_exc_top = max(0.0, float(t_top - 26.0))
                    t_exc_bot = max(0.0, float(t_bot - 26.0))
                    layer_dz = float(z_bot - z_top)
                    integral_val += 0.5 * (t_exc_top + t_exc_bot) * layer_dz
                else:
                    # Interval crosses D26: integrate from z_top to exact d26
                    t_exc_top = max(0.0, float(t_top - 26.0))
                    t_exc_bot = 0.0  # By definition at D26, T = 26.0°C
                    partial_dz = float(d26 - z_top)
                    integral_val += 0.5 * (t_exc_top + t_exc_bot) * partial_dz
                    break

            # Convert to Joules / m^2
            heat_joules_m2 = rho * cp * integral_val

            if output_units.lower() == "kj_cm2":
                # 1 J/m^2 = 1e-7 kJ/cm^2
                tchp_map[b, y, x] = heat_joules_m2 * 1e-7
            elif output_units.lower() == "mj_m2":
                # 1 J/m^2 = 1e-6 MJ/m^2
                tchp_map[b, y, x] = heat_joules_m2 * 1e-6
            else:
                tchp_map[b, y, x] = heat_joules_m2

    if squeeze_output:
        return tchp_map[0]
    return tchp_map


# ==============================================================================
# 3. Mixed Layer Depth (MLD) Calculation
# ==============================================================================
def compute_mld(
    temp_3d: Union[np.ndarray, torch.Tensor],
    depth_levels: list = STANDARD_DEPTH_LEVELS_M,
    delta_t: float = 0.2,
    ref_depth: float = 10.0,
) -> np.ndarray:
    """
    Computes Mixed Layer Depth (MLD) based on temperature criterion (de Boyer Montégut et al. 2004).
    
    Criterion:
        MLD is the depth where T(z) = T(z_ref) - delta_t
        Default: z_ref = 10.0m, delta_t = 0.2°C
        
    Parameters:
    -----------
    temp_3d : (15, H, W) or (B, 15, H, W) temperature field (°C).
    depth_levels : Standard vertical depth levels (m).
    delta_t : Temperature threshold drop (default 0.2°C).
    ref_depth : Reference depth (default 10.0m) to bypass diurnal surface heating skin.
    
    Returns:
    --------
    mld_map : (H, W) or (B, H, W) array of MLD values in meters.
    """
    is_torch = isinstance(temp_3d, torch.Tensor)
    if is_torch:
        t_arr = temp_3d.detach().cpu().numpy()
    else:
        t_arr = np.array(temp_3d, copy=True)

    squeeze_output = False
    if t_arr.ndim == 3:
        t_arr = t_arr[np.newaxis, ...]
        squeeze_output = True

    B, Nz, H, W = t_arr.shape
    depths = np.asarray(depth_levels, dtype=np.float32).flatten()
    max_depth = float(depths[-1])

    # Find reference depth index (closest to ref_depth, default 10m -> index 2)
    ref_idx = int(np.argmin(np.abs(depths - ref_depth)))

    mld_map = np.full((B, H, W), np.nan, dtype=np.float32)

    for b in range(B):
        valid_mask = ~np.isnan(t_arr[b, 0, :, :])
        if not np.any(valid_mask):
            continue

        b_temp = t_arr[b]
        y_coords, x_coords = np.where(valid_mask)

        for y, x in zip(y_coords, x_coords):
            profile = b_temp[:, y, x]
            t_ref = profile[ref_idx]
            t_threshold = t_ref - delta_t

            # Look below the reference depth
            sub_profile = profile[ref_idx:]
            sub_depths = depths[ref_idx:]

            idx_below = np.where(sub_profile < t_threshold)[0]
            if len(idx_below) == 0:
                mld_map[b, y, x] = max_depth
                continue

            k = idx_below[0]
            if k == 0:
                mld_map[b, y, x] = float(sub_depths[0])
                continue

            z_above, z_below = sub_depths[k - 1], sub_depths[k]
            t_above, t_below = sub_profile[k - 1], sub_profile[k]

            if abs(t_above - t_below) < 1e-6:
                mld_map[b, y, x] = float(z_above)
            else:
                alpha = (t_above - t_threshold) / (t_above - t_below)
                alpha = np.clip(alpha, 0.0, 1.0)
                mld_map[b, y, x] = float(z_above + alpha * (z_below - z_above))

    if squeeze_output:
        return mld_map[0]
    return mld_map


# ==============================================================================
# 4. Sound Velocity Profiles (SVP) & SOFAR Channel Minimum
# ==============================================================================
def compute_sound_velocity(
    temp_3d: Union[np.ndarray, torch.Tensor],
    depth_levels: list = STANDARD_DEPTH_LEVELS_M,
    salinity_3d: Optional[Union[np.ndarray, torch.Tensor]] = None,
    sss_2d: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> np.ndarray:
    """
    Computes 3D Speed of Sound in Seawater c(T, S, z) using Mackenzie (1981) formulation.
    
    Formula:
        c(T, S, z) = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3
                     + 1.340*(S - 35.0) + 1.630e-2*z + 1.675e-7*z^2
                     - 1.025e-2*T*(S - 35.0) - 7.139e-13*T*z^3
                     
    Parameters:
    -----------
    temp_3d : (15, H, W) or (B, 15, H, W) temperature field (°C).
    depth_levels : Standard vertical depth levels (m).
    salinity_3d : (Optional) 3D Salinity field in PSU.
    sss_2d : (Optional) 2D Sea Surface Salinity map in PSU.
    
    Returns:
    --------
    sound_speed_3d : (15, H, W) or (B, 15, H, W) Speed of Sound in meters/second (m/s).
    """
    is_torch = isinstance(temp_3d, torch.Tensor)
    if is_torch:
        t_arr = temp_3d.detach().cpu().numpy()
    else:
        t_arr = np.array(temp_3d, copy=True)

    squeeze_output = False
    if t_arr.ndim == 3:
        t_arr = t_arr[np.newaxis, ...]
        squeeze_output = True

    B, Nz, H, W = t_arr.shape
    depths = np.asarray(depth_levels, dtype=np.float32).flatten().reshape(1, Nz, 1, 1)

    # Prepare salinity field
    if salinity_3d is not None:
        if isinstance(salinity_3d, torch.Tensor):
            s_arr = salinity_3d.detach().cpu().numpy()
        else:
            s_arr = np.array(salinity_3d, copy=True)
        if s_arr.ndim == 3:
            s_arr = s_arr[np.newaxis, ...]
    elif sss_2d is not None:
        if isinstance(sss_2d, torch.Tensor):
            s_2d = sss_2d.detach().cpu().numpy()
        else:
            s_2d = np.array(sss_2d, copy=True)
        if s_2d.ndim == 0 or s_2d.size == 1:
            s_arr = np.full_like(t_arr, float(s_2d.flatten()[0]))
        elif s_2d.ndim == 2:
            s_2d = s_2d[np.newaxis, np.newaxis, ...]
            s_arr = np.repeat(s_2d, Nz, axis=1)
        elif s_2d.ndim == 3:
            s_2d = s_2d[:, np.newaxis, ...]
            s_arr = np.repeat(s_2d, Nz, axis=1)
        else:
            s_arr = np.full_like(t_arr, 35.0)
    else:
        # Default standard oceanic salinity (35.0 PSU)
        s_arr = np.full_like(t_arr, 35.0)

    T = t_arr
    S = s_arr
    Z = depths
    dS = S - 35.0

    # Mackenzie (1981) Equation for Speed of Sound in Seawater
    c = (
        1448.96
        + 4.591 * T
        - 5.304e-2 * (T ** 2)
        + 2.374e-4 * (T ** 3)
        + 1.340 * dS
        + 1.630e-2 * Z
        + 1.675e-7 * (Z ** 2)
        - 1.025e-2 * T * dS
        - 7.139e-13 * T * (Z ** 3)
    )

    if squeeze_output:
        return c[0]
    return c


def detect_sofar_channel(
    sound_speed_3d: Union[np.ndarray, torch.Tensor],
    depth_levels: list = STANDARD_DEPTH_LEVELS_M,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detects the SOFAR (Sound Fixing and Ranging) Channel Axis depth and duct strength.
    
    The SOFAR axis is the depth where sound velocity reaches its minimum:
        z_SOFAR(x, y) = argmin_z c(x, y, z)
        Duct Strength = c(surface) - c(z_SOFAR)
        
    Parameters:
    -----------
    sound_speed_3d : (15, H, W) or (B, 15, H, W) sound speed volume (m/s).
    depth_levels : Standard vertical depth levels (m).
    
    Returns:
    --------
    sofar_depth_map : (H, W) or (B, H, W) SOFAR axis depth in meters.
    duct_strength_map : (H, W) or (B, H, W) acoustic trapping strength in m/s.
    """
    is_torch = isinstance(sound_speed_3d, torch.Tensor)
    if is_torch:
        c_arr = sound_speed_3d.detach().cpu().numpy()
    else:
        c_arr = np.array(sound_speed_3d, copy=True)

    squeeze_output = False
    if c_arr.ndim == 3:
        c_arr = c_arr[np.newaxis, ...]
        squeeze_output = True

    B, Nz, H, W = c_arr.shape
    depths = np.asarray(depth_levels, dtype=np.float32).flatten()

    sofar_depth = np.full((B, H, W), np.nan, dtype=np.float32)
    duct_strength = np.full((B, H, W), np.nan, dtype=np.float32)

    for b in range(B):
        valid_mask = ~np.isnan(c_arr[b, 0, :, :])
        if not np.any(valid_mask):
            continue

        b_c = c_arr[b]
        # Find minimum index along depth axis
        min_idx = np.argmin(b_c, axis=0)  # (H, W)

        sofar_depth[b, valid_mask] = depths[min_idx[valid_mask]]
        c_surface = b_c[0]
        c_min = np.take_along_axis(b_c, min_idx[np.newaxis, ...], axis=0)[0]
        duct_strength[b, valid_mask] = np.maximum(0.0, c_surface[valid_mask] - c_min[valid_mask])

    if squeeze_output:
        return sofar_depth[0], duct_strength[0]
    return sofar_depth, duct_strength


# ==============================================================================
# 5. Master Pipeline: Compute All Products Simultaneously
# ==============================================================================
def compute_all_derived_products(
    temp_3d: Union[np.ndarray, torch.Tensor],
    depth_levels: list = STANDARD_DEPTH_LEVELS_M,
    sss_2d: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, np.ndarray]:
    """
    Generates the complete operational suite of oceanographic products.
    
    Returns:
    --------
    Dictionary containing:
      • TCHP_kJ_cm2      : Tropical Cyclone Heat Potential (kJ/cm²)
      • D26_m            : 26°C Isotherm Depth (m)
      • D20_m            : 20°C Isotherm Depth (m)
      • MLD_m            : Mixed Layer Depth (m)
      • Sound_Velocity_ms: 3D Sound Speed Volume (m/s)
      • SOFAR_Axis_m     : Deep Sound Channel Axis Depth (m)
      • Duct_Strength_ms : Acoustic Waveguide Trapping Strength (m/s)
    """
    d26 = compute_isotherm_depth(temp_3d, depth_levels=depth_levels, target_temp=26.0)
    d20 = compute_isotherm_depth(temp_3d, depth_levels=depth_levels, target_temp=20.0)
    tchp = compute_tchp(temp_3d, depth_levels=depth_levels, output_units="kJ_cm2")
    mld = compute_mld(temp_3d, depth_levels=depth_levels, delta_t=0.2, ref_depth=10.0)
    svp = compute_sound_velocity(temp_3d, depth_levels=depth_levels, sss_2d=sss_2d)
    sofar_depth, duct_strength = detect_sofar_channel(svp, depth_levels=depth_levels)

    return {
        "TCHP_kJ_cm2": tchp,
        "D26_m": d26,
        "D20_m": d20,
        "MLD_m": mld,
        "Sound_Velocity_ms": svp,
        "SOFAR_Axis_m": sofar_depth,
        "Duct_Strength_ms": duct_strength,
        # Normalized snake_case aliases for API convenience
        "tchp_kj_cm2": tchp,
        "isotherm_d26_depth_m": d26,
        "thermocline_d20_depth_m": d20,
        "mixed_layer_depth_m": mld,
        "sound_velocity_ms": svp,
        "sofar_sound_channel_axis_m": sofar_depth,
        "acoustic_duct_trapping_strength_ms": duct_strength,
    }
