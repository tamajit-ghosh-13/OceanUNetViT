"""
================================================================================
OceanEmbed - Temporal Harmonizer & Multi-Source Fusion (preprocessing/harmonize.py)
================================================================================
PURPOSE:
  Different ocean datasets are delivered at different time frequencies:
    - SST L4 composites → already daily (good)
    - SSH/SLA (DUACS)  → already daily (good)
    - Winds (ERA5)     → 1-hourly (must be averaged to daily)
    - GLORYS target    → daily (good)
    - ARGO floats      → irregular (sparse point observations)

  This module handles THREE things:
    1. TEMPORAL ALIGNMENT:   Convert all datasets to a common daily time axis.
    2. DEPTH SELECTION:      Extract only the 15 standard depth levels from GLORYS.
    3. MULTI-SOURCE FUSION:  Combine SST, SSS, SSH, Currents, Winds into one
                             aligned (T, 7, H, W) tensor per day.

BEGINNER EXPLANATION OF TIME ALIGNMENT:
  Think of each dataset like a playlist of songs (daily ocean maps). 
  Some playlists have one song per hour (winds), others one song per day (SST).
  We need every playlist to have exactly one song per day, so we average 
  hourly recordings into a "best-of-the-day" daily summary.
================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

try:
    import xarray as xr
    import pandas as pd
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from config import (
    STANDARD_DEPTH_LEVELS_M,
    N_DEPTH_LEVELS,
    INPUT_VARIABLES,
    N_INPUT_CHANNELS,
    GRID_LAT_SIZE,
    GRID_LON_SIZE,
    TEMPORAL_RESOLUTION,
)


# ==============================================================================
# 1. Resample Any Dataset to Daily Temporal Resolution
# ==============================================================================
def align_to_daily(
    ds: Any,
    time_dim: str = "time",
    method: str = "mean",
) -> Any:
    """
    Resamples an xarray Dataset or DataArray to daily (1D) temporal resolution.

    Parameters:
    -----------
    ds : xr.Dataset or xr.DataArray
        Input dataset with a time dimension.
    time_dim : str
        Name of the time dimension (usually 'time', sometimes 't').
    method : str
        Aggregation method: 'mean' (default for meteorological fields) or 'max', 'min'.

    Returns:
    --------
    xr.Dataset or xr.DataArray resampled to daily.

    Example: Wind data arriving at hourly resolution (24 maps/day) is reduced
    to ONE daily mean wind map by averaging all 24 hourly fields.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required for temporal alignment. Run: pip install xarray")

    # Rename non-standard time dim if needed
    if time_dim != "time" and time_dim in ds.dims:
        ds = ds.rename({time_dim: "time"})

    # Resample: group all observations within each calendar day and aggregate
    resampled = ds.resample(time="1D")
    if method == "mean":
        return resampled.mean(dim="time", skipna=True)
    elif method == "max":
        return resampled.max(dim="time", skipna=True)
    elif method == "min":
        return resampled.min(dim="time", skipna=True)
    else:
        raise ValueError(f"Unknown resampling method: '{method}'. Choose 'mean', 'max', or 'min'.")


# ==============================================================================
# 2. Extract Standard Depth Levels from GLORYS
# ==============================================================================
def select_standard_depths(
    ds: Any,
    depth_variable: str = "depth",
    tolerance_m: float = 10.0,
) -> Any:
    """
    Subsets a GLORYS Dataset to only the 15 required standard depth levels.

    The GLORYS model outputs temperature on ~50 depth levels (overkill).
    We select only the 15 standard depths specified by the hackathon authority:
      [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters

    Parameters:
    -----------
    ds : xr.Dataset or xr.DataArray
        GLORYS dataset with a depth/level dimension.
    depth_variable : str
        Name of the depth coordinate. Common names: 'depth', 'lev', 'level'.
    tolerance_m : float
        Maximum allowed difference (meters) when matching GLORYS depths to standard levels.
        GLORYS may have 0.494m as the first level (matches our 0m within tolerance).

    Returns:
    --------
    xr.Dataset subsetted to the 15 standard depth levels.
    """
    if not HAS_XARRAY:
        raise ImportError("xarray required.")

    # Find the depth coordinate name
    actual_depth_dim = None
    for candidate in ["depth", "lev", "level", "deptht", "z"]:
        if candidate in ds.dims or candidate in ds.coords:
            actual_depth_dim = candidate
            break

    if actual_depth_dim is None:
        raise KeyError(
            f"Could not find a depth dimension in the dataset. "
            f"Available dimensions: {list(ds.dims)}"
        )

    available_depths = ds[actual_depth_dim].values

    # For each required standard depth, find the closest available GLORYS depth
    selected_indices = []
    matched_depths = []
    for target_depth in STANDARD_DEPTH_LEVELS_M:
        differences = np.abs(available_depths - target_depth)
        closest_idx = int(np.argmin(differences))
        min_diff = differences[closest_idx]

        if min_diff > tolerance_m:
            print(
                f"   ⚠️  Standard depth {target_depth}m: "
                f"closest GLORYS depth is {available_depths[closest_idx]:.2f}m "
                f"(difference: {min_diff:.2f}m > tolerance {tolerance_m}m). Using nearest."
            )
        selected_indices.append(closest_idx)
        matched_depths.append(float(available_depths[closest_idx]))

    # Select the matched depth levels
    ds_depths = ds.isel({actual_depth_dim: selected_indices})
    print(f"   ✅ Extracted {len(selected_indices)}/{len(available_depths)} GLORYS depth levels")
    print(f"   Matched depths (m): {matched_depths}")
    return ds_depths


# ==============================================================================
# 3. Fuse Multiple Source Datasets into a Single Aligned Array
# ==============================================================================
def fuse_datasets(
    datasets: Dict[str, Any],
    date_range: Optional[Tuple[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Merges all 7 surface input channels from multiple Copernicus sources
    into a single aligned (T, 7, H, W) numpy array.

    Parameters:
    -----------
    datasets : dict
        Keys: variable names from INPUT_VARIABLES (e.g. 'SST', 'SSH', 'U_CUR')
        Values: xr.DataArray or np.ndarray for each variable,
                all already regridded to the standard 0.25° grid.
    date_range : tuple of str, optional
        ("YYYY-MM-DD", "YYYY-MM-DD") time window to crop all datasets to.

    Returns:
    --------
    fused_inputs  : np.ndarray shape (T, 7, H, W) — daily surface inputs
    common_dates  : np.ndarray of datetime strings for each time step T
    variable_order: list of variable names in the order they appear in axis 1
    """
    if not HAS_XARRAY:
        # Fallback: numpy-only fusion (all arrays must already be aligned)
        return _fuse_numpy(datasets)

    # Find the common time axis across all datasets
    all_time_axes = []
    for var_name, da in datasets.items():
        if isinstance(da, xr.DataArray) and "time" in da.dims:
            times = pd.DatetimeIndex(da.time.values)
            all_time_axes.append(times)

    if not all_time_axes:
        raise ValueError("No xarray DataArrays with 'time' dimension found in datasets dict.")

    # Common time axis = intersection of all datasets' time axes
    common_times = all_time_axes[0]
    for t_axis in all_time_axes[1:]:
        common_times = common_times.intersection(t_axis)

    if date_range is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        common_times = common_times[(common_times >= start) & (common_times <= end)]

    T = len(common_times)
    print(f"   Fusion: {T} common daily time steps across {len(datasets)} variables")

    # Assemble the (T, 7, H, W) array in the order specified by INPUT_VARIABLES
    fused = np.full((T, N_INPUT_CHANNELS, GRID_LAT_SIZE, GRID_LON_SIZE), np.nan, dtype=np.float32)

    for ch_idx, var_name in enumerate(INPUT_VARIABLES):
        if var_name not in datasets:
            print(f"   ⚠️  Variable '{var_name}' missing from datasets dict. Filling with zeros.")
            fused[:, ch_idx] = 0.0
            continue

        da = datasets[var_name]

        if isinstance(da, xr.DataArray):
            # Align to common time axis
            da_aligned = da.sel(time=common_times)
            fused[:, ch_idx] = da_aligned.values
        elif isinstance(da, np.ndarray):
            if da.shape[0] == T:
                fused[:, ch_idx] = da
            else:
                print(f"   ⚠️  Numpy array for '{var_name}' has shape {da.shape}, expected T={T}. Skipping.")

    common_dates = [str(t.date()) for t in common_times]
    return fused, np.array(common_dates), INPUT_VARIABLES[:]


def _fuse_numpy(datasets: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Simple numpy-only fusion when xarray is not available.
    Assumes all arrays are already shape (T, H, W) and temporally aligned.
    """
    T = None
    for arr in datasets.values():
        if isinstance(arr, np.ndarray) and arr.ndim == 3:
            T = arr.shape[0]
            break

    if T is None:
        raise ValueError("Could not determine time dimension T from datasets dict.")

    fused = np.full((T, N_INPUT_CHANNELS, GRID_LAT_SIZE, GRID_LON_SIZE), np.nan, dtype=np.float32)
    for ch_idx, var_name in enumerate(INPUT_VARIABLES):
        if var_name in datasets:
            arr = datasets[var_name]
            if arr.shape == (T, GRID_LAT_SIZE, GRID_LON_SIZE):
                fused[:, ch_idx] = arr
    dates = np.array([f"day_{i}" for i in range(T)])
    return fused, dates, INPUT_VARIABLES[:]


# ==============================================================================
# 4. Build Synthetic Aligned Dataset (for offline testing)
# ==============================================================================
def generate_synthetic_aligned_dataset(
    n_days: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a fully aligned synthetic dataset for offline pipeline testing.

    Returns:
    --------
    surface_inputs : np.ndarray shape (n_days, 7, 101, 241)   — 7-channel surface observations
    subsurface_targets : np.ndarray shape (n_days, 15, 101, 241) — 15-level subsurface temperature
    dates : np.ndarray of date strings, length n_days
    """
    H, W = GRID_LAT_SIZE, GRID_LON_SIZE
    lat = np.linspace(5.0, 30.0, H)
    lon = np.linspace(45.0, 105.0, W)

    # Create physically plausible latitude thermal gradient
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")  # (H, W)

    print(f"   Generating {n_days} days of synthetic ocean data ({H}×{W} = {H*W:,} grid cells)...")

    # Surface inputs: (n_days, 7, H, W)
    surface_inputs = np.zeros((n_days, N_INPUT_CHANNELS, H, W), dtype=np.float32)
    for t in range(n_days):
        # Seasonal phase (days shift temperature slightly)
        phase = 2 * np.pi * t / 365.0

        # SST: warm equatorial (28-30°C) declining northward, with seasonal oscillation
        sst = 30.0 - 0.3 * (lat_grid - 5.0) + 1.5 * np.sin(phase) + \
              0.5 * np.sin(2 * np.pi * lon_grid / 60.0) + \
              np.random.normal(0, 0.2, (H, W))

        # SSS: slightly fresher near coasts / rivers (Bay of Bengal)
        sss = 35.5 - 1.5 * np.exp(-((lon_grid - 88.0)**2 / 400)) + \
              np.random.normal(0, 0.1, (H, W))

        # SSH: mesoscale eddy signature
        ssh = 0.1 * np.sin(2 * np.pi * (lat_grid - 15.0) / 10.0) * \
              np.cos(2 * np.pi * (lon_grid - 75.0) / 20.0) + \
              np.random.normal(0, 0.02, (H, W))

        # Surface currents (geostrophic components, small)
        u_cur = -0.3 * np.gradient(ssh, axis=1) + np.random.normal(0, 0.02, (H, W))
        v_cur =  0.3 * np.gradient(ssh, axis=0) + np.random.normal(0, 0.02, (H, W))

        # Winds: SW monsoon in summer (positive U), NE monsoon in winter (negative U)
        u_wind = 4.0 * np.sin(phase) + np.random.normal(0, 1.0, (H, W))
        v_wind = 2.0 * np.cos(phase) + np.random.normal(0, 1.0, (H, W))

        surface_inputs[t] = np.stack([sst, sss, ssh, u_cur, v_cur, u_wind, v_wind], axis=0)

    # Subsurface targets: (n_days, 15, H, W)
    subsurface_targets = np.zeros((n_days, N_DEPTH_LEVELS, H, W), dtype=np.float32)
    for t in range(n_days):
        sst_day = surface_inputs[t, 0]  # SST for this day
        for d_idx, depth_m in enumerate(STANDARD_DEPTH_LEVELS_M):
            # Exponential thermocline decay: deeper = colder
            # Mixed layer depth ≈ 50m in North Indian Ocean
            if depth_m <= 50:
                # Above thermocline — fairly uniform SST
                decay = np.exp(-depth_m / 80.0)
                deep_temp = 4.0 + (sst_day - 4.0) * decay
            else:
                # Below thermocline — rapid cooling
                decay = np.exp(-depth_m / 300.0)
                deep_temp = 4.0 + (sst_day - 4.0) * decay

            subsurface_targets[t, d_idx] = deep_temp + np.random.normal(0, 0.1, (H, W))

    start = datetime(2023, 1, 1)
    dates = np.array([(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)])
    print(f"   ✅ Generated: inputs {surface_inputs.shape}, targets {subsurface_targets.shape}")
    return surface_inputs, subsurface_targets, dates


# ==============================================================================
# 5. Verification Self-Test
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Testing preprocessing/harmonize.py...")
    inputs, targets, dates = generate_synthetic_aligned_dataset(n_days=10)
    assert inputs.shape == (10, 7, GRID_LAT_SIZE, GRID_LON_SIZE), f"Unexpected shape: {inputs.shape}"
    assert targets.shape == (10, 15, GRID_LAT_SIZE, GRID_LON_SIZE), f"Unexpected shape: {targets.shape}"
    print(f"   Surface inputs shape:      {inputs.shape}  ← (days, 7_channels, lat, lon)")
    print(f"   Subsurface targets shape:  {targets.shape} ← (days, 15_depths, lat, lon)")
    print(f"   Date range: {dates[0]} to {dates[-1]}")
    print("✅ harmonize.py verified successfully!")
