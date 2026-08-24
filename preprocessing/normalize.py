"""
================================================================================
OceanEmbed - Normalization & Masking (preprocessing/normalize.py)
================================================================================
PURPOSE:
  Neural networks learn fastest and most stably when all input values are
  in a similar numeric range. Raw ocean data has wildly different scales:
    - SST is roughly 20°C to 32°C
    - SSH is roughly -0.5m to +0.5m
    - U_WIND can be -20 m/s to +20 m/s

  If we fed these raw values into the model, the network would "pay more attention"
  to the big-number variables (winds) and practically ignore the small-number ones
  (SSH). Normalization fixes this.

TECHNIQUE USED — Z-Score Standardization:
  normalized = (value - mean) / standard_deviation
  → All variables are rescaled so mean ≈ 0 and std ≈ 1.
  → Reverting: denormalized = (normalized * std) + mean

LAND-SEA MASKING:
  Over land (e.g., India, Arabian Peninsula), satellite ocean measurements are
  meaningless noise. We identify land grid cells and replace them with zeros
  (a neutral value after normalization) so the AI ignores them.

NaN FILLING:
  Satellites miss data when clouds cover the ocean. We fill these missing cells
  using temporal interpolation (fill from neighboring days) or spatial median.
================================================================================
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, List

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from config import INPUT_VARIABLES, NORMALIZATION_STATS, GRID_LAT_SIZE, GRID_LON_SIZE


# ==============================================================================
# 1. Normalize a Multi-Channel Input Array
# ==============================================================================
def normalize_inputs(
    data: np.ndarray,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Z-score normalizes the 7-channel surface input array.

    Parameters:
    -----------
    data : np.ndarray
        Shape: (7, H, W) or (T, 7, H, W) where T = time steps.
        Channel order matches INPUT_VARIABLES in config.py:
          [SST, SSS, SSH, U_CUR, V_CUR, U_WIND, V_WIND]
    stats : dict, optional
        Pre-computed statistics {variable: {mean, std}}.
        If None, uses climatological defaults from config.py.
        If you're computing from real data, pass the actual statistics.

    Returns:
    --------
    normalized_data : np.ndarray — same shape as input, values ≈ N(0, 1)
    used_stats      : dict       — the mean/std values applied (for denormalization later)
    """
    if stats is None:
        stats = NORMALIZATION_STATS

    normalized = data.copy().astype(np.float32)
    used_stats = {}

    # Number of channels present in the data
    n_channels = data.shape[1] if data.ndim == 4 else data.shape[0]

    for ch_idx in range(min(n_channels, len(INPUT_VARIABLES))):
        var_name = INPUT_VARIABLES[ch_idx]
        if var_name not in stats:
            used_stats[var_name] = {"mean": 0.0, "std": 1.0}
            continue

        mean = stats[var_name]["mean"]
        std  = stats[var_name]["std"]

        if std == 0:
            raise ValueError(f"Standard deviation for '{var_name}' is zero — cannot normalize.")

        # Apply normalization along the channel dimension
        if data.ndim == 3:
            normalized[ch_idx] = (data[ch_idx] - mean) / std
        elif data.ndim == 4:
            normalized[:, ch_idx] = (data[:, ch_idx] - mean) / std

        used_stats[var_name] = {"mean": mean, "std": std}

    return normalized, used_stats


# ==============================================================================
# 2. Denormalize Model Outputs (Convert Predictions Back to °C)
# ==============================================================================
def denormalize_outputs(
    predictions: np.ndarray,
    stats: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
) -> np.ndarray:
    """
    Converts the model's normalized output back to physical temperature in °C.
    Supports both unified global stats and per-depth normalization stats (Feature 3).

    Parameters:
    -----------
    predictions : np.ndarray
        Shape: (B, 15, H, W) or (15, H, W) — model output with 15 depth levels.
    stats : dict or list of dicts, optional
        If list of 15 dicts: applies per-depth (mean, std) individually.
        If dict: applies global (mean, std).
        Defaults to config.TEMP_TARGET_STATS_PER_DEPTH.

    Returns:
    --------
    np.ndarray in °C, same shape as input.
    """
    from config import TEMP_TARGET_STATS_PER_DEPTH, NORMALIZATION_STATS

    if stats is None:
        stats = TEMP_TARGET_STATS_PER_DEPTH

    denorm = predictions.copy()

    # Per-depth normalization list (Feature 3)
    if isinstance(stats, list) and len(stats) == 15:
        for d_idx, d_stat in enumerate(stats):
            m, s = d_stat["mean"], d_stat["std"]
            if denorm.ndim == 4:
                denorm[:, d_idx] = denorm[:, d_idx] * s + m
            elif denorm.ndim == 3:
                denorm[d_idx] = denorm[d_idx] * s + m
        return denorm

    # Fallback to single global dict
    if isinstance(stats, dict):
        mean = stats.get("mean", 16.0)
        std  = stats.get("std", 10.0)
        return denorm * std + mean

    return denorm


# ==============================================================================
# 3. Build Land-Sea Mask for North Indian Ocean Grid
# ==============================================================================
def build_land_sea_mask(
    reference_field: Optional[np.ndarray] = None,
    method: str = "nan_based",
) -> np.ndarray:
    """
    Creates a boolean land-sea mask for the standard 0.25° North Indian Ocean grid.

    Mask convention: True = ocean (valid data), False = land (ignore).

    Parameters:
    -----------
    reference_field : np.ndarray, optional
        A 2D (H, W) array from any real dataset.
        Land cells typically have NaN in ocean datasets.
        If provided, NaN positions are flagged as land.
    method : str
        'nan_based': Land = cells that are NaN in the reference field (recommended).
        'synthetic': Creates a simplified rectangular ocean mask (for testing).

    Returns:
    --------
    mask : np.ndarray of shape (GRID_LAT_SIZE, GRID_LON_SIZE), dtype bool
        True where ocean water exists.
    """
    if method == "nan_based" and reference_field is not None:
        # Any grid cell that is NaN in the reference SST/SSH field is land
        ocean_mask = ~np.isnan(reference_field)
        return ocean_mask

    # Synthetic fallback mask (for offline testing)
    # Approximates the North Indian Ocean with a simple rectangular ocean region
    mask = np.ones((GRID_LAT_SIZE, GRID_LON_SIZE), dtype=bool)
    # Approximate Indian subcontinent landmass (rough polygon)
    # Rows correspond to latitudes 5°N–30°N, Cols to 45°E–105°E
    lat_arr = np.linspace(5.0, 30.0, GRID_LAT_SIZE)
    lon_arr = np.linspace(45.0, 105.0, GRID_LON_SIZE)

    for i, lat in enumerate(lat_arr):
        for j, lon in enumerate(lon_arr):
            # Very rough approximation of Indian Peninsula land
            if lat > 8.0 and lat < 23.0 and lon > 68.0 and lon < 88.0:
                mask[i, j] = False  # Mark as land
            # Arabian Peninsula
            if lat > 15.0 and lat < 30.0 and lon > 45.0 and lon < 60.0:
                mask[i, j] = False

    print(f"   Synthetic Land-Sea Mask: {mask.sum():,} ocean cells "
          f"/ {mask.size:,} total ({100*mask.mean():.1f}% ocean)")
    return mask


# ==============================================================================
# 4. Apply Land-Sea Mask to Input Data
# ==============================================================================
def apply_land_sea_mask(
    data: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Sets all land grid cells to a neutral fill value (default: 0.0 after normalization).

    Parameters:
    -----------
    data : np.ndarray
        Shape: (7, H, W) or (B, 7, H, W) or (B, 15, H, W)
    mask : np.ndarray
        Shape: (H, W), dtype bool. True = ocean (keep), False = land (mask out).
    fill_value : float
        Value to assign to land cells. After z-score normalization, 0.0 is neutral.

    Returns:
    --------
    np.ndarray — same shape as data with land cells zeroed out.
    """
    masked_data = data.copy()
    # Broadcast mask to match data dimensions
    land_cells = ~mask  # Invert: True = land

    if data.ndim == 2:
        masked_data[land_cells] = fill_value
    elif data.ndim == 3:
        # (C, H, W): apply mask to all channels
        masked_data[:, land_cells] = fill_value
    elif data.ndim == 4:
        # (B, C, H, W): apply mask across batch and channels
        masked_data[:, :, land_cells] = fill_value

    return masked_data


# ==============================================================================
# 5. NaN Fill Strategies
# ==============================================================================
def fill_nans(
    data: np.ndarray,
    method: str = "spatial_median",
) -> np.ndarray:
    """
    Fills missing (NaN) values in ocean fields caused by cloud cover or sensor gaps.

    Parameters:
    -----------
    data : np.ndarray
        Shape: (..., H, W). Last two dims are spatial.
    method : str
        'spatial_median': Replace each NaN with the median of its spatial neighbors.
        'zero':           Replace NaNs with 0.0 (safe after normalization).
        'forward_fill':   For time-series, carry forward the last valid value.

    Returns:
    --------
    np.ndarray — NaN-free array of same shape.
    """
    filled = data.copy()

    if method == "zero":
        filled = np.where(np.isnan(filled), 0.0, filled)

    elif method == "spatial_median":
        # Process each spatial slice independently
        shape_prefix = filled.shape[:-2]
        H, W = filled.shape[-2], filled.shape[-1]
        flat = filled.reshape(-1, H, W)

        for t in range(flat.shape[0]):
            slice_2d = flat[t]
            nan_mask = np.isnan(slice_2d)
            if nan_mask.any():
                # Fill with spatial median of the valid (non-NaN) values
                spatial_median = float(np.nanmedian(slice_2d))
                flat[t][nan_mask] = spatial_median

        filled = flat.reshape(filled.shape)

    elif method == "forward_fill":
        # Only valid for time-series arrays with time as first dimension
        for t in range(1, filled.shape[0]):
            nan_mask = np.isnan(filled[t])
            filled[t][nan_mask] = filled[t - 1][nan_mask]
        # Fill any remaining NaN at t=0 with zeros
        filled = np.where(np.isnan(filled), 0.0, filled)

    else:
        raise ValueError(f"Unknown fill method: '{method}'. Choose 'zero', 'spatial_median', or 'forward_fill'.")

    return filled


# ==============================================================================
# 6. Compute Normalization Statistics from Real Data
# ==============================================================================
def compute_normalization_stats(
    data: np.ndarray,
    variable_names: list,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes per-channel mean and standard deviation from a real dataset.
    Use this ONCE on your training data to get data-driven normalization stats
    instead of using the hardcoded climatological defaults in config.py.

    Parameters:
    -----------
    data : np.ndarray
        Shape: (T, C, H, W) where T=time steps, C=channels.
    variable_names : list
        List of variable name strings corresponding to channels (length C).
    mask : np.ndarray, optional
        Ocean mask (H, W). If provided, statistics are only computed over ocean cells.

    Returns:
    --------
    dict of {variable_name: {"mean": float, "std": float}}
    """
    T, C, H, W = data.shape
    stats = {}

    for ch_idx, var_name in enumerate(variable_names):
        channel_data = data[:, ch_idx]  # Shape: (T, H, W)

        if mask is not None:
            # Only use ocean pixels for statistics
            ocean_vals = channel_data[:, mask]  # Shape: (T, N_ocean)
        else:
            ocean_vals = channel_data.reshape(T, -1)

        # Remove any NaNs before computing stats
        valid_vals = ocean_vals[~np.isnan(ocean_vals)]
        mean = float(np.mean(valid_vals))
        std  = float(np.std(valid_vals))

        print(f"   {var_name:10s}: mean={mean:8.3f}, std={std:6.3f}")
        stats[var_name] = {"mean": mean, "std": std}

    return stats


# ==============================================================================
# 7. Full Preprocessing Pipeline (Convenience Wrapper)
# ==============================================================================
def preprocess_inputs(
    raw_data: np.ndarray,
    stats: Optional[Dict] = None,
    mask: Optional[np.ndarray] = None,
    nan_fill_method: str = "spatial_median",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Runs the complete preprocessing pipeline for model inputs in one call:
      Step 1: Fill NaN values (cloud gaps, sensor dropouts)
      Step 2: Apply land-sea mask (silence land areas)
      Step 3: Z-score normalize each channel

    Parameters:
    -----------
    raw_data : np.ndarray
        Shape: (7, H, W) — raw satellite input for one time step.
    stats : dict, optional
        Normalization statistics. Uses config defaults if None.
    mask : np.ndarray, optional
        Ocean mask (H, W). If None, creates synthetic mask.
    nan_fill_method : str
        NaN filling strategy (see fill_nans() docstring).

    Returns:
    --------
    processed : np.ndarray — shape (7, H, W), normalized, masked, NaN-free
    land_mask : np.ndarray — shape (H, W), True = ocean
    used_stats : dict      — normalization statistics applied (needed for denormalization)
    """
    # Step 1: Build land mask if not provided
    if mask is None:
        # Use first channel (SST) to detect land as NaN positions
        ref = raw_data[0, 0] if raw_data.ndim == 4 else (raw_data[0] if raw_data.ndim == 3 else raw_data)
        mask = build_land_sea_mask(reference_field=ref, method="nan_based")

    # Step 2: Fill NaN values before normalization
    filled = fill_nans(raw_data, method=nan_fill_method)

    # Step 3: Apply land-sea mask (zero out land cells)
    masked = apply_land_sea_mask(filled, mask, fill_value=0.0)

    # Step 4: Z-score normalize
    normalized, used_stats = normalize_inputs(masked, stats=stats)

    return normalized, mask, used_stats


# ==============================================================================
# 8. Verification Self-Test
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Testing preprocessing/normalize.py...")

    # Create fake 7-channel input with some NaNs
    fake_input = np.random.randn(7, GRID_LAT_SIZE, GRID_LON_SIZE).astype(np.float32)
    # Scale each channel to approximate physical ranges
    fake_input[0] = fake_input[0] * 2.5 + 28.5   # SST
    fake_input[1] = fake_input[1] * 1.5 + 35.0   # SSS
    fake_input[2] = fake_input[2] * 0.15          # SSH
    fake_input[3] = fake_input[3] * 0.3           # U_CUR
    fake_input[4] = fake_input[4] * 0.3           # V_CUR
    fake_input[5] = fake_input[5] * 5.0           # U_WIND
    fake_input[6] = fake_input[6] * 5.0           # V_WIND

    # Inject 5% NaNs to simulate cloud cover
    nan_positions = np.random.rand(*fake_input.shape) < 0.05
    fake_input[nan_positions] = np.nan

    print(f"   Input shape: {fake_input.shape}, NaN count: {np.isnan(fake_input).sum()}")
    processed, mask, stats = preprocess_inputs(fake_input)
    print(f"   After preprocessing: NaN count={np.isnan(processed).sum()}")
    print(f"   SST channel: mean={processed[0].mean():.3f}, std={processed[0].std():.3f} (should be ≈0, 1)")
    assert not np.isnan(processed).any(), "NaNs still present after preprocessing!"
    print("✅ normalize.py verified successfully!")
