"""
================================================================================
OceanEmbed - Spatial Regridder (preprocessing/regrid.py)
================================================================================
PURPOSE:
  Different satellite products come on different spatial grids:
    - GLORYS reanalysis → 0.083° × 0.083° (very high resolution)
    - Surface currents (DUACS) → 0.25° × 0.25° (already correct)
    - Wind products (ERA5/Copernicus) → 0.125° × 0.125° (too fine)
    - SST L4 composites → 0.05° × 0.05° (too fine)

  This module resamples ALL inputs onto the single STANDARD 0.25° grid
  required by the hackathon specification, covering:
    Latitude:  5.00°N → 30.00°N (101 points)
    Longitude: 45.00°E → 105.00°E (241 points)

HOW REGRIDDING WORKS (Beginner Explanation):
  Imagine you have a detailed weather map drawn on graph paper with tiny squares
  (0.083° = ~9km). The AI needs all maps drawn on standard squares (0.25° = ~28km).
  We "zoom out" by averaging or interpolating so every map has exactly 101 × 241 cells.

METHODS USED:
  - Bilinear Interpolation: Best for smooth fields (SST, SSH, SSS).
    Like stretching or shrinking a rubber photograph smoothly.
  - Nearest Neighbor: Used for categorical fields (land masks).
================================================================================
"""

import numpy as np
from typing import Optional, Dict, Any

# Safe imports — graceful fallback if xarray/scipy not yet installed
try:
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

# Import our project configuration
from config import BBOX, GRID_RESOLUTION_DEG, GRID_LAT_SIZE, GRID_LON_SIZE


# ==============================================================================
# 1. Build the Target Standard Grid
# ==============================================================================
def build_standard_grid() -> Dict[str, np.ndarray]:
    """
    Creates the authoritative 0.25° target grid arrays for the North Indian Ocean.

    Returns a dict with:
      'lat': 1D array of 101 latitude values  (5.0° to 30.0°)
      'lon': 1D array of 241 longitude values (45.0° to 105.0°)
    """
    lat = np.linspace(BBOX["min_lat"], BBOX["max_lat"], GRID_LAT_SIZE)
    lon = np.linspace(BBOX["min_lon"], BBOX["max_lon"], GRID_LON_SIZE)
    return {"lat": lat, "lon": lon}


# ==============================================================================
# 2. Core Regridding Function
# ==============================================================================
def regrid_to_standard_grid(
    data: Any,
    source_lat: Optional[np.ndarray] = None,
    source_lon: Optional[np.ndarray] = None,
    method: str = "bilinear",
) -> "xr.DataArray":
    """
    Regrids a 2D or 3D field onto the standard 0.25° North Indian Ocean grid.

    Supports two input types:
      1. xarray.DataArray (recommended — carries coordinate metadata)
      2. numpy ndarray (must supply source_lat and source_lon separately)

    Parameters:
    -----------
    data : xr.DataArray or np.ndarray
        Input spatial field to regrid.
        Expected shape: (..., lat, lon) where lat/lon are the LAST two dimensions.
    source_lat : np.ndarray, optional
        Source latitude 1D array. Required only if data is a numpy array.
    source_lon : np.ndarray, optional
        Source longitude 1D array. Required only if data is a numpy array.
    method : str
        Interpolation method. 'bilinear' (default) or 'nearest'.

    Returns:
    --------
    xr.DataArray with coordinates remapped to the standard 0.25° grid.
    """
    target_grid = build_standard_grid()

    if HAS_XARRAY and isinstance(data, xr.DataArray):
        return _regrid_xarray(data, target_grid, method)
    elif isinstance(data, np.ndarray):
        if source_lat is None or source_lon is None:
            raise ValueError(
                "When passing a numpy array to regrid_to_standard_grid(), "
                "you MUST also supply source_lat and source_lon coordinate arrays."
            )
        return _regrid_numpy(data, source_lat, source_lon, target_grid, method)
    else:
        raise TypeError(
            f"Expected xr.DataArray or np.ndarray, got {type(data).__name__}"
        )


def _regrid_xarray(
    da: "xr.DataArray",
    target_grid: Dict[str, np.ndarray],
    method: str,
) -> "xr.DataArray":
    """
    Regrids an xarray.DataArray to the target grid using xarray's interp().

    Beginner Note: xarray.interp() is like asking a GPS to re-plot your route
    on a different scale map — it smoothly fills in intermediate values.
    """
    # Detect the lat/lon dimension names (handles 'lat', 'latitude', 'y', etc.)
    lat_dim = _find_dim(da, ["lat", "latitude", "y"])
    lon_dim = _find_dim(da, ["lon", "longitude", "x"])

    # Crop to bounding box FIRST (server-side subset reduces data size)
    da = da.sel({
        lat_dim: slice(BBOX["min_lat"], BBOX["max_lat"]),
        lon_dim: slice(BBOX["min_lon"], BBOX["max_lon"]),
    })

    # Interpolate to the standard 0.25° target grid
    interp_kwargs = {method: "linear"} if method == "bilinear" else {method: "nearest"}
    regridded = da.interp(
        {
            lat_dim: target_grid["lat"],
            lon_dim: target_grid["lon"],
        },
        method="linear" if method == "bilinear" else "nearest",
    )

    return regridded


def _regrid_numpy(
    data: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_grid: Dict[str, np.ndarray],
    method: str,
) -> "xr.DataArray":
    """
    Regrids a raw numpy array using scipy's RegularGridInterpolator.

    This is the fallback path when xarray metadata is not available.
    Works for 2D arrays (lat × lon) or 3D arrays (time/depth × lat × lon).
    """
    # Clip source coordinates to bounding box
    lat_mask = (source_lat >= BBOX["min_lat"]) & (source_lat <= BBOX["max_lat"])
    lon_mask = (source_lon >= BBOX["min_lon"]) & (source_lon <= BBOX["max_lon"])
    data = data[..., lat_mask, :][..., lon_mask]
    src_lat = source_lat[lat_mask]
    src_lon = source_lon[lon_mask]

    # Build target meshgrid
    target_lat_grid, target_lon_grid = np.meshgrid(
        target_grid["lat"], target_grid["lon"], indexing="ij"
    )
    target_points = np.stack([target_lat_grid.ravel(), target_lon_grid.ravel()], axis=-1)

    fill_val = "extrapolate" if method == "bilinear" else None
    scipy_method = "linear" if method == "bilinear" else "nearest"

    if data.ndim == 2:
        # Single 2D spatial slice
        interp = RegularGridInterpolator(
            (src_lat, src_lon), data, method=scipy_method, bounds_error=False, fill_value=np.nan
        )
        result = interp(target_points).reshape(GRID_LAT_SIZE, GRID_LON_SIZE)
        return xr.DataArray(
            result,
            dims=["lat", "lon"],
            coords={"lat": target_grid["lat"], "lon": target_grid["lon"]},
        )
    else:
        # 3D array: first dim = time or depth
        results = []
        for i in range(data.shape[0]):
            interp = RegularGridInterpolator(
                (src_lat, src_lon), data[i], method=scipy_method, bounds_error=False, fill_value=np.nan
            )
            results.append(interp(target_points).reshape(GRID_LAT_SIZE, GRID_LON_SIZE))
        return xr.DataArray(
            np.stack(results, axis=0),
            dims=["z", "lat", "lon"],
            coords={"lat": target_grid["lat"], "lon": target_grid["lon"]},
        )


# ==============================================================================
# 3. Convenience Helper: Detect Coordinate Dimension Name
# ==============================================================================
def _find_dim(da: "xr.DataArray", candidates: list) -> str:
    """
    Searches a DataArray's dimensions for a matching coordinate name.
    Handles different naming conventions across datasets.
    """
    for name in candidates:
        if name in da.dims or name in da.coords:
            return name
    raise KeyError(
        f"Could not find any of {candidates} in DataArray dims: {list(da.dims)}. "
        f"Please rename the dimension manually."
    )


# ==============================================================================
# 4. Regrid a Full Multi-Variable Dataset
# ==============================================================================
def regrid_dataset(
    ds: "xr.Dataset",
    variable_mapping: Optional[Dict[str, str]] = None,
    method: str = "bilinear",
) -> "xr.Dataset":
    """
    Regrids an entire xarray Dataset (multiple variables at once).

    Parameters:
    -----------
    ds : xr.Dataset
        Source dataset from Copernicus or any other source.
    variable_mapping : dict, optional
        Maps source variable names to our standard names.
        e.g. {"analysed_sst": "SST", "sla": "SSH"}
    method : str
        Interpolation method ('bilinear' or 'nearest').

    Returns:
    --------
    xr.Dataset on the standard 0.25° grid with renamed variables.
    """
    target_grid = build_standard_grid()

    lat_dim = _find_dim(ds, ["lat", "latitude", "y"])
    lon_dim = _find_dim(ds, ["lon", "longitude", "x"])

    # Crop to bounding box
    ds = ds.sel({
        lat_dim: slice(BBOX["min_lat"], BBOX["max_lat"]),
        lon_dim: slice(BBOX["min_lon"], BBOX["max_lon"]),
    })

    # Interpolate all variables together
    ds_regridded = ds.interp(
        {
            lat_dim: target_grid["lat"],
            lon_dim: target_grid["lon"],
        },
        method="linear" if method == "bilinear" else "nearest",
    )

    # Rename variables to our standard names
    if variable_mapping:
        rename_dict = {k: v for k, v in variable_mapping.items() if k in ds_regridded}
        ds_regridded = ds_regridded.rename(rename_dict)

    return ds_regridded


# ==============================================================================
# 5. Verification Self-Test
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Testing preprocessing/regrid.py...")
    grid = build_standard_grid()
    print(f"   Standard Grid Lat: {grid['lat'][0]}° → {grid['lat'][-1]}° ({len(grid['lat'])} points)")
    print(f"   Standard Grid Lon: {grid['lon'][0]}° → {grid['lon'][-1]}° ({len(grid['lon'])} points)")
    assert len(grid["lat"]) == GRID_LAT_SIZE, f"Expected {GRID_LAT_SIZE} lat points"
    assert len(grid["lon"]) == GRID_LON_SIZE, f"Expected {GRID_LON_SIZE} lon points"

    # Simulate regridding from a coarser 0.5° grid
    src_lat = np.arange(4.0, 31.5, 0.5)
    src_lon = np.arange(44.0, 106.5, 0.5)
    fake_field = np.random.rand(len(src_lat), len(src_lon)).astype(np.float32)
    result = _regrid_numpy(fake_field, src_lat, src_lon, grid, method="bilinear")
    print(f"   Regrid test: {fake_field.shape} (0.5°) → {result.shape} (0.25°)")
    assert result.shape == (GRID_LAT_SIZE, GRID_LON_SIZE)
    print("✅ regrid.py verified successfully!")
