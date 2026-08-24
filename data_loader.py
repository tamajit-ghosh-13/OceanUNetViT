"""
================================================================================
OceanEmbed - Data Pipeline (data_loader.py)  [UPGRADED v2]
================================================================================
CHANGES FROM v1:
  - Input channels: 3 → 7 (added U_CUR, V_CUR, U_WIND, V_WIND)
  - Depth levels: 14 custom → 15 standard hackathon levels
  - Grid size: 64×128 arbitrary → 101×241 (0.25° North Indian Ocean)
  - Added: full preprocessing pipeline integration (normalize, mask, fill)
  - Added: GLORYS real data loading stub for all 7 channels + target

TARGET: North Indian Ocean, 5°N–30°N, 45°E–105°E, 0.25° × 0.25°, daily
================================================================================
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any, List

try:
    import xarray as xr
    import copernicusmarine
    HAS_COPERNICUS_LIBS = True
except ImportError:
    HAS_COPERNICUS_LIBS = False

# Project-level imports
from config import (
    BBOX, GRID_LAT_SIZE, GRID_LON_SIZE,
    N_INPUT_CHANNELS, N_DEPTH_LEVELS,
    INPUT_VARIABLES, STANDARD_DEPTH_LEVELS_M,
    COPERNICUS_DATASETS, NORMALIZATION_STATS,
    TRAINING,
)
from preprocessing.normalize import preprocess_inputs, denormalize_outputs
from preprocessing.harmonize import (
    generate_synthetic_aligned_dataset,
    select_standard_depths,
    align_to_daily,
)
from preprocessing.regrid import regrid_dataset, build_standard_grid


# ==============================================================================
# 1. Copernicus Marine Streaming — 7-Channel Surface Observations
# ==============================================================================
def open_surface_observations(
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-31",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Streams all 7 required surface observation channels from Copernicus Marine.

    Performs server-side spatial subsetting to the North Indian Ocean bounding box
    BEFORE any data is transferred, keeping downloads small.

    Returns a dict mapping variable name → xr.DataArray (daily, 0.25° grid, NIO).
    """
    if not HAS_COPERNICUS_LIBS:
        raise ImportError(
            "copernicusmarine and xarray are required. Run: pip install copernicusmarine xarray"
        )

    if username and password:
        copernicusmarine.login(username=username, password=password, overwrite_configuration=True)

    common_kwargs = dict(
        minimum_latitude=BBOX["min_lat"],
        maximum_latitude=BBOX["max_lat"],
        minimum_longitude=BBOX["min_lon"],
        maximum_longitude=BBOX["max_lon"],
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
    )

    datasets_out = {}

    # --- Channel 0: SST (Sea Surface Temperature) ---
    print("📡 Fetching SST...")
    cfg = COPERNICUS_DATASETS["SST"]
    ds_sst = copernicusmarine.open_dataset(dataset_id=cfg["dataset_id"], variables=[cfg["variable"]], **common_kwargs)
    ds_sst = regrid_dataset(ds_sst, variable_mapping={cfg["variable"]: "SST"})
    datasets_out["SST"] = align_to_daily(ds_sst["SST"])

    # --- Channel 1: SSS (Sea Surface Salinity) ---
    print("📡 Fetching SSS...")
    cfg = COPERNICUS_DATASETS["SSS"]
    ds_sss = copernicusmarine.open_dataset(dataset_id=cfg["dataset_id"], variables=[cfg["variable"]], **common_kwargs)
    ds_sss = regrid_dataset(ds_sss, variable_mapping={cfg["variable"]: "SSS"})
    datasets_out["SSS"] = align_to_daily(ds_sss["SSS"])

    # --- Channels 2, 3, 4: SSH + U_CUR + V_CUR (from DUACS SSH product) ---
    print("📡 Fetching SSH, U_CUR, V_CUR from DUACS...")
    cfg = COPERNICUS_DATASETS["SSH"]
    var_map = {"sla": "SSH", "ugos": "U_CUR", "vgos": "V_CUR"}
    ds_ssh = copernicusmarine.open_dataset(
        dataset_id=cfg["dataset_id"],
        variables=["sla", "ugos", "vgos"],
        **common_kwargs
    )
    ds_ssh = regrid_dataset(ds_ssh, variable_mapping=var_map)
    datasets_out["SSH"]   = align_to_daily(ds_ssh["SSH"])
    datasets_out["U_CUR"] = align_to_daily(ds_ssh["U_CUR"])
    datasets_out["V_CUR"] = align_to_daily(ds_ssh["V_CUR"])

    # --- Channels 5, 6: U_WIND + V_WIND (ERA5/Copernicus winds, originally 0.125°) ---
    print("📡 Fetching U_WIND, V_WIND (will regrid from 0.125° → 0.25°)...")
    cfg = COPERNICUS_DATASETS["WINDS"]
    ds_wind = copernicusmarine.open_dataset(
        dataset_id=cfg["dataset_id"],
        variables=cfg["variable"],
        **common_kwargs
    )
    ds_wind = regrid_dataset(ds_wind, variable_mapping={
        "eastward_wind": "U_WIND",
        "northward_wind": "V_WIND",
    })
    datasets_out["U_WIND"] = align_to_daily(ds_wind["U_WIND"])
    datasets_out["V_WIND"] = align_to_daily(ds_wind["V_WIND"])

    print(f"✅ All 7 surface channels loaded.")
    return datasets_out


# ==============================================================================
# 2. GLORYS Subsurface Temperature Target Loader
# ==============================================================================
def open_glorys_target(
    start_date: str = "2023-01-01",
    end_date: str = "2023-01-31",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Any:
    """
    Streams GLORYS12 3D ocean temperature (thetao) for the NIO region.
    Selects the 15 standard depth levels and regrids to 0.25°.

    This is the GROUND TRUTH that the model learns to predict.
    """
    if not HAS_COPERNICUS_LIBS:
        raise ImportError("copernicusmarine required.")

    if username and password:
        copernicusmarine.login(username=username, password=password, overwrite_configuration=True)

    cfg = COPERNICUS_DATASETS["GLORYS_TEMP"]
    print(f"📡 Fetching GLORYS target temperature (thetao) — will extract 15 standard depths...")
    ds = copernicusmarine.open_dataset(
        dataset_id=cfg["dataset_id"],
        variables=[cfg["variable"]],
        minimum_latitude=BBOX["min_lat"],
        maximum_latitude=BBOX["max_lat"],
        minimum_longitude=BBOX["min_lon"],
        maximum_longitude=BBOX["max_lon"],
        start_datetime=f"{start_date}T00:00:00",
        end_datetime=f"{end_date}T23:59:59",
    )
    # Select only the 15 standard depths
    ds = select_standard_depths(ds)
    # Regrid from 0.083° → 0.25°
    ds = regrid_dataset(ds, variable_mapping={"thetao": "TEMP"})
    ds = align_to_daily(ds)
    print("✅ GLORYS target loaded.")
    return ds["TEMP"]  # Shape: (time, 15, lat, lon)


# ==============================================================================
# 3. PyTorch Dataset Class
# ==============================================================================
class OceanDataset(Dataset):
    """
    PyTorch Dataset for 3D Ocean Temperature Reconstruction.

    Input (Surface 2D Observations):  (7, 101, 241)
      Channel 0: SST   — Sea Surface Temperature (°C)
      Channel 1: SSS   — Sea Surface Salinity (PSU)
      Channel 2: SSH   — Sea Surface Height / SLA (m)
      Channel 3: U_CUR — Zonal geostrophic surface current (m/s)
      Channel 4: V_CUR — Meridional geostrophic surface current (m/s)
      Channel 5: U_WIND — Zonal surface wind at 10m (m/s)
      Channel 6: V_WIND — Meridional surface wind at 10m (m/s)

    Target (3D Subsurface Temperature): (15, 101, 241)
      Temperature in °C at the 15 standard depth levels (0m to 1000m).
    """

    def __init__(
        self,
        surface_inputs: Optional[np.ndarray] = None,
        subsurface_targets: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
        use_mock_data: bool = True,
        n_mock_days: int = 100,
        norm_stats: Optional[Dict] = None,
        land_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.norm_stats = norm_stats or NORMALIZATION_STATS
        self.land_mask = land_mask

        if use_mock_data or surface_inputs is None:
            print(f"ℹ️ [OceanDataset] Mock Mode — generating {n_mock_days} days of synthetic NIO data.")
            self.inputs, self.targets, self.dates = generate_synthetic_aligned_dataset(n_mock_days)
        else:
            self.inputs = surface_inputs.astype(np.float32)
            self.targets = subsurface_targets.astype(np.float32)
            self.dates = dates

        self.n_samples = self.inputs.shape[0]
        print(f"   Dataset ready: {self.n_samples} samples | "
              f"Input {self.inputs.shape[1:]} | Target {self.targets.shape[1:]}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns one daily training pair after full preprocessing."""
        raw_inputs = self.inputs[idx]     # (C, 101, 241)
        raw_targets = self.targets[idx]   # (15, 101, 241)

        # 1. Preprocess physical channels (0 to 6)
        p_phys, mask, _ = preprocess_inputs(
            raw_inputs[:7],
            stats=self.norm_stats,
            mask=self.land_mask,
            nan_fill_method="spatial_median",
        )

        # 2. If extra channels exist (DOY harmonics, SST Anomaly, MLD proxy), mask & zero-fill NaNs
        if raw_inputs.shape[0] > 7:
            extra_ch = raw_inputs[7:].copy()
            # Replace NaNs over land with 0.0
            extra_ch = np.where(np.isnan(extra_ch), 0.0, extra_ch)
            # Apply ocean mask to extra channels as well
            for ch_i in range(extra_ch.shape[0]):
                extra_ch[ch_i][~mask] = 0.0
            processed_inputs = np.concatenate([p_phys, extra_ch], axis=0)
        else:
            processed_inputs = p_phys

        # Ensure entire input tensor is completely free of NaNs and Infs
        processed_inputs = np.nan_to_num(processed_inputs, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Per-Depth Target Normalization (Feature 3)
        from config import TEMP_TARGET_STATS_PER_DEPTH
        processed_targets = np.zeros_like(raw_targets, dtype=np.float32)
        for d_idx in range(min(15, raw_targets.shape[0])):
            d_mean = TEMP_TARGET_STATS_PER_DEPTH[d_idx]["mean"]
            d_std  = TEMP_TARGET_STATS_PER_DEPTH[d_idx]["std"]
            processed_targets[d_idx] = (raw_targets[d_idx] - d_mean) / d_std

        # Fill any NaN in targets (over land / below seafloor) with 0.0
        processed_targets = np.nan_to_num(processed_targets, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            torch.from_numpy(processed_inputs.astype(np.float32)),
            torch.from_numpy(processed_targets.astype(np.float32)),
        )


# ==============================================================================
# 4. DataLoader Factory
# ==============================================================================
def create_ocean_dataloaders(
    batch_size: int = TRAINING["batch_size"],
    n_train_days: int = 80,
    n_val_days: int = 20,
    use_mock_data: bool = True,
    surface_inputs: Optional[np.ndarray] = None,
    subsurface_targets: Optional[np.ndarray] = None,
    dates: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train and validation PyTorch DataLoaders.

    In mock mode: generates synthetic North Indian Ocean physics data.
    In real mode: accepts pre-loaded numpy arrays from Copernicus.
    """
    if use_mock_data:
        train_ds = OceanDataset(use_mock_data=True, n_mock_days=n_train_days)
        val_ds   = OceanDataset(use_mock_data=True, n_mock_days=n_val_days)
    else:
        split_idx = n_train_days
        train_ds = OceanDataset(
            surface_inputs=surface_inputs[:split_idx],
            subsurface_targets=subsurface_targets[:split_idx],
            dates=dates[:split_idx] if dates is not None else None,
            use_mock_data=False,
        )
        val_ds = OceanDataset(
            surface_inputs=surface_inputs[split_idx:],
            subsurface_targets=subsurface_targets[split_idx:],
            dates=dates[split_idx:] if dates is not None else None,
            use_mock_data=False,
        )

    num_workers = 0  # MPS on Apple Silicon requires num_workers=0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ==============================================================================
# 5. Self-Test
# ==============================================================================
if __name__ == "__main__":
    print("🧪 Running data_loader.py self-test...")
    train_loader, val_loader = create_ocean_dataloaders(
        batch_size=2, n_train_days=6, n_val_days=2, use_mock_data=True
    )
    inputs, targets = next(iter(train_loader))
    print(f"   Batch Input Shape:   {inputs.shape}   → (B, 7_channels, 101_lat, 241_lon)")
    print(f"   Batch Target Shape:  {targets.shape}  → (B, 15_depths, 101_lat, 241_lon)")
    assert inputs.shape == (2, 7, GRID_LAT_SIZE, GRID_LON_SIZE)
    assert targets.shape == (2, 15, GRID_LAT_SIZE, GRID_LON_SIZE)
    assert not torch.isnan(inputs).any(), "NaN in inputs!"
    assert not torch.isnan(targets).any(), "NaN in targets!"
    print("✅ data_loader.py verified successfully!")
