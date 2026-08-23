"""
================================================================================
OceanEmbed - Central Configuration (config.py)
================================================================================
This file is the single source of truth for every constant in the project.
If you need to change a depth level, the bounding box, or a dataset ID,
change it HERE and it will automatically propagate everywhere.

Beginner Tip: Think of this file like the "settings menu" for the entire AI.
================================================================================
"""

# ==============================================================================
# 1. Geographic Domain — North Indian Ocean
# ==============================================================================
BBOX = {
    "min_lat": 5.0,    # 5°N  (Equatorial Indian Ocean)
    "max_lat": 30.0,   # 30°N (Northern Arabian Sea / Persian Gulf)
    "min_lon": 45.0,   # 45°E (Horn of Africa / Red Sea)
    "max_lon": 105.0,  # 105°E (Andaman Sea / Strait of Malacca)
}

# Standard 0.25° grid dimensions derived from the bounding box:
#   Latitude steps:  (30.0 - 5.0) / 0.25 + 1 = 101 points
#   Longitude steps: (105.0 - 45.0) / 0.25 + 1 = 241 points
GRID_RESOLUTION_DEG = 0.25
GRID_LAT_SIZE = 101   # Number of latitude grid points
GRID_LON_SIZE = 241   # Number of longitude grid points

# ==============================================================================
# 2. Temporal Configuration
# ==============================================================================
TEMPORAL_RESOLUTION = "1D"   # Daily temporal resolution as required by spec
DATE_FORMAT = "%Y-%m-%d"

# ==============================================================================
# 3. Input Surface Variables (7 Channels)
# ==============================================================================
# These are the satellite "clues" we feed INTO the neural network.
# Order here determines the channel index in the tensor: (B, 7, H, W)
INPUT_VARIABLES = [
    "SST",      # Channel 0: Sea Surface Temperature (°C)
    "SSS",      # Channel 1: Sea Surface Salinity (PSU)
    "SSH",      # Channel 2: Sea Surface Height / Sea Level Anomaly (m)
    "U_CUR",    # Channel 3: Surface Ocean Current - Zonal (East-West) component (m/s)
    "V_CUR",    # Channel 4: Surface Ocean Current - Meridional (North-South) component (m/s)
    "U_WIND",   # Channel 5: Surface Wind - Zonal component (m/s) at 10m height
    "V_WIND",   # Channel 6: Surface Wind - Meridional component (m/s) at 10m height
]
N_INPUT_CHANNELS = len(INPUT_VARIABLES)  # 7

# ==============================================================================
# 4. Output Depth Levels (15 Levels)
# ==============================================================================
# These are the exact STANDARD depth levels specified by the hackathon authority.
# The model must output temperature at EACH of these 15 depths simultaneously.
# Tensor shape: (Batch, 15, H=101, W=241)
STANDARD_DEPTH_LEVELS_M = [
    0,    # Sea surface (thin skin)
    5,    # Shallow mixed layer
    10,   # Upper mixed layer
    20,   # Base of mixed layer (tropics)
    30,   # Transition zone start
    50,   # Top of thermocline
    75,   # Upper thermocline
    100,  # Core thermocline (tropical Indian Ocean)
    125,  # Mid-thermocline
    150,  # Deep thermocline
    200,  # Below thermocline
    300,  # Intermediate water
    500,  # Deep intermediate
    700,  # AAIW (Antarctic Intermediate Water depth)
    1000, # Upper deep water
]
N_DEPTH_LEVELS = len(STANDARD_DEPTH_LEVELS_M)  # 15

# ==============================================================================
# 5. Copernicus Marine Dataset IDs
# ==============================================================================
# Each dataset_id corresponds to an official Copernicus product.
# These are passed to copernicusmarine.open_dataset().

COPERNICUS_DATASETS = {
    # Sea Surface Temperature — high-resolution L4 daily composite
    "SST": {
        "dataset_id": "cmems_P1D-m_SST-EUR-SST_L4_MULT_NRT",
        "variable": "analysed_sst",
        "note": "May need fallback to 'cmems_obs-sst_glo_phy_l4_my_0.05deg_P1D-m'"
    },
    # Sea Surface Salinity — Copernicus-ESA SMOS derived L4
    "SSS": {
        "dataset_id": "cmems_obs-mob_glo_phy-so_my_l4_P1D",
        "variable": "so",
        "note": "ESA SMOS surface salinity product"
    },
    # Sea Surface Height / Sea Level Anomaly — altimetry merged gridded
    "SSH": {
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        "variable": "sla",
        "note": "DUACS multi-mission merged SLA at 0.25°"
    },
    # Surface Ocean Currents (U, V) — geostrophic from altimetry
    "CURRENTS": {
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
        "variable": ["ugos", "vgos"],
        "note": "Geostrophic velocity components from DUACS (same product as SSH)"
    },
    # Surface Winds (U, V) — ERA5 10m winds regridded to 0.25°
    "WINDS": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H-i",
        "variable": ["eastward_wind", "northward_wind"],
        "note": "Requires regridding from 0.125° to 0.25°"
    },
    # GLORYS Subsurface Temperature — our TRAINING TARGET
    "GLORYS_TEMP": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variable": "thetao",
        "note": "GLORYS12 reanalysis ocean temperature at all depth levels. Subsample to standard depths."
    },
}

# ==============================================================================
# 6. Normalization Statistics
# ==============================================================================
# Z-score parameters: normalized = (value - mean) / std
# These are climatological approximations for the North Indian Ocean.
# They will be overwritten if you compute statistics from real data.
NORMALIZATION_STATS = {
    "SST":    {"mean": 28.5,  "std": 2.5},    # °C
    "SSS":    {"mean": 35.0,  "std": 1.5},    # PSU
    "SSH":    {"mean": 0.0,   "std": 0.15},   # meters
    "U_CUR":  {"mean": 0.0,   "std": 0.3},    # m/s
    "V_CUR":  {"mean": 0.0,   "std": 0.3},    # m/s
    "U_WIND": {"mean": 0.0,   "std": 5.0},    # m/s
    "V_WIND": {"mean": 0.0,   "std": 5.0},    # m/s
    # Target variable normalization (per depth, simplified to surface-reference)
    "TEMP_TARGET": {"mean": 16.0, "std": 10.0},  # °C (average across all depths)
}

# ==============================================================================
# 7. Training Hyperparameters (Defaults)
# ==============================================================================
TRAINING = {
    "batch_size": 4,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "val_split": 0.2,       # 20% of data reserved for validation
    "random_seed": 42,
    "checkpoint_dir": "./checkpoints",
    "log_interval": 5,      # Print metrics every N epochs
}

# ==============================================================================
# 8. Model Architecture Defaults
# ==============================================================================
MODEL = {
    "in_channels": N_INPUT_CHANNELS,     # 7 input surface channels
    "out_depth_levels": N_DEPTH_LEVELS,  # 15 standard output depths
    "base_filters": 64,                  # Upscaled from 32 → 64 for 7-channel input
    "vit_heads": 8,                      # Increased to 8 for richer attention
    "vit_mlp_ratio": 4.0,
    "embedding_dim": 256,                # Latent embedding vector size
}
