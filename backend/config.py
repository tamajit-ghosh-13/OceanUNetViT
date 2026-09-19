"""
================================================================================
OceanEmbed - Central Configuration & Physical Constants (config.py)
================================================================================
This module serves as the single source of truth for the entire OceanEmbed pipeline.

PHYSICAL DOMAIN SPECIFICATION:
  • Region: North Indian Ocean (Arabian Sea, Bay of Bengal, Equatorial Indian Ocean)
  • Latitude (phi):   5.0°N to 30.0°N  (101 points at delta_phi = 0.25°)
  • Longitude (lambda): 45.0°E to 105.0°E (241 points at delta_lambda = 0.25°)
  • Spatial Dimensions: H = 101, W = 241, Total Surface Grid Nodes = 24,341

INPUT FEATURE CHANNELS (12 Surface Variables):
  1.  SST            : Sea Surface Temperature (°C) — Surface thermal boundary condition
  2.  SSS            : Sea Surface Salinity (PSU) — Surface haline boundary condition
  3.  SSH            : Sea Surface Height (m) — Altimetric baroclinic thermocline pumping
  4.  U_CUR          : Zonal Surface Ocean Current (m/s) — East-West advective transport
  5.  V_CUR          : Meridional Surface Ocean Current (m/s) — North-South advective transport
  6.  U_WIND         : Geostrophic Zonal Wind (m/s) — u_w = -(g/f) * (d_SSH/dy)
  7.  V_WIND         : Geostrophic Meridional Wind (m/s) — v_w = (g/f) * (d_SSH/dx)
  8.  WIND_MAG       : Mechanical Wind Magnitude (m/s) — |w| = sqrt(u_w^2 + v_w^2)
  9.  DOY_SIN        : Solar Insolation Phase — sin(2 * pi * DayOfYear / 365.25)
  10. DOY_COS        : Monsoon Seasonal Cycle — cos(2 * pi * DayOfYear / 365.25)
  11. SST_ANOM       : Climatological SST Anomaly (°C) — SST(x,y,t) - mean_SST(x,y)
  12. DENSITY_SIGMA0 : Potential Density Anomaly (kg/m^3) — TEOS-10: sigma_0 = rho(S_A, Theta, 0) - 1000

OUTPUT VERTICAL TARGETS (15 Standard Depth Levels):
  z in [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters.
  • Mixed Layer: 0m - 30m (wind-driven turbulent homogenization)
  • Thermocline Core: 50m - 150m (steepest vertical temperature gradient dT/dz)
  • Sub-Thermocline: 200m - 300m (intermediate water mass transition)
  • Deep Ocean: 500m - 1000m (Antarctic Intermediate & deep water mass stratification)
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
# 3. Input Surface Variables (12 Physical & Thermodynamic Channels)
# ==============================================================================
# Order here determines the channel index in the tensor: (B, 12, H, W)
INPUT_VARIABLES = [
    "SST",           # Channel 0: Sea Surface Temperature (°C)
    "SSS",           # Channel 1: Sea Surface Salinity (PSU)
    "SSH",           # Channel 2: Sea Surface Height / Sea Level Anomaly (m)
    "U_CUR",         # Channel 3: Surface Ocean Current - Zonal (East-West) (m/s)
    "V_CUR",         # Channel 4: Surface Ocean Current - Meridional (North-South) (m/s)
    "U_WIND",        # Channel 5: 10m Wind - Zonal component (m/s)
    "V_WIND",        # Channel 6: 10m Wind - Meridional component (m/s)
    "WIND_MAG",      # Channel 7: Mechanical Wind Stress / Mixing Magnitude (m/s)
    "DOY_SIN",       # Channel 8: Seasonal Harmonic Component sin(2*pi*DOY/365)
    "DOY_COS",       # Channel 9: Seasonal Harmonic Component cos(2*pi*DOY/365)
    "SST_ANOM",      # Channel 10: Climatological Temporal SST Anomaly (°C)
    "DENSITY_SIGMA0",# Channel 11: TEOS-10 Potential Density Anomaly sigma_0 (kg/m^3)
]
N_INPUT_CHANNELS = len(INPUT_VARIABLES)  # 12

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
NORMALIZATION_STATS = {
    "SST":            {"mean": 28.5,  "std": 2.5},    # °C
    "SSS":            {"mean": 35.0,  "std": 1.5},    # PSU
    "SSH":            {"mean": 0.0,   "std": 0.15},   # meters
    "U_CUR":          {"mean": 0.0,   "std": 0.3},    # m/s
    "V_CUR":          {"mean": 0.0,   "std": 0.3},    # m/s
    "U_WIND":         {"mean": 0.0,   "std": 5.0},    # m/s
    "V_WIND":         {"mean": 0.0,   "std": 5.0},    # m/s
    "WIND_MAG":       {"mean": 6.0,   "std": 3.5},    # m/s
    "DOY_SIN":        {"mean": 0.0,   "std": 0.707},  # dimensionless
    "DOY_COS":        {"mean": 0.0,   "std": 0.707},  # dimensionless
    "SST_ANOM":       {"mean": 0.0,   "std": 1.0},    # °C
    "DENSITY_SIGMA0": {"mean": 22.5,  "std": 1.5},    # kg/m^3 (TEOS-10 surface potential density)
    "TEMP_TARGET":    {"mean": 16.0,  "std": 10.0},
}



# Per-Depth Target Normalization Stats (Feature 3)
# Based on physical North Indian Ocean thermocline depth distributions
TEMP_TARGET_STATS_PER_DEPTH = [
    {"depth_m": 0,    "mean": 29.0, "std": 2.5},
    {"depth_m": 5,    "mean": 28.8, "std": 2.5},
    {"depth_m": 10,   "mean": 28.5, "std": 2.6},
    {"depth_m": 20,   "mean": 28.2, "std": 2.8},
    {"depth_m": 30,   "mean": 27.8, "std": 3.0},
    {"depth_m": 50,   "mean": 26.5, "std": 3.8},  # Top of thermocline - higher variance
    {"depth_m": 75,   "mean": 24.5, "std": 4.5},
    {"depth_m": 100,  "mean": 22.0, "std": 5.0},  # Core thermocline - highest dynamic variance
    {"depth_m": 125,  "mean": 19.5, "std": 4.5},
    {"depth_m": 150,  "mean": 17.5, "std": 3.8},
    {"depth_m": 200,  "mean": 15.5, "std": 3.0},
    {"depth_m": 300,  "mean": 12.5, "std": 2.2},
    {"depth_m": 500,  "mean": 10.5, "std": 1.5},
    {"depth_m": 700,  "mean": 9.5,  "std": 1.2},
    {"depth_m": 1000, "mean": 7.0,  "std": 0.8},  # Abyssal water - low variance
]

# ==============================================================================
# 7. Training Hyperparameters (Defaults)
# ==============================================================================
TRAINING = {
    "batch_size": 4,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "val_split": 0.2,
    "random_seed": 42,
    "checkpoint_dir": "./checkpoints",
    "log_interval": 5,
}

# ==============================================================================
# 8. Model Architecture Defaults
# ==============================================================================
MODEL = {
    "in_channels": 12,                   # 12 pure physical & thermodynamic channels
    "out_depth_levels": N_DEPTH_LEVELS,  # 15 standard output depths
    "base_filters": 64,
    "vit_heads": 8,
    "vit_mlp_ratio": 4.0,
    "embedding_dim": 256,
}
