"""
OceanEmbed Preprocessing Package
=================================
This package handles all data cleaning, harmonization, and
standardization steps BEFORE the data reaches the neural network.

Sub-modules:
  - regrid:     Resamples any irregular or coarse spatial grid → 0.25° standard
  - normalize:  Z-score standardization per channel + land-sea masking + NaN filling
  - harmonize:  Temporal alignment across multiple datasets → daily resolution
"""

from preprocessing.regrid import regrid_to_standard_grid
from preprocessing.normalize import (
    normalize_inputs,
    denormalize_outputs,
    apply_land_sea_mask,
    fill_nans,
)
from preprocessing.harmonize import (
    align_to_daily,
    select_standard_depths,
    fuse_datasets,
)

__all__ = [
    "regrid_to_standard_grid",
    "normalize_inputs",
    "denormalize_outputs",
    "apply_land_sea_mask",
    "fill_nans",
    "align_to_daily",
    "select_standard_depths",
    "fuse_datasets",
]
