"""
================================================================================
OceanEmbed - Derived Physical Products Engine (products/__init__.py)
================================================================================
Exports real-time derived oceanographic products computed from 3D subsurface
thermal and haline fields:
  • Tropical Cyclone Heat Potential (TCHP)
  • Mixed Layer Depth (MLD)
  • Isotherm Depths (D26, D20, D12)
  • Sound Velocity Profiles (SVP) & SOFAR Channel Detection
================================================================================
"""

from products.derived_products import (
    compute_isotherm_depth,
    compute_tchp,
    compute_mld,
    compute_sound_velocity,
    detect_sofar_channel,
    compute_all_derived_products,
)

__all__ = [
    "compute_isotherm_depth",
    "compute_tchp",
    "compute_mld",
    "compute_sound_velocity",
    "detect_sofar_channel",
    "compute_all_derived_products",
]
