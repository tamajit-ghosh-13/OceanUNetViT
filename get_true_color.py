import matplotlib.pyplot as plt
import numpy as np

cmap = plt.get_cmap("Spectral_r")
norm = plt.Normalize(vmin=8.0, vmax=40.0)
color = cmap(norm(15.993))
print(f"Color for 15.993 (RGB): {color[:3]}")

color20 = cmap(norm(20.758))
print(f"Color for 20.758 (RGB): {color20[:3]}")
