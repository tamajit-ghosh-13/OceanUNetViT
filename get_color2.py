import matplotlib.pyplot as plt
import numpy as np
cmap = plt.get_cmap("Spectral_r")
norm = plt.Normalize(vmin=8.0, vmax=40.0)
print(f"8.00 (RGB): {cmap(norm(8.0))[:3]}")
print(f"11.76 (RGB): {cmap(norm(11.76))[:3]}")
