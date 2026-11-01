import matplotlib.pyplot as plt
import numpy as np
cmap = plt.get_cmap("Spectral_r")
norm = plt.Normalize(vmin=8.0, vmax=40.0)
print(f"18.42 (RGB): {cmap(norm(18.42))[:3]}")
