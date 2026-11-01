import matplotlib.pyplot as plt
import numpy as np
cmap = plt.get_cmap("Spectral_r")
norm = plt.Normalize(vmin=2.0, vmax=20.0)
print(f"12.641 (RGB): {cmap(norm(12.641))[:3]}")
print(f"16.82 (RGB): {cmap(norm(16.82))[:3]}")
