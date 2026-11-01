import matplotlib.pyplot as plt
import numpy as np

depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200])
t_vals = np.array([37.0, 34.436, 33.795, 32.873, 31.924, 29.712, 26.85, 23.719, 20.758, 18.42, 15.993])
T_col = np.tile(t_vals, (2, 1)).T
dummy_x = np.array([0, 1])

fig, ax = plt.subplots(figsize=(4, 6))
cf = ax.contourf(dummy_x, depths, T_col, levels=np.linspace(8.0, 40.0, 35), cmap="Spectral_r", extend="both")
ax.invert_yaxis()

# Save the raw image
fig.savefig("test_shallow.png")
