import matplotlib.pyplot as plt
import numpy as np

depths = np.array([300, 500, 700, 1000])
t_vals = np.array([12.641, 10.512, 9.501, 7.0])
T_col = np.tile(t_vals, (2, 1)).T
dummy_x = np.array([0, 1])

fig, ax = plt.subplots(figsize=(4, 6))
cf = ax.contourf(dummy_x, depths, T_col, levels=np.linspace(2.0, 20.0, 35), cmap="Spectral_r", extend="both")
ax.invert_yaxis()
plt.colorbar(cf)
fig.savefig("test_contourf.png")
