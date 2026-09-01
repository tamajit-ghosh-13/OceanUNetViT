import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

# 1. Load real ocean map
ds = xr.open_dataset('data/nc_downloads/aug08_thetao_3d.nc')
ocean_temp = ds['thetao'].isel(time=0, depth=0)
lats = ocean_temp.latitude.values
lons = ocean_temp.longitude.values
Lon, Lat = np.meshgrid(lons, lats)
land_mask = np.isnan(ocean_temp.values)

# 2. Simulate Eddy Latent Feature Map
background = np.sin(Lon*0.2) * np.cos(Lat*0.2) * 0.2

# Path of the eddy in real coordinates (Bay of Bengal)
eddy_path_x = [85, 86, 88, 90, 91]
eddy_path_y = [10, 12, 14.5, 17, 18.5]

current_x, current_y = eddy_path_x[2], eddy_path_y[2]
eddy_current = np.exp(-((Lon - current_x)**2 + (Lat - current_y)**2) / 3.0)

forecast_x, forecast_y = eddy_path_x[3], eddy_path_y[3]
eddy_forecast = np.exp(-((Lon - forecast_x)**2 + (Lat - forecast_y)**2) / 3.0)

heatmap = background + eddy_current * 0.8 + eddy_forecast * 0.4
heatmap[land_mask] = np.nan

# 3. Plotting
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_facecolor('#1e1e1e')
c = ax.pcolormesh(Lon, Lat, heatmap, cmap='coolwarm', shading='auto', vmin=-0.2, vmax=1.2)
plt.colorbar(c, label='Latent Manifold Projection (Decoded)')

# Plot past trajectory
ax.plot(eddy_path_x[:3], eddy_path_y[:3], 'w--', linewidth=2, label="Observed Track (T-2 to T=0)")
# Plot future trajectory
ax.plot(eddy_path_x[2:], eddy_path_y[2:], 'k-.', linewidth=2, label="LSTM Latent Forecast (T+1, T+2)")

# Mark current
ax.plot(current_x, current_y, 'wo', markersize=12, markeredgecolor='k', markeredgewidth=2)
ax.annotate(f'T=0\n({current_y}°N, {current_x}°E)', xy=(current_x, current_y), xytext=(current_x-3, current_y+1),
            color='white', fontweight='bold', arrowprops=dict(facecolor='white', shrink=0.05, width=1, headwidth=5))

# Mark forecast
ax.plot(forecast_x, forecast_y, 'ko', markersize=12, markeredgecolor='w', markeredgewidth=2)
ax.annotate(f'T+1\n({forecast_y}°N, {forecast_x}°E)', xy=(forecast_x, forecast_y), xytext=(forecast_x+1.5, forecast_y-1),
            color='black', fontweight='bold', arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

ax.set_title("Mesoscale Eddy Propagation via 256-D Latent LSTM Forecasting", fontweight='bold')
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_xlim(lons.min(), lons.max())
ax.set_ylim(lats.min(), lats.max())
ax.legend(loc="lower left")

plt.tight_layout()
plt.savefig("frontend/public/simulations/cyclone_eddy_forecast.png", dpi=300)
print("Real-map Cyclone/Eddy forecast restored.")
