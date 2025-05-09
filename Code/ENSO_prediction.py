import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from func_for_prediction import *

for t in [0, 2, 5]: # 對應提前 6、9、12 個月的預測
sst = cut_var_pred[t, :, sst_lev, lat_idx_equator, :] # SST 沿赤道
taux = cut_var_pred[t, :, 0, lat_idx_equator, :] # τx

plt.figure(figsize=(12, 5))
plt.contourf(lon_vals, time_vals, sst, levels=np.linspace(-2, 2, 21), cmap='RdBu_r')
plt.quiver(lon_vals[::4], time_vals[::4], taux[::4, ::4], np.zeros_like(taux[::4, ::4]), scale=5)
plt.title(f'SST + τx prediction, lead = {lead_months[t]}')
plt.xlabel('Longitude')
plt.ylabel('Time')
plt.colorbar(label='SST anomaly (°C)')
plt.show()
