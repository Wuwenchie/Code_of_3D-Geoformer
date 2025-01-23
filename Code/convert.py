import netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# 讀取數據檔案
file_path = "./Data/3DGeoformer_predictions_Jan1983Dec2021.nc"
ds = xr.open_dataset(file_path)

# 提取經度、緯度、深度等變數
lat = ds['lat']
lon = ds['lon']
depth = ds['depth']

# 設置 lead 時間步長與變數名稱
leads = [6, 9, 12]
variables = ['taux', 'tauy', 'sea temperature']

# 創建子圖
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12), constrained_layout=True)

# 遍歷 lead 時間步長
for i, lead in enumerate(leads):
    # 限制地理範圍（120°E-90°W，20°S-20°N）
    lon_range = slice(120, 270)
    lat_range = slice(-20, 20)
    
    # 繪製 tau_x
    ax = axes[0, i]
    data = ds['taux'].sel(lead_time=lead, lon=lon_range, lat=lat_range).isel(time=0)
    im = ax.contourf(data['lon'], data['lat'], data.values, cmap='RdBu_r', levels=np.linspace(0, 1, 5))
    ax.set_title(f"$\\tau_x$; lead = {lead}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([-20, 0, 20])
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

    # 繪製 tau_y
    ax = axes[1, i]
    data = ds['tauy'].sel(lead_time=lead, lon=lon_range, lat=lat_range).isel(time=0)
    im = ax.contourf(data['lon'], data['lat'], data.values, cmap='RdBu_r', levels=np.linspace(0, 1, 5))
    ax.set_title(f"$\\tau_y$; lead = {lead}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([-20, 0, 20])
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

    # 繪製 SST（海表溫度）
    ax = axes[2, i]
    data = ds['sea temperature'].sel(lead_time=lead, lon=lon_range, lat=lat_range).isel(depth=0, time=0)
    im = ax.contourf(data['lon'], data['lat'], data.values, cmap='RdBu_r', levels=np.linspace(0, 1, 5))
    ax.set_title(f"SST; lead = {lead}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([-20, 0, 20])
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

    # 繪製縱剖面溫度（緯度平均）
    ax = axes[3, i]
    data = ds['sea temperature'].sel(lead_time=lead, lon=lon_range, lat=lat_range).mean(dim='lat').isel(time=0)
    im = ax.contourf(data['lon'], depth, data.values, cmap='RdBu_r', levels=np.linspace(0, 1, 5))
    ax.set_title(f"Temperature (Depth); lead = {lead}", fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Depth (m)", fontsize=10)
    fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)

# 整體標題
plt.suptitle('1983-01-01', fontsize=16, fontweight='normal', ha='center')
plt.savefig(f"1983-01-01.png")
plt.show()
