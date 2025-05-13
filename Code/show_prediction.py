import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# data_in = xr.open_dataset('./model_3_level/Geoformer_1983_2021_output_3_level.nc')
adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_kb.nc"
data_in = xr.open_dataset(adr_oridata)

# 提取 sea_temperature 數據
sea_temp = data_in['sea temperature']
taux = data_in['taux']
tauy = data_in['tauy']

# 檢查數據形狀
print("data shape:", sea_temp.shape)
print("taux_data shape:", taux.shape)
print("tauy_data shape:", tauy.shape)

# 選擇 120°E 到 90°W 的經度範圍
# sea_temp = sea_temp.sel(lon=slice(120, 270), lat=slice(-20, 20))
# taux = taux.sel(lon=slice(120, 270), lat=slice(-20, 20))
# tauy = tauy.sel(lon=slice(120, 270), lat=slice(-20, 20))
lon = sea_temp['lon']
# 將 lon 轉換為 -180° 到 180° 的表示
# lon = sea_temp['lon'].values
# lon_adjusted = np.where(lon > 180, lon - 360, lon)
# sea_temp = sea_temp.assign_coords(lon=lon_adjusted)
# taux = taux.assign_coords(lon=lon_adjusted)
# tauy = tauy.assign_coords(lon=lon_adjusted)

# 緯度：-20° 到 20°，分段規則
lat1 = np.arange(-20, -5, 1)  # -20° 到 -5°，步長 1°
lat2 = np.arange(-5, 5.5, 0.5)  # -5° 到 5°，步長 0.5°
lat3 = np.arange(6, 21, 1)  # 6° 到 20°，步長 1°
lat_new = np.concatenate([lat1, lat2, lat3])


# 插值到新網格
sea_temp = sea_temp.interp(lon=lon, lat=lat_new, method='linear')
taux = taux.interp(lon=lon, lat=lat_new, method='linear')
tauy = tauy.interp(lon=lon, lat=lat_new, method='linear')
print(sea_temp)

fig, axes = plt.subplots(3, 4, figsize=(12, 4))
depth = 0   # for wind
# lat = 0     # for sea temperature
times = ['2015-04-01', '2015-05-01', '2015-06-01', '2015-07-01', '2015-08-01', 
          '2015-09-01', '2015-10-01', '2015-11-01', '2015-12-01', '2016-01-01', 
          '2016-02-01', '2016-03-01'
        ]


lead_time = 19

for i, time in enumerate(times):
    data = sea_temp.sel(time=time, depth=depth, method='nearest')[lead_time]  # lead_time=0
    taux_data = taux.sel(time=time, method='nearest')[lead_time]  # lead_time=0
    tauy_data = tauy.sel(time=time, method='nearest')[lead_time]  # lead_time=0
    print(data.shape)

    # # for sea temperature
    # lon, depth = np.meshgrid(data['lon'], data['depth'])
    # # 使用 2D 索引
    # row, col = divmod(i, 4)
    # contour = axes[row, col].contourf(lon, depth, data.values, cmap='RdBu_r', levels=20)
    # # ct1 = axes[row, col].contour(data.values, [-1.5, 0.0, 1.5, 3.0, 4.5], colors="k", linewidths=1)
    # # axes[row, col].clabel(contour, fmt='%1.1f', inline=True, fontsize=8, colors='black')

    # axes[row, col].invert_yaxis()
    # # axes[row, col].set_yticklabels(np.array[5, 20], fontsize=9)
    # axes[row, col].set_title(f'{time}')
    # axes[row, col].set_xlabel('Longitude (lon)')
    # # axes[row, col].set_xticks(lon[::5])  # 每 10 個點顯示一個標籤
    # # axes[row, col].set_xticklabels([f'{abs(l)}°E' for l in lon[::5]])
    # axes[row, col].set_ylabel('Depth (m)')

    # tauxy
    Lon, Lat = np.meshgrid(data['lon'], data['lat'])

    # 使用 2D 索引
    row, col = divmod(i, 4)
    contour = axes[row, col].contourf(Lon, Lat, data.values, cmap='RdBu_r', levels=20)
    # ct1 = axes[row, col].contour(data.values, [-1.5, 0.0, 1.5, 3.0, 4.5], colors="k", linewidths=1)
    # axes[row, col].clabel(contour, fmt='%1.1f', inline=True, fontsize=8, colors='black')

    # 繪製風應力向量（在最淺深度層，即 depth=5 米）
    # 創建用於風應力繪圖的網格（僅在最淺層）
    lon_quiver, lat_quiver = np.meshgrid(lon[::10], lat_new[::10])
    U = taux_data.values[::10, ::10]  # 經度方向風應力
    V = tauy_data.values[::10, ::10]  # 緯度方向風應力
    magnitude = np.sqrt(U**2 + V**2)

    # 檢查 quiver 數據形狀
    print("lon_quiver shape:", lon_quiver.shape)
    print("lat_quiver shape:", lat_quiver.shape)
    print("U shape:", U.shape)
    print("V shape:", V.shape)

    # 在圖表頂部（深度 5 米）繪製風應力向量
    scale = 1000000  # 調整箭頭大小
    axes[row, col].quiver(lon_quiver, lat_quiver, U, V, 
                   scale=scale, color='black', width=0.01)

    # axes[row, col].set_yticklabels(np.array[5, 20], fontsize=9)
    axes[row, col].set_title(f'{time}')
    axes[row, col].set_xlabel('Longitude (°E)')
    # axes[row, col].set_xticks(lon[::5])  # 每 10 個點顯示一個標籤
    # axes[row, col].set_xticklabels([f'{abs(l)}°E' for l in lon[::5]])
    axes[row, col].set_ylabel('Latitude')

    # 設置 X 軸標籤為經度格式 (E/W)
    # custom_ticks = np.linspace(120, -90, 5)  # 例如 -90, -60, -30, 0, 30, 60, 120
    # axes[row, col].set_xticks(lon[::5])
    # axes[row, col].set_xticklabels([f'{abs(l)}°E' if l >= 0 else f'{abs(l)}°W' for l in lon[::5]])
    # axes[row, col].set_xticklabels([f'{abs(l)}°E' for l in lon[::5]])
    # lon = data['lon']


# 添加色條，放在底部
fig.subplots_adjust(bottom=0.85)  # 增加底部空間以容納色條
cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', pad=0.1, aspect=50, 
                    label='(°C)')
cbar.ax.set_position([0.15, 0.075, 0.7, 0.01])  # [left, bottom, width, height]
# fig.colorbar(contour, ax=axes.ravel().tolist(), label='Sea Temperature Anomaly (°C)')
plt.tight_layout()
fig.subplots_adjust(bottom=0.15, top=0.85)
plt.suptitle('Sea Temperature Anomaly (lead_time=20)', fontsize=10, fontweight='bold')
# plt.suptitle('Sea Temperature Anomaly and Wind Stress (lead_time=1)', fontsize=10, fontweight='bold')
# plt.savefig("./model_3_level/tauxy_prediction_20.png")
plt.show()
