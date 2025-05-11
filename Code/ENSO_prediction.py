from myconfig import mypara
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from my_tools import cal_ninoskill2, runmean
from func_for_prediction import func_pre

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L


# --------------------------------------------------------
files = file_name("./model")
file_num = len(files)
lead_max = mypara.output_length
adr_datain = (
    "./data/GODAS_group_up150_temp_tauxy_8021_kb.nc"
)
adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_kb.nc"
# ---------------------------------------------------------
def save_prediction_to_netcdf(
        cut_var_pred,
        mypara,
        adr_oridata,
        output_path,
        start_time,
        freq="MS",
    ):
    """
    儲存 Geoformer 預測輸出為 NetCDF 檔案
    - cut_var_pred: shape = [lead, time, channel, lat, lon]
        - channel 0: taux
        - channel 1: tauy
        - channel 2+: sea temperature at depth
    - mypara: 包含 lat_range, lon_range, lev_range
    - adr_oridata: 原始檔案，用於取得 lat/lon 資訊
    - output_path: 輸出的 .nc 檔路徑
    - start_time: 第一個預測起始月（ex. "1990-01"）
    - freq: 時間頻率（預設 "MS" = 月初）
    """
        lead, time_len, ch_n, lat_n, lon_n = cut_var_pred.shape
        assert ch_n >= 3, "通道數必須至少包含 taux, tauy, 與一層溫度"
        # 建立座標
        ds_ref = xr.open_dataset(adr_oridata)
        lat_vals = ds_ref['lat'].values[mypara.lat_range[0]:mypara.lat_range[1]]
        lon_vals = ds_ref['lon'].values[mypara.lon_range[0]:mypara.lon_range[1]]
        ds_ref.close()
        
        lev_vals = np.arange(mypara.lev_range[0], mypara.lev_range[1])
        lead_vals = np.arange(1, lead_max+1)
        time_vals = pd.date_range(start=start_time, periods=time_len, freq=freq)

        # 建立資料變數
        taux = (("lead_time", "time", "lat", "lon"), cut_var_pred[:, :, 0, :, :])
        tauy = (("lead_time", "time", "lat", "lon"), cut_var_pred[:, :, 1, :, :])
        sea_temp = (
            ("lead_time", "time", "depth", "lat", "lon"),
            cut_var_pred[:, :, 2:, :, :]
        )

        # 建立 xarray Dataset
        ds = xr.Dataset(
            data_vars={
                "taux": taux,
                "tauy": tauy,
                "sea temperature": sea_temp
            },
            coords={
                "lead_time": lead_vals,
                "time": time_vals,
                "depth": lev_vals,
                "lat": lat_vals,
                "lon": lon_vals
            },
            attrs={
                "Production model": "3D-Geoformer",
                "Description": "Multivariate prediction results in the tropical ocean from a self-attention-based neural network.",
            }
        )
        
        # 儲存
        ds.to_netcdf(output_path)
        print(f"✅ Saved NetCDF in expected Geoformer format: {output_path}")

# ---------------------------------------------------------
for i_file in files[: file_num + 1]:
    (cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true,) = func_pre(
        mypara=mypara,
        adr_model=i_file,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
    )
    # ---------------------------------------------------------
    save_prediction_to_netcdf(
        cut_var_pred=cut_var_pred,
        mypara=mypara,
        adr_oridata=adr_oridata,
        output_path="Geoformer_1983_2021_output.nc",
        start_time="1983-01-01"
    )
    # ---------------------------------------------------------    
    # 第一步：提取赤道上的 SST 和 τx 時間序列（橫跨所有年份）
    # 取得經度、時間軸
    lon_vals = mypara.lon_range
    time_vals = np.arange(len(cut_var_true)) # 或從原始資料讀 datetime
    
    # 找赤道最近緯度索引
    lat_vals = mypara.lat_range
    equ_idx = np.argmin(np.abs(lat_vals)) # 緯度最接近 0°
    
    # 取出真實 SST/τx 與預測值，沿赤切片
    sst_lev = 2
    sst_true = cut_var_true[:, sst_lev, equ_idx, :]
    taux_true = cut_var_true[:, 0, equ_idx, :]
    
    # 同樣針對預測：以 lead = 6, 9, 12 個月為例
    leads = [5, 8, 11] # 對應 lead=6, 9, 12
    sst_preds = [cut_var_pred[lead, :, sst_lev, equ_idx, :] for lead in leads]
    taux_preds = [cut_var_pred[lead, :, 0, equ_idx, :] for lead in leads]
    
    # 第二步：繪圖 S8
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    
    all_panels = [("a", "Analysis", sst_true, taux_true)]
    for i, lead in enumerate([6, 9, 12]):
        all_panels.append((chr(ord("b")+i), f"Lead={lead}", sst_preds[i], taux_preds[i]))
    
        for ax, (label, title, sst, taux) in zip(axes, all_panels):
            X, Y = np.meshgrid(lon_vals, np.arange(sst.shape[0]))
            ax.contourf(lon_vals, np.arange(sst.shape[0]), sst, levels=np.linspace(-2.5, 2.5, 21), cmap="RdBu_r", extend='both')
            ax.quiver(
            lon_vals[::3], np.arange(sst.shape[0])[::3],
            taux[::3, ::3], np.zeros_like(taux[::3, ::3]),
            scale=0.04, width=0.0025, headwidth=3
            )
            ax.axvline(x=180, color='k', linestyle='--') # dateline
            ax.set_title(f"({label}) {title}")
            ax.set_xlabel("Longitude")
            ax.set_xlim([120, 280])
            ax.set_xticks([120, 160, 200, 240, 280])
            axes[0].set_ylabel("Time (months since start)")
            plt.colorbar(axes[0].collections[0], ax=axes.ravel().tolist(), orientation='horizontal', pad=0.1, label='SST anomaly (°C)')
            plt.tight_layout()
            plt.savefig("ENSO_prediction.png", dpi=300)
