from myconfig import mypara
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from my_tools import cal_ninoskill2, runmean
from func_for_prediction import func_pre

mpl.use("Agg")
plt.rc("font", family="Arial")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


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
for i_file in files[: file_num + 1]:
    fig1 = plt.figure(figsize=(5, 2.5), dpi=300)
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    (cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true,) = func_pre(
        mypara=mypara,
        adr_model=i_file,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
    )
    # ---------------------------------------------------------
    for l in range(lead_max):
        aa = runmean(cut_nino_pred_jx[l], 3)
        corr[l] = np.corrcoef(aa, bb)[0, 1]
        mse[l] = mean_squared_error(aa, bb)
        mae[l] = mean_absolute_error(aa, bb)
        del aa, bb
    # 第一步：提取赤道上的 SST 和 τx 時間序列（橫跨所有年份）
    # 取得經度、時間軸
    lon_vals = mypara.lon_values
    time_vals = np.arange(len(cut_var_true)) # 或從原始資料讀 datetime
    
    # 找赤道最近緯度索引
    lat_vals = mypara.lat_values
    equ_idx = np.argmin(np.abs(lat_vals)) # 緯度最接近 0°
    
    # 取出真實 SST/τx 與預測值，沿赤道切片
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
            plt.savefig("figure_S8_like.png", dpi=300)
