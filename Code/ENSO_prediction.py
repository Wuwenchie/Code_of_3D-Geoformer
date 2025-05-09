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
