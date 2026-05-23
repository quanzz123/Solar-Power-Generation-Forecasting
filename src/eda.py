import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "datasets" / "cleaned"
RESULT_DIR = BASE_DIR / "Result"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình phong cách biểu đồ cao cấp
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6.5)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def perform_eda():
    print("=== [EDA] Start Exploratory Data Analysis & Plotting ===")
    
    # 1. Đọc dữ liệu đã làm sạch
    p1_file = CLEANED_DIR / "plant_1_cleaned.csv"
    p2_file = CLEANED_DIR / "plant_2_cleaned.csv"
    
    if not p1_file.exists() or not p2_file.exists():
        raise FileNotFoundError("Chưa có dữ liệu làm sạch. Hãy chạy cleaning.py trước!")
        
    p1 = pd.read_csv(p1_file)
    p2 = pd.read_csv(p2_file)
    
    p1["DATE_TIME"] = pd.to_datetime(p1["DATE_TIME"])
    p2["DATE_TIME"] = pd.to_datetime(p2["DATE_TIME"])
    
    # Trích xuất cột giờ phục vụ phân tích
    p1["Hour"] = p1["DATE_TIME"].dt.hour
    p2["Hour"] = p2["DATE_TIME"].dt.hour
    
    p1_day = p1[p1["AC_POWER"] > 0.5]
    p2_day = p2[p2["AC_POWER"] > 0.5]
    
    # -------------------------------------------------------------
    # BIỂU ĐỒ 1: Phân phối sản lượng AC/DC ban ngày
    # -------------------------------------------------------------
    print("-> Plotting Chart 1: AC/DC Power Distribution...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    sns.histplot(p1_day["AC_POWER"], bins=30, kde=True, color="orange", ax=axes[0, 0])
    axes[0, 0].set_title("Nhà máy 1 - Phân phối Công suất xoay chiều AC (Daylight)")
    axes[0, 0].set_xlabel("AC Power (kW)")
    
    sns.histplot(p1_day["DC_POWER"], bins=30, kde=True, color="goldenrod", ax=axes[0, 1])
    axes[0, 1].set_title("Nhà máy 1 - Phân phối Công suất một chiều DC (Daylight)")
    axes[0, 1].set_xlabel("DC Power (kW)")
    
    sns.histplot(p2_day["AC_POWER"], bins=30, kde=True, color="cyan", ax=axes[1, 0])
    axes[1, 0].set_title("Nhà máy 2 - Phân phối Công suất xoay chiều AC (Daylight)")
    axes[1, 0].set_xlabel("AC Power (kW)")
    
    sns.histplot(p2_day["DC_POWER"], bins=30, kde=True, color="teal", ax=axes[1, 1])
    axes[1, 1].set_title("Nhà máy 2 - Phân phối Công suất một chiều DC (Daylight)")
    axes[1, 1].set_xlabel("DC Power (kW)")
    
    plt.tight_layout()
    plot_path_1 = RESULT_DIR / "distribution_ac_dc.png"
    plt.savefig(plot_path_1, dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # BIỂU ĐỒ 2: Chu kỳ ngày đêm (Diurnal Profiles) cho Nhà máy 1
    # -------------------------------------------------------------
    print("-> Plotting Chart 2: Diurnal Hourly Profiles...")
    p1_hourly = p1.groupby("Hour")[["AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE"]].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(14, 6.5))
    
    # Trục Y trái: AC Power
    color_power = "tab:orange"
    ax1.set_xlabel("Giờ trong ngày (Hour)")
    ax1.set_ylabel("Công suất AC trung bình (kW)", color=color_power)
    line1 = ax1.plot(p1_hourly["Hour"], p1_hourly["AC_POWER"], color=color_power, linewidth=3, marker="o", label="AC Power (kW)")
    ax1.tick_params(axis="y", labelcolor=color_power)
    ax1.set_xticks(range(0, 24))
    
    # Trục Y phải: Bức xạ & Nhiệt độ
    ax2 = ax1.twinx()
    color_irrad = "tab:blue"
    line2 = ax2.plot(p1_hourly["Hour"], p1_hourly["IRRADIATION"], color=color_irrad, linewidth=2, linestyle="--", label="Bức xạ (W/m²)")
    ax2.tick_params(axis="y")
    ax2.set_ylabel("Cường độ bức xạ (W/m²) | Nhiệt độ (°C)", color="tab:blue")
    
    color_temp = "tab:red"
    line3 = ax2.plot(p1_hourly["Hour"], p1_hourly["MODULE_TEMPERATURE"], color=color_temp, linewidth=2, linestyle=":", label="Nhiệt độ tấm pin (°C)")
    
    # Ghép legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")
    
    plt.title("Nhà máy 1 - Chu kỳ tương quan mật thiết giữa Công suất, Bức xạ và Nhiệt độ theo Giờ")
    
    plot_path_2 = RESULT_DIR / "diurnal_profiles.png"
    plt.savefig(plot_path_2, dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # BIỂU ĐỒ 3: Ma trận tương quan đa biến (Pearson Correlation Heatmap)
    # -------------------------------------------------------------
    print("-> Plotting Chart 3: Correlation Heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    
    # Chọn lọc một số đặc trưng để vẽ heatmap dễ nhìn
    corr_cols = [
        "AC_POWER", "DC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION",
        "OM_TEMPERATURE_2M", "OM_RELATIVE_HUMIDITY_2M", "OM_CLOUD_COVER", "OM_WIND_SPEED_10M"
    ]
    
    # Clean column names for neat visual display
    display_cols = [col.replace("OM_", "").replace("_", " ").title() for col in corr_cols]
    
    # Plant 1 correlation
    df1_corr = p1[corr_cols].corr()
    df1_corr.columns = display_cols
    df1_corr.index = display_cols
    sns.heatmap(df1_corr, annot=True, cmap="Oranges", fmt=".2f", square=True, ax=axes[0], cbar_kws={"shrink": 0.8})
    axes[0].set_title("Nhà máy 1 - Ma trận tương quan Pearson")
    
    # Plant 2 correlation
    df2_corr = p2[corr_cols].corr()
    df2_corr.columns = display_cols
    df2_corr.index = display_cols
    sns.heatmap(df2_corr, annot=True, cmap="YlGnBu", fmt=".2f", square=True, ax=axes[1], cbar_kws={"shrink": 0.8})
    axes[1].set_title("Nhà máy 2 - Ma trận tương quan Pearson")
    
    plt.tight_layout()
    plot_path_3 = RESULT_DIR / "correlation_heatmaps.png"
    plt.savefig(plot_path_3, dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # BIỂU ĐỒ 4: Khai phá dị thường Scatter Plot Bức xạ vs AC Power
    # -------------------------------------------------------------
    print("-> Plotting Chart 4: Anomalies Scatter Plots...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.scatterplot(data=p1, x="IRRADIATION", y="AC_POWER", alpha=0.25, color="orange", ax=axes[0])
    axes[0].set_title("Nhà máy 1 - Biểu đồ phân tán Bức xạ vs AC Power (kW)")
    axes[0].set_xlabel("Cường độ bức xạ (W/m²)")
    axes[0].set_ylabel("Công suất xoay chiều AC (kW)")
    
    sns.scatterplot(data=p2, x="IRRADIATION", y="AC_POWER", alpha=0.25, color="teal", ax=axes[1])
    axes[1].set_title("Nhà máy 2 - Biểu đồ phân tán Bức xạ vs AC Power (kW)")
    axes[1].set_xlabel("Cường độ bức xạ (W/m²)")
    axes[1].set_ylabel("Công suất xoay chiều AC (kW)")
    
    plt.tight_layout()
    plot_path_4 = RESULT_DIR / "anomalies_scatter.png"
    plt.savefig(plot_path_4, dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # BIỂU ĐỒ 5: Xếp hạng tổng sản lượng Inverter (Outlier Yield Rankings)
    # -------------------------------------------------------------
    print("-> Plotting Chart 5: Inverter Yield Rankings...")
    p1_inv = p1.groupby("SOURCE_KEY")["AC_POWER"].sum().sort_values().reset_index()
    p1_inv["Yield_MWh"] = p1_inv["AC_POWER"] / 4.0 / 1000.0
    
    p2_inv = p2.groupby("SOURCE_KEY")["AC_POWER"].sum().sort_values().reset_index()
    p2_inv["Yield_MWh"] = p2_inv["AC_POWER"] / 4.0 / 1000.0
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    sns.barplot(data=p1_inv, x="Yield_MWh", y="SOURCE_KEY", ax=axes[0], palette="Oranges_r")
    axes[0].set_title("Nhà máy 1 - Tổng sản lượng tích lũy của các Inverter (34 ngày)")
    axes[0].set_xlabel("Sản lượng điện năng phát lên lưới (MWh)")
    axes[0].set_ylabel("SOURCE_KEY")
    
    sns.barplot(data=p2_inv, x="Yield_MWh", y="SOURCE_KEY", ax=axes[1], palette="GnBu_r")
    axes[1].set_title("Nhà máy 2 - Tổng sản lượng tích lũy của các Inverter (34 ngày)")
    axes[1].set_xlabel("Sản lượng điện năng phát lên lưới (MWh)")
    axes[1].set_ylabel("SOURCE_KEY")
    
    plt.tight_layout()
    plot_path_5 = RESULT_DIR / "inverter_yield_rankings.png"
    plt.savefig(plot_path_5, dpi=150)
    plt.close()
    
    print("=== [EDA] Completed successfully! All 5 plots saved to Result/ ===\n")

if __name__ == "__main__":
    perform_eda()
