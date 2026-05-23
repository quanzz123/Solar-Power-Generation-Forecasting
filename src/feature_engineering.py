import numpy as np
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "datasets" / "cleaned"
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def engineer_features_for_plant(plant_no: int):
    print(f"=== [FEATURE ENGINEERING] Start engineering features for Plant {plant_no} ===")
    
    cleaned_file = CLEANED_DIR / f"plant_{plant_no}_cleaned.csv"
    if not cleaned_file.exists():
        raise FileNotFoundError("Chưa có dữ liệu làm sạch. Hãy chạy cleaning.py trước!")
        
    df = pd.read_csv(cleaned_file)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    
    # 1. Đặc trưng ngày đêm vật lý (is_day) & Chênh lệch nhiệt độ tấm pin (Temp_Diff)
    # Xác định có ngày bằng cảm biến bức xạ mặt trời
    df["is_day"] = (df["IRRADIATION"] > 1e-4).astype(int)
    
    # Chênh lệch nhiệt độ phản ánh tản nhiệt và lão hóa
    df["Temp_Diff"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    
    # 2. Đặc trưng tuần hoàn lượng giác (Sin/Cos Cyclical Encoding)
    df["Hour"] = df["DATE_TIME"].dt.hour
    df["Minute"] = df["DATE_TIME"].dt.minute
    df["Day_of_Year"] = df["DATE_TIME"].dt.dayofyear
    
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["Day_of_Year_Sin"] = np.sin(2 * np.pi * df["Day_of_Year"] / 365.25)
    df["Day_of_Year_Cos"] = np.cos(2 * np.pi * df["Day_of_Year"] / 365.25)
    
    # 3. Đặc trưng trễ (Lags) & Cửa sổ cuộn (Rolling Window) theo TỪNG INVERTER riêng lẻ
    # Sắp xếp thời gian theo từng SOURCE_KEY trước khi tính toán để tránh rò rỉ dữ liệu chéo thiết bị
    df = df.sort_values(by=["SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)
    
    # A. Trích chọn đặc trưng trễ (Lags)
    df["IRRADIATION_lag_1"] = df.groupby("SOURCE_KEY")["IRRADIATION"].shift(1)
    df["IRRADIATION_lag_2"] = df.groupby("SOURCE_KEY")["IRRADIATION"].shift(2)
    df["AMBIENT_TEMP_lag_1"] = df.groupby("SOURCE_KEY")["AMBIENT_TEMPERATURE"].shift(1)
    df["MODULE_TEMP_lag_1"] = df.groupby("SOURCE_KEY")["MODULE_TEMPERATURE"].shift(1)
    
    # B. Trích chọn thống kê cửa sổ cuộn trượt 1 giờ (4 mốc 15 phút)
    # Trung bình cuộn và Độ lệch chuẩn cuộn phản ánh độ biến động thời tiết
    df["IRRADIATION_roll_mean_1h"] = df.groupby("SOURCE_KEY")["IRRADIATION"].transform(lambda x: x.shift(1).rolling(4).mean())
    df["IRRADIATION_roll_std_1h"] = df.groupby("SOURCE_KEY")["IRRADIATION"].transform(lambda x: x.shift(1).rolling(4).std())
    
    # C. Xử lý các giá trị NaN ở biên của từng nhóm inverter do dịch chuyển trễ
    lag_and_roll_cols = [
        "IRRADIATION_lag_1", "IRRADIATION_lag_2", "AMBIENT_TEMP_lag_1", "MODULE_TEMP_lag_1",
        "IRRADIATION_roll_mean_1h", "IRRADIATION_roll_std_1h"
    ]
    df[lag_and_roll_cols] = df.groupby("SOURCE_KEY")[lag_and_roll_cols].bfill().ffill()
    
    # 4. Ràng buộc vật lý ban đêm: Nếu is_day == 0 thì công suất bắt buộc bằng 0.0 kW
    # Tránh nhiễu cảm biến trạm đo
    df.loc[df["is_day"] == 0, "AC_POWER"] = 0.0
    df.loc[df["is_day"] == 0, "DC_POWER"] = 0.0
    
    # 5. Lưu bộ dữ liệu đặc trưng nâng cao
    output_path = PROCESSED_DIR / f"plant_{plant_no}_processed.csv"
    df.to_csv(output_path, index=False)
    
    print(f"-> Completed! Processed file saved at: {output_path}")
    print(f"-> Data shape: {df.shape}\n")

def main():
    for p in [1, 2]:
        engineer_features_for_plant(p)

if __name__ == "__main__":
    main()
