import os
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"
CLEANED_DIR = DATA_DIR / "cleaned"

CLEANED_DIR.mkdir(parents=True, exist_ok=True)

def clean_and_merge_plant(plant_no: int):
    print(f"=== [CLEANING] Start cleaning and merging for Plant {plant_no} ===")
    
    # 1. Định nghĩa đường dẫn file
    gen_file = DATA_DIR / f"Plant_{plant_no}_Generation_Data.csv"
    weather_file = DATA_DIR / f"Plant_{plant_no}_Weather_Sensor_Data.csv"
    api_file = DATA_DIR / f"open_meteo_weather_plant_{plant_no}.csv"
    
    # Đọc dữ liệu
    gen_df = pd.read_csv(gen_file)
    weather_df = pd.read_csv(weather_file)
    api_df = pd.read_csv(api_file)
    
    # 2. Chuẩn hóa DateTime
    if plant_no == 1:
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], format="%d-%m-%Y %H:%M", errors="coerce")
    else:
        gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], errors="coerce")
        
    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"], errors="coerce")
    api_df["DATE_TIME"] = pd.to_datetime(api_df["DATE_TIME"], errors="coerce")
    
    # Loại bỏ dòng lỗi DateTime
    gen_df = gen_df.dropna(subset=["DATE_TIME"])
    
    # 3. Gộp bước 1: Sản lượng + Cảm biến thời tiết thực địa
    # Lược bỏ các cột định danh trùng lặp ở file thời tiết
    weather_subset = weather_df[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    merged_df = pd.merge(gen_df, weather_subset, on="DATE_TIME", how="left")
    
    # 4. Gộp bước 2: Hợp nhất với thời tiết Open-Meteo đã crawl
    # Lược bỏ các cột metadata thừa
    api_drop_cols = ["PLANT_NO", "LATITUDE", "LONGITUDE", "OPEN_METEO_TIMEZONE", "OPEN_METEO_UTC_OFFSET_SECONDS"]
    api_subset = api_df.drop(columns=[col for col in api_drop_cols if col in api_df.columns])
    
    final_df = pd.merge(merged_df, api_subset, on="DATE_TIME", how="left")
    
    # 5. Nội suy tuyến tính xử lý khuyết thiếu (NaN) cho toàn bộ cột thời tiết
    weather_cols = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
    om_cols = [col for col in final_df.columns if col.startswith("OM_")]
    all_weather_cols = weather_cols + om_cols
    
    final_df[all_weather_cols] = final_df[all_weather_cols].interpolate(method="linear").bfill().ffill()
    
    # 6. Lưu file đã làm sạch
    output_path = CLEANED_DIR / f"plant_{plant_no}_cleaned.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"-> Completed! Cleaned file saved at: {output_path}")
    print(f"-> Data shape: {final_df.shape}\n")

def main():
    for p in [1, 2]:
        clean_and_merge_plant(p)

if __name__ == "__main__":
    main()
