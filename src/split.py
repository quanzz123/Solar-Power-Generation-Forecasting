import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SPLIT_DIR = BASE_DIR / "datasets" / "split"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def split_time_series_for_plant(plant_no: int):
    print(f"=== [SPLIT] Start Train/Val/Test splitting for Plant {plant_no} ===")
    
    processed_file = PROCESSED_DIR / f"plant_{plant_no}_processed.csv"
    if not processed_file.exists():
        raise FileNotFoundError("Chưa có dữ liệu trích chọn đặc trưng. Hãy chạy feature_engineering.py trước!")
        
    df = pd.read_csv(processed_file)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    
    # Sắp xếp thời gian tăng dần
    df = df.sort_values(by="DATE_TIME").reset_index(drop=True)
    
    # Xác định các mốc thời gian phân tách
    min_date = df["DATE_TIME"].min()
    max_date = df["DATE_TIME"].max()
    
    # Mốc thời gian 1: Hết ngày thứ 20 (Dùng 20 ngày đầu để train)
    val_start_date = min_date + pd.Timedelta(days=20)
    # Mốc thời gian 2: Hết ngày thứ 27 (Dùng 7 ngày tiếp theo để validation)
    test_start_date = val_start_date + pd.Timedelta(days=7)
    
    # Thực hiện phân tách
    train_df = df[df["DATE_TIME"] < val_start_date]
    val_df = df[(df["DATE_TIME"] >= val_start_date) & (df["DATE_TIME"] < test_start_date)]
    test_df = df[df["DATE_TIME"] >= test_start_date]
    
    # 5. Lưu các tập dữ liệu phân mảnh
    train_df.to_csv(SPLIT_DIR / f"plant_{plant_no}_train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / f"plant_{plant_no}_val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / f"plant_{plant_no}_test.csv", index=False)
    
    # In báo cáo chi tiết
    print(f"-> Splitting dates:")
    print(f"   - Entire dataset: {min_date} to {max_date}")
    print(f"   - Training set (Train): {train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()} ({train_df.shape[0]} rows)")
    print(f"   - Validation set (Val): {val_df['DATE_TIME'].min()} to {val_df['DATE_TIME'].max()} ({val_df.shape[0]} rows)")
    print(f"   - Testing set (Test): {test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()} ({test_df.shape[0]} rows)")
    print(f"-> Splitted CSV fragments saved successfully to: {SPLIT_DIR}\n")

def main():
    for p in [1, 2]:
        split_time_series_for_plant(p)

if __name__ == "__main__":
    main()
