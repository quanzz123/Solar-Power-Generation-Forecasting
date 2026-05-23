import sys
from pathlib import Path

# Thêm thư mục src vào hệ thống để Python nhận diện các module nội bộ
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

# Import các hàm từ các module trong thư mục src/
from cleaning import clean_and_merge_plant
from eda import perform_eda
from feature_engineering import engineer_features_for_plant
from split import split_time_series_for_plant
from model import train_and_evaluate_selected

def main():
    print("=====================================================================")
    print("    SOLAR POWER FORECASTING & DATA MINING PIPELINE (CRISP-DM)        ")
    print("=====================================================================")
    
    # BƯỚC 1: Làm sạch và Gộp dữ liệu
    print("\n[STEP 1] Running Data Cleaning & Merging...")
    for p in [1, 2]:
        clean_and_merge_plant(p)
        
    # BƯỚC 2: Khảo sát khám phá dữ liệu (EDA) và lưu biểu đồ vào Result/
    print("\n[STEP 2] Running Exploratory Data Analysis (EDA) & Saving Plots...")
    perform_eda()
    
    # BƯỚC 3: Kỹ thuật trích chọn đặc trưng (Feature Engineering)
    print("\n[STEP 3] Running Advanced Feature Engineering...")
    for p in [1, 2]:
        engineer_features_for_plant(p)
        
    # BƯỚC 4: Phân chia tập dữ liệu theo mốc thời gian chuỗi thời gian
    print("\n[STEP 4] Running Chronological Time-Series Splitting...")
    for p in [1, 2]:
        split_time_series_for_plant(p)
        
    # BƯỚC 5: Huấn luyện mô hình, kiểm thử, đánh giá & xuất đồ thị sai số
    print("\n[STEP 5] Running Model Training, Evaluation & Benchmarking...")
    # Cấu hình danh sách các mô hình bạn muốn chạy tại đây.
    # Lựa chọn có sẵn: ["linear_regression", "random_forest", "xgboost"]
    # Ví dụ: chỉ chạy ["xgboost"] nếu bạn chỉ muốn tập trung tối ưu hóa và chạy XGBoost.
    MODELS_TO_RUN = ["linear_regression", "random_forest", "xgboost"]
    
    train_and_evaluate_selected(models_to_run=MODELS_TO_RUN)
    
    print("=====================================================================")
    print("  SUCCESS: Full Data Mining Pipeline completed successfully end-to-end!")
    print("  Datasets are saved in: datasets/")
    print("  Reports and plots are saved in: Result/")
    print("=====================================================================")

if __name__ == "__main__":
    main()
