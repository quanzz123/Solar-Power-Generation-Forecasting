import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "datasets" / "split"
RESULT_DIR = BASE_DIR / "Result"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6.5)

# =====================================================================
# HÀM BỔ TRỢ: ĐÁNH GIÁ VÀ ÁP DỤNG RÀNG BUỘC VẬT LÝ HẬU DỰ BÁO
# =====================================================================
def evaluate_predictions(y_true, y_pred_raw, df_split, model_name, plant_no):
    # Sao chép để tránh thay đổi dữ liệu gốc
    y_pred = y_pred_raw.copy()
    
    # Ràng buộc vật lý: nếu là ban đêm (is_day == 0) thì công suất bắt buộc bằng 0
    y_pred[df_split["is_day"] == 0] = 0.0
    
    # Cắt cụt toàn bộ giá trị âm phi vật lý thành 0
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    
    # Tính toán các chỉ số sai số khoa học
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Tính toán MAPE chỉ trong thời gian ban ngày (y_true > 5 kW) để tránh lỗi chia cho 0
    daylight_mask = y_true > 5.0
    mape = np.mean(np.abs((y_true[daylight_mask] - y_pred[daylight_mask]) / y_true[daylight_mask])) * 100 if daylight_mask.sum() > 0 else 0.0
    
    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "predictions": y_pred
    }

# =====================================================================
# MÔ HÌNH HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)
# =====================================================================
def train_linear_regression(X_train, y_train, X_val, y_val, X_test, y_test, val_df, test_df, plant_no):
    print("   [TRAIN] Training model: Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Dự đoán và Đánh giá
    val_metrics = evaluate_predictions(y_val, model.predict(X_val), val_df, "Linear Regression", plant_no)
    test_metrics = evaluate_predictions(y_test, model.predict(X_test), test_df, "Linear Regression", plant_no)
    
    # Lưu mô hình
    model_path = RESULT_DIR / f"plant_{plant_no}_linear_regression.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    return model, val_metrics, test_metrics

# =====================================================================
# MÔ HÌNH RANDOM FOREST (Với khả năng truyền tham số kwargs tối ưu hóa)
# =====================================================================
def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, val_df, test_df, plant_no, **kwargs):
    print("   [TRAIN] Training model: Random Forest...")
    
    # Các tham số mặc định, có thể ghi đè bởi kwargs để phục vụ tối ưu hóa tham số sau này
    rf_params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1
    }
    rf_params.update(kwargs)
    
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    
    # Dự đoán và Đánh giá
    val_metrics = evaluate_predictions(y_val, model.predict(X_val), val_df, "Random Forest", plant_no)
    test_metrics = evaluate_predictions(y_test, model.predict(X_test), test_df, "Random Forest", plant_no)
    
    # Lưu mô hình
    model_path = RESULT_DIR / f"plant_{plant_no}_random_forest.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    return model, val_metrics, test_metrics

# =====================================================================
# MÔ HÌNH GRADIENT BOOSTING XGBOOST (Với khả năng tối ưu hóa tham số)
# =====================================================================
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, val_df, test_df, plant_no, **kwargs):
    print("   [TRAIN] Training model: XGBoost...")
    
    # Các tham số mặc định, có thể ghi đè bởi kwargs phục vụ Tuning sau này
    xgb_params = {
        "n_estimators": 120,
        "max_depth": 6,
        "learning_rate": 0.08,
        "random_state": 42,
        "n_jobs": -1
    }
    xgb_params.update(kwargs)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # Dự đoán và Đánh giá
    val_metrics = evaluate_predictions(y_val, model.predict(X_val), val_df, "XGBoost", plant_no)
    test_metrics = evaluate_predictions(y_test, model.predict(X_test), test_df, "XGBoost", plant_no)
    
    # Lưu mô hình
    model_path = RESULT_DIR / f"plant_{plant_no}_xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    return model, val_metrics, test_metrics

# =====================================================================
# HÀM ĐIỀU PHỐI CHÍNH: CHỈ CHẠY CÁC MÔ HÌNH ĐƯỢC CHỌN (SELECTIVE)
# =====================================================================
def train_and_evaluate_selected(models_to_run=["linear_regression", "random_forest", "xgboost"], **tuning_args):
    print(f"=== [MODELING] Start training selected models: {models_to_run} ===")
    
    # 16 Đặc trưng đầu vào hoàn chỉnh
    features = [
        "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "Temp_Diff",
        "Hour_Sin", "Hour_Cos", "Day_of_Year_Sin", "Day_of_Year_Cos",
        "IRRADIATION_lag_1", "IRRADIATION_lag_2", "AMBIENT_TEMP_lag_1", "MODULE_TEMP_lag_1",
        "IRRADIATION_roll_mean_1h", "IRRADIATION_roll_std_1h",
        "OM_TEMPERATURE_2M", "OM_RELATIVE_HUMIDITY_2M"
    ]
    target = "AC_POWER"
    
    xgb_importances = {}
    report_rows = [] # Danh sách thu thập kết quả phục vụ xuất file report
    
    for plant_no in [1, 2]:
        print(f"\n--- PLANT {plant_no} ---")
        
        # Đọc dữ liệu phân mảnh
        train_df = pd.read_csv(SPLIT_DIR / f"plant_{plant_no}_train.csv")
        val_df = pd.read_csv(SPLIT_DIR / f"plant_{plant_no}_val.csv")
        test_df = pd.read_csv(SPLIT_DIR / f"plant_{plant_no}_test.csv")
        
        available_features = [col for col in features if col in train_df.columns]
        
        X_train, y_train = train_df[available_features], train_df[target]
        X_val, y_val = val_df[available_features], val_df[target]
        X_test, y_test = test_df[available_features], test_df[target]
        
        # Áp dụng StandardScaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=available_features)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=available_features)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=available_features)
        
        # Bộ lưu trữ chỉ số và dự báo
        val_results = {}
        test_results = {}
        test_predictions = {}
        
        # ----------------------------------------------------
        # CHỈ CHẠY CÁC MÔ HÌNH CÓ TRONG DANH SÁCH models_to_run
        # ----------------------------------------------------
        
        # A. Hồi quy tuyến tính
        if "linear_regression" in models_to_run:
            lr_model, val_m, test_m = train_linear_regression(
                X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, val_df, test_df, plant_no
            )
            val_results["Linear Regression"] = val_m
            test_results["Linear Regression"] = test_m
            test_predictions["Linear Regression"] = test_m["predictions"]
            
        # B. Random Forest
        if "random_forest" in models_to_run:
            rf_kwargs = tuning_args.get("random_forest", {})
            rf_model, val_m, test_m = train_random_forest(
                X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, val_df, test_df, plant_no, **rf_kwargs
            )
            val_results["Random Forest"] = val_m
            test_results["Random Forest"] = test_m
            test_predictions["Random Forest"] = test_m["predictions"]
            
        # C. XGBoost
        if "xgboost" in models_to_run:
            xgb_kwargs = tuning_args.get("xgboost", {})
            xgb_model, val_m, test_m = train_xgboost(
                X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, val_df, test_df, plant_no, **xgb_kwargs
            )
            val_results["XGBoost"] = val_m
            test_results["XGBoost"] = test_m
            test_predictions["XGBoost"] = test_m["predictions"]
            
            # Lưu Feature Importance của XGBoost
            if hasattr(xgb_model, "feature_importances_"):
                importances = xgb_model.feature_importances_
                xgb_importances[f"plant_{plant_no}"] = sorted(
                    [{"feature": f, "importance": imp} for f, imp in zip(available_features, importances)],
                    key=lambda x: x["importance"],
                    reverse=True
                )

        # Thu thập các dòng kết quả cho báo cáo
        for name in val_results.keys():
            report_rows.append({
                "Plant": f"Plant {plant_no}",
                "Model": name,
                "Val_R2": val_results[name]["R2"],
                "Val_MAE": val_results[name]["MAE"],
                "Val_RMSE": val_results[name]["RMSE"],
                "Val_MAPE": val_results[name]["MAPE"],
                "Test_R2": test_results[name]["R2"],
                "Test_MAE": test_results[name]["MAE"],
                "Test_RMSE": test_results[name]["RMSE"],
                "Test_MAPE": test_results[name]["MAPE"]
            })

        # ----------------------------------------------------
        # IN BẢNG CHỈ SỐ SAI SỐ HỌC THUẬT LÊN CONSOLE
        # ----------------------------------------------------
        if val_results:
            print("\n   [EVALUATION ON VALIDATION SET]")
            print("   " + "="*80)
            print(f"   {'Algorithm (Model)':<22} | {'R2 Score':<10} | {'MAE (kW)':<10} | {'RMSE (kW)':<11} | {'MAPE (Daylight %)'}")
            print("   " + "-"*80)
            for name in val_results.keys():
                m = val_results[name]
                print(f"   {name:<22} | {m['R2']:<10.4f} | {m['MAE']:<10.2f} | {m['RMSE']:<11.2f} | {m['MAPE']:.2f}%")
            print("   " + "="*80)
            
            print("\n   [EVALUATION ON TEST SET (HOLDOUT FUTURE)]")
            print("   " + "="*80)
            print(f"   {'Algorithm (Model)':<22} | {'R2 Score':<10} | {'MAE (kW)':<10} | {'RMSE (kW)':<11} | {'MAPE (Daylight %)'}")
            print("   " + "-"*80)
            for name in test_results.keys():
                m = test_results[name]
                print(f"   {name:<22} | {m['R2']:<10.4f} | {m['MAE']:<10.2f} | {m['RMSE']:<11.2f} | {m['MAPE']:.2f}%")
            print("   " + "="*80 + "\n")
            
        # ----------------------------------------------------
        # VẼ ĐỒ THỊ SO SÁNH THỰC TẾ VS DỰ BÁO CỦA MÔ HÌNH MẪU
        # ----------------------------------------------------
        # Lấy mô hình tốt nhất có sẵn trong models_to_run để vẽ đồ thị mẫu
        best_model_name = None
        for candidate in ["XGBoost", "Random Forest", "Linear Regression"]:
            if candidate in test_results:
                best_model_name = candidate
                break
                
        if best_model_name:
            print(f"-> Plotting actual vs. predicted curves for {best_model_name}...")
            test_df["DATE_TIME"] = pd.to_datetime(test_df["DATE_TIME"])
            sample_inverter = test_df["SOURCE_KEY"].iloc[0]
            sample_mask = test_df["SOURCE_KEY"] == sample_inverter
            
            plot_df = test_df[sample_mask].sort_values(by="DATE_TIME")
            start_date = plot_df["DATE_TIME"].min()
            end_date = start_date + pd.Timedelta(days=3)
            plot_subset = plot_df[(plot_df["DATE_TIME"] >= start_date) & (plot_df["DATE_TIME"] <= end_date)]
            
            plt.figure(figsize=(14, 6.5))
            plt.plot(plot_subset["DATE_TIME"], plot_subset["AC_POWER"], label="Actual", color="black", linewidth=2.5)
            plt.plot(
                plot_subset["DATE_TIME"], 
                test_predictions[best_model_name][(test_df["SOURCE_KEY"] == sample_inverter) & (test_df["DATE_TIME"] >= start_date) & (test_df["DATE_TIME"] <= end_date)],
                label=f"Predicted ({best_model_name})", 
                color="orange", 
                linewidth=2, 
                linestyle="--"
            )
            
            plt.title(f"Plant {plant_no} - Actual vs. Predicted 3-Day Test Curve (Inverter: {sample_inverter})")
            plt.xlabel("DateTime")
            plt.ylabel("AC Power (kW)")
            plt.legend()
            plt.tight_layout()
            
            plot_path = RESULT_DIR / f"plant_{plant_no}_actual_vs_predicted.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

    # ----------------------------------------------------
    # VẼ TẦM QUAN TRỌNG ĐẶC TRƯNG NẾU XGBOOST ĐƯỢC CHẠY
    # ----------------------------------------------------
    if "xgboost" in models_to_run and xgb_importances:
        print("-> Plotting feature importance bar charts...")
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        for idx, plant_no in enumerate([1, 2]):
            imp_data = xgb_importances[f"plant_{plant_no}"][:10]
            features_lbl = [item["feature"] for item in imp_data]
            importances_val = [item["importance"] for item in imp_data]
            
            sns.barplot(x=importances_val, y=features_lbl, ax=axes[idx], palette="Blues_r")
            axes[idx].set_title(f"Plant {plant_no} - Feature Importance (XGBoost)")
            axes[idx].set_xlabel("F-Score (Gini Importance)")
            
        plt.tight_layout()
        plt.savefig(RESULT_DIR / "feature_importance.png", dpi=150)
        plt.close()
        
    # ----------------------------------------------------
    # XUẤT CÁC BÁO CÁO KẾT QUẢ ĐỘ ĐO RA THƯ MỤC RESULT/
    # ----------------------------------------------------
    if report_rows:
        print("-> Writing evaluation reports to Result/...")
        report_df = pd.DataFrame(report_rows)
        
        # A. Xuất CSV phục vụ mở Excel tạo bảng
        report_df.to_csv(RESULT_DIR / "metrics_report.csv", index=False)
        
        # B. Xuất TXT căn lề ASCII tuyệt đẹp để copy Word trực tiếp
        with open(RESULT_DIR / "metrics_report.txt", "w", encoding="utf-8") as f:
            f.write("=========================================================================\n")
            f.write("               DATA MINING EVALUATION METRICS REPORT                     \n")
            f.write("=========================================================================\n\n")
            
            for p_name in ["Plant 1", "Plant 2"]:
                f.write(f"--- {p_name.upper()} ---\n")
                p_df = report_df[report_df["Plant"] == p_name]
                
                f.write("\n[EVALUATION ON VALIDATION SET]\n")
                f.write(p_df[["Model", "Val_R2", "Val_MAE", "Val_RMSE", "Val_MAPE"]].to_string(index=False))
                f.write("\n\n[EVALUATION ON TEST SET (HOLDOUT FUTURE)]\n")
                f.write(p_df[["Model", "Test_R2", "Test_MAE", "Test_RMSE", "Test_MAPE"]].to_string(index=False))
                f.write("\n\n" + "="*80 + "\n\n")
                
        print("-> Successfully saved metrics_report.csv and metrics_report.txt to Result/!")
        
    print("=== [MODELING] Completed successfully! ===\n")

if __name__ == "__main__":
    # Mặc định chạy cả 3 mô hình nếu gọi trực tiếp
    train_and_evaluate_selected()
