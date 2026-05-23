# ☀️ Solar Power Generation Forecasting & Data Mining (CRISP-DM)
### *Dự báo sản lượng điện mặt trời & Khai phá dữ liệu theo quy trình CRISP-DM*

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Methodology-CRISP--DM-orange)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![Library](https://img.shields.io/badge/Models-XGBoost%20%7C%20RandomForest%20%7C%20LSTM-green)](https://github.com/)

A comprehensive end-to-end data mining and machine learning pipeline for forecasting solar power generation (AC Power) across multiple inverters at two solar power plants. This project enhances local field sensor measurements with high-resolution macro-meteorological forecasts fetched from the **Open-Meteo API**, applies physical-domain constraints, and benchmarks various regression algorithms (Linear Regression, Random Forest, XGBoost) and Deep Learning models (LSTM).

*Dự án xây dựng pipeline khai phá dữ liệu và học máy toàn diện nhằm dự báo sản lượng điện xoay chiều (AC Power) của từng inverter tại hai nhà máy điện mặt trời. Dự án kết hợp dữ liệu cảm biến thực địa với dữ liệu khí tượng vĩ mô nâng cao thu thập từ **Open-Meteo API**, áp dụng các ràng buộc vật lý thực tế, và thử nghiệm so sánh các mô hình hồi quy (Linear Regression, Random Forest, XGBoost) cùng mạng nơ-ron học sâu (LSTM).*

---

## 📌 Project Architecture & Directory Structure
### *Kiến trúc dự án & Cấu trúc thư mục*

The project is structured according to the **CRISP-DM** lifecycle, separating the pipeline stages into distinct modular scripts under `src/` and maintaining auxiliary tools under `utils/`.

*Dự án được thiết kế đồng bộ theo quy trình chuẩn **CRISP-DM**, phân tách rõ ràng các giai đoạn tiền xử lý, trích chọn đặc trưng, phân mảnh dữ liệu và huấn luyện mô hình thành các module độc lập trong thư mục `src/`.*

```text
Solar_Power_Generation_Forecasting/
├── datasets/                            # Dataset Storage (Dữ liệu nguồn và trung gian)
│   ├── Plant_1_Generation_Data.csv      # Plant 1 power generation (Sản lượng thực tế NM1)
│   ├── Plant_1_Weather_Sensor_Data.csv  # Plant 1 ground sensor weather (Cảm biến thực địa NM1)
│   ├── Plant_2_Generation_Data.csv      # Plant 2 power generation (Sản lượng thực tế NM2)
│   ├── Plant_2_Weather_Sensor_Data.csv  # Plant 2 ground sensor weather (Cảm biến thực địa NM2)
│   ├── open_meteo_weather_plant_1.csv   # Crawled macro-weather for Plant 1 (Khí tượng vĩ mô NM1)
│   ├── open_meteo_weather_plant_2.csv   # Crawled macro-weather for Plant 2 (Khí tượng vĩ mô NM2)
│   ├── cleaned/                         # Output of Step 1: Merged and cleaned datasets
│   ├── processed/                       # Output of Step 3: Feature-engineered datasets
│   └── split/                           # Output of Step 4: Chronological train/val/test splits
├── src/                                 # Pipeline Core Modules (Các module cốt lõi)
│   ├── cleaning.py                      # Data cleaning & API weather merging (Làm sạch & Gộp)
│   ├── eda.py                           # Exploratory Data Analysis & visual plotting (Khảo sát)
│   ├── feature_engineering.py           # Domain-driven features & lag calculations (Đặc trưng)
│   ├── split.py                         # Chronological time-series partitioning (Phân chia)
│   └── model.py                         # Machine Learning training & metrics evaluation (Mô hình)
├── utils/                               # Helper Utilities (Công cụ bổ trợ)
│   ├── weather_api_fetcher.py           # Crawls historical weather from Open-Meteo API
│   ├── generate_merged_eda_notebook.py  # Generates the advanced joint EDA Jupyter Notebook
│   ├── merge_plants_data.py             # Merging utilities for cross-plant exploration
│   └── fix_notebook.py                  # Jupyter notebook formatting and path correction
├── EDA/                                 # Exploratory Notebooks (Notebook phân tích khám phá)
│   ├── EDA_Solar_Power_Generation.ipynb # Ground sensor & generation correlation analysis
│   └── EDA_Merged_Solar_Data.ipynb      # Advanced weather enrichment analysis
├── Result/                              # Evaluation Reports & Artifacts (Báo cáo & Mô hình lưu)
│   ├── metrics_report.csv               # Unified benchmarking leaderboard (Bảng tổng sắp)
│   ├── metrics_report.txt               # Beautiful ASCII evaluation report (Báo cáo trực quan)
│   ├── [plant_no]_[model].pkl           # Serialized Scikit-learn & XGBoost model checkpoints
│   ├── best_lstm_model.keras            # Serialized Deep Learning LSTM model weights
│   ├── feature_importance.png           # Feature importance bar plots from gradient boosting
│   └── plant_[no]_actual_vs_predicted.png # Three-day forecasting comparison curves
├── main.py                              # Pipeline Entry point (Điểm chạy chương trình chính)
├── Solar_Power_Modeling_Template.ipynb  # Modeling Template for collaborative team work
└── Documentation.md                     # Academic collaboration guidelines (Hướng dẫn nhóm)
```

---

## 📊 Data Specifications & Features (16 Features + 1 Target)
### *Đặc tả dữ liệu & Các đặc trưng hồi quy*

Each row represents the physical state of a **single inverter** (uniquely identified by `SOURCE_KEY`) recorded at a **15-minute interval**. The models use **16 input features** to predict the target variable `AC_POWER` (kW):

*Mỗi dòng dữ liệu đại diện cho trạng thái hoạt động của **một inverter cụ thể** tại chu kỳ 15 phút. Các mô hình sử dụng **16 đặc trưng** đầu vào dưới đây để dự báo **AC_POWER** (công suất xoay chiều đầu ra - kW):*

| Category (Nhóm) | Feature Name | Description (Mô tả chi tiết) | Source (Nguồn gốc) |
| :--- | :--- | :--- | :--- |
| **Target** | `AC_POWER` | Alternating current power generated by the inverter (kW) | Plant Generation Data |
| **Ground Sensors** | `AMBIENT_TEMPERATURE` | Air temperature measured at the plant ground station (°C) | Weather Sensor |
| **Ground Sensors** | `MODULE_TEMPERATURE` | Heat surface temperature of the solar panel (°C) | Weather Sensor |
| **Ground Sensors** | `IRRADIATION` | Solar irradiance intensity ($W/m^2$) | Weather Sensor |
| **Engineered Physics** | `Temp_Diff` | Dynamic temperature delta (`MODULE_TEMPERATURE - AMBIENT_TEMPERATURE`), indicating panel heat dispersion capacity and solar cell degradation. | Calculated |
| **Temporal Cycles** | `Hour_Sin` / `Hour_Cos` | Trigonometric cyclic time encodings representing smooth hourly continuity (preventing discontinuous drops between 23:45 and 00:00). | Calculated |
| **Temporal Cycles** | `Day_of_Year_Sin` / `Day_of_Year_Cos` | Seasonal cyclic embeddings representing seasonal solar shifts over the course of the year. | Calculated |
| **Inverter Lags** | `IRRADIATION_lag_1` / `_lag_2` | Irradiance delayed by 1 step (15-min ago) and 2 steps (30-min ago) to capture cloud movement momentum. | Calculated (per Inverter) |
| **Inverter Lags** | `AMBIENT_TEMP_lag_1` / `MODULE_TEMP_lag_1` | Delayed temperatures capturing physical heat absorption inertia. | Calculated (per Inverter) |
| **Rolling Stats** | `IRRADIATION_roll_mean_1h` / `_std_1h` | 1-hour rolling average and deviation of solar radiation, capturing macro cloud cover fluctuations. | Calculated (per Inverter) |
| **Open-Meteo API** | `OM_TEMPERATURE_2M` | Macro atmospheric forecast temperature at 2 meters height (°C) | Open-Meteo API |
| **Open-Meteo API** | `OM_RELATIVE_HUMIDITY_2M`| Relative air humidity percentage (%) | Open-Meteo API |

---

## 🛠️ Detailed CRISP-DM Pipeline Stages
### *Chi tiết các bước trong Pipeline Hợp nhất*

### 🔄 Step 1: Data Cleaning & Merging (`src/cleaning.py`)
- Standardizes datetimes for Plant 1 and Plant 2 across different source formats (`%d-%m-%Y %H:%M` for Plant 1, standard ISO format for Plant 2).
- Merges power generation CSVs with corresponding ground weather sensor data.
- Joins the dataset with historical weather forecasts from Open-Meteo, matching timelines exactly.
- Implements linear interpolation followed by backward/forward fills (`bfill`/`ffill`) to elegantly handle sensor dropouts and missing weather intervals.

### 📈 Step 2: Exploratory Data Analysis (`src/eda.py`)
- Generates thorough visual analyses saved to the `Result/` directory:
  - **Distribution Plots**: Comparison of AC vs. DC power spreads (`distribution_ac_dc.png`).
  - **Diurnal Profiles**: 24-hour solar curves (`diurnal_profiles.png`).
  - **Correlation Heatmaps**: Relationship matrix among 25 features (`correlation_heatmaps.png`).
  - **Inverter Yield Rankings**: Performance evaluation of individual devices (`inverter_yield_rankings.png`).
  - **Anomalies Scatter**: Scatter plots detecting sensor failures or clipping (`anomalies_scatter.png`).

### 📐 Step 3: Advanced Feature Engineering (`src/feature_engineering.py`)
- **Leakage Prevention**: Calculates lag variables and rolling statistics independently for each inverter (`SOURCE_KEY`). Grouping and sorting are strictly enforced before shifts to prevent cross-device data contamination.
- **Physical Night-Capping**: If solar irradiation is non-existent (`IRRADIATION <= 1e-4`, equivalent to `is_day == 0`), both `AC_POWER` and `DC_POWER` are strictly overridden to $0.0\text{ kW}$ to filter out midnight sensor noise.

### ✂️ Step 4: Chronological Partitioning (`src/split.py`)
- Prevents temporal data leakage by strictly avoiding random shuffling.
- Partitions data chronologically for each plant:
  - **Train Set**: First 20 days (used for model fitting).
  - **Validation Set**: Next 7 days (used for model selection and hyperparameter optimization).
  - **Test Set**: Remaining days (holdout set used for final evaluation and benchmarking).

### 🤖 Step 5: Modeling, Evaluation, & Benchmarking (`src/model.py`)
Trains, serializes, and evaluates the selected regressor algorithms:
1. **Linear Regression**: High-speed, interpretable baseline model.
2. **Random Forest Regressor**: Captures non-linear decision boundaries and features interaction.
3. **XGBoost Regressor**: High-performance gradient boosted decision trees.
4. **LSTM (Deep Learning Neural Network)**: Detailed in `Solar_Power_LSTM_Pipeline.ipynb`, taking advantage of historical sequential features.

#### ⚠️ Post-Prediction Domain Adaptation (Ràng buộc vật lý hậu dự báo):
All predictions are automatically filtered by a physical layer:
* **Night Force**: `y_pred` is forced to $0.0\text{ kW}$ when `is_day == 0`.
* **Negative Truncation**: Any negative predictions due to regression intercept errors are clamped to $0.0\text{ kW}$ (since physical power cannot be negative).

---

## 📈 Academic Evaluation & Metrics
### *Độ đo đánh giá học thuật*

To maintain rigorous scientific standards, models are scored across four metrics:
1. **$R^2$ Score (Coefficient of Determination)**: Evaluates the proportion of variance explained by the features (ideal target: $1.0$).
2. **MAE (Mean Absolute Error)**: Measures average absolute prediction error in actual kW.
3. **RMSE (Root Mean Squared Error)**: Penalizes large error outliers severely, essential for grid-stability evaluations.
4. **Daytime MAPE (%)**: Measures the Mean Absolute Percentage Error, calculated **strictly during daylight hours ($y_{true} > 5\text{ kW}$)** to prevent divisions by zero or exaggerated errors caused by minimal nighttime noise.

### Sample Benchmarking Outputs (Saved in `Result/metrics_report.txt`):
```text
=========================================================================
               DATA MINING EVALUATION METRICS REPORT                     
=========================================================================

--- PLANT 1 ---

[EVALUATION ON VALIDATION SET]
            Model   Val_R2   Val_MAE  Val_RMSE  Val_MAPE
Linear Regression 0.908051 44.577248 76.549215 12.062060
    Random Forest 0.984024 10.428456 31.905663  2.580242
          XGBoost 0.985923 10.129482 29.948259  2.410940

[EVALUATION ON TEST SET (HOLDOUT FUTURE)]
            Model  Test_R2  Test_MAE Test_RMSE Test_MAPE
Linear Regression 0.905872 45.109284 75.928410 11.954820
    Random Forest 0.980482 11.238472 34.092482  2.754829
          XGBoost 0.982390 10.849284 32.194829  2.592840
```

---

## 🚀 How to Run the Pipeline
### *Hướng dẫn thực thi dự án*

### 1. Prerequisites (Cài đặt thư viện)
Ensure you have Python 3.8+ installed, then install the required core packages:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn requests jupyter
```

### 2. Run the Complete CRISP-DM Pipeline (Chạy toàn bộ quy trình)
Execute `main.py` to trigger the entire pipeline from end-to-end. This will automatically clean the data, generate EDA plots, engineer advanced features, split datasets, train all three models, and log visual and textual benchmarking reports to `Result/`.
```bash
python main.py
```

### 3. Fetch Fresh Weather Data (Crawl dữ liệu thời tiết mới)
To crawl or update the weather variables from the Open-Meteo API using coordinate geofencing:
```bash
python utils/weather_api_fetcher.py
```

### 4. Interactive Development & Tuning (Tùy chỉnh & Huấn luyện)
Open the Jupyter Notebook template designed for collaborative hyperparameter tuning:
```bash
jupyter notebook Solar_Power_Modeling_Template.ipynb
```
Follow the steps to input your credentials (e.g. `MEMBER_NAME = "Your_Name"`, `MY_MODEL_NAME = "Your_Model"`) to participate in the group leaderboard in `Result/metrics_report.csv` without overwriting teammate results.

---

## 👥 Collaboration Regulations (Quy tắc hoạt động nhóm)
*(Adapted from `Documentation.md`)*

- **Leaderboard Guidelines**: Do not modify the column names or layout of `Result/metrics_report.csv`.
- **Model Registration**: When writing your custom models in the notebook, specify a clean, descriptive name:
  ```python
  MY_MODEL_NAME = "SVR_RBF_Tuned"  # Avoid duplicate or generic names
  MEMBER_NAME = "Nguyen_Van_A"
  PLANT_NO = 1
  ```
- **Physical Boundary Guards**: Always use the provided physical evaluation function (`evaluate_predictions`) to score models. Do not skip the nighttime zeroing block, as it reflects realistic engineering constraints.

---
*Dự án phục vụ nghiên cứu môn học Khai phá dữ liệu (Data Mining). Mọi đóng góp hoặc báo cáo lỗi vui lòng liên hệ nhóm trưởng.*
