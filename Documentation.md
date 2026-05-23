# ☀️ HƯỚNG DẪN HỢP TÁC VÀ HUẤN LUYỆN MÔ HÌNH NHÓM
## Dự án: Khai phá dữ liệu & Dự báo sản lượng điện mặt trời (CRISP-DM)

Chào mừng các thành viên trong nhóm nghiên cứu! Tài liệu này được biên soạn để hướng dẫn các bạn cách sử dụng **Jupyter Notebook mẫu** [Solar_Power_Modeling_Template.ipynb](file:///d:/WORKSPACE/KPDL/Solar_Power_Generation_Forecasting/Solar_Power_Modeling_Template.ipynb) nhằm xây dựng, tinh chỉnh tham số và đánh giá mô hình của mình một cách đồng bộ và chính xác nhất.

---

## 1. Tổng quan cấu trúc thư mục Dự án
Để đảm bảo mã nguồn của bạn chạy mượt mà, cấu trúc thư mục làm việc của dự án phải được tổ chức thống nhất như sau:
```text
Solar_Power_Generation_Forecasting/
├── datasets/
│   └── split/                     <-- Nơi chứa dữ liệu đã phân chia sẵn của nhóm trưởng
│       ├── plant_1_train.csv
│       ├── plant_1_val.csv
│       ├── plant_1_test.csv
│       ├── plant_2_train.csv
│       ├── plant_2_val.csv
│       └── plant_2_test.csv
├── Result/                        <-- Thư mục tự động lưu biểu đồ, mô hình và báo cáo chung
│       └── metrics_report.csv     <-- Bảng xếp hạng điểm số dùng chung của cả 5 thành viên
├── Solar_Power_Modeling_Template.ipynb <-- Notebook mẫu gửi cho các thành viên
└── src/                           <-- Các module tiền xử lý dữ liệu và thiết lập pipeline tĩnh
```

---

## 2. Đặc tả dữ liệu & Các đặc trưng (16 Features + 1 Target)
Mỗi hàng dữ liệu đại diện cho trạng thái của **một inverter cụ thể** tại một thời điểm $t$ (chu kỳ 15 phút). Các bạn sẽ sử dụng **16 đặc trưng đầu vào (Features)** dưới đây đã được chuẩn hóa để dự báo **biến mục tiêu (Target)**:

### 🌟 Biến mục tiêu (Target):
* `AC_POWER`: Công suất phát điện xoay chiều của inverter đó tại thời điểm $t$ (đơn vị: kW).

### 📊 16 Đặc trưng đầu vào (Features):
1. **Nhóm đặc trưng khí tượng thực địa (Ground Sensors):**
   * `AMBIENT_TEMPERATURE`: Nhiệt độ môi trường đo được tại nhà máy (°C).
   * `MODULE_TEMPERATURE`: Nhiệt độ bề mặt tấm pin mặt trời (°C).
   * `IRRADIATION`: Cường độ bức xạ mặt trời ($W/m^2$).
2. **Nhóm đặc trưng vật lý tự thiết kế (Engineered Features):**
   * `Temp_Diff`: Chênh lệch nhiệt độ (`MODULE_TEMPERATURE - AMBIENT_TEMPERATURE`), phản ánh khả năng tản nhiệt và mức độ lão hóa của tấm pin.
3. **Nhóm đặc trưng tuần hoàn lượng giác (Cyclical Time Encodings):**
   * `Hour_Sin` & `Hour_Cos`: Ánh xạ giờ trong ngày lên đường tròn lượng giác để bảo toàn tính liên tục tuần hoàn (tránh lỗi ngắt quãng giữa giờ 23:45 và 00:00).
   * `Day_of_Year_Sin` & `Day_of_Year_Cos`: Ánh xạ ngày trong năm để mô hình hóa sự thay đổi cường độ nắng theo mùa (xuân, hạ, thu, đông).
4. **Nhóm đặc trưng trễ riêng biệt của từng Inverter (Lag Features):**
   * `IRRADIATION_lag_1` & `IRRADIATION_lag_2`: Bức xạ mặt trời trễ 1 bước (15 phút trước) và 2 bước (30 phút trước).
   * `AMBIENT_TEMP_lag_1` & `MODULE_TEMP_lag_1`: Nhiệt độ trễ 1 bước để mô phỏng quán tính nhiệt vật lý của tấm pin.
5. **Nhóm thống kê trượt (Rolling Statistics):**
   * `IRRADIATION_roll_mean_1h` & `IRRADIATION_roll_std_1h`: Trung bình trượt và độ lệch chuẩn của bức xạ trong 1 giờ gần nhất để đo lường độ biến động của mây che phủ.
6. **Nhóm khí tượng vĩ mô (Crawl từ Open-Meteo):**
   * `OM_TEMPERATURE_2M`: Nhiệt độ không khí dự báo ở độ cao 2m (°C).
   * `OM_RELATIVE_HUMIDITY_2M`: Độ ẩm không khí tương đối (%).

---

## 3. Hướng dẫn từng bước huấn luyện và tùy chỉnh trên Notebook mẫu

### 📍 Bước 1: Khai báo thông tin mô hình và tác giả (Mục 3 trong Notebook)
Mỗi thành viên cần khai báo chính xác tên của mình và tên mô hình để hệ thống tự động ghi nhận vào báo cáo chung mà không đè lên kết quả của người khác:
```python
MY_MODEL_NAME = "XGBoost_Tuned"      # Điền tên mô hình của bạn (Ví dụ: Support Vector Regressor, Random Forest,...)
MEMBER_NAME = "Nguyen_Van_A"        # Điền tên của bạn
PLANT_NO = 1                        # Chọn nhà máy chạy thử nghiệm (1 hoặc 2)
```

### 📍 Bước 2: Khởi tạo và Tối ưu hóa siêu tham số (Hyperparameter Tuning)
Bạn có thể tự do thử nghiệm nhiều thuật toán khác nhau (ví dụ: Random Forest, SVR, XGBoost, Neural Network, LightGBM, CatBoost...). Dưới đây là các ví dụ mã nguồn hướng dẫn cách viết code tối ưu hóa siêu tham số:

> [!NOTE]
> Để tránh rò rỉ dữ liệu (Data Leakage), bạn chỉ được phép dùng tập **Train** để huấn luyện và tập **Validation** để thử nghiệm tham số. **Tuyệt đối không dùng tập Test trong bước tối ưu hóa siêu tham số này.**

#### 💡 Ví dụ 1: Tối ưu siêu tham số cho Support Vector Regressor (SVR) sử dụng `GridSearchCV`
```python
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# 1. Khai báo không gian tham số cần tìm kiếm
param_grid = {
    'C': [10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.2]
}

# 2. Khởi tạo mô hình nền tảng
svr_base = SVR(kernel='rbf')

# 3. Tìm kiếm bằng Grid Search trên dữ liệu Train
grid_search = GridSearchCV(estimator=svr_base, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# 4. Gán mô hình tốt nhất vào biến my_model
my_model = grid_search.best_estimator_
print(f"Bộ tham số SVR tối ưu nhất: {grid_search.best_params_}")
```

#### 💡 Ví dụ 2: Huấn luyện nhanh mô hình Random Forest (Điều chỉnh thủ công)
```python
from sklearn.ensemble import RandomForestRegressor

my_model = RandomForestRegressor(
    n_estimators=250,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
my_model.fit(X_train_scaled, y_train)
```

---

## 4. Cơ chế Đánh giá & Ràng buộc Vật lý Tự động
Sau khi bạn kết thúc phần code huấn luyện ở Mục 3, Notebook sẽ tự động chạy các phần còn lại với cấu hình chuẩn hóa tuyệt đối:

### ⚠️ Ràng buộc vật lý (Physical Constraints):
* **Cưỡng bức ban đêm:** Khi bức xạ mặt trời $IRRADIATION \le 10^{-4}$ (tương ứng biến chỉ thị `is_day == 0`), công suất phát của inverter sẽ được tự động gán cứng bằng $0.0\text{ kW}$ (để loại bỏ nhiễu cảm biến trạm khí tượng hoặc dòng điện dò rỉ cực nhỏ phi thực tế).
* **Cắt cụt dưới:** Toàn bộ các giá trị dự đoán âm (nếu có) sẽ được chuyển về $0.0\text{ kW}$ vì công suất điện xoay chiều thực tế không thể âm.

### 📈 Các độ đo đánh giá tiêu chuẩn:
Mô hình của bạn sẽ được đánh giá tự động trên cả 2 tập **Validation** và **Test** qua các độ đo:
1. **$R^2$ Score (Hệ số xác định):** Đo lường tỷ lệ biến thiên của thực tế được giải thích bởi mô hình. Càng tiệm cận $1.0$ càng tốt.
2. **MAE (Sai số tuyệt đối trung bình - kW):** Phản ánh sai lệch trung bình theo đơn vị kW thực tế của mỗi điểm dự báo.
3. **RMSE (Căn sai số bình phương trung bình - kW):** Phạt nặng các sai số lớn. Cực kỳ quan trọng để đánh giá độ ổn định của hệ thống điện.
4. **Daytime MAPE (%):** Chỉ số sai số phần trăm tuyệt đối trung bình, được tính toán riêng trên các mốc thời gian ban ngày có công suất thực tế $y_{true} > 5\text{ kW}$ để tránh lỗi chia cho $0$ vào ban đêm.

---

## 5. Quy tắc xuất kết quả và Lưu trữ
Để phục vụ việc tổng hợp kết quả của cả nhóm, khi chạy xong cell cuối cùng của Notebook:
1. **Model Serialization:** Đối tượng mô hình của bạn sẽ được đóng gói và lưu tự động vào `Result/plant_[PLANT_NO]_[tên_mô_hình].pkl`.
2. **Plot Saving:** Đồ thị đường cong so sánh Actual vs Predicted của mô hình trên tập Test trong 3 ngày liên tiếp sẽ được lưu thành file ảnh `Result/plant_[PLANT_NO]_[tên_mô_hình]_actual_vs_predicted.png`.
3. **Metrics Logger:** Kết quả điểm số của mô hình của bạn sẽ được ghi/cập nhật vào bảng tổng sắp chung `Result/metrics_report.csv`. 

> [!IMPORTANT]
> **Quy định chung của nhóm:**
> * Không thay đổi tên cột hoặc cấu trúc file CSV kết quả `metrics_report.csv`.
> * Đặt tên mô hình `MY_MODEL_NAME` độc lập, ngắn gọn và có ý nghĩa (ví dụ: `SVR_RBF`, `XGBoost_Tuned`, `Deep_ANN`).
> * Chúc các bạn huấn luyện mô hình thành công và đạt kết quả tối ưu nhất cho báo cáo học phần!
