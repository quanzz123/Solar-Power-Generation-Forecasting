
# ==========================================
# CELL
# ==========================================
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Thiết lập đường dẫn thư mục gốc
BASE_DIR = Path(".").resolve()
SPLIT_DIR = BASE_DIR / "datasets" / "split"
RESULT_DIR = BASE_DIR / "Result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Cấu hình biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 6.5)

print("Khai báo thư viện thành công!")
try:
    print("Phiên bản PyTorch:", torch.__version__)
    print("GPU khả dụng:", torch.cuda.is_available())
except Exception:
    print("PyTorch chưa được cài đặt trong môi trường hiện tại. Bạn hãy cài đặt bằng lệnh: pip install torch")

# ==========================================
# CELL
# ==========================================
import sys
# Thêm thư mục src vào hệ thống để Python nhận diện các module nội bộ
sys.path.append(str(BASE_DIR / "src"))

from cleaning import clean_and_merge_plant
from feature_engineering import engineer_features_for_plant
from split import split_time_series_for_plant

# Bước 1: Làm sạch dữ liệu
print("[STEP 1] Data Cleaning...")
for p in [1, 2]:
    clean_and_merge_plant(p)

# Bước 2: Kỹ thuật đặc trưng
print("\n[STEP 2] Feature Engineering...")
for p in [1, 2]:
    engineer_features_for_plant(p)

# Bước 3: Phân chia tập dữ liệu
print("\n[STEP 3] Chronological Splitting...")
for p in [1, 2]:
    split_time_series_for_plant(p)

print("\n--- HOÀN TẤT BƯỚC TIỀN XỬ LÝ DỮ LIỆU ĐẦU VÀO ---")

# ==========================================
# CELL
# ==========================================
# Cấu hình chu kỳ thời gian
STEPS_PER_DAY = 96  # 24 giờ * 4 mốc (15 phút)
LOOKBACK_STEPS = 2 * STEPS_PER_DAY  # 2 ngày lịch sử = 192 bước
FORECAST_STEPS = 1 * STEPS_PER_DAY  # 1 ngày dự báo = 96 bước

# 4 Đặc trưng thời gian thực quan trọng nhất đối với LSTM
FEATURES_LSTM = ["AC_POWER", "IRRADIATION", "MODULE_TEMPERATURE", "Temp_Diff"]
TARGET_COL = "AC_POWER"

def create_sequences_for_lstm(df, lookback=LOOKBACK_STEPS, forecast=FORECAST_STEPS):
    X_seq, y_seq = [], []
    
    # Gom nhóm theo từng inverter riêng lẻ để tạo chuỗi độc lập
    for inverter_id, group in df.groupby("SOURCE_KEY"):
        # Sắp xếp theo trình tự thời gian
        group_sorted = group.sort_values(by="DATE_TIME").reset_index(drop=True)
        data_values = group_sorted[FEATURES_LSTM].values
        target_values = group_sorted[TARGET_COL].values
        
        # Tạo cửa sổ trượt
        total_len = len(group_sorted)
        for i in range(total_len - lookback - forecast + 1):
            # Lấy chuỗi lịch sử của 4 đặc trưng trong 2 ngày qua
            x_win = data_values[i : (i + lookback)]
            # Lấy chuỗi công suất thực tế cần dự báo của 1 ngày tiếp theo
            y_win = target_values[(i + lookback) : (i + lookback + forecast)]
            
            X_seq.append(x_win)
            y_seq.append(y_win)
            
    return np.array(X_seq), np.array(y_seq)

print("Hàm tạo chuỗi thời gian cho LSTM đã sẵn sàng!")

# ==========================================
# CELL
# ==========================================
# Chọn nhà máy 1 để chạy mô hình LSTM tiêu biểu
PLANT_NO = 1

# Đọc các mảnh dữ liệu đã phân tách sẵn
train_df = pd.read_csv(SPLIT_DIR / f"plant_{PLANT_NO}_train.csv")
val_df = pd.read_csv(SPLIT_DIR / f"plant_{PLANT_NO}_val.csv")
test_df = pd.read_csv(SPLIT_DIR / f"plant_{PLANT_NO}_test.csv")

# 1. Áp dụng MinMaxScaler
scaler = MinMaxScaler()
# Khớp bộ biến đổi trên tập Train
scaler.fit(train_df[FEATURES_LSTM])

# Bản sao để tránh thay đổi dữ liệu gốc
train_scaled_df = train_df.copy()
val_scaled_df = val_df.copy()
test_scaled_df = test_df.copy()

train_scaled_df[FEATURES_LSTM] = scaler.transform(train_df[FEATURES_LSTM])
val_scaled_df[FEATURES_LSTM] = scaler.transform(val_df[FEATURES_LSTM])
test_scaled_df[FEATURES_LSTM] = scaler.transform(test_df[FEATURES_LSTM])

# 2. Tạo chuỗi thời gian 3D Tensor cho LSTM
print("Bắt đầu tạo chuỗi thời gian...")
X_train_lstm, y_train_lstm = create_sequences_for_lstm(train_scaled_df)
X_val_lstm, y_val_lstm = create_sequences_for_lstm(val_scaled_df)
X_test_lstm, y_test_lstm = create_sequences_for_lstm(test_scaled_df)

print("\n--- KÍCH THƯỚC DỮ LIỆU ĐẦU VÀO CHO MẠNG LSTM ---")
print(f"Tập Train: X_shape = {X_train_lstm.shape}, y_shape = {y_train_lstm.shape}")
print(f"Tập Val  : X_shape = {X_val_lstm.shape}, y_shape = {y_val_lstm.shape}")
print(f"Tập Test : X_shape = {X_test_lstm.shape}, y_shape = {y_test_lstm.shape}")

# ==========================================
# CELL
# ==========================================
class SolarLSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size_1=64, hidden_size_2=32, dense_size=64, output_size=96):
        super(SolarLSTMModel, self).__init__()
        # LSTM Layer 1: học tuần tự
        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        # LSTM Layer 2: nén thông tin
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        # Lớp ẩn
        self.fc1 = nn.Linear(hidden_size_2, dense_size)
        self.relu = nn.ReLU()
        # Lớp đầu ra
        self.fc2 = nn.Linear(dense_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        # Lấy trạng thái ẩn cuối cùng giống return_sequences=False
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Khởi tạo mô hình và thiết bị chạy (CPU / GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        # Kiểm tra tính tương thích thực tế của CUDA/cuDNN với phần cứng
        # Khởi tạo một dummy LSTM để kích hoạt cuDNN kiểm tra tương thích
        dummy_lstm = torch.nn.LSTM(1, 1).to(device)
    except Exception:
        print("Cảnh báo: CUDA khả dụng nhưng không tương thích với phần cứng/cuDNN hiện tại. Tự động chuyển sang CPU.")
        device = torch.device("cpu")

input_size = X_train_lstm.shape[2] # 4 đặc trưng
output_size = FORECAST_STEPS # 96 bước dự báo
lstm_model = SolarLSTMModel(input_size=input_size, output_size=output_size).to(device)

print(lstm_model)
total_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"Tổng số tham số huấn luyện: {total_params:,}")

# ==========================================
# CELL
# ==========================================
class SolarLSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Tạo bộ nạp DataLoader
train_dataset = SolarLSTMDataset(X_train_lstm, y_train_lstm)
val_dataset = SolarLSTMDataset(X_val_lstm, y_val_lstm)

EPOCHS = 2
BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=15, path='best_lstm_model.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
            
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

# Định nghĩa Loss & Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
early_stopping = EarlyStopping(patience=15, path=str(RESULT_DIR / 'best_lstm_model.pth'))

history = {'loss': [], 'val_loss': []}

print(f"Bắt đầu huấn luyện mạng PyTorch LSTM (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...")
try:
    for epoch in range(1, EPOCHS + 1):
        lstm_model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_dataset)
        
        lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_dataset)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch:02d}/{EPOCHS} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
        
        early_stopping(val_loss, lstm_model)
        if early_stopping.early_stop:
            print(f"Dừng sớm kích hoạt ở epoch {epoch}!")
            break
            
    lstm_model.load_state_dict(torch.load(early_stopping.path))
    print("Huấn luyện thành công và đã tải lại mô hình tốt nhất!")
except Exception as e:
    print("Không thể huấn luyện mô hình. Lý do:", e)

# ==========================================
# CELL
# ==========================================
try:
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Train MSE Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation MSE Loss', color='orange')
    plt.title('LSTM Model Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'lstm_learning_curves.png', dpi=150)
    plt.show()
except Exception:
    print("Chưa có đồ thị vì chưa thực hiện huấn luyện thành công.")

# ==========================================
# CELL
# ==========================================
try:
    # Dự báo tập Test bằng PyTorch
    lstm_model.eval()
    X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_test_pred_raw = lstm_model(X_test_tensor).cpu().numpy()

    # Áp dụng ràng buộc vật lý
    y_test_pred = np.clip(y_test_pred_raw, a_min=0, a_max=None)

    # Khôi phục tỷ lệ về đơn vị gốc (kW)
    min_ac = scaler.data_min_[0]
    max_ac = scaler.data_max_[0]
    y_test_lstm_orig = y_test_lstm * (max_ac - min_ac) + min_ac
    y_test_pred_orig = y_test_pred * (max_ac - min_ac) + min_ac

    # Tính toán các độ đo
    r2_lstm = r2_score(y_test_lstm_orig.flatten(), y_test_pred_orig.flatten())
    mae_lstm = mean_absolute_error(y_test_lstm_orig.flatten(), y_test_pred_orig.flatten())
    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_orig.flatten(), y_test_pred_orig.flatten()))

    print("\n" + "="*80)
    print(f" KẾT QUẢ ĐÁNH GIÁ MẠNG PYTORCH LSTM TRÊN TẬP TEST")
    print("="*80)
    print(f" - R2 Score (Hệ số xác định): {r2_lstm:.4f}")
    print(f" - MAE (Sai số tuyệt đối):    {mae_lstm:.2f} kW")
    print(f" - RMSE (Căn sai số bình phương): {rmse_lstm:.2f} kW")
    print("="*80)
except Exception as e:
    print("Chưa thể đánh giá mô hình. Lỗi:", e)

# ==========================================
# CELL
# ==========================================
try:
    # Chọn ngẫu nhiên một mẫu chuỗi trong tập kiểm thử
    sample_idx = 100

    # Khôi phục tỷ lệ về đơn vị gốc (kW) trước khi trực quan hóa
    min_ac = scaler.data_min_[0]
    max_ac = scaler.data_max_[0]
    y_test_lstm_orig_sample = y_test_lstm[sample_idx] * (max_ac - min_ac) + min_ac
    y_test_pred_orig_sample = y_test_pred[sample_idx] * (max_ac - min_ac) + min_ac

    plt.figure(figsize=(14, 6.5))
    plt.plot(y_test_lstm_orig_sample, label="Actual Power (Thực tế)", color="black", linewidth=2.5)
    plt.plot(y_test_pred_orig_sample, label="LSTM Predicted Power (Dự báo 24h tiếp theo)", color="blue", linewidth=2, linestyle="--")

    plt.title(f"Plant {PLANT_NO} - LSTM Multi-Step Forecast (96 Steps ahead / Next 24 Hours)")
    plt.xlabel("Future Time Steps (15-min intervals)")
    plt.ylabel("AC Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULT_DIR / 'lstm_sample_forecast.png', dpi=150)
    plt.show()
except Exception:
    print("Chưa vẽ được đồ thị thực tế vs dự báo do thiếu kết quả dự báo.")
