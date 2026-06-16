import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KERAS_NOTEBOOK_PATH = BASE_DIR / "Solar_Power_LSTM_Pipeline.ipynb"
PYTORCH_NOTEBOOK_PATH = BASE_DIR / "Solar_Power_PyTorch_LSTM_Pipeline.ipynb"

def generate_pytorch_notebook():
    if not KERAS_NOTEBOOK_PATH.exists():
        print(f"Không tìm thấy file Notebook gốc tại: {KERAS_NOTEBOOK_PATH}")
        return

    with open(KERAS_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Modify markdown title cells to reflect PyTorch
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            new_source = []
            for line in source:
                line = line.replace("DEEP LEARNING LSTM PIPELINE", "PYTORCH LSTM PIPELINE")
                line = line.replace("EarlyStopping giúp chống overfitting tuyệt đối", "EarlyStopping tự viết bằng PyTorch")
                new_source.append(line)
            cell["source"] = new_source

        elif cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            
            # Check cell type/contents
            is_import_cell = any("import tensorflow as tf" in line or "TensorFlow / Keras" in line for line in source)
            is_model_cell = any("def build_lstm_model(" in line or "build_lstm_model(" in line for line in source)
            is_train_cell = any("EarlyStopping(monitor='val_loss'" in line for line in source)
            is_curve_cell = any("history.history['loss']" in line for line in source)
            is_eval_cell = any("lstm_model.predict(" in line for line in source)
            
            if is_import_cell:
                print("Modifying Import cell...")
                new_source = [
                    "import os\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "from pathlib import Path\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.preprocessing import MinMaxScaler\n",
                    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                    "\n",
                    "# PyTorch\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "from torch.utils.data import Dataset, DataLoader\n",
                    "\n",
                    "# Thiết lập đường dẫn thư mục gốc\n",
                    "BASE_DIR = Path(\".\").resolve()\n",
                    "SPLIT_DIR = BASE_DIR / \"datasets\" / \"split\"\n",
                    "RESULT_DIR = BASE_DIR / \"Result\"\n",
                    "RESULT_DIR.mkdir(parents=True, exist_ok=True)\n",
                    "\n",
                    "# Cấu hình biểu đồ\n",
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "plt.rcParams['figure.figsize'] = (14, 6.5)\n",
                    "\n",
                    "print(\"Khai báo thư viện thành công!\")\n",
                    "try:\n",
                    "    print(\"Phiên bản PyTorch:\", torch.__version__)\n",
                    "    print(\"GPU khả dụng:\", torch.cuda.is_available())\n",
                    "except Exception:\n",
                    "    print(\"PyTorch chưa được cài đặt trong môi trường hiện tại. Bạn hãy cài đặt bằng lệnh: pip install torch\")"
                ]
                cell["source"] = new_source
                cell["outputs"] = []
                
            elif is_model_cell:
                print("Modifying Model Architecture cell...")
                new_source = [
                    "class SolarLSTMModel(nn.Module):\n",
                    "    def __init__(self, input_size=4, hidden_size_1=64, hidden_size_2=32, dense_size=64, output_size=96):\n",
                    "        super(SolarLSTMModel, self).__init__()\n",
                    "        # LSTM Layer 1: học tuần tự\n",
                    "        self.lstm1 = nn.LSTM(input_size, hidden_size_1, batch_first=True)\n",
                    "        self.dropout1 = nn.Dropout(0.2)\n",
                    "        # LSTM Layer 2: nén thông tin\n",
                    "        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)\n",
                    "        self.dropout2 = nn.Dropout(0.1)\n",
                    "        # Lớp ẩn\n",
                    "        self.fc1 = nn.Linear(hidden_size_2, dense_size)\n",
                    "        self.relu = nn.ReLU()\n",
                    "        # Lớp đầu ra\n",
                    "        self.fc2 = nn.Linear(dense_size, output_size)\n",
                    "        \n",
                    "    def forward(self, x):\n",
                    "        # x shape: (batch_size, seq_len, input_size)\n",
                    "        out, _ = self.lstm1(x)\n",
                    "        out = self.dropout1(out)\n",
                    "        out, _ = self.lstm2(out)\n",
                    "        # Lấy trạng thái ẩn cuối cùng giống return_sequences=False\n",
                    "        out = out[:, -1, :]\n",
                    "        out = self.dropout2(out)\n",
                    "        out = self.fc1(out)\n",
                    "        out = self.relu(out)\n",
                    "        out = self.fc2(out)\n",
                    "        return out\n",
                    "\n",
                    "# Khởi tạo mô hình và thiết bị chạy (CPU / GPU)\n",
                    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                    "if device.type == \"cuda\":\n",
                    "    try:\n",
                    "        # Kiểm tra tính tương thích thực tế của CUDA/cuDNN với phần cứng\n",
                    "        # Khởi tạo một dummy LSTM để kích hoạt cuDNN kiểm tra tương thích\n",
                    "        dummy_lstm = torch.nn.LSTM(1, 1).to(device)\n",
                    "    except Exception:\n",
                    "        print(\"Cảnh báo: CUDA khả dụng nhưng không tương thích với phần cứng/cuDNN hiện tại. Tự động chuyển sang CPU.\")\n",
                    "        device = torch.device(\"cpu\")\n",
                    "\n",
                    "input_size = X_train_lstm.shape[2] # 4 đặc trưng\n",
                    "output_size = FORECAST_STEPS # 96 bước dự báo\n",
                    "lstm_model = SolarLSTMModel(input_size=input_size, output_size=output_size).to(device)\n",
                    "\n",
                    "print(lstm_model)\n",
                    "total_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)\n",
                    "print(f\"Tổng số tham số huấn luyện: {total_params:,}\")"
                ]
                cell["source"] = new_source
                cell["outputs"] = []
                
            elif is_train_cell:
                print("Modifying Training cell...")
                new_source = [
                    "class SolarLSTMDataset(Dataset):\n",
                    "    def __init__(self, X, y):\n",
                    "        self.X = torch.tensor(X, dtype=torch.float32)\n",
                    "        self.y = torch.tensor(y, dtype=torch.float32)\n",
                    "        \n",
                    "    def __len__(self):\n",
                    "        return len(self.X)\n",
                    "        \n",
                    "    def __getitem__(self, idx):\n",
                    "        return self.X[idx], self.y[idx]\n",
                    "\n",
                    "# Tạo bộ nạp DataLoader\n",
                    "train_dataset = SolarLSTMDataset(X_train_lstm, y_train_lstm)\n",
                    "val_dataset = SolarLSTMDataset(X_val_lstm, y_val_lstm)\n",
                    "\n",
                    "EPOCHS = 50\n",
                    "BATCH_SIZE = 256\n",
                    "train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\n",
                    "val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)\n",
                    "\n",
                    "class EarlyStopping:\n",
                    "    def __init__(self, patience=15, path='best_lstm_model.pth'):\n",
                    "        self.patience = patience\n",
                    "        self.path = path\n",
                    "        self.counter = 0\n",
                    "        self.best_loss = None\n",
                    "        self.early_stop = False\n",
                    "        \n",
                    "    def __call__(self, val_loss, model):\n",
                    "        if self.best_loss is None:\n",
                    "            self.best_loss = val_loss\n",
                    "            self.save_checkpoint(model)\n",
                    "        elif val_loss >= self.best_loss:\n",
                    "            self.counter += 1\n",
                    "            if self.counter >= self.patience:\n",
                    "                self.early_stop = True\n",
                    "        else:\n",
                    "            self.best_loss = val_loss\n",
                    "            self.save_checkpoint(model)\n",
                    "            self.counter = 0\n",
                    "            \n",
                    "    def save_checkpoint(self, model):\n",
                    "        torch.save(model.state_dict(), self.path)\n",
                    "\n",
                    "# Định nghĩa Loss & Optimizer\n",
                    "criterion = nn.MSELoss()\n",
                    "optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)\n",
                    "early_stopping = EarlyStopping(patience=15, path=str(RESULT_DIR / 'best_lstm_model.pth'))\n",
                    "\n",
                    "history = {'loss': [], 'val_loss': []}\n",
                    "\n",
                    "print(f\"Bắt đầu huấn luyện mạng PyTorch LSTM (Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE})...\")\n",
                    "try:\n",
                    "    for epoch in range(1, EPOCHS + 1):\n",
                    "        lstm_model.train()\n",
                    "        train_loss = 0.0\n",
                    "        for batch_X, batch_y in train_loader:\n",
                    "            batch_X, batch_y = batch_X.to(device), batch_y.to(device)\n",
                    "            optimizer.zero_grad()\n",
                    "            outputs = lstm_model(batch_X)\n",
                    "            loss = criterion(outputs, batch_y)\n",
                    "            loss.backward()\n",
                    "            optimizer.step()\n",
                    "            train_loss += loss.item() * batch_X.size(0)\n",
                    "        train_loss /= len(train_dataset)\n",
                    "        \n",
                    "        lstm_model.eval()\n",
                    "        val_loss = 0.0\n",
                    "        with torch.no_grad():\n",
                    "            for batch_X, batch_y in val_loader:\n",
                    "                batch_X, batch_y = batch_X.to(device), batch_y.to(device)\n",
                    "                outputs = lstm_model(batch_X)\n",
                    "                loss = criterion(outputs, batch_y)\n",
                    "                val_loss += loss.item() * batch_X.size(0)\n",
                    "        val_loss /= len(val_dataset)\n",
                    "        \n",
                    "        history['loss'].append(train_loss)\n",
                    "        history['val_loss'].append(val_loss)\n",
                    "        \n",
                    "        print(f\"Epoch {epoch:02d}/{EPOCHS} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}\")\n",
                    "        \n",
                    "        early_stopping(val_loss, lstm_model)\n",
                    "        if early_stopping.early_stop:\n",
                    "            print(f\"Dừng sớm kích hoạt ở epoch {epoch}!\")\n",
                    "            break\n",
                    "            \n",
                    "    lstm_model.load_state_dict(torch.load(early_stopping.path))\n",
                    "    print(\"Huấn luyện thành công và đã tải lại mô hình tốt nhất!\")\n",
                    "except Exception as e:\n",
                    "    print(\"Không thể huấn luyện mô hình. Lý do:\", e)"
                ]
                cell["source"] = new_source
                cell["outputs"] = []
                
            elif is_curve_cell:
                print("Modifying Learning Curves cell...")
                for line in source:
                    line = line.replace("history.history['loss']", "history['loss']")
                    line = line.replace("history.history['val_loss']", "history['val_loss']")
                    new_source.append(line)
                cell["source"] = new_source
                cell["outputs"] = []
                
            elif is_eval_cell:
                print("Modifying Test Evaluation cell...")
                new_source = [
                    "try:\n",
                    "    # Dự báo tập Test bằng PyTorch\n",
                    "    lstm_model.eval()\n",
                    "    X_test_tensor = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)\n",
                    "    with torch.no_grad():\n",
                    "        y_test_pred_raw = lstm_model(X_test_tensor).cpu().numpy()\n",
                    "\n",
                    "    # Áp dụng ràng buộc vật lý\n",
                    "    y_test_pred = np.clip(y_test_pred_raw, a_min=0, a_max=None)\n",
                    "\n",
                    "    # Khôi phục tỷ lệ về đơn vị gốc (kW)\n",
                    "    min_ac = scaler.data_min_[0]\n",
                    "    max_ac = scaler.data_max_[0]\n",
                    "    y_test_lstm_orig = y_test_lstm * (max_ac - min_ac) + min_ac\n",
                    "    y_test_pred_orig = y_test_pred * (max_ac - min_ac) + min_ac\n",
                    "\n",
                    "    # Tính toán các độ đo\n",
                    "    r2_lstm = r2_score(y_test_lstm_orig.flatten(), y_test_pred_orig.flatten())\n",
                    "    mae_lstm = mean_absolute_error(y_test_lstm_orig.flatten(), y_test_pred_orig.flatten())\n",
                    "    rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm_orig.flatten(), y_test_pred_orig.flatten()))\n",
                    "\n",
                    "    print(\"\\n\" + \"=\"*80)\n",
                    "    print(f\" KẾT QUẢ ĐÁNH GIÁ MẠNG PYTORCH LSTM TRÊN TẬP TEST\")\n",
                    "    print(\"=\"*80)\n",
                    "    print(f\" - R2 Score (Hệ số xác định): {r2_lstm:.4f}\")\n",
                    "    print(f\" - MAE (Sai số tuyệt đối):    {mae_lstm:.2f} kW\")\n",
                    "    print(f\" - RMSE (Căn sai số bình phương): {rmse_lstm:.2f} kW\")\n",
                    "    print(\"=\"*80)\n",
                    "except Exception as e:\n",
                    "    print(\"Chưa thể đánh giá mô hình. Lỗi:\", e)"
                ]
                cell["source"] = new_source
                cell["outputs"] = []

    with open(PYTORCH_NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully generated PyTorch LSTM notebook!")

if __name__ == "__main__":
    generate_pytorch_notebook()
