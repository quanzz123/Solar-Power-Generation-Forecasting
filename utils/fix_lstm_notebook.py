import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "Solar_Power_LSTM_Pipeline.ipynb"

def fix_lstm_notebook():
    if not NOTEBOOK_PATH.exists():
        print(f"Không tìm thấy file Notebook tại: {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            
            # 1. Modify cell 6: callbacks, EPOCHS, BATCH_SIZE
            has_callbacks = any("EarlyStopping(monitor='val_loss'" in line for line in source)
            # 2. Modify cell 8: test set evaluation metrics
            has_evaluation = any("mae_lstm = mean_absolute_error(" in line for line in source)
            # 3. Modify cell 9: plotting sample curves
            has_plotting = any("plt.plot(y_test_lstm[sample_idx]" in line for line in source)
            
            if has_callbacks:
                print("Found callbacks cell. Modifying patience, EPOCHS, and BATCH_SIZE...")
                for line in source:
                    if "EarlyStopping(monitor='val_loss', patience=7," in line:
                        line = line.replace("patience=7", "patience=15")
                        print("-> Changed patience to 15")
                    elif "EPOCHS = 30" in line:
                        line = line.replace("EPOCHS = 30", "EPOCHS = 50")
                        print("-> Changed EPOCHS to 50")
                    elif "BATCH_SIZE = 128" in line:
                        line = line.replace("BATCH_SIZE = 128", "BATCH_SIZE = 256")
                        print("-> Changed BATCH_SIZE to 256")
                    new_source.append(line)
                cell["source"] = new_source
                
            elif has_evaluation:
                print("Found evaluation cell. Inserting inverse scaling code...")
                for line in source:
                    if "r2_lstm = r2_score(y_test_lstm.flatten(), y_test_pred.flatten())" in line:
                        # Insert inverse scaling code right before this line
                        new_source.append("    # Khôi phục tỷ lệ (Inverse transform) về đơn vị gốc (kW) trước khi đánh giá\n")
                        new_source.append("    min_ac = scaler.data_min_[0]\n")
                        new_source.append("    max_ac = scaler.data_max_[0]\n")
                        new_source.append("    y_test_lstm_orig = y_test_lstm * (max_ac - min_ac) + min_ac\n")
                        new_source.append("    y_test_pred_orig = y_test_pred * (max_ac - min_ac) + min_ac\n\n")
                        # Modify the metric calculations to use the orig arrays
                        line = line.replace("y_test_lstm.flatten(), y_test_pred.flatten()", "y_test_lstm_orig.flatten(), y_test_pred_orig.flatten()")
                        print("-> Inverted scaling for evaluation metrics")
                    elif "mae_lstm = mean_absolute_error(y_test_lstm.flatten(), y_test_pred.flatten())" in line:
                        line = line.replace("y_test_lstm.flatten(), y_test_pred.flatten()", "y_test_lstm_orig.flatten(), y_test_pred_orig.flatten()")
                    elif "rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm.flatten(), y_test_pred.flatten()))" in line:
                        line = line.replace("y_test_lstm.flatten(), y_test_pred.flatten()", "y_test_lstm_orig.flatten(), y_test_pred_orig.flatten()")
                    
                    new_source.append(line)
                cell["source"] = new_source
                
            elif has_plotting:
                print("Found plotting cell. Inserting sample inverse scaling code...")
                for line in source:
                    if "sample_idx = 100" in line:
                        new_source.append(line)
                        # Insert inverse scaling code right after sample_idx
                        new_source.append("\n    # Khôi phục tỷ lệ về đơn vị gốc (kW) trước khi trực quan hóa\n")
                        new_source.append("    min_ac = scaler.data_min_[0]\n")
                        new_source.append("    max_ac = scaler.data_max_[0]\n")
                        new_source.append("    y_test_lstm_orig_sample = y_test_lstm[sample_idx] * (max_ac - min_ac) + min_ac\n")
                        new_source.append("    y_test_pred_orig_sample = y_test_pred[sample_idx] * (max_ac - min_ac) + min_ac\n")
                        print("-> Inverted scaling for sample plot values")
                        continue
                    elif "plt.plot(y_test_lstm[sample_idx]" in line:
                        line = line.replace("y_test_lstm[sample_idx]", "y_test_lstm_orig_sample")
                    elif "plt.plot(y_test_pred[sample_idx]" in line:
                        line = line.replace("y_test_pred[sample_idx]", "y_test_pred_orig_sample")
                    
                    new_source.append(line)
                cell["source"] = new_source

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print("Successfully updated the LSTM notebook!")

if __name__ == "__main__":
    fix_lstm_notebook()
