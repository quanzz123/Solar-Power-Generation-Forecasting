import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "EDA_Solar_Power_Generation.ipynb"

def fix_notebook():
    if not NOTEBOOK_PATH.exists():
        print(f"Không tìm thấy file Notebook tại: {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    fixed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            for line in source:
                # Kiem tra dong bi loi
                if 'p1_merged["Hour"] = p1_merged["DATE_TIME"], p1_merged["DATE_TIME"].dt.hour' in line:
                    print(f"Found bad line and removed: {line.strip()}")
                    fixed = True
                    continue
                new_source.append(line)
            cell["source"] = new_source

    if fixed:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)
        print("Successfully fixed the notebook!")
    else:
        print("Bad line not found in notebook.")

if __name__ == "__main__":
    fix_notebook()
