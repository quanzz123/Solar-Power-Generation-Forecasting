import json
import sys
from pathlib import Path

def convert_and_run(notebook_name, script_name):
    notebook_path = Path(notebook_name)
    script_path = Path(script_name)
    
    if not notebook_path.exists():
        print(f"Notebook not found: {notebook_path}")
        sys.exit(1)
        
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
        
    code_lines = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            # Join lines in cell
            cell_code = "".join(source)
            # Add a separator comment
            code_lines.append(f"\n# ==========================================\n# CELL\n# ==========================================\n")
            code_lines.append(cell_code)
            code_lines.append("\n")
            
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.writelines(code_lines)
        
    print(f"Successfully compiled notebook to script: {script_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_notebook_as_script.py <notebook_path> <output_script_path>")
        sys.exit(1)
    convert_and_run(sys.argv[1], sys.argv[2])
