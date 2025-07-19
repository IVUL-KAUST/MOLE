from glob import glob
import os
import json
from src.utils import get_arxiv_id

files = glob("MagedAnnotations/*.json")
os.makedirs("MagedAnnotations_updated", exist_ok=True)
for file in files:
    new_data = {}
    with open(file, "r") as f:
        data = json.load(f)
    new_data["metadata"] = data
    new_data["config"] = {
        "model_name": "maged",
    }
    arxiv_id = get_arxiv_id(data["Paper Link"])
    path = f"static/results_maged/{arxiv_id}/zero_shot"
    file_name = file.split("/")[-1]
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{file_name}", "w") as f:
        json.dump(new_data, f, indent=4)