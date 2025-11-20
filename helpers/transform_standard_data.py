import glob
import json
import os
import sys
sys.path.append("src")
from utils import create_hash
directory = "helpers/results_kimi"
files = glob.glob(f"helpers/kimi_annotated_data/*.json")
print(files)
for file in files:
    metadata = json.load(open(file))
    schema_name = file.split("/")[-1].split("_")[0]
    data = {
        "metadata": metadata,
        "config": {
        "model_name": "moonshotai/kimi-k2",
        "few_shot": 0,
        "link": metadata["Paper_Link"],
        "schema_name": schema_name,
        "context": "all",
        "format": "pdf_plumber",
        "max_model_len": 32768,
        "max_output_len": 2048,
        "browse_web": False,
        "backend": "openrouter"
    }
    }
    folder = f'{directory}/{create_hash(metadata["Paper_Link"])}'
    os.makedirs(folder, exist_ok=True)
    json.dump(data, open(f'{folder}/results.json', "w"), indent=4)


# fix keys in human annotated data
files = glob.glob(f"helpers/human_annotated_data/**/**/*.json")
for file in files:
    metadata = json.load(open(file))
    # remove the space in the keys
    metadata = {k.replace(" ", "_"): v for k, v in metadata.items()}
    json.dump(metadata, open(file, "w"), indent=4)
