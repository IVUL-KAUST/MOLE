from glob import glob 
import json
import os
model_name = "Qwen2.5-3B-Instruct-sft-16384"
json_files = glob("static/results/**/*.json")
for json_file in json_files:
    results = json.load(open(json_file))
    if results["config"]["model_name"] == model_name:
        print(json_file)
        os.remove(json_file)
    