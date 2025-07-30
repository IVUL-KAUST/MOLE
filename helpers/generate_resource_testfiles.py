from glob import glob
import json
import os
files = glob("evals/**/**/**.json")

for file in files:
    data = json.load(open(file))
    file_name = file.split("/")[-1]
    schema_name = file.split("/")[-3]
    split = file.split("/")[2]
    os.makedirs(f"evals/resource/{split}", exist_ok=True)
    with open(f"evals/resource/{split}/{file_name}", "w") as f:
        f.write(json.dumps({
            "Name": data["Name"],
            "Category": schema_name,
            "Paper_Title": data["Paper_Title"],
            "Paper_Link": data["Paper_Link"],
            "Year": data["Year"],
            "Link": data["Link"],
            "Abstract": data["Abstract"],
            "annotations_from_paper": {
                "Name": 0,
                "Category": 1,
                "Paper_Title": 0,
                "Paper_Link": 0,
                "Year": 0,
                "Link": 0,
                "Abstract": 0,
            }
        }, indent=4))
