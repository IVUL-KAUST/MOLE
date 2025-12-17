import json 
import requests
import os
import shutil
from search import download_paper


path = "https://raw.githubusercontent.com/allenai/s2orc/refs/heads/master/data/metadata/sample.jsonl"
jsonl_data = requests.get(path).text.split("\n")
keys = ["title", "abstract", "authors", "year", "field"]
for i, line in enumerate(jsonl_data):
    if line:
        json_data = json.loads(line)
        if json_data["arxiv_id"]:
            fixed_json = {}
            paper_link = f"https://arxiv.org/pdf/{json_data['arxiv_id']}"
            fixed_json['Paper_Link'] = paper_link
            # Download the paper
            try:
                success, paper_path = download_paper(paper_link, download_path="static/papers/", log=True)
                # shutil.rmtree(paper_path)
            except Exception as e:
                print(f"Error downloading paper {i}: {paper_link} - {e}")
                continue
            if not success:
                print(f"Failed to download paper {i}: {paper_link}")
                continue
            for key in keys:
                if key == "field":
                    fixed_json[key.capitalize()] = json_data["mag_field_of_study"]
                elif key == "authors":
                    authors = []
                    for author in json_data["authors"]:
                       full_name = f"{author['first']} {' '.join(author['middle'])} {author['last']} {author['suffix']}".strip()
                       # remove extra spaces 
                       full_name = " ".join([name for name in full_name.split() if name])
                       authors.append(full_name)
                    fixed_json[key.capitalize()] = authors
                else:
                    fixed_json[key.capitalize()] = json_data[key]
            # os.makedirs(f"evals/s2orc/test", exist_ok=True)
            # with open(f"evals/s2orc/test/{i}.json", "w") as f:
            #     json.dump(fixed_json, f, indent=4)