from glob import glob 
import json 
import os
import sys
from search import download_paper
import tqdm

json_files = sorted(glob("../S2ORC_Exp500v1/val/Annotations/*.json"))
pdf_links = sorted(glob("../S2ORC_Exp500v1/val/PDF-Links/*.txt"))
all_keys = ['title', 'link', 'abstract', 'author', 'authoraffiliation', 'doi', 'email', 'date']
for i, json_file in enumerate(tqdm.tqdm(json_files)):
    json_data = json.load(open(json_file))
    fixed_json = {}
    paper_link = open(pdf_links[i]).read().strip()
    
    # Download the paper
    try:
        success, paper_path = download_paper(paper_link, download_path="static/papers/", log=True)
    except Exception as e:
        print(f"Error downloading paper {i}: {paper_link} - {e}")
        continue
    if not success:
        print(f"Failed to download paper {i}: {paper_link}")
        continue
    
    print(f"Downloaded paper {i}: {paper_link}")
    
    fixed_json['Paper_Link'] = paper_link
    for attribute in all_keys:
        if attribute == "title":
            fixed_json[attribute.capitalize()] = json_data[attribute][0]["text"]
        elif attribute == "abstract":
            fixed_json[attribute.capitalize()] = json_data[attribute][0]["text"]
        elif attribute not in json_data:
            fixed_json[attribute.capitalize()] = []
        else:
            fixed_json[attribute.capitalize()] = [val["text"] for val in json_data[attribute]]
    keys = fixed_json.keys()
    fixed_json['annotations_from_paper'] = {}
    for key in keys:
        fixed_json['annotations_from_paper'][key] = 1
    os.makedirs(f"evals/msed/test", exist_ok=True)
    with open(f"evals/msed/test/{i}.json", "w") as f:
        json.dump(fixed_json, f, indent=4)
