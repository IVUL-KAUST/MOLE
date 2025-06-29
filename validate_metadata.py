import json
import glob
import os
from utils import evaluate_lengths
for lang in os.listdir('schema'):
    lang = lang.split('.')[0]
    schema = json.load(open(f'schema/{lang}.json'))
    for file in glob.glob(f'evals/{lang}/**/*.json'):
        metadata = json.load(open(file))
        for key in metadata:
            if key == 'annotations_from_paper':
                continue
            if key not in schema:
                print(f"{file} is missing {key}")
            if 'options' in schema[key]:
                if 'List' in schema[key]['answer_type']:
                    for item in metadata[key]:
                        if item not in schema[key]['options']:
                            print(f"{file} is missing {item} in {key}")
                else:
                    if metadata[key] not in schema[key]['options']:
                        print(f"{file} is missing {metadata[key]} in {key}")
                    
        annotations_from_paper = metadata["annotations_from_paper"]
        for key in annotations_from_paper:
            if key not in schema:
                print(f"{file} is missing {key} in annotations_from_paper")
        length = evaluate_lengths(metadata, schema = lang)
        if abs(length - 1) > 0.01:
            print(f"{file} has length {length}")
