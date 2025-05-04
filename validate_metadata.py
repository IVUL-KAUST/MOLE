import json
import glob
import os
from utils import evaluate_lengths
for schema in os.listdir('schema'):
    lang = schema.split('.')[0]
    metadata = json.load(open(f'schema/{schema}'))
    for file in glob.glob(f'evals/{lang}/**/*.json'):
        new_metadata = json.load(open(file))
        for key in metadata:
            if key not in new_metadata:
                print(f"{file} is missing {key}")
        annotations_from_paper = new_metadata["annotations_from_paper"]
        for key in metadata:
            if key not in annotations_from_paper:
                print(f"{file} is missing {key} in annotations_from_paper")
        evaluate_lengths(new_metadata, schema = lang)
