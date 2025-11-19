from datasets import load_dataset, Dataset, DatasetDict
from glob import glob
import pandas as pd
import json
base_path = "evals"
schemas = ["ar", "en", "jp", "ru", "fr", "multi"]
dfs = []
metadata = ['Name', 'Subsets', 'HF Link', 'Link', 'License', 'Year', 'Language', 'Dialect', 'Domain', 'Form', 'Collection Style', 'Description', 'Volume', 'Unit', 'Ethical Risks', 'Provider', 'Derived From', 'Paper Title', 'Paper Link', 'Script', 'Tokenized', 'Host', 'Access', 'Cost', 'Test Split', 'Tasks', 'Venue Title', 'Venue Type', 'Venue Name', 'Authors', 'Affiliations', 'Abstract']
for schema in schemas:
    for split in ["valid", "test"]:
        for file in glob(f"{base_path}/{schema}/{split}/*.json"):
            data = json.load(open(file, "r"))
            new_data = {"category": schema, "split": split}
            for k in metadata:
                if k in data:
                    if isinstance(data[k], list):
                        new_data[k] = str(data[k])
                    else:
                        new_data[k] = data[k]
                else:
                    new_data[k] = None
            
            for k in metadata:
                if k in data["annotations_from_paper"]:
                    new_data[k + "_exist"] = data["annotations_from_paper"][k]
                else:
                    new_data[k + "_exist"] = None

            df = pd.DataFrame.from_dict(new_data, orient="index").transpose()
            dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
dataset = Dataset.from_pandas(df)

dataset.push_to_hub("IVUL-KAUST/MOLE")








