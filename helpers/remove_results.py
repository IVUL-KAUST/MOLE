import argparse
import glob
import os
from tqdm import tqdm
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default = "results", type=str)
    parser.add_argument("--model", default = "llama-4-maverick-results", type=str)
    parser.add_argument("--schema", default = "all", type=str)
    args = parser.parse_args()


    for file in tqdm(glob.glob(f"static/{args.results_path}/**/**.json")):
        model_name = json.load(open(file))['config']['model_name']
        schema_name = json.load(open(file))['config']['schema_name']
        if args.schema == "all":
            if model_name == args.model:
                os.remove(file)
        else:
            if model_name == args.model and schema_name == args.schema:
                os.remove(file)
if __name__ == "__main__":
    main()

