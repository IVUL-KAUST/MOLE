import argparse
import glob
import os
from constants import eval_datasets_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default = "results_latex", type=str)
    parser.add_argument("--model", default = "llama-4-maverick-results", type=str)
    parser.add_argument("--schema", default = "en", type=str)
    args = parser.parse_args()

    ids = eval_datasets_ids[args.schema]["test"]

    for file in glob.glob(f"static/{args.results_path}/**/**/**.json"):
        if "human" not in file:
            if args.model in file and any(id in file for id in ids):
                print(file)
                os.remove(file)
if __name__ == "__main__":
    main()

