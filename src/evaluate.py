from search import run, create_args
from tabulate import tabulate  # type: ignore
from utils import fix_arxiv_link
from constants import *
import numpy as np
from schema import get_schema
from utils import show_info

if __name__ == "__main__":
    args = create_args()
    metric_results = {}
    paper_links = []
        
    dataset = get_schema(args.schema_name).get_eval_datasets(split = args.split)
    
    for idx, data in enumerate(dataset):
        show_info(f"Processing paper {idx+1}/{len(dataset)}")
        model_results = run(
            data['Paper_Link'],
            args.model,
            browse_web=args.browse_web,
            overwrite=args.overwrite,
            schema_name = args.schema_name,
            few_shot = args.few_shot,
            results_path = args.results_path,
            repeat_on_error = args.repeat_on_error,
            context = args.context,
            format = args.format,
        )

        metrics = ['precision', 'recall', 'f1', 'length']
        for model_name in model_results:
            results = model_results[model_name]

            if model_name not in metric_results:
                metric_results[model_name] = []
            metric_results[model_name].append(
                [results["validation"][m] for m in results["validation"] if m in metrics]
            )
    results = []
    for model_name in metric_results:
        if len(metric_results[model_name]) == len(dataset):
            results.append(
                [model_name]
                + (np.mean(metric_results[model_name], axis=0) * 100).tolist()
            )
    headers = ["MODEL"] + metrics 
    print(
        tabulate(
            sorted(results, key=lambda x: x[-1]),
            headers=headers,
            tablefmt="grid",
            floatfmt=".2f",
        )
    )
