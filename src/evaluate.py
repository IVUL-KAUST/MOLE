from search import run, create_args
from tabulate import tabulate  # type: ignore
from utils import fix_arxiv_link
from constants import *
import numpy as np
from schema import get_schema
from utils import show_info
import asyncio
import concurrent.futures

async def main():
    args = create_args()
    metric_results = {}
    paper_links = []           
    dataset = get_schema(args.schema_name).get_eval_datasets(split = args.split)
    
    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(dataset)) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        
        for idx, data in enumerate(dataset):
            show_info(f"🔍 Processing paper {idx+1}/{len(dataset)}")
            # Run the synchronous function in a thread
            task = loop.run_in_executor(executor, run,
                data['Paper_Link'],
                args.model,
                args.overwrite,
                args.browse_web,
                args.schema_name,
                args.few_shot,
                args.results_path,
                args.repeat_on_error,
                args.context,
                args.format,
                args.backend,
                {   
                    "title": data["Paper_Title"],
                    "abstract": data["Abstract"],
                }
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    metrics = ['precision', 'recall', 'f1', 'length']
    for r in results:
        model_name = list(r.keys())[0]
        results = r[model_name]

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

if __name__ == "__main__":
    asyncio.run(main())