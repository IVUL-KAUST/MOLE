from search import run
from tabulate import tabulate  # type: ignore
import numpy as np
from schema import get_schema
import asyncio
import concurrent.futures
from arg_utils import args
from utils import TextLogger

async def main(args):
    metric_results = {}
    dataset = get_schema(args.schema_name).get_eval_datasets(split = args.split)
    logger = TextLogger(log = args.log)
    
    # Create a thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(dataset)) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        
        for idx, data in enumerate(dataset):
            logger.show_info(f"🔍 Processing paper {idx+1}/{len(dataset)}")
            # Run the synchronous function in a thread
            task = loop.run_in_executor(executor, run,
                data['Paper_Link'],
                args
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
    asyncio.run(main(args))