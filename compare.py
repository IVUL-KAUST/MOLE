from prompt import run, create_args
from tabulate import tabulate # type: ignore


if __name__ == "__main__":
    

    # Parse arguments
    args = create_args()
    data = []
    for model_name in args.models.split(','):
        data_results = []
        for data_name in args.datasets.split(','):
            args.model_name = model_name
            args.keywords = data_name
            metadata = run(args)
            if metadata:
                data_results.append(round(metadata['validation']* 100, 2))
        
        data.append([model_name]+data_results+[round(sum([x for x in data_results])/len(data_results), 2)])

    # Build the markdown table
    headers = ['Model']+args.datasets.split(',')+['Average']
    data = sorted(data, key=lambda x: x[-1])

    # Combine all parts
    print(tabulate(data, headers=headers, tablefmt="grid"))
