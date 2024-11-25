from prompt import run, create_args
from tabulate import tabulate


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
        
        data.append([model_name]+data_results)

    # Build the markdown table
    headers = ['Model']+args.datasets.split(',')

    # Combine all parts
    print(tabulate(data, headers=headers, tablefmt="grid"))
