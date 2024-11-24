from prompt import run
import argparse
from tabulate import tabulate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process keywords, month, and year parameters')
    
    # Add arguments
    parser.add_argument('-k', '--datasets', 
                        type=str, 
                        required=True,
                        help='Comma-separated list of keywords')
    
    parser.add_argument('-m', '--month', 
                        type=int, 
                        required=True,
                        help='Month (1-12)')
    
    parser.add_argument('-y', '--year', 
                        type=int, 
                        required=True,
                        help='Year (4-digit format)')

    parser.add_argument('-n', '--model_names', 
                        type=str, 
                        required=False,
                        default = 'claude-3-5-sonnet-latest',
                        help='Name of the model to use')
    
    parser.add_argument('-c', '--check_abstract', 
                        type=bool, 
                        required= False,
                        default = True,
                        help='whether to check the abstract')
    
    parser.add_argument('-v', '--verbose', 
                        type=bool, 
                        required= False,
                        default = True,
                        help='whether to check the abstract')

    # Parse arguments
    args = parser.parse_args()
    data = []
    for model_name in args.model_names.split(','):
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
