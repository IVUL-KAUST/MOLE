from prompt import run, create_args
from tabulate import tabulate # type: ignore
from utils import get_masader_data

if __name__ == "__main__":
    args = create_args()
    data = []
    data_names = []
    for model_name in args.models.split(','):
        data_results = []
        if args.masader_validate:
            masader_data = get_masader_data()
            titles = [str(x['Paper Title']) for idx,x in masader_data.iterrows()][0:2]
            data_names = [str(x['Name']) for idx,x in masader_data.iterrows()][0:2]            
        else:
            data_names = args.datasets.split(',')
            titles = ['' for _ in data_names]

        for data_name,title in zip(data_names, titles):
            args.model_name = model_name
            if args.title != '':
                title = title.replace('\r\n', ' ')
                title = title.replace(':', '')
                args.keywords = title
            else:
                args.keywords = data_name
            metadata = run(args)
            if metadata:
                data_results.append(round(metadata['validation']* 100, 2))
        
        data.append([model_name]+data_results+[round(sum([x for x in data_results])/len(data_results), 2)])

    # Build the markdown table
    headers = ['Model']+data_names+['Average']
    data = sorted(data, key=lambda x: x[-1])

    # Combine all parts
    print(tabulate(data, headers=headers, tablefmt="grid"))
