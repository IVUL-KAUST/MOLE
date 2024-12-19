from prompt import run, create_args
from tabulate import tabulate # type: ignore
from utils import get_masader_data

if __name__ == "__main__":
    args = create_args()
    data_results = []
    metric_results = []
    data_names = []
    for model_name in args.models.split(','):
        per_data_results = []
        per_metric_results = []
        if args.masader_validate:
            masader_data = get_masader_data()
            titles = [str(x['Paper Title']) for idx,x in masader_data.iterrows()][0:2]
            data_names = [str(x['Name']) for idx,x in masader_data.iterrows()][0:2]            
        else:
            data_names = args.keywords.split(',')
            titles = ['' for _ in data_names]

        for data_name,title in zip(data_names, titles):
            args.model_name = model_name
            if title != '':
                title = title.replace('\r\n', ' ')
                title = title.replace(':', '')
                args.keywords = title
            else:
                args.keywords = data_name
            metadata = run(args)
            if metadata:
                per_data_results.append(metadata['validation']['AVERAGE'])
                per_metric_results.append([metadata['validation'][m] for m in metadata['validation']])
        
        data_results.append([model_name]+per_data_results+[(sum([x for x in per_data_results])/len(per_data_results))*100])
        metric_results.append([model_name]+[sum(elements) / len(per_metric_results) * 100 for elements in zip(*per_metric_results)])

    if not args.aggergate:
        # Build the markdown table
        headers = ['Model']+data_names+['Average']

        # Combine all parts
        print(tabulate(sorted(data_results, key=lambda x: x[-1]), headers=headers, tablefmt="grid", floatfmt=".2f"))
    
    else:
        headers = ['MODEL', 'PUBLICATION', 'CONTENT', 'ACCESSABILITY', 'DIVERSITY','EVALUATION','AVERAGE']

        # Combine all parts
        print(tabulate(sorted(metric_results, key=lambda x: x[-1]), headers=headers, tablefmt="grid",floatfmt=".2f"))
