from pages.search import run, create_args
from tabulate import tabulate # type: ignore
from utils import fix_arxiv_link
import numpy as np
from constants import *

if __name__ == "__main__":
    args = create_args()
    metric_results = {}
    data_names = []
    len_data = 0
    use_split = None

    if args.masader_validate or args.masader_test:
        titles = []
        data_names = []
        paper_links = []

        if args.masader_validate:
            dataset = masader_valid_dataset
            use_split = 'valid'
        else:
            use_split = 'test'
            dataset = masader_test_dataset

        for x in dataset:
            titles.append(str(x['Paper Title']))
            data_names.append(str(x['Name']))
            paper_links.append(str(x['Paper Link']))      
    else:
        data_names = args.keywords.split(',')
        titles = ['' for _ in data_names]
        paper_links = ['' for _ in data_names]
    len_data = len(data_names)
    
    for data_name,title,paper_link in zip(data_names, titles, paper_links):
        if title != '':
            title = title.replace('\r\n', ' ')
            title = title.replace(':', '')
            args.keywords = title
        else:
            args.keywords = data_name
        
        if paper_link != '':
            link = fix_arxiv_link(paper_link)
            model_results = run(mode = 'api', link = link, year = None, month = None, models = args.models.split(','), browse_web=args.browse_web, overwrite=args.overwrite, use_split = use_split)
        else:
            model_results = run(mode = 'api', keywords = args.keywords, year = None, month = None, models = args.models.split(','), browse_web=args.browse_web, overwrite= args.overwrite, use_split = use_split)

        for model_name in model_results:
            results = model_results[model_name]

            if model_name not in metric_results:
               metric_results[model_name] = []
            metric_results[model_name].append([results['validation'][m] for m in results['validation']])
    results = []
    for model_name in metric_results:
        if len(metric_results[model_name]) == len_data:
            results.append([model_name]+(np.mean(metric_results[model_name], axis = 0)*100).tolist())
    headers = ['MODEL', 'CONTENT', 'ACCESSABILITY', 'DIVERSITY','EVALUATION','AVERAGE']
    print(tabulate(sorted(results, key=lambda x: x[-1]), headers=headers, tablefmt="grid",floatfmt=".2f"))
