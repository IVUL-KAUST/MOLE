from pages.search import run, create_args
from tabulate import tabulate # type: ignore
from utils import get_masader_test, get_masader_valid
import numpy as np

def fix_arxiv_link(link):
    if link.endswith('.pdf'):
        link = link.replace('.pdf', '')
        _id = link.split('/')[-1]
        return f'https://arxiv.org/abs/{_id}'
    else:
        _id = link.split('/')[-1]
        return f'https://arxiv.org/abs/{_id}'

if __name__ == "__main__":
    args = create_args()
    metric_results = {}
    data_names = []
    len_data = 0

    if args.masader_validate:
        masader_data = get_masader_valid()
        titles = [str(x['Paper Title']) for x in masader_data]
        data_names = [str(x['Name']) for x in masader_data]
        paper_links = [str(x['Paper Link']) for x in masader_data]

    elif args.masader_test:
        masader_data = get_masader_test()
        titles = [str(x['Paper Title']) for x in masader_data]
        data_names = [str(x['Name']) for x in masader_data]
        paper_links = [str(x['Paper Link']) for x in masader_data]             
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
            model_results = run(mode = 'api', link = link, year = None, month = None, models = args.models.split(','), browse_web=args.browse_web, overwrite=args.overwrite)
        else:
            model_results = run(mode = 'api', keywords = args.keywords, year = None, month = None, models = args.models.split(','), browse_web=args.browse_web, overwrite= args.overwrite)

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
