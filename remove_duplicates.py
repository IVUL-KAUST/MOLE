from glob import glob
import os
import json
format = "results_latex"
for paper_path in os.listdir(f'static/{format}'):
    results_path = f'static/{format}/{paper_path}/zero_shot'
    f_file = 'google_gemini-2.5-pro-preview-03-25-results.json'
    s_file = 'google_gemini-2.5-pro-results.json'
    if os.path.exists(f'{results_path}/{f_file}'):
        if os.path.exists(f'{results_path}/{s_file}'):
            os.remove(f'{results_path}/{s_file}')
        
    else:
        if os.path.exists(f'{results_path}/{s_file}'):
            json_data = json.load(open(f'{results_path}/{s_file}'))
            json_data['config']['model_name'] = 'google_gemini-2.5-pro-preview-03-25'
            with open(f'{results_path}/{f_file}', 'w') as f:
                json.dump(json_data, f)
            os.remove(f'{results_path}/{s_file}')

        
