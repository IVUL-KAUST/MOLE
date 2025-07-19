
import os
from glob import glob
from plots import remap_names
from src.utils import get_id_from_path
import shutil

files = glob('static/results_latex/**/**/**.json')
cut_off_period= {
    "Gemma 3 27B": ['2502.07455', '2504.13161', '2501.14249', '2504.21677', '2504.15941', '2505.04851', '2503.13102'],
    "Gemini 2.5 Pro": ['2504.13161', '2504.21677', '2504.15941', '2505.04851', '2503.13102'],
    "DeepSeek V3": ['2504.13161', '2504.21677', '2504.15941', '2505.04851', '2503.13102'],
    "Llama 4 Maverick": ['2504.13161', '2504.21677', '2504.15941', '2505.04851'],
    "Claude 3.5 Sonnet": ['2502.07455', '2504.13161', '2501.14249', '2504.21677', '2504.15941', '2501.17117', '2505.04851', '2503.13102', '2501.05841'],
    "Qwen 2.5 72B": ['2502.07455', '2504.13161', '2501.14249', '2504.21677', '2504.15941', '2501.17117', '2505.04851', '2503.13102', '2501.05841'],
    "GPT 4o": ['2502.07455', '2504.13161', '2501.14249', '2504.21677', '2504.15941', '2501.17117', '2505.04851', '2503.13102', '2501.05841'],
}
os.makedirs('static/results_cutoff', exist_ok=True)

for file in files:
    file_name = file.split('/')[-1]
    model_name = file_name.replace('.json', '').replace('-results', '')
    model_name = remap_names(model_name)
    arxiv_id = get_id_from_path(file)
    if model_name in cut_off_period and arxiv_id in cut_off_period[model_name]:
        # copy to results_latex/results_cutoff
        os.makedirs(f'static/results_cutoff/{arxiv_id}/zero_shot', exist_ok=True)
        shutil.copy(file, f'static/results_cutoff/{arxiv_id}/zero_shot/{file_name}')
