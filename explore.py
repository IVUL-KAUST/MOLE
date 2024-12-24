import streamlit as st
import os
import json
from glob import glob
import pandas as pd
from utils import *

st.set_page_config(layout="wide")
def load_json_files(folder_path, model_name=None):
    """Load JSON files from the specified folder and filter by model name."""
    json_files = {}
    for file in glob('static/results/**/**.json'):
        if file.endswith(".json"):
            with open(file, "r") as f:
                try:
                    data = json.load(f)
                    data['relative_path'] = file
                    arxiv_id = file.split('/')[-2].replace('_arXiv','')
                    if arxiv_id not in json_files:
                        json_files[arxiv_id] = []
                    json_files[arxiv_id].append(data)
                except json.JSONDecodeError:
                    st.warning(f"Failed to read {file}. Ensure it is a valid JSON.")
    return json_files

def main():
    col1, col2 = st.columns([1, 2])

    folder_path = 'static/results'
    all_jsons = load_json_files(folder_path)

    if 'output' not in st.session_state:
        st.session_state['output'] = ''
    with col1:
        for arxiv_id in all_jsons:
            metadata = all_jsons[arxiv_id][0]['metadata']
            title = metadata["Paper Title"]
            with st.expander(title):
                models = st.multiselect("Select a model:", [r['config']['model_name'] for r in all_jsons[arxiv_id]], key = f'{arxiv_id}_model')
                compare = st.button('Compare', key = f'{arxiv_id}_compare_btn')
                eval_all = st.button('Eval Table', key = f'{arxiv_id}_compare_all_btn')
                if compare:
                    if len(models) == 2:
                        r1, r2 = [r for r in all_jsons[arxiv_id] if r['config']['model_name'] in models]
                        st.session_state['output'] = compare_results(r1, r2)
                    elif len(models) == 1:
                        for model in models:
                            for result in all_jsons[arxiv_id]:
                                if result['config']['model_name'] == model:
                                    st.link_button("Open using Masader Form", f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{result['relative_path']}")
                                    st.session_state['output'] = result['metadata']
                    else:
                        raise()
                if eval_all:
                    scores = {}
                    for result in all_jsons[arxiv_id]:
                        scores[result['config']['model_name']] = result['validation']
                    df = pd.DataFrame(scores)
                    df = df.map(lambda x: x * 100)
                    df = df.map("{0:.2f}".format)

                    st.session_state['output'] = df.transpose().sort_values('AVERAGE')
    with col2:
        st.write(st.session_state['output'])
            

if __name__ == "__main__":
    main()