import streamlit as st
import os
import json
from glob import glob

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
    folder_path = 'static/results'
    all_jsons = load_json_files(folder_path)

    for arxiv_id in all_jsons:
        metadata = all_jsons[arxiv_id][0]['metadata']
        title = metadata["Paper Title"]
        with st.expander(title):
            model = st.selectbox("Select a model:", [r['config']['model_name'] for r in all_jsons[arxiv_id]], key = f'{arxiv_id}_model')
            if model:
                for result in all_jsons[arxiv_id]:
                    if result['config']['model_name'] == model:
                        st.link_button("Open using Masader Form", f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{result['relative_path']}")
                        st.json(result['metadata'])

if __name__ == "__main__":
    main()