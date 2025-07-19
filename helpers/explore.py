import streamlit as st #type: ignore
import os
import json
from glob import glob
import pandas as pd
from src.utils import *
import numpy as np
from src.constants import TEST_DATASETS_IDS, VALID_DATASETS_IDS

st.set_page_config(layout="wide")


def load_json_files(foulder_path="static/results"):
    """Load JSON files from the specified folder and filter by model name."""
    json_files = {}
    for file in glob(f"{foulder_path}/**/**.json"):
        if file.endswith(".json"):
            with open(file, "r") as f:
                try:
                    data = json.load(f)
                    data["relative_path"] = file
                    arxiv_id = file.split("/")[-2].replace("_arXiv", "")
                    if arxiv_id not in json_files:
                        json_files[arxiv_id] = []
                    json_files[arxiv_id].append(data)
                except json.JSONDecodeError:
                    st.warning(f"Failed to read {file}. Ensure it is a valid JSON.")
    return json_files


def main():
    folder_path = "static/results"
    all_jsons = load_json_files(folder_path)

    activate = st.toggle('Show test')
    ids = VALID_DATASETS_IDS
    if activate:
        ids = TEST_DATASETS_IDS
    
    json_files = {}
    
    for arxiv_id in all_jsons:
        if arxiv_id not in ids:
            continue
        else:
            json_files[arxiv_id] = all_jsons[arxiv_id]
    model_results = {}
    len_datasets = len(json_files)
    for arxiv_id in json_files:
        scores = {}
        for result in json_files[arxiv_id]:
            scores[result["config"]["model_name"]] = result["validation"]

        for model_name in scores:
            if model_name == "human":
                continue
            if model_name not in model_results:
                model_results[model_name] = []
            model_results[model_name].append([v for k, v in scores[model_name].items()])
    results = {}
    for model_name in model_results:
        if len(model_results[model_name]) == len_datasets:
            results[model_name] = np.mean(model_results[model_name], axis=0)

    df = pd.DataFrame(results).transpose()
    df.columns = ["CONTENT", "ACCESSABILITY", "DIVERSITY", "EVALUATION", "AVERAGE"]
    df = df.map(lambda x: x * 100)
    df = df.sort_values("AVERAGE")
    df = df.map("{0:.2f}".format)
    st.write(df)

    col1, col2 = st.columns([1, 2])

    if "output" not in st.session_state:
        st.session_state["output"] = ""
    for arxiv_id in json_files:
        metadata = json_files[arxiv_id][0]["metadata"]
        title = metadata["Paper Title"]
        with st.expander(title):
            models = st.multiselect(
                "Select a model:",
                [r["config"]["model_name"] for r in json_files[arxiv_id]],
                key=f"{arxiv_id}_model",
            )
            compare = st.button("Compare", key=f"{arxiv_id}_compare_btn")
            show_diff = st.toggle("show diff", key=f"{arxiv_id}_show_diff")
            eval_all = st.button("Eval Table", key=f"{arxiv_id}_compare_all_btn")
            if compare:
                if len(models) >= 1:
                    results = [
                        r
                        for r in all_jsons[arxiv_id]
                        if r["config"]["model_name"] in models
                    ]
                    st.session_state["output"] = compare_results(
                        results, show_diff=show_diff
                    )
                    st.write(st.session_state["output"])
                elif len(models) == 1:
                    for model in models:
                        for result in json_files[arxiv_id]:
                            if result["config"]["model_name"] == model:
                                st.link_button(
                                    "Open using Masader Form",
                                    f"https://masaderform-production.up.railway.app/?json_url=https://masaderbot-production.up.railway.app/app/{result['relative_path']}",
                                )
                                st.session_state["output"] = result["metadata"]
                                st.write(st.session_state["output"])
                else:
                    raise ()
            if eval_all:
                scores = {}
                for result in json_files[arxiv_id]:
                    scores[result["config"]["model_name"]] = result["validation"]
                df = pd.DataFrame(scores)
                df = df.map(lambda x: x * 100)
                df = df.transpose().sort_values("AVERAGE")
                df = df.map("{0:.2f}".format)
                st.session_state["output"] = df
                st.write(df)


if __name__ == "__main__":
    main()
