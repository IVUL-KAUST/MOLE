for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models google/gemini-2.5-pro-preview-03-25 -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done