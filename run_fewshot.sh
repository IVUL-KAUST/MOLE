for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models google/gemini-2.5-pro-preview-03-25 -mt --schema $lang --few_shot $1 --results_path results_latex &
done