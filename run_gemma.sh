for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models google/gemma-3-27b-it -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done