for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models meta-llama/llama-4-maverick -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done