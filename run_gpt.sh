for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models openai/gpt-4o -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done