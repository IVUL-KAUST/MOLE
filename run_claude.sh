for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models anthropic/claude-3.5-sonnet -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done