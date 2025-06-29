for lang in ar en fr ru jp multi
do
    python evaluate.py --models anthropic/claude-3.5-sonnet -mt --schema $lang --few_shot 0 --results_path results_latex &
done