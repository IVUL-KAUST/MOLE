for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models qwen/qwen-2.5-72b-instruct -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done