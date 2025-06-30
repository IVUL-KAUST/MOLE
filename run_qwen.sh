for lang in ar en fr ru jp multi
do
    python evaluate.py --models qwen/qwen-2.5-72b-instruct -mt --schema $lang --few_shot 0 --results_path results_latex &
done