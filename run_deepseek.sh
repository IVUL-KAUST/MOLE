for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models deepseek/deepseek-chat-v3-0324 -mt --schema $lang --few_shot 0 -b --results_path results_latex &
done