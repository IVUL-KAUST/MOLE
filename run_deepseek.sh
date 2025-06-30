for lang in ar en fr ru jp multi
do
    python evaluate.py --models deepseek/deepseek-chat-v3-0324 -mt --schema $lang --few_shot 0 --results_path results_latex &
done