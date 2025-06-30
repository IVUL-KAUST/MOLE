for lang in ar en fr ru jp multi
do
    python evaluate.py --models openai/gpt-4o -mt --schema $lang --few_shot 0 --results_path results_latex &
done