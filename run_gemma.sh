for lang in ar en fr ru jp multi
do
    python evaluate.py --models google/gemma-3-27b-it -mt --schema $lang --few_shot 0 --results_path results_latex &
done