for lang in ar en fr ru jp multi
do
    python evaluate.py --models baseline-keyword -mt --schema $lang -o --few_shot 0 --results_path results_latex
done