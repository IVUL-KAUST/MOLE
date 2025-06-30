for lang in ar en fr ru jp multi
do
    python evaluate.py --models google/gemini-2.5-pro-preview-03-25 -mt --schema $lang --few_shot 0 --results_path results_title --use_title &
done