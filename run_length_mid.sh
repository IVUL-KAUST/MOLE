
for lang in ar en fr ru jp multi 
do
    uv run evaluate.py --models $1 -mt --schema $lang --few_shot 0 --results_path results_length_mid &
done