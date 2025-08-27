for lang in ar en fr ru jp multi
do
    uv run evaluate.py --models $1 -mt --schema $lang --few_shot 0 --context_size $2 --results_path results_context_$2 &
done