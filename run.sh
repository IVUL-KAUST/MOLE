for schema in ar en fr ru jp multi; do
    python src/evaluate.py --model gemma-3-4b-it-sft-4096 --backend vllm --split test --schema_name $schema --max_model_len 4096 --max_output_len 2048 --overwrite
done