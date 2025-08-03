for schema in ar en fr ru jp multi; do
    python src/evaluate.py --model NuExtract-2.0-8B --backend vllm --split test --schema_name $schema --max_model_len 8912 --max_output_len 2048 --overwrite
done