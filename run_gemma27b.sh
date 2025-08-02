for schema in ar en fr ru jp multi; do
    python src/evaluate.py --model google/gemma-3-27b-it --backend openrouter --split test --schema_name $schema
done
