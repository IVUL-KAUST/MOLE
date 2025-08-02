for schema in ar en fr ru jp multi; do
    python src/evaluate.py --model baseline-random --split test --schema_name $schema
done