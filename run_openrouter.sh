for schema in ar en ru jp fr multi; do
    uv run src/evaluate.py --model deepseek/deepseek-chat-v3.1 --backend openrouter --split test --schema_name $schema --log
done
