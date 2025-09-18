#!/bin/bash

# Array of models (one per line for readability)
models=(
    "openai/gpt-4o"
    "anthropic/claude-3.5-sonnet"
    "google/gemini-2.5-pro"
    "deepseek/deepseek-chat-v3-0324"
    "meta-llama/llama-4-maverick"
    "google/gemma-3-27b-it"
    "qwen/qwen-2.5-72b-instruct"
)

# Run evaluation for each model and language combination
for model in "${models[@]}"; do
    echo "Running evaluation for model: $model"
    for lang in ar en fr ru jp multi; do
        echo "  Language: $lang"
        uv run evaluate.py --models "$model" -mt --schema "$lang" --few_shot 0 --results_path results_pdf --pdf_mode plumber
    done
    echo "Completed evaluation for model: $model"
    echo "----------------------------------------"
done

echo "All evaluations completed!"