#!/bin/bash

# Start VLLM server
echo -e "Launching VLLM"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
uv run vllm serve /hdd/shared_models/Qwen2.5-0.5B-Instruct --enable-lora --max-lora-rank 256 --max_model_len 8192 --port 8787 > logs/vllm_server.log 2>&1 &
sleep 150

# Initialize results file
> adapters_results.txt

# Loop through adapters
for r in 4 8 16 32 64 128; do
    for alpha in 4 8 16 32 64 128; do
        if [ $alpha -ge $r ]; then
            adapter_path="lora_tuning_adapters/r_${r}_alpha_${alpha}/Qwen2.5-0.5B-Instruct-kimi-k2-sft-8192-r-${r}-alpha-${alpha}"
            adapter_name="lora_r${r}_alpha${alpha}"
            
            [ ! -d "$adapter_path" ] && continue
            
            echo -e "\nProcessing $adapter_name with path: $adapter_path"
            
            # Load adapter
            curl -X POST http://localhost:8787/v1/load_lora_adapter \
                -H "Content-Type: application/json" \
                -d @- << EOF
{
    "lora_name": "$adapter_name",
    "lora_path": "$adapter_path"
}
EOF
            sleep 5
            
            # Evaluate and capture output
            echo -e "=== $adapter_name (r=$r, alpha=$alpha) ===" >> adapters_results.txt
            for schema in ar en fr ru jp multi model; do
                uv run python src/evaluate.py --model "$adapter_name" --schema_name "$schema" --backend vllm --split test --overwrite --max_model_len 8192 --max_output_len 2048
            done
            uv run src/plots.py --split test --group_by_x category >> adapters_results.txt 2>&1
            
            # Unload adapter
            curl -s -X POST "http://localhost:8787/v1/unload_lora_adapter" \
                -H "Content-Type: application/json" \
                -d "{\"lora_name\": \"$adapter_name\", \"top_p\":\"0.8\", \"top_k\":\"20\", \"temperature\":\"0.7\", \"repetition_penalty\":\"1.05\", \"max_tokens\":\"8192\"}"
        fi
    done
done

# Cleanup
echo -e "\nshutting down vllm"
pkill -f "vllm serve"