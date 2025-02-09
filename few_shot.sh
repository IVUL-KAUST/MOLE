for i in 1 3 5; do
    python evaluate.py --models 'gemini-1.5-pro' -mv --schema ar --few_shot ${i}
done