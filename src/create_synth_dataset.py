from datasets import load_dataset
from search import run
import requests
import time
import sys
from openai import OpenAI
from tqdm import tqdm

dataset = load_dataset('csv', data_files='train_dataset.csv', split='train')

def check_server_status(model, HOST = "localhost", PORT = 8787):
        url = f"http://{HOST}:{PORT}/v1"
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url=url,
            )
            print("running inference")
            chat_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "ping"},
                ]
            )
            print("inference done")
            print(chat_response)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

if __name__ == "__main__":
    model_name = sys.argv[1]
    while not check_server_status(model_name):
        print("Server is not ready, waiting for 1 second")
        time.sleep(1)
    print("Server is ready")
    for example in tqdm(dataset):
        try:
            url = example['url']
            schema_name = example['schema_name']
            result = run(url, model_name=model_name, schema_name=schema_name, backend='vllm', results_path='synth_dataset', format='pdf_plumber')
        except Exception as e:
            print(f"Error: {e}")
            continue


