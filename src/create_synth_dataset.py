from datasets import load_dataset
from search import run
import requests
import time
import sys
from openai import OpenAI
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=False, default=6)
parser.add_argument("--num_proc", type=int, required=False, default=6)
parser.add_argument("--backend", type=str, required=False, default="vllm")
args = parser.parse_args()

dataset = load_dataset('csv', data_files='train_dataset.csv', split='train')
dataset = dataset.filter(lambda x: x['schema_name'] in ['ar', 'ru', 'en', 'jp', 'fr', 'multi'])
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
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

def inference(examples):
    urls = examples['url']
    schema_names = examples['schema_name']
    for url, schema_name in zip(urls, schema_names):
        try:
            run(url, model_name=args.model_name, schema_name=schema_name, backend=args.backend, results_path='synth_datasetv2', format='pdf_plumber', max_model_len=32768, max_output_len=2048, log = False)
        except Exception as e:
            continue
    return examples

if __name__ == "__main__":
    if args.backend == "vllm":
        while not check_server_status(args.model_name):
            print("Server is not ready, waiting for 1 second")
            time.sleep(1)
        print("Server is ready")

    # I need to speed up the process using threads (50 threads at maximum)
    dataset.map(inference, batched=True, batch_size=args.batch_size, num_proc=args.num_proc)
        


