from datasets import load_dataset
from search import run
import requests
import time
import sys
from openai import OpenAI
from tqdm import tqdm
import concurrent.futures   
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

    # I need to speed up the process using threads (50 threads at maximum)
    with concurrent.futures.ThreadPoolExecutor(max_workers= 6) as executor:
        for example in tqdm(dataset):
            url = example['url']
            schema_name = example['schema_name']
            executor.submit(run, url, model_name=model_name, schema_name=schema_name, backend='vllm', results_path='synth_datasetv2', format='pdf_plumber', max_model_len=32768, max_output_len=2048)
        


