from search import run 
from arg_utils import args
import os 
import json 
import asyncio
import concurrent.futures

# include src in the path 


models = {
    "DeepSeek-V3": "https://arxiv.org/pdf/2412.19437",
    "BLOOM": "https://arxiv.org/pdf/2211.05100",
    "Samba": "https://arxiv.org/pdf/2406.07522?",
    "RWKV": "https://arxiv.org/pdf/2305.13048",
    "Mistral 7B": "https://arxiv.org/pdf/2310.06825",
    "GPT-OSS": "https://arxiv.org/pdf/2508.10925",
    "Janus-Pro": "https://arxiv.org/pdf/2501.17811",
    "Phi-4": "https://arxiv.org/pdf/2412.08905",
    "StarCoder": "https://arxiv.org/pdf/2305.06161",
    "Qwen 2.5": "https://arxiv.org/pdf/2412.15115",
    "SoundStrom": "https://arxiv.org/pdf/2305.09636",
    "Whisper": "https://arxiv.org/pdf/2212.04356",
    "Falcon": "https://arxiv.org/pdf/2311.16867",
    "BERT": "https://arxiv.org/pdf/1810.04805",
    "KIMI K2": "https://arxiv.org/pdf/2507.20534",
    "Llama 3": "https://arxiv.org/pdf/2407.21783",
    "ChatGLM": "https://arxiv.org/pdf/2406.12793",
    "Qwen Image": "https://arxiv.org/pdf/2508.02324",
    "Smollm2": "https://arxiv.org/pdf/2502.02737",
    "Florence-2": "https://arxiv.org/pdf/2311.06242",
    "Gemma 3": "https://arxiv.org/pdf/2503.19786"
}

async def main():
    args.model_name='moonshotai/kimi-k2'
    args.schema_name='model'
    args.backend='openrouter'
    args.results_path='synth_dataset_models'
    args.format='pdf_plumber'
    args.log=True
    urls = list(models.values())

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        
        for idx, url in enumerate(urls):
            task = loop.run_in_executor(executor, run,
                url,
                args
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
    
    for i, r in enumerate(results):
        metadata = r['moonshotai/kimi-k2']['metadata']
        metadata['annotations_from_paper'] = {key: 1 for key in metadata.keys()}
        name = metadata['Name']
        metadata['Paper_Link'] = urls[i]
        os.makedirs('evals/model/test', exist_ok=True)
        json.dump(metadata, open(f'evals/model/test/{name}.json', 'w'), indent=4)

if __name__ == "__main__":
    asyncio.run(main())