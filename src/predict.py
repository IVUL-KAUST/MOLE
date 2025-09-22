from search import get_metadata
from rich import print

def extract(text, model_name, schema_name, backend = "openrouter", max_model_len = 8192, max_output_len = 2084):
    message, metadata, cost, error = get_metadata(
        text, model_name, schema_name=schema_name, backend = backend, log = False, max_model_len = max_model_len, max_output_len = max_output_len
    )
    return metadata

text = """
Mextract 1.0 is a 3B model that can extract metadata from any text. The model has been evaluated on the MOLE
benchmark and achieved impressive results. The arhitecture is based on Transformer model. The model is released
under the Apache 2.0 license. The model can process up to 8192 tokens. The model was released in 2025 from KAUST.
"""

metadata = extract(
    text, "Qwen2.5-3B-Instruct-kimi-k2-sft-8192", schema_name="model", backend = "vllm"   
)
print(metadata)

metadata = extract(
    text, "moonshotai/kimi-k2", schema_name="model", backend = "openrouter"   
)
print(metadata)