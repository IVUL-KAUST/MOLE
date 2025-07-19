from  src.search import run
from src.schema import Schema, Attribute
from src.schema import Str, URL, Int, Float, Bool, List, Dict

link = 'https://arxiv.org/pdf/2402.03177'
custom_schema = Schema(schema_name='custom', schema_path='schema/custom.json')
metadata = {
    'Name': 'test',
    'Link': 'https://www.github.com/full',
    'Free': 'true',
    'Size': 100.0,
    'Languages': "English, Arabic, French",
}
print(custom_schema)
casted_metadata = custom_schema.cast(metadata)
print(casted_metadata)

# print(custom_schema.validate(casted_metadata))

raise()

def predict(link, model_name='google/gemini-2.5-pro', schema= 'sample'):
    schema.load()
    results = run(link = link, models = model_name.split(','), pdf_mode = 'plumber', schema = schema)

    model_name = model_name.replace('/', '_')
    return results[model_name]['metadata']

metadata = predict(link)
print(metadata)
