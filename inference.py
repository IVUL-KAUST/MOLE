from  pages.search import run
link = 'https://arxiv.org/pdf/2402.03177'

def predict(link, model_name='google/gemini-2.5-pro', schema= 'arxiv'):
    results = run(link = link, models = model_name.split(','), pdf_mode = 'plumber', schema = schema)

    model_name = model_name.replace('/', '_')
    return results[model_name]['metadata']

metadata = predict(link)
print(metadata)
