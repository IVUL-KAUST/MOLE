from fastapi import FastAPI, UploadFile, File, Form # type: ignore
from src.search import run
import json

app = FastAPI()

@app.post("/run")
async def func(link: str =  Form(''), schema: str = Form(''), file: UploadFile = File(None)):
    if file != None:
        pdf_content = file.file
    else:
        pdf_content = None

    browse_web = False
    model_name = 'google/gemini-2.5-pro'

    # Call your processing function with the file content and link
    results = run(link = link, paper_pdf=pdf_content, models = model_name.split(','), overwrite=False, few_shot = 0, schema = schema, pdf_mode = 'plumber')
    
    # print(results)
    # results = json.load(open('/Users/zaidalyafeai/Documents/Development/masader_bot/static/results_latex/1410.3791/zero_shot/google_gemma-3-27b-it-browsing-results.json'))
    model_name = model_name.replace('/', '_')
    print(results[model_name]['metadata'])
    return {'model_name': model_name, 'metadata': results[model_name]['metadata']}

@app.post("/schema")
async def func(name: str =  Form('')):

    with open(f"schema/{name}.json", "r") as f:
        data = json.load(f)
    return data
    