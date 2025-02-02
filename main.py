from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pages.search import run
import json

app = FastAPI()

@app.post("/run")
async def func(link: str =  Form(''), file: UploadFile = File(None)):
    if file != None:
        pdf_content = file.file
    else:
        pdf_content = None

    browse_web = False
    model_name = 'gemini-1.5-flash'
    # Call your processing function with the file content and link
    results = run(link = link, paper_pdf=pdf_content, models = model_name.split(','), overwrite=True)
    print(results)
    return {'model_name': model_name, 'metadata': results[model_name]['metadata']}

@app.post("/schema")
async def func(name: str =  Form('')):

    with open(f"schema/{name}.json", "r") as f:
        data = json.load(f)
    return data
    