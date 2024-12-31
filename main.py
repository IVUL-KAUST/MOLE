from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from pages.search import run

app = FastAPI()

@app.post("/run")
async def func(link:str = '', file: UploadFile = File(...)):
    pdf_content = file.file
    
    model_name = 'claude-3-5-haiku-latest'
    # Call your processing function with the file content and link
    results = run(link = link, paper_pdf=pdf_content, models = model_name.split(','))
    
    return {'model_name': model_name, 'metadata': results[model_name]['metadata']}
    