from fastapi import FastAPI
from pages.search import run_by_link

app = FastAPI()

@app.get("/run")
async def run(link):
    results = run_by_link(link = link)
    return results['gemini-1.5-flash']['metadata']
    