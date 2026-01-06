from fastapi import FastAPI, UploadFile, File, Form # type: ignore
# add absolute path from src 
import sys
sys.path.append('src')
from search import run
from schema import get_schema
import arg_utils
import json

keys_order_datasets = ["Name", "Subsets", "HF_Link", "Link", "License", "Year", "Language", "Dialect", "Domain", "Form", "Collection_Style", "Description", "Volume", "Unit", "Ethical_Risks", "Provider", "Derived_From", "Paper_Title", "Paper_Link", "Script", "Tokenized", "Host", "Access", "Cost", "Test_Split", "Tasks", "Venue_Title", "Venue_Type", "Venue_Name", "Authors", "Affiliations", "Abstract", "Added_By"]
keys_order_tool = ["Name", "Link", "License", "Year", "Access", "Version", "Description", "Provider", "Paper_Title", "Paper_Link", "Tasks", "Interface", "Host", "Code_Execution", "Supported_OS", "Programming_Language"]
keys_order_msed = ["Title", "Paper_Link", "Link", "Author", "Authoraffiliation", "Doi", "Email", "Date", "Abstract", "Added_By"]
app = FastAPI()

@app.post("/run")
async def func(link: str =  Form(''), schema_name: str = Form(''), file: UploadFile = File(None)):
    browse_web = False
    model_name = 'moonshotai/kimi-k2'

    # Call your processing function with the file content and link
    _args = arg_utils.args
    _args.model_name = model_name
    _args.schema_name = schema_name
    _args.format = 'pdf_plumber'
    _args.overwrite = True
    _args.log = True
    print(link)
    results = run(link, _args)
    
    # print(results)
    # results = json.load(open('/Users/zaidalyafeai/Documents/Development/masader_bot/static/results_latex/1410.3791/zero_shot/google_gemma-3-27b-it-browsing-results.json'))
    metadata = results[model_name]['metadata']
    metadata['Added_By'] = model_name
    print(metadata)
    return {'model_name': model_name, 'metadata': metadata}

@app.post("/schema")
async def func(name: str =  Form('')):
    schema = get_schema(name)
    schema_dict = json.loads(schema.schema())
    readme = 'GUIDELINES.md'
    keys_order = keys_order_datasets
    if name == 'tool':
        readme = 'GUIDELINES_TOOL.md'
        keys_order = keys_order_tool
    elif name == 'msed':
        readme = 'GUIDELINES_MSED.md'
        keys_order = keys_order_msed
    
    for line in open(readme, 'r').readlines():
        if '**' in line:
            key = line.strip().split('**')[1].replace(' ', '_')
            schema_dict[key]['description'] = line.strip().split('**')[-1].strip().capitalize()
    schema_dict ['Added_By'] = {
        "answer_type": "str",
        "answer_min": 1,
        "answer_max": 1,
        "description": "Your full name"
    }
    return {key: schema_dict[key] for key in keys_order}
    