from glob import glob
import json
import requests
import re
from datetime import datetime

def get_bibtex(url, name):
    # Extract arXiv ID from the URL
    arxiv_id = None
    if 'arxiv.org' in url:
        # Handle different arXiv URL formats
        if '/abs/' in url:
            arxiv_id = url.split('/abs/')[-1]
        elif '/pdf/' in url:
            arxiv_id = url.split('/pdf/')[-1].replace('.pdf', '')
    
    if not arxiv_id:
        return "Error: Invalid arXiv URL"
    
    # Clean up arXiv ID (remove version number if present)
    arxiv_id = arxiv_id.split('v')[0]
    
    # Query arXiv API
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    response = requests.get(api_url)
    
    if response.status_code != 200:
        return "Error: Failed to fetch paper metadata"
    
    # Parse the XML response
    from xml.etree import ElementTree as ET
    root = ET.fromstring(response.content)
    
    # Extract paper metadata
    entry = root.find('{http://www.w3.org/2005/Atom}entry')
    if entry is None:
        return "Error: Paper not found"
    
    title = entry.find('{http://www.w3.org/2005/Atom}title').text
    authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
              for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
    published = entry.find('{http://www.w3.org/2005/Atom}published').text
    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text
    
    # Format date
    date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
    year = date.year
    
    # Create BibTeX entry
    bibtex = f"""@article{{{name},
  title = {{{title}}},
  author = {{{' and '.join(authors)}}},
  year = {{{year}}},
  archivePrefix = {{arXiv}},
  journal={{arXiv preprint arXiv: {arxiv_id}}},
  primaryClass = {{cs.AI}},
  eprint = {{{arxiv_id}}},
  url = {{{url}}}
}}"""
    
    return bibtex

write_up = ""
for file in glob("evals/*/*/*.json"):
    with open(file, "r") as f:
        data = json.load(f)
    link = data["Paper Link"]
    name = data["Name"]
    name = name.replace(" ", "_")
    bibtex = get_bibtex(link, name)
    write_up += f"{name.replace('_', ' ')} \cite{{{name}}}, "
    with open(f"custom.bib", "a") as f:
        f.write(bibtex)

print(write_up)