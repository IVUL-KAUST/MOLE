from glob import glob
import json 
import re
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup

files = glob("evals/**/test/*.json")

def get_arxiv_abstract_from_pdf_link(pdf_url):
    # Extract arXiv ID from the PDF link
    if '.pdf' not in pdf_url:
        pdf_url = pdf_url + ".pdf"
    match = re.search(r'arxiv\.org/pdf/(\d{4}\.\d{4,5})(v\d+)?\.pdf', pdf_url)
    if not match:
        raise ValueError("Invalid arXiv PDF URL")
    arxiv_id = match.group(1)

    # Query arXiv API
    api_url = f'http://export.arxiv.org/api/query?id_list={arxiv_id}'
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch from arXiv API")

    # Parse XML to get the abstract
    root = ET.fromstring(response.text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    abstract = root.find('atom:entry/atom:summary', ns).text.strip()

    return abstract


if __name__ == "__main__":
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        arxiv_link = data["Paper Link"]
        if '.pdf' not in arxiv_link:
            arxiv_link = arxiv_link + ".pdf"
        # get the abstract from the arxiv link
        abstract = get_arxiv_abstract_from_pdf_link(arxiv_link)
        abstract = abstract.replace("\n", " ")
        data["Abstract"] = abstract
        with open(file, "w") as f:
            json.dump(data, f, indent=4)