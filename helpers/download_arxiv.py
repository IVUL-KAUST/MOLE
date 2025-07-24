import urllib, urllib.request

import pandas as pd
from tqdm import tqdm
import time
import xml.etree.ElementTree as ET
import requests
from bs4 import BeautifulSoup as bs4

if __name__ == "__main__":
    url = "http://export.arxiv.org/oai2?verb=ListSets"
    results = requests.get(url).text
    soup = bs4(results, 'xml')
    total = 100000
    tqdm_bar = tqdm(total=total, desc="Downloading Arxiv Papers")
    data = []
    _id = 0
    for year in range(2010, 2026):
        for month in range(1, 13):
            month = f'0{month}' if month < 10 else str(month)
            url = f"http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv&set=cs:cs:CL&from={year}-{month}-01&until={year}-{month}-31"
            results = requests.get(url).text
            soup = bs4(results, 'xml')
            records = soup.find_all('record')
            for r in range(0, min(total - len(data), len(records))):
                record = records[r]
                title = record.find('title').text
                abstract = record.find('abstract').text
                year = record.find('created').text.split('-')[0]
                arxiv_id = record.find('identifier').text.split(':')[-1]
                url = f"https://arxiv.org/pdf/{arxiv_id}"
                data.append({
                    'id': _id,
                    'title': title,
                    'year': year,
                    'url': url,
                    'abstract': abstract,
                })
                _id += 1
                tqdm_bar.update(1)
                tqdm_bar.set_description(f"Month: {month}, Year: {year}, Total: {len(data)}")
        if len(data) >= total:
            break
    tqdm_bar.close()
    pd.DataFrame(data).to_csv('arxiv_papers.csv', index=False)            
