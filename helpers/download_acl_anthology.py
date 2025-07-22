
from acl_anthology import Anthology
import argparse
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    anthology = Anthology.from_repo()
    results = []
    bpar = tqdm(total=100000, desc="Downloading ACL Anthology", position=0)
    _id = 0
    for paper in anthology.papers():
        if len(results) >= 100000:
            break
        if paper.pdf is None or paper.title is None or paper.abstract is None:
            continue
        bpar.update(1)
        results.append({
            'id': _id,
            'title': paper.title,
            'year': paper.year,
            'url': paper.pdf.url if paper.pdf else None,
            'abstract': paper.abstract,
        })
        _id += 1
    df = pd.DataFrame(results)
    df.to_csv('acl_anthology.csv', index=False)
    print(f"Downloaded {len(df)} papers")