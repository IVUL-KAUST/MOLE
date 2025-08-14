
from acl_anthology import Anthology
import argparse
import pandas as pd
from tqdm import tqdm
from utils import create_hash, TextLogger
import os
import requests
from typing import Tuple
import time

top_100_languages =[
    "afrikaans", "albanian", "amharic", "arabic", "armenian", "aymara", "azerbaijani", "bengali",
    "bosnian", "bulgarian", "burmese", "chinese", "croatian", "czech", "danish", "dari",
    "dutch", "english", "estonian", "filipino", "finnish", "french", "georgian", "german",
    "greek", "guarani", "hebrew", "hindi", "hungarian", "icelandic", "indonesian", "irish",
    "italian", "japanese", "kazakh", "khmer", "kinyarwanda", "korean", "kurdish", "kyrgyz",
    "lao", "latvian", "lithuanian", "luxembourgish", "macedonian", "malagasy", "malay",
    "maltese", "mandarin", "maori", "mongolian", "montenegrin", "nepali", "norwegian",
    "pashto", "persian", "polish", "portuguese", "quechua", "romanian", "romansh", "russian",
    "serbian", "shona", "sinhala", "slovak", "slovene", "somali", "spanish", "swahili",
    "swedish", "tagalog", "tajik", "tamil", "thai", "turkish", "turkmen", "ukrainian",
    "urdu", "uzbek", "vietnamese", "xhosa", "zulu"
]
arabic_dialects = ["moroccan", "egyptian", "levantine", "palestinian", "syrian", "lebanese", "jordanian", "iraqi", "palestinian", "syrian", "lebanese", "jordanian", "iraqi", "palestinian", "syrian", "lebanese", "jordanian", "iraqi"]
id2lang = { 'ar': 'arabic', 'en': 'english', 'fr': 'french', 'jp': 'japanese', 'ru': 'russian', 'multi': 'multilingual', 'other': 'other'}
lang2id = {v: k for k, v in id2lang.items()}

class Downloader:
    def __init__(self, download_path: str = "static/papers/", log = True):
        self.download_path = download_path
        self.log = log
        self.logger = TextLogger(log = log)

    def download_paper(self, identifier: str, download_pdf: bool = True) -> Tuple[bool, str]:
        paper_dir = os.path.join(self.download_path, create_hash(identifier))
        os.makedirs(paper_dir, exist_ok=True)

        # download the pdf
        if download_pdf:
            response = None
            if os.path.exists(os.path.join(paper_dir, f"paper.pdf")):
                self.logger.show_info(f"📄 PDF already exists at {paper_dir}")
                return True, paper_dir
            for i in range(3):
                try:
                    response = requests.get(identifier, timeout=10)
                    break
                except Exception as e:
                    self.logger.show_warning(f"Error downloading paper {identifier}: {e}")
                    time.sleep(1)
            if response is not None and response.status_code == 200:
                with open(os.path.join(paper_dir, f"paper.pdf"), "wb") as f:
                    f.write(response.content)
                self.logger.show_info(f"📄 PDF downloaded successfully to {paper_dir}")
            else:
                self.logger.show_warning(f"Failed to download PDF for {identifier}")
                return False, paper_dir
        return True, paper_dir

class ACLDownloader(Downloader):
    def __init__(self, download_path: str = "static/papers/", log = True):
        super().__init__(download_path, log)

def get_words_from_paper(title, abstract, title_only=False):
    words = [] 
    if title_only:
        paper = title
    else:
        paper = title + ' ' + abstract
    for word in paper.split(' '):
        word = word.lower().strip()
        words.extend(word.split('-'))
    return words

def filter_schema(paper):
    data_keywords = ['dataset', 'datasets', 'benchmark', 'corpus', 'corpora', 'collection', 'collections', 'data']
    if paper.title is None or paper.abstract is None:
        return 'discard'
    paper_words = get_words_from_paper(str(paper.title), str(paper.abstract), title_only=False)
    title_words = get_words_from_paper(str(paper.title), str(paper.abstract), title_only=True)
    if not any(word in paper_words for word in data_keywords):
        return 'other'

    included_langs = []
    for lang in top_100_languages+arabic_dialects:
        if lang.lower() in paper_words:
            included_langs.append(lang)
    
    if len(included_langs) == 0 and any(word in title_words for word in data_keywords):
        return 'en'
    elif len(included_langs) == 1:
        if included_langs[0] in lang2id:
            return lang2id[included_langs[0]]
    elif len(included_langs) > 2 and ('multilingual' in paper_words or 'cross-lingual' in paper_words):
        return 'multi' 
    else:
        return 'discard'

class Paper:
    def __init__(self, id, title, year, url, abstract):
        self.id = id
        self.title = title
        self.year = year
        self.url = url
        self.abstract = abstract

def get_cached_papers():
    data  = []
    df = pd.read_csv('acl_anthology.csv')
    for index, row in df.iterrows():
        data.append(Paper(row['id'], row['title'], row['year'], row['url'], row['abstract']))
    return data

def annotate_schema(limit=None, redownload=False):
    if not redownload:
        papers = get_cached_papers()
    else:
        anthology = Anthology.from_repo()
        papers = anthology.papers()

    results = []
    # pbar = tqdm(total=limit, desc="Searching ACL Anthology")
    if limit is None:
        limit = len(papers)
    bpar = tqdm(total=limit, desc="Annotating schema", position=0)
    for paper in papers:
        bpar.update(1)
        if limit is not None and len(results) >= limit:
            break
        schema = filter_schema(paper)
        if schema == 'discard':
            continue
        # pbar.update(1)
        results.append({
            'id': paper.id,
            'title': paper.title,
            'year': paper.year,
            'url': paper.url,
            'abstract': paper.abstract,
            'schema_name': schema
        })
        
    bpar.close()
    return results

def count_all_papers():
    anthology = Anthology.from_repo()
    return len(list(anthology.papers()))

def count_papers_per_venue():
    anthology = Anthology.from_repo()
    venue_counts = {}
    for paper in anthology.papers():
        if paper.venue_ids:
            for v_id in paper.venue_ids:
                if v_id in anthology.venues:
                    venue_name = anthology.venues[v_id].name
                    venue_counts[venue_name] = venue_counts.get(venue_name, 0) + 1
    return venue_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search the ACL Anthology or count all papers.')
    parser.add_argument('--annotate_schema', action='store_true', help='Annotate the schema of the papers.')
    parser.add_argument('--limit', type=int, default=10, help='The maximum number of papers to return.')
    parser.add_argument('--venue', type=str, help='The venue to search for (e.g., LREC). Use "LREC" as a shortcut for "International Conference on Language Resources and Evaluation".')
    parser.add_argument('--count_all', action='store_true', help='Count all papers in the ACL Anthology.')
    parser.add_argument('--count_per_venue', action='store_true', help='Count papers per venue in the ACL Anthology.')
    parser.add_argument('--top_n', type=int, help='Display only the top N venues when counting papers per venue.')
    parser.add_argument('--venue_name', type=str, help='Display the number of papers for a specific venue.')
    parser.add_argument('--output_csv', type=str, nargs='?', const='results', help='Save search results to a CSV file. Optionally provide a filename (e.g., --output_csv my_results.csv). If no filename is provided, it defaults to the first keyword or "results.csv".')
    args = parser.parse_args()
    
    if args.count_all:
        num_papers = count_all_papers()
        print(f"Total number of papers in ACL Anthology: {num_papers}")
    elif args.count_per_venue:
        venue_counts = count_papers_per_venue()
        top_n = args.top_n if args.top_n else len(venue_counts)
        for venue, count in sorted(venue_counts.items(), key=lambda item: item[1], reverse=True)[:top_n]:
            print(f"{venue}: {count}")
    elif args.venue_name:
        venue_counts = count_papers_per_venue()
        found = False
        for venue, count in venue_counts.items():
            if venue.lower() == args.venue_name.lower():
                print(f"Number of papers in {venue}: {count}")
                found = True
                break
        if not found:
            print(f"Venue '{args.venue_name}' not found.")
    elif args.annotate_schema:
        papers = annotate_schema()
        
        if papers:
            df = pd.DataFrame(papers)
            if args.output_csv:
                output_filename = args.output_csv if args.output_csv != 'results' else f"{args.keyword[0].lower()}.csv"
                df[['title', 'pdf', 'abstract']].to_csv(output_filename, index=False)
                print(f"Results saved to {output_filename}")
            else:
                print(df[['title', 'pdf']])
        else:
            print("No papers found matching the criteria.")
    else:
        parser.print_help()
