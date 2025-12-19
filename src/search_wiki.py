import os
from utils import create_hash
import wikipediaapi

class WikiDownloader:
    def __init__(self, download_path="static/papers/", log = True):
        self.download_path = download_path
        self.log = log
        self.wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="my-research-script/1.0"
        )
    
    def _create_download_dir(self, paper_id: str) -> str:
        """Create and return the download directory path."""
        paper_dir = os.path.join(self.download_path, create_hash(paper_id))
        os.makedirs(paper_dir, exist_ok=True)
        return paper_dir
    
    def _download_file(self, url, path):
        page_title = url.split("/")[-1]
        page = self.wiki.page(page_title)

        with open(path, "w", encoding="utf-8") as f:
            f.write(page.text)

    def download_paper(self, paper_link):
        paper_dir = self._create_download_dir(paper_link)
        try:
            self._download_file(paper_link, os.path.join(paper_dir, "paper_text.txt"))
            return True, paper_dir
        except Exception as e:
            return False, None
        
        