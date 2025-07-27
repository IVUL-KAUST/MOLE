from datasets import load_dataset
from utils import create_hash
from search import download_paper, extract_and_save_paper_text
from tqdm import tqdm
import time

dataset = load_dataset("csv", data_files = "train_dataset.csv", split = "train")

for example in tqdm(dataset):
    success, paper_path = download_paper(example['url'], log = False)
    paper_text = extract_and_save_paper_text(paper_path, context = "all", format = "pdf_plumber", save_paper_text = True, log = False)
    time.sleep(1)