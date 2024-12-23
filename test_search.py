import arxiv

def search_paper_by_title(title):
    client = arxiv.Client()
    search = arxiv.Search(
        query=' AND '.join(f'ti:{k}' for k in title.split(' ')),  # Title-based search
        max_results=5,  # You can adjust this to get more results
        sort_by=arxiv.SortCriterion.Relevance
    )
    print(search)
    for result in client.results(search):
        print(f"Title: {result.title}")
        print(f"URL: {result.entry_id}")
        print(f"Published: {result.published}")
        print(f"Summary: {result.summary}\n")

# Example usage:
search_paper_by_title('Natural Language Inference for Arabic Using Extended Tree Edit Distance with Subtrees')