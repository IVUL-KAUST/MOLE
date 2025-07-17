
from acl_anthology import Anthology
import argparse
import pandas as pd

def search_acl_by_keyword(keywords, limit=10, venue_filter=None):
    if venue_filter and venue_filter.lower() == "lrec":
        venue_filter = "International Conference on Language Resources and Evaluation"
    """
    Search for papers in the ACL Anthology by keyword.
    
    Args:
        keywords (list): A list of keywords to search for.
        limit (int): The maximum number of papers to return.
        venue_filter (str): The venue to filter by (e.g., 'LREC').
        
    Returns:
        list: A list of dictionaries, where each dictionary contains the paper's information.
    """
    anthology = Anthology.from_repo()

    

    results = []
    for paper in anthology.papers():
        paper_venue_names = []
        if paper.venue_ids:
            for v_id in paper.venue_ids:
                if v_id in anthology.venues:
                    paper_venue_names.append(anthology.venues[v_id].name)
                else:
                    print(f"Warning: Venue ID {v_id} not found in anthology.venues")
        
        current_paper_venue = ', '.join(paper_venue_names) if paper_venue_names else None

        # Filter by venue first if venue_filter is provided
        if venue_filter and (not current_paper_venue or venue_filter.lower() not in current_paper_venue.lower()):
            continue

        # Filter by all keywords
        match = True
        paper_text = str(paper.title).lower() + " " + str(paper.abstract).lower()
        for keyword in keywords:
            if keyword.lower() not in paper_text:
                match = False
                break
        
        if match:
            results.append({
                'id': paper.id,
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'year': paper.year,
                'pdf': paper.pdf.url if paper.pdf else None,
                'abstract': paper.abstract,
                'venue': current_paper_venue
            })
            if len(results) >= limit:
                break
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
    parser.add_argument('--keyword', type=str, nargs='+', help='The keyword(s) to search for.')
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
    elif args.keyword:
        papers = search_acl_by_keyword(args.keyword, args.limit, args.venue)
        
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
