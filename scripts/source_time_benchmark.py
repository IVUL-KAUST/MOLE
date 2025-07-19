#!/usr/bin/env python3
"""
Script to benchmark paper extraction methods: LaTeX, PDF Plumber, and Docling.
Mimics the exact extraction process from process.py to ensure accurate comparisons.
"""

import os
import time
import json
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import sys
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary functions from the project
from src.utils import setup_logger
from src.constants import eval_datasets_ids

# Import the extract_paper_text function directly from process.py
from src.search import extract_paper_text
from src.utils import get_paper_content_from_docling

# Load environment variables and set up logger
load_dotenv()
logger = setup_logger()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Benchmark paper extraction methods using the exact approach from process.py"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        choices=["test", "valid"],
        help="Dataset split to use for benchmarking (test or valid)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--papers_dir", 
        type=str, 
        default="static/papers",
        help="Directory containing paper sources"
    )
    parser.add_argument(
        "--force_download", 
        action="store_true",
        help="Force download papers even if they already exist"
    )
    parser.add_argument(
        "--plot_only", 
        action="store_true",
        help="Only generate plots from existing benchmark data"
    )
    parser.add_argument(
        "-n", "--num_papers",
        type=int,
        default=None,
        help="Number of papers to process (for testing/debugging)"
    )
    return parser.parse_args()

def download_paper(paper_id, download_path="static/papers/"):
    """Download paper from arXiv if not already downloaded"""
    from src.search_arxiv import ArxivSourceDownloader
    
    # Check if paper already exists
    paper_path = f"{download_path}/{paper_id}"
    if os.path.exists(paper_path) and os.path.exists(f"{paper_path}/paper.pdf"):
        logger.info(f"Paper {paper_id} already exists at {paper_path}")
        return paper_path
    
    # Download paper
    logger.info(f"Downloading paper {paper_id}...")
    downloader = ArxivSourceDownloader(download_path=download_path)
    success, paper_path = downloader.download_paper(paper_id, verbose=True)
    
    if success:
        # Copy PDF to arXiv folder if it exists
        if os.path.exists(f"{paper_path}/paper.pdf"):
            import shutil
            os.makedirs(f"{paper_path}_arXiv", exist_ok=True)
            shutil.copy(f"{paper_path}/paper.pdf", f"{paper_path}_arXiv/paper.pdf")
        logger.info(f"Successfully downloaded paper {paper_id}")
    else:
        logger.error(f"Failed to download paper {paper_id}")
    
    return paper_path if success else None

def benchmark_extraction(paper_path, method, context_size="all"):
    """Benchmark a specific extraction method by calling extract_paper_text with appropriate parameters"""
    start_time = time.time()
    
    try:
        if method == "latex":
            # For latex extraction, use_pdf should be False and pdf_mode should be None
            paper_text = extract_paper_text(
                path=paper_path, 
                use_pdf=False, 
                pdf_mode=None, 
                context_size=context_size
            )
        elif method == "pdfplumber":
            # For PDF Plumber, use_pdf should be True and pdf_mode should be "plumber"
            paper_text = extract_paper_text(
                path=paper_path, 
                use_pdf=True, 
                pdf_mode="plumber", 
                context_size=context_size
            )
        elif method == "docling":
            # paper_text = extract_paper_text(
            #     path=paper_path, 
            #     use_pdf=True, 
            #     pdf_mode="docling", 
            #     use_cached_docling=False,
            #     context_size=context_size,
            # )
            paper_text = get_paper_content_from_docling(paper_path=f'{paper_path}/paper.pdf')
        else:
            logger.error(f"Unknown extraction method: {method}")
            return "", 0
            
    except Exception as e:
        logger.error(f"Error during {method} extraction: {str(e)}")
        paper_text = ""
    
    extraction_time = time.time() - start_time
    return paper_text, extraction_time

def save_benchmark_results(results, output_path):
    """Save benchmark results to a JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Benchmark results saved to {output_path}")

def plot_benchmark_results(results, output_dir, split):
    """Generate plots for the benchmark results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to pandas DataFrame for easier plotting
    data = []
    for paper_id, metrics in results.items():
        for method, values in metrics.items():
            if method != "paper_id" and method != "paper_path":
                data.append({
                    "paper_id": paper_id,
                    "method": method,
                    "time": values["time"]
                })
    
    df = pd.DataFrame(data)
    
    # Create figure directory
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot 1: Average extraction time by method
    plt.figure(figsize=(10, 6))
    avg_times = df.groupby('method')['time'].mean()
    bars = plt.bar(avg_times.index, avg_times.values)
    plt.ylabel('Average Extraction Time (seconds)')
    plt.title(f'Average Extraction Time by Method ({split} split)')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}s', ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{split}_avg_extraction_time.png"), dpi=300)
    plt.close()
    
    # Plot 2: Box plot of extraction times
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[df['method'] == m]['time'] for m in df['method'].unique()], 
                tick_labels=df['method'].unique())
    plt.ylabel('Extraction Time (seconds)')
    plt.title(f'Distribution of Extraction Times ({split} split)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{split}_time_boxplot.png"), dpi=300)
    plt.close()
    
    # Generate summary statistics
    summary = {
        "dataset_split": split,
        "total_papers": len(results),
        "methods": {}
    }
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        summary["methods"][method] = {
            "avg_time": method_df['time'].mean(),
            "median_time": method_df['time'].median(),
            "min_time": method_df['time'].min(),
            "max_time": method_df['time'].max(),
            "std_dev": method_df['time'].std()
        }
    
    # Save summary
    with open(os.path.join(output_dir, f"{split}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create a summary table as CSV
    summary_rows = []
    for method, stats in summary["methods"].items():
        summary_rows.append({
            "Method": method,
            "Avg Time (s)": f"{stats['avg_time']:.2f}",
            "Median Time (s)": f"{stats['median_time']:.2f}",
            "Min Time (s)": f"{stats['min_time']:.2f}",
            "Max Time (s)": f"{stats['max_time']:.2f}",
            "Std Dev": f"{stats['std_dev']:.2f}"
        })
    
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, f"{split}_summary_table.csv"), 
        index=False
    )
    
    # Create detailed paper-by-paper CSV for easy copy/paste to spreadsheets
    paper_rows = []
    methods = sorted(list(set([item['method'] for item in data])))
    
    # Create a pivot table-like structure with papers as rows and methods as columns
    for paper_id in sorted(results.keys()):
        row = {"paper_id": paper_id}
        for method in methods:
            method_time = 0
            if method in results[paper_id]:
                method_time = results[paper_id][method]["time"]
            row[f"{method}_time"] = f"{method_time:.2f}"
        paper_rows.append(row)
    
    # Save detailed results as CSV
    pd.DataFrame(paper_rows).to_csv(
        os.path.join(output_dir, f"{split}_detailed_times.csv"), 
        index=False
    )
    
    logger.info(f"Plots and summaries saved to {output_dir}")
    return summary

def benchmark_paper(paper_id, papers_dir, force_download=False):
    """Benchmark extraction methods for a single paper"""
    # Download paper if needed
    if force_download:
        paper_path = download_paper(paper_id, download_path=papers_dir)
    else:
        paper_path = f"{papers_dir}/{paper_id}"
        if not os.path.exists(paper_path) or not os.path.exists(f"{paper_path}/paper.pdf"):
            paper_path = download_paper(paper_id, download_path=papers_dir)
    
    if not paper_path:
        logger.error(f"Could not process paper {paper_id}")
        return None
    
    results = {"paper_id": paper_id, "paper_path": paper_path}
    
    # Benchmark each extraction method
    methods = ["latex", "pdfplumber", "docling"]
    for method in methods:
        try:
            logger.info(f"Benchmarking {method} extraction for paper {paper_id}")
                
            # Run the benchmark for this method
            paper_text, extraction_time = benchmark_extraction(
                paper_path=paper_path,
                method=method
            )
            
            # Store the results
            results[method] = {"time": extraction_time}
            logger.info('='*120)
            
        except Exception as e:
            logger.error(f"Error benchmarking {method} for paper {paper_id}: {str(e)}")
            results[method] = {"time": 0}
    
    return results

def get_all_paper_ids(split):
    """Get all paper IDs from all languages for the specified split"""
    all_paper_ids = []
    for language in eval_datasets_ids:
        if split in eval_datasets_ids[language]:
            all_paper_ids.extend(eval_datasets_ids[language][split])
    return all_paper_ids

def main():
    """Main function to run the benchmark"""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all paper IDs for the specified split across all languages
    paper_ids = get_all_paper_ids(args.split)
    
    if not paper_ids:
        logger.error(f"No paper IDs found for {args.split} split")
        return
    
    # Limit the number of papers if specified
    if args.num_papers is not None:
        paper_ids = paper_ids[:args.num_papers]
        logger.info(f"Limiting benchmark to {args.num_papers} papers")
    
    output_file = os.path.join(args.output_dir, f"{args.split}_benchmark.json")
    
    if not args.plot_only:
        # Run benchmark
        benchmark_results = {}
        
        for paper_id in tqdm(paper_ids, desc=f"Benchmarking {args.split} papers"):
            logger.info(f"Processing paper {paper_id}")
            results = benchmark_paper(paper_id, args.papers_dir, args.force_download)
            if results:
                benchmark_results[paper_id] = results
            
            # Save incremental results
            save_benchmark_results(benchmark_results, output_file)
    else:
        # Load existing benchmark results
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                benchmark_results = json.load(f)
            logger.info(f"Loaded existing benchmark results from {output_file}")
        else:
            logger.error(f"No existing benchmark results found at {output_file}")
            return
    
    # Generate plots and summary statistics
    summary = plot_benchmark_results(benchmark_results, args.output_dir, args.split)
    
    # Print summary
    logger.info("\n=== BENCHMARK SUMMARY ===")
    logger.info(f"Dataset split: {args.split}")
    logger.info(f"Total papers: {summary['total_papers']}")
    
    for method, stats in summary["methods"].items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Average extraction time: {stats['avg_time']:.2f} seconds")
        logger.info(f"  Median extraction time: {stats['median_time']:.2f} seconds")
    
    logger.info(f"\nDetailed extraction times for each paper saved to {args.output_dir}/{args.split}_detailed_times.csv")

if __name__ == "__main__":
    main()