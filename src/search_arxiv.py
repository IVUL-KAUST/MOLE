import gzip
import arxiv
from datetime import datetime, date
from typing import List, Dict, Optional, Union, Tuple
import shutil
import requests
import tarfile
import os
from urllib.parse import urlparse
import gzip
from utils import *
import hashlib

class ArxivSourceDownloader:
    """
    Downloads source files and PDFs from arXiv papers.
    Handles multiple source formats including gzip, tar, pdf, and ps.
    """
    
    def __init__(self, download_path: str = "resultsv2", log = True):
        """
        Initialize the downloader.
        
        Args:
            download_path (str): Directory where files will be downloaded
        """
        self.download_path = download_path
        self.log = log
        self.logger = TextLogger(log = log)
        self.client = arxiv.Client()

    def _get_paper_id(self, identifier: str) -> str:
        """Extract paper ID from various forms of arXiv identifiers."""
        if identifier.startswith('http'):
            path = urlparse(identifier).path
            identifier = path.split('/')[-1]
            
        if identifier.startswith('arxiv:'):
            identifier = identifier[6:]
            
        return identifier.split('v')[0].strip()

    def _detect_file_type(self, file_path: str) -> str:
        """
        Detect the type of downloaded file using file signatures.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            str: Detected file type
        """
        with open(file_path, 'rb') as f:
            # Read first few bytes for signature checking
            signature = f.read(4)
            
            # Reset file pointer
            f.seek(0)
            
            # Check for gzip signature (1f 8b)
            if signature.startswith(b'\x1f\x8b'):
                return 'gzip'
                
            # Check for PDF signature (%PDF)
            if signature.startswith(b'%PDF'):
                return 'pdf'
                
            # Try to open as tar
            try:
                with tarfile.open(file_path, 'r:*') as _:
                    return 'tar'
            except:
                pass
                
            # Check if it might be a TeX file
            try:
                content = f.read().decode('utf-8')
                if '\\document' in content or '\\begin{document}' in content:
                    return 'tex'
            except:
                pass
                
            return 'unknown'

    def _handle_gzip(self, file_path: str, extract_path: str) -> bool:
        """Handle gzip compressed files."""
        try:
            # Try to extract as tar.gz first
            try:
                with tarfile.open(file_path, 'r:gz') as tar:
                    def is_safe_path(members):
                        for member in members:
                            if not os.path.abspath(os.path.join(extract_path, member.name)).startswith(
                                os.path.abspath(extract_path)
                            ):
                                self.logger.warning(f"Skipping potentially unsafe path: {member.name}")
                                continue
                            yield member
                    
                    tar.extractall(path=extract_path, members=is_safe_path(tar))
                return True
            except:
                # If not a tar.gz, try as simple gzip
                with gzip.open(file_path, 'rb') as f_in:
                    extracted_path = os.path.join(extract_path, 'source.tex')
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return True
                
        except Exception as e:
            self.logger.error(f"Error extracting gzip file: {e}")
            return False

    def _handle_tar(self, file_path: str, extract_path: str) -> bool:
        """Handle tar archives."""
        try:
            with tarfile.open(file_path, 'r:*') as tar:
                def is_safe_path(members):
                    for member in members:
                        if not os.path.abspath(os.path.join(extract_path, member.name)).startswith(
                            os.path.abspath(extract_path)
                        ):
                            self.logger.warning(f"Skipping potentially unsafe path: {member.name}")
                            continue
                        yield member
                
                tar.extractall(path=extract_path, members=is_safe_path(tar))
            return True
        except Exception as e:
            self.logger.error(f"Error extracting tar file: {e}")
            return False

    def _save_direct(self, file_path: str, extract_path: str, file_type: str) -> bool:
        """Handle direct files (PDF, TeX, etc.)."""
        try:
            new_path = os.path.join(extract_path, f'source.{file_type}')
            shutil.copy2(file_path, new_path)
            return True
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
            return False

    def _download_file(self, url: str, output_path: str) -> bool:
        """
        Download a file from URL to specified path.
        
        Args:
            url (str): URL to download from
            output_path (str): Path to save the file
            
        Returns:
            bool: True if download successful
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading file from {url}: {e}")
            return False

    def _download_source(self, url: str, paper_dir: str) -> Optional[str]:
        """Download source files from URL."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save the downloaded file
            temp_path = os.path.join(paper_dir, "source.download")
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return temp_path
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download source files: {e}")
            return None

    def _get_pdf_url(self, paper_id: str) -> Optional[str]:
        """Get PDF URL for a given paper ID."""
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))
            return paper.pdf_url
        except Exception as e:
            self.logger.error(f"Error getting PDF URL: {e}")
            return None

    def _process_source_file(self, file_path: str, extract_path: str) -> bool:
        """
        Process the downloaded source file based on its type.
        
        Args:
            file_path (str): Path to downloaded file
            extract_path (str): Directory to extract/save files
            
        Returns:
            bool: True if processing successful
        """
        try:
            file_type = self._detect_file_type(file_path)
            # self.logger.info(f"Detected file type: {file_type}")
            
            self._save_direct(file_path, extract_path, file_type)
            
            if file_type == 'gzip':
                return self._handle_gzip(file_path, extract_path)
            elif file_type == 'tar':
                return self._handle_tar(file_path, extract_path)
                
        except Exception as e:
            self.logger.error(f"Error processing source file: {e}")
            return False

    def download_paper(self, identifier: str, download_pdf: bool = True, download_source: bool = True) -> Tuple[bool, str]:
        """
        Download paper files (PDF and/or source files).
        
        Args:
            identifier (str): ArXiv paper identifier (ID, URL, or DOI)
            download_pdf (bool): Whether to download PDF
            download_source (bool): Whether to download source files
            
        Returns:
            Tuple[bool, str]: (Success status, Path to downloaded files)
        """
        paper_id = self._get_paper_id(identifier)
        # if verbose:
        #     self.logger.info(f"🔄 Processing paper ID: {paper_id} ...")
        
        paper_dir = self._create_download_dir(identifier)
        success = True
        
        if download_pdf:
            if os.path.exists(os.path.join(paper_dir, f"paper.pdf")):
                return True, paper_dir
            pdf_url = self._get_pdf_url(paper_id)
            if pdf_url:
                pdf_path = os.path.join(paper_dir, f"paper.pdf")
                if os.path.exists(pdf_path):
                    self.logger.show_info(f"📄 PDF already exists at {paper_dir}")
                    success = True
                else:
                    pdf_success = self._download_file(pdf_url, pdf_path)
                    if pdf_success:
                        self.logger.show_info(f"📄 PDF downloaded successfully to {paper_dir}")
                    else:
                        self.logger.show_warning("Failed to download PDF")
                        success = False
            else:
                self.logger.show_warning("PDF URL not found")
                success = False
        
        if download_source:
            source_url = self._get_source_url(paper_id)
            downloaded_file = self._download_source(source_url, paper_dir)
            
            if downloaded_file:
                source_success = self._process_source_file(downloaded_file, paper_dir)
                if os.path.exists(downloaded_file):
                    os.remove(downloaded_file)
                
                if not source_success:
                    self.logger.show_warning("Failed to process source files")
                    success = False
            else:
                self.logger.show_warning("Failed to download source files")
                success = False
        
        return success, paper_dir


    def _create_download_dir(self, paper_id: str) -> str:
        """Create and return the download directory path."""
        paper_dir = os.path.join(self.download_path, create_hash(paper_id))
        os.makedirs(paper_dir, exist_ok=True)
        return paper_dir

    def _get_source_url(self, paper_id: str) -> str:
        """Get the e-print URL for a given paper ID."""
        return f"https://arxiv.org/e-print/{paper_id}"

    def get_paper_metadata(self, identifier: str) -> Optional[dict]:
        """Get paper metadata from arXiv."""
        paper_id = self._get_paper_id(identifier)
        
        try:
            search = arxiv.Search(id_list=[paper_id])
            paper = next(self.client.results(search))
            
            return {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'published': paper.published.strftime('%Y-%m-%d'),
                'categories': paper.categories,
                'abstract': paper.summary
            }
            
        except StopIteration:
            self.logger.error(f"Paper not found: {paper_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching paper metadata: {e}")
            return None

class ArxivSearcher:
    """
    A utility class to search arXiv for papers based on keywords and criteria.
    """
    
    def __init__(self, max_results: int = 10):
        """
        Initialize the searcher with maximum number of results to return.
        
        Args:
            max_results (int): Maximum number of papers to return per search
        """
        self.max_results = max_results
        self.client = arxiv.Client()
    
    def _build_date_query(self, 
                         month: Optional[int] = None, 
                         year: Optional[int] = None) -> str:
        """
        Build date range query for arXiv search.
        
        Args:
            month (int, optional): Month to search for (1-12)
            year (int, optional): Year to search for
            
        Returns:
            str: Date range query string
        """
        if year is None:
            return ""
            
        # Validate month if provided
        if month is not None and not (1 <= month <= 12):
            raise ValueError("Month must be between 1 and 12")
            
        # Create start date
        if month is None:
            # If only year provided, search whole year
            start_date = date(year, 1, 1)
            end_date = date(year + 1, 1, 1)
        else:
            # If month and year provided, search specific month
            start_date = date(year, month, 1)
            # Move to first day of next month
            if month == 12:
                end_date = date(year + 1, 1, 1)
            else:
                end_date = date(year, month + 1, 1)
        
        # Format dates for arXiv query
        return f"submittedDate:[{start_date.strftime('%Y%m%d')}0000 TO {end_date.strftime('%Y%m%d')}0000]"
    
    def search(self, 
               keywords: List[str],
               categories: List[str] = None,
               month: Optional[int] = None,
               year: Optional[int] = None,
               sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
               sort_order: arxiv.SortOrder = arxiv.SortOrder.Descending) -> List[Dict]:
        """
        Search arXiv for papers matching the given keywords and criteria.
        
        Args:
            keywords (str): Search query string
            categories (List[str]): List of arXiv categories to search in (e.g., ['cs.AI', 'cs.LG'])
            month (int, optional): Month to search for (1-12)
            year (int, optional): Year to search for
            sort_by (arxiv.SortCriterion): How to sort results (Relevance, LastUpdatedDate, SubmittedDate)
            sort_order (arxiv.SortOrder): Order of sorting (Ascending or Descending)
            
        Returns:
            List[Dict]: List of papers with their details
        """
        # Construct search query
        search_parts = [' AND '.join(f'ti:{k}' for k in keywords)]
        
        # Add category filter if provided
        if categories:
            cat_query = ' OR '.join(f'cat:{cat}' for cat in categories)
            search_parts.append(f'({cat_query})')
                    
        # Add date filter if provided
        date_query = self._build_date_query(month, year)
        if date_query:
            search_parts.append(date_query)
            
        # Combine all parts with AND
        search_query = ' AND '.join(f'({part})' for part in search_parts if part)

        # Create search parameters
        search = arxiv.Search(
            query=search_query,
            max_results=self.max_results,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        # Execute search and format results
        results = []
        for result in self.client.results(search):
            paper = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'published': result.published.strftime('%Y-%m-%d'),
                'summary': result.summary,
                'pdf_url': result.pdf_url,
                'article_url': result.entry_id,
                'categories': result.categories
            }
            results.append(paper)
            
        return results
    
    def print_results(self, results: List[Dict]) -> None:
        """
        Print search results in a formatted way.
        
        Args:
            results (List[Dict]): List of paper results to print
        """
        if not results:
            print("No results found.")
            return
            
        for i, paper in enumerate(results, 1):
            print(f"\n{'-'*80}\n{i}. {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Published: {paper['published']}")
            print(f"Categories: {', '.join(paper['categories'])}")
            print(f"PDF: {paper['pdf_url']}")
            print(f"Article: {paper['article_url']}")
            print("\nAbstract:")
            print(paper['summary'])
