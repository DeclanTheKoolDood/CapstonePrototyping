
from typing import List, Literal, Optional, Dict, Generator, Tuple, Union, Any
from pydantic import BaseModel, Field
from mcp.server.fastmcp import Context
from crawl4ai import AsyncWebCrawler, CrawlResult
from googlesearch import search as google_search, SearchResult
from paperscraper.citations import get_citations_by_doi, get_citations_from_title
from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv, arxiv
from paperscraper.impact import Impactor
from paperscraper.load_dumps import QUERY_FN_DICT
from paperscraper.pdf import save_pdf
from paperscraper.plotting import plot_comparison, plot_venn_two, plot_venn_three
from paperscraper.postprocessing import aggregate_paper
from paperscraper.scholar import get_and_dump_scholar_papers
from paperscraper.utils import get_filename_from_query, load_jsonl
from fastmcp import FastMCP, Context

import json
import io
import os
import logging
import wikipedia
import asyncio
import fitz  # PyMuPDF
import aiohttp

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
	name="web_search_tools",
	dependencies=[],
	debug=True,
	log_level="DEBUG"
)

class DocumentProcessor:
	"""Process different document types and extract text content."""

	@staticmethod
	def extract_text_from_pdf(content: bytes) -> str:
		"""Extract text from a PDF file."""
		try:
			text : str = ""
			with fitz.Document(stream=io.BytesIO(content)) as doc:
				for page in doc:
					text += page.get_text()
			return text
		except Exception as e:
			return f"Error extracting text from PDF: {str(e)}"

class GoogleWebPage(BaseModel):
	"""Class to store information from Crawl4AI Crawler Result"""
	url : str
	pdf : Optional[bytes] = Field(None)
	media : Dict[str, List[Dict]] = Field(None)
	markdown : Optional[str] = Field(None)
	html : Optional[str] = Field(None)
	cleaned_html : Optional[str] = Field(None)

class WikipediaScraper:

	async def get_summary(query : str) -> Optional[str]:
		try:
			loop = asyncio.get_event_loop()
			f = loop.run_in_executor(None, wikipedia.summary, query)
			return await f
		except Exception as e:
			logging.error(e)
			return None

	async def search(query : str) -> Optional[List[str]]:
		try:
			loop = asyncio.get_event_loop()
			f = loop.run_in_executor(None, wikipedia.search, query)
			return await f
		except Exception as e:
			logging.error(e)
			return None

	async def get_page(query : str) -> Optional[wikipedia.WikipediaPage]:
		try:
			loop = asyncio.get_event_loop()
			f = loop.run_in_executor(None, wikipedia.page, query)
			return await f
		except Exception as e:
			logging.error(e)
			return None

class GoogleSearchScraper:

	async def search_query(query : str, num_results : int = 30) -> List[SearchResult]:
		def internal_search() -> List[SearchResult]:
			nonlocal query, num_results
			gen : Generator[Union[SearchResult, str], Any, None] = google_search(
				query, num_results=num_results, sleep_interval=2, unique=True, lang='en', safe=None, advanced=True
			)
			return [item for item in gen]
		loop = asyncio.get_event_loop()
		return await loop.run_in_executor(None, internal_search)

	async def download_google_pages(urls : List[str]) -> List[Optional[GoogleWebPage]]:
		pages : List[Optional[GoogleWebPage]] = []
		async with AsyncWebCrawler() as crawler:
			for url in urls:
				try:
					result : CrawlResult = await crawler.arun(url)
				except Exception as e:
					logger.warning(f"Failed to download page: {url} due to error: {e}")
					continue
				if result.status_code != 200:
					logger.warning(f"Failed to download page: {url} with status code: {result.status_code} {result.error_message}")
					continue
				page : GoogleWebPage = GoogleWebPage(
					url=result.url,
					pdf=result.pdf,
					media=result.media,
					markdown=result.markdown,
					html=result.html,
					cleaned_html=result.cleaned_html,
				)
				pages.append(page)
		return pages

class PaperScraperWrapper:
	"""
	A simple wrapper for the paperscraper package that makes it easy to search
	across all available sources for academic papers.

	This wrapper downloads and utilizes all available dumps (arXiv, bioRxiv,
	medRxiv, chemRxiv) and provides a unified interface to search them all.
	"""

	def __init__(self, dumps_directory: str = "server_dumps", download_dumps: bool = True):
		"""
		Initialize the PaperScraper wrapper.

		Args:
			dumps_directory: Directory to store downloaded dumps
			download_dumps: Whether to download all available dumps on initialization
		"""

		self.query_functions = QUERY_FN_DICT
		self.dumps_directory = dumps_directory

		# Create dumps directory if it doesn't exist
		os.makedirs(dumps_directory, exist_ok=True)

		# Download all dumps if requested
		if download_dumps:
			self.download_all_dumps()

	def download_all_dumps(self, max_retries: int = 10) -> None:
		"""
		Download all available dumps from preprint servers.

		Args:
			max_retries: Maximum number of retries for API connections
		"""
		print("Downloading dumps. This may take some time...")

		# Import all the dump downloaders

		# Download each dump with retries
		print("Downloading medRxiv dump (~35 MB, ~30 min)...")
		medrxiv(max_retries=max_retries)

		print("Downloading bioRxiv dump (~350 MB, ~1 hour)...")
		biorxiv(max_retries=max_retries)

		print("Downloading chemRxiv dump (~20 MB, ~45 min)...")
		chemrxiv(max_retries=max_retries)

		print("Downloading arXiv dump...")
		arxiv()

		print("All dumps downloaded successfully!")

	def download_dump_for_period(self,
								dump_type: str,
								start_date: str,
								end_date: Optional[str] = None,
								max_retries: int = 10) -> None:
		"""
		Download a specific dump for a given time period.

		Args:
			dump_type: Type of dump to download ('biorxiv', 'medrxiv', 'chemrxiv', 'arxiv')
			start_date: Start date in format 'YYYY-MM-DD'
			end_date: End date in format 'YYYY-MM-DD' (None for current date)
			max_retries: Maximum number of retries for API connections
		"""

		dump_functions = {
			'biorxiv': biorxiv,
			'medrxiv': medrxiv,
			'chemrxiv': chemrxiv,
			'arxiv': arxiv
		}

		if dump_type not in dump_functions:
			raise ValueError(f"Invalid dump type: {dump_type}. Valid types are: {list(dump_functions.keys())}")

		print(f"Downloading {dump_type} dump for period {start_date} to {end_date or 'current date'}...")

		if dump_type == 'arxiv':
			dump_functions[dump_type](start_date=start_date, end_date=end_date)
		else:
			dump_functions[dump_type](start_date=start_date, end_date=end_date, max_retries=max_retries)

		print(f"{dump_type} dump downloaded successfully!")

	def search_all_sources(self,
						query: List[List[str]],
						output_directory: str = "search_results",
						output_basename: Optional[str] = None) -> Dict[str, str]:
		"""
		Search all available sources with the given query.

		Args:
			query: Query in the format [[term1, synonym1, ...], [term2, synonym2, ...], ...]
				Each sublist represents terms connected by OR, and the sublists are connected by AND
			output_directory: Directory to save the search results
			output_basename: Base filename for output files (without extension)
							If None, will generate from query

		Returns:
			Dictionary mapping source names to output file paths
		"""
		os.makedirs(output_directory, exist_ok=True)

		if output_basename is None:

			output_basename = get_filename_from_query(query)

		results = {}

		# Search each source
		for source, query_fn in self.query_functions.items():
			output_file = os.path.join(output_directory, f"{source}_{output_basename}")

			try:
				print(f"Searching {source}...")
				query_fn(query, output_filepath=output_file)
				results[source] = output_file
				print(f"Search results for {source} saved to {output_file}")
			except Exception as e:
				print(f"Error searching {source}: {str(e)}")

		# Also search Google Scholar
		try:
			# For Scholar, we need to flatten the query
			flat_query = " ".join([" OR ".join(sublist) for sublist in query])
			scholar_output = os.path.join(output_directory, f"scholar_{output_basename}")

			print("Searching Google Scholar...")
			get_and_dump_scholar_papers(flat_query, output_filepath=scholar_output)
			results['scholar'] = scholar_output
			print(f"Search results for Google Scholar saved to {scholar_output}")
		except Exception as e:
			print(f"Error searching Google Scholar: {str(e)}")

		return results

	def search_source(self,
					source: str,
					query: List[List[str]],
					output_directory: str = "search_results",
					output_basename: Optional[str] = None) -> str:
		"""
		Search a specific source with the given query.

		Args:
			source: Source to search ('pubmed', 'arxiv', 'biorxiv', 'medrxiv', 'chemrxiv', 'scholar')
			query: Query in the format [[term1, synonym1, ...], [term2, synonym2, ...], ...]
				  Each sublist represents terms connected by OR, and the sublists are connected by AND
			output_directory: Directory to save the search results
			output_basename: Base filename for output files (without extension)
							 If None, will generate from query

		Returns:
			Path to the output file
		"""
		os.makedirs(output_directory, exist_ok=True)

		if output_basename is None:
			output_basename = get_filename_from_query(query)

		output_file = os.path.join(output_directory, f"{source}_{output_basename}")

		if source == 'scholar':


			# For Scholar, we need to flatten the query
			flat_query = " ".join([" OR ".join(sublist) for sublist in query])

			print("Searching Google Scholar...")
			get_and_dump_scholar_papers(flat_query, output_filepath=output_file)
		elif source in self.query_functions:
			print(f"Searching {source}...")
			self.query_functions[source](query, output_filepath=output_file)
		else:
			raise ValueError(f"Invalid source: {source}. Valid sources are: {list(self.query_functions.keys()) + ['scholar']}")

		print(f"Search results for {source} saved to {output_file}")
		return output_file

	def download_pdfs(
		self,
		result_file: str,
		output_directory: str = "pdfs",
		naming_key: str = "doi",
		api_keys_file: Optional[str] = None
	) -> List[str]:
		"""
		Download PDFs for papers in a search result file and return the exact file paths.

		Args:
			result_file: Path to the search result file (.jsonl)
			output_directory: Directory to save the PDFs
			naming_key: Key to use for naming the PDF files ('doi', 'title', etc.)
			api_keys_file: Path to a file containing API keys for Wiley/Elsevier

		Returns:
			List of paths to successfully downloaded PDFs
		"""
		os.makedirs(output_directory, exist_ok=True)

		# Load the papers from the search results
		papers = load_jsonl(result_file)
		print(f"Attempting to download {len(papers)} PDFs from {result_file} to {output_directory}...")

		# Track successfully downloaded PDFs
		pdf_paths = []

		# Download each paper individually to track success
		for i, paper in enumerate(papers):
			try:
				# Create a filename based on the naming key
				if naming_key in paper and paper[naming_key]:
					# Clean the key value to make it a valid filename
					key_value = str(paper[naming_key])
					key_value = "".join([c if c.isalnum() or c in ['-', '.'] else "_" for c in key_value])

					# If the key is too long, truncate it
					if len(key_value) > 100:
						key_value = key_value[:100]

					pdf_file = os.path.join(output_directory, f"{key_value}.pdf")

					# Attempt to download the PDF
					try:
						success = save_pdf(paper, filepath=pdf_file, api_keys=api_keys_file)
						if success and os.path.exists(pdf_file):
							pdf_paths.append(pdf_file)
							print(f"[{i+1}/{len(papers)}] Successfully downloaded: {pdf_file}")
						else:
							print(f"[{i+1}/{len(papers)}] Failed to download PDF for {key_value}")
					except Exception as e:
						print(f"[{i+1}/{len(papers)}] Error downloading PDF for {key_value}: {str(e)}")
				else:
					print(f"[{i+1}/{len(papers)}] Skipping paper without {naming_key}")
			except Exception as e:
				print(f"[{i+1}/{len(papers)}] Unexpected error: {str(e)}")

		print(f"Successfully downloaded {len(pdf_paths)} PDFs out of {len(papers)} papers")
		return pdf_paths

	def get_citations(self, papers: Union[str, List[Dict[str, Any]]], by_doi: bool = True) -> List[Dict[str, Any]]:
		"""
		Get citation counts for papers.

		Args:
			papers: Either a path to a .jsonl file or a list of paper dictionaries
			by_doi: Whether to use DOI (True) or title (False) for citation lookup

		Returns:
			List of papers with added citation information
		"""
		# Load papers if a file path is provided
		if isinstance(papers, str):
			papers = load_jsonl(papers)

		results = []
		for paper in papers:
			try:
				if by_doi and 'doi' in paper and paper['doi']:
					citations = get_citations_by_doi(paper['doi'])
					paper['citations'] = citations
				elif 'title' in paper and paper['title']:
					citations = get_citations_from_title(paper['title'])
					paper['citations'] = citations
			except Exception as e:
				print(f"Error getting citations for paper: {str(e)}")
				paper['citations'] = None

			results.append(paper)

		return results

	def get_impact_factors(self, papers: Union[str, List[Dict[str, Any]]], threshold: int = 85) -> List[Dict[str, Any]]:
		"""
		Add journal impact factors to papers.

		Args:
			papers: Either a path to a .jsonl file or a list of paper dictionaries
			threshold: Threshold for fuzzy matching journal names (0-100)

		Returns:
			List of papers with added impact factor information
		"""


		# Load papers if a file path is provided
		if isinstance(papers, str):
			papers = load_jsonl(papers)

		impactor = Impactor()
		results = []

		for paper in papers:
			try:
				if 'journal' in paper and paper['journal']:
					impact_results = impactor.search(paper['journal'], threshold=threshold)
					if impact_results:
						paper['impact_factor'] = impact_results[0]['factor']
						paper['impact_score'] = impact_results[0]['score']
			except Exception as e:
				print(f"Error getting impact factor for {paper.get('journal')}: {str(e)}")

			results.append(paper)

		return results

	def visualize_results(self,
						data_dict: Dict[str, Dict[str, List[int]]],
						data_keys: List[str],
						title_text: str,
						keyword_text: List[str],
						output_file: str) -> str:
		"""
		Create a visualization comparing search results.

		Args:
			data_dict: Data dictionary as returned by aggregate_paper
			data_keys: Keys to extract from data_dict
			title_text: Title for the plot
			keyword_text: Text labels for the different data_keys
			output_file: Output file path (without extension)

		Returns:
			Path to the created visualization
		"""

		plot_comparison(
			data_dict,
			data_keys,
			title_text=title_text,
			keyword_text=keyword_text,
			figpath=output_file
		)

		return f"{output_file}.png"

	def create_venn_diagram(self,
						sizes: Tuple,
						labels: List[str],
						title: str = "",
						output_file: str = "venn_diagram") -> str:
		"""
		Create a Venn diagram visualization.

		Args:
			sizes: Tuple of set sizes (for 2-way: (A, B, AB), for 3-way: (A, B, C, AB, AC, BC, ABC))
			labels: List of labels for the sets
			title: Title for the diagram
			output_file: Output file path (without extension)

		Returns:
			Path to the created visualization
		"""
		if len(labels) == 2:
			plot_venn_two(sizes, labels, title=title, figname=output_file)
		elif len(labels) == 3:
			plot_venn_three(sizes, labels, title=title, figname=output_file)
		else:
			raise ValueError("Only 2-way or 3-way Venn diagrams are supported")

		return f"{output_file}.png"

	def aggregate_results(self,
						result_file: str,
						start_year: int,
						bins_per_year: int = 1,
						filtering: bool = True,
						filter_keys: Optional[List[List[str]]] = None) -> Tuple[List[int], List[Dict]]:
		"""
		Aggregate papers by publication date into bins.

		Args:
			result_file: Path to the search result file (.jsonl)
			start_year: First year to consider
			bins_per_year: Number of bins per year
			filtering: Whether to filter papers that don't match the query
			filter_keys: Query terms to use for filtering

		Returns:
			Tuple of (aggregated_counts, filtered_papers)
		"""

		print(f"Aggregating results from {result_file}...")
		return aggregate_paper(
			result_file,
			start_year,
			bins_per_year=bins_per_year,
			filtering=filtering,
			filter_keys=filter_keys,
			return_filtered=True
		)

	def load_results(self, result_file: str) -> List[Dict[str, Any]]:
		"""
		Load search results from a file.

		Args:
			result_file: Path to the search result file (.jsonl)

		Returns:
			List of paper dictionaries
		"""
		return load_jsonl(result_file)

	def save_results(self, papers: List[Dict[str, Any]], output_file: str) -> None:
		"""
		Save papers to a JSONL file.

		Args:
			papers: List of paper dictionaries
			output_file: Output file path
		"""
		# Make sure the directory exists
		os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

		# Write the papers to the file
		with open(output_file, 'w', encoding='utf-8') as f:
			for paper in papers:
				f.write(json.dumps(paper) + '\n')

paper_scraper = PaperScraperWrapper(dumps_directory="papers_dir", download_dumps=False)

@mcp.tool(description="Download the given URL and return the page text contents. If the page is a PDF file, it will be the text of the PDF file, otherwise for webpages, it will be the Markdown or HTML.")
async def download_webpage(ctx : Context, urls : List[str]) -> Dict[str, Optional[str]]:
	"""Download a webpage and return its content."""
	# download the webpage
	results : Dict[str, Optional[str]] = {}
	async with AsyncWebCrawler() as crawler:
		for url in urls:
			result : CrawlResult = await crawler.arun(url)
			if result.status_code != 200:
				err_msg = f"Failed to download page: {url} with status code: {result.status_code} {result.error_message}"
				logger.warning(err_msg)
				ctx.error(err_msg)
				results[url] = None
				continue
			results[url] = result.markdown or result.cleaned_html or result.html or None
	return results

@mcp.tool(description="Search Google for any generic term. Returns the top 5 results. Returns a List of URLs, and a List of each page's content (PDF text content, webpage markdown or html as backup).")
async def search_google(ctx : Context, queries : List[str]) -> Tuple[List[str], List[str]]:
	# search google
	google_searched : List[str] = []
	google_results : List[GoogleWebPage] = []
	topk_google : int = 3
	for query in queries:
		try:
			results : List[SearchResult] = GoogleSearchScraper.search_query(query, num_results=topk_google)
		except Exception as e:
			logger.error(f"Failed to search google with query {query} due to error: {e}")
			continue
		results : List[Optional[SearchResult]] = [result for result in results if result not in google_searched]
		google_searched.extend([item.url for item in results if item is not None])
		results : List[Optional[GoogleWebPage]] = GoogleSearchScraper.download_google_pages([item.url for item in results if item is not None])
		results : List[GoogleWebPage] = [page for page in results if page is not None]
		google_results.extend(results)
	ctx.info(f"Found a total of {len(google_results)} google search results for query {queries}.")
	# extract page contents and return them
	urls : List[str] = []
	contents : List[str] = []
	for page in google_results:
		item_content : str = page.html
		if page.pdf:
			item_content = DocumentProcessor.extract_text_from_pdf(page.pdf)
		elif page.markdown:
			item_content = page.markdown
		elif page.cleaned_html:
			item_content = page.cleaned_html
		urls.append(page.url)
		contents.append(f"{page.url}\n\n{item_content}")
	return urls, contents

@mcp.tool(description="Search Wikipedia for individual words and nouns. Returns the top 5 results.")
async def search_wikipedia(ctx : Context, queries : List[str]) -> Tuple[List[str], List[str]]:
	# search wikipedia
	wikipedia_searched : List[str] = []
	wikipedia_results : List[wikipedia.WikipediaPage] = []
	topk_wikipedia : int = 5
	for query in queries:
		try:
			titles : List[str] = await WikipediaScraper.search(query, results=topk_wikipedia)
		except Exception as e:
			logger.error(f"Failed to search wikipedia with query {query} due to error: {e}")
			continue
		pages : List[Optional[wikipedia.WikipediaPage]] = [WikipediaScraper.get_page(page_name) for page_name in titles if page_name not in wikipedia_searched]
		wikipedia_searched.extend(titles)
		pages : List[wikipedia.WikipediaPage] = [page for page in pages if page is not None]
		wikipedia_results.extend(pages)
	ctx.info(f"Found a total of {len(wikipedia_results)} wikipedia search results for query {queries}.")
	# extract page contents and return them
	urls : List[str] = []
	contents : List[str] = []
	for page in wikipedia_results:
		item_content : str = page.content
		urls.append(page.url)
		contents.append(f"{page.url}\n\n{item_content}")
	return urls, contents

# Search Internet for Results (PubMed, Wikipedia, Google)
@mcp.tool(description="Search for research papers given a word or phrase. Returns the top 5 results.")
async def search_research_papers(ctx : Context, queries : List[str]) -> Tuple[List[str], List[str]]:
	"""Search for research papers given a word or phrase. Returns the top 5 results."""
	# search papers (list of pdf file filepaths)
	papers_results : List[str] = []
	topk_papers : int = 3
	for query in queries:
		query_formatted = [[query]]
		query_dir = "_".join(query.lower().split())
		output_dir = os.path.join("results", query_dir)
		ctx.info(f"Searching for: {query}")
		results = paper_scraper.search_all_sources(query_formatted, output_directory=output_dir)
		papers_added_for_query : int = 0
		for source, result_file in results.items():
			if papers_added_for_query >= topk_papers:
				break
			pdf_dir = os.path.join("pdfs", query_dir, source)
			ctx.info(f"Downloading PDFs for {query} from {source}...")
			pdf_paths = paper_scraper.download_pdfs(
				result_file, output_directory=pdf_dir, naming_key="doi"
			)
			papers_results.extend(pdf_paths)
	papers_results : List[str] = list(set(papers_results))
	ctx.info(f"Found a total of {len(papers_results)} research paper results for query {queries}.")
	# extract page contents and return them
	urls : List[str] = []
	contents : List[str] = []
	for pdf_path in papers_results:
		urls.append(pdf_path)
		try:
			with open(pdf_path, "rb") as f:
				pdf_content : str = DocumentProcessor.extract_text_from_pdf(f.read())
				contents.append(f"{pdf_path}\n\n{pdf_content}")
		except Exception as e:
			print(f"Error reading PDF file {pdf_path}: {e}")
			contents.append(f"Error reading PDF file {pdf_path}: {e}")
	return urls, contents

# # Search GitHub repositories (repository README, individual nested files, etc)
class GitHubRepository(BaseModel):
	full_name : str
	user : str
	repo_id : str
	description : Optional[str]
	stars : int
	url : str
	license : str

async def compile_github_repository(ctx : Context, repo_data : Dict) -> Optional[GitHubRepository]:
	try:
		license : str = "No License Found"
		if repo_data.get('license', None):
			license = repo_data['license']['name']
		repo = GitHubRepository(
			full_name=repo_data['full_name'],
			repo_id=repo_data['name'],
			user=repo_data['owner']['login'],
			description=repo_data.get('description', None),
			stars=repo_data.get('stargazers_count', 0),
			url=repo_data['url'],
			license=license
		)
		return repo
	except Exception as e:
		ctx.error(f"Failed to prepare GitHub repository data: {e}")
		return None

@mcp.tool(description="Search for GitHub repositories using a query.")
async def tool_search_github_repositories(ctx : Context, query : str, max_items : int = 10) -> List[GitHubRepository]:
	try:
		# Construct the API URL with query and limit
		api_url = f"https://api.github.com/search/repositories?q={query}&per_page=10"
		# Make the asynchronous request
		async with aiohttp.ClientSession() as session:
			async with session.get(api_url) as response:
				if response.status == 200:
					data = await response.json()
					items = data.get('items', [])[:max_items]
					ctx.info(f"Found a total of {len(items)} repositories for query {query}.")
					repos = [compile_github_repository(ctx, item) for item in items]
					return await asyncio.gather(*repos)
				else:
					raise Exception(f"Error {response.status}: {await response.text()}")
	except aiohttp.ClientError as e:
		ctx.error(f"Network error: {str(e)}")
		return []

# TODO: Download YouTube video
# TODO: Download video
# TODO: Download image
