
# TODO - replace with langchain RAG workflow (more efficient and much smaller)

from typing import Any, Dict, List, Optional
from fastmcp import FastMCP

mcp = FastMCP(
	name="document_tools",
	dependencies=[],
	debug=True,
	log_level="DEBUG"
)

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.workflow import Context, Workflow, StartEvent, StopEvent, step
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from pydantic import BaseModel, Field

import unicodedata
import asyncio
import os
import chromadb

chroma_client = chromadb.Client(
	chromadb.Settings(persist_directory=".chroma_db", anonymized_telemetry=False)
)

PERSIST_DIR = ".documents_chroma"

RAG_WORKFLOW = None
RAG_INDEX = None

def set_rag_workflow(value):
	global RAG_WORKFLOW
	RAG_WORKFLOW = value

def set_rag_index(value):
	global RAG_INDEX
	RAG_INDEX = value

class DocumentResult(BaseModel):
	file_name : str= Field(description="The filename of the document that was matched.")
	file_path : str = Field(description="The filepath of the document that was matched.")
	embed_ids : List[str] = Field(default_factory=list, description="List of embedding ids used from this document.")
	text : str = Field(description="The content of which this document contributed towards. NOTE: SHARED WITH ADDITIONAL RETURNED DOCUMENTS.")

class RetrieverEvent(Event):
	"""Result of running retrieval"""
	nodes: list[NodeWithScore]

class RerankEvent(Event):
	"""Result of running reranking on retrieved nodes"""
	nodes: list[NodeWithScore]

def clean_text(text: str) -> str:
	# Normalize Unicode characters to remove or replace problematic ones
	text = unicodedata.normalize("NFKD", text)
	# Encode to UTF-8 and ignore errors, then decode back to string
	text = text.encode("utf-8", errors="ignore").decode("utf-8")
	return text

def get_file_metadata(x : str) -> dict:
	return {"chunk_size": 4096, "chunk_overlap": 256, "file_path": x}

def create_or_load_index(documents, embed_model):
	collection = chroma_client.get_or_create_collection(name="llm_documents")
	vector_store = ChromaVectorStore(chroma_collection=collection)
	storage_context = StorageContext.from_defaults(
		docstore=SimpleDocumentStore(batch_size=128),
		index_store=SimpleIndexStore(),
		vector_store=vector_store,
	)
	if os.path.exists(PERSIST_DIR):
		print("Loading index from disk...")
		index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embed_model, show_progress=True)
		return index
	else:
		print("Creating new index...")
		index = VectorStoreIndex.from_documents(documents=documents, storage_context=storage_context, embed_model=embed_model, show_progress=True)
		index.storage_context.persist(persist_dir=PERSIST_DIR)
		return index

class RAGWorkflow(Workflow):

	@step
	async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
		"""Entry point to ingest a document, triggered by a StartEvent with `dirname`."""
		dirname = ev.get("dirname")
		if not dirname:
			print("No directory provided.")
			return None
		# Configure SimpleDirectoryReader with encoding error handling
		reader = SimpleDirectoryReader(
			input_dir=dirname,
			recursive=True,
			encoding="utf-8",
			errors="ignore",
			file_metadata=get_file_metadata,
			exclude=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff"]  # Exclude images
		)
		documents = reader.load_data(show_progress=True, num_workers=16)
		# Preprocess documents to clean text
		for doc in documents:
			try:
				doc.set_content(clean_text(doc.get_content())) # Clean text
			except Exception as e:
				print(f"Error cleaning document {doc.metadata.get('file_path', 'unknown')}: {e}")
				doc.set_content("") # Set to empty
		# Initialize embedding model
		embed_model = OllamaEmbedding(model_name="all-minilm:l6-v2", embed_batch_size=256)
		# Create index
		index = create_or_load_index(documents=documents, embed_model=embed_model)
		return StopEvent(result=index)

	@step
	async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
		"""Entry point for RAG, triggered by a StartEvent with `query`."""
		query : str = ev.get("query")
		index : VectorStoreIndex = ev.get("index")
		if not query:
			return None
		print(f"Query the database with: {query}")
		await ctx.set("query", query)
		if index is None:
			print("Index is empty, load some documents before querying!")
			return None
		retriever = index.as_retriever(similarity_top_k=3)
		nodes = await retriever.aretrieve(query)
		print(f"Retrieved {len(nodes)} nodes.")
		return RetrieverEvent(nodes=nodes)

	@step
	async def rerank(self, ctx: Context, ev: RetrieverEvent) -> RerankEvent:
		# Rerank the nodes
		llm = Ollama(model="qwen3:4b")
		ranker = LLMRerank(choice_batch_size=10, top_n=5, llm=llm)
		print(await ctx.get("query", default=None), flush=True)
		new_nodes = await ranker.apostprocess_nodes(ev.nodes, query_str=await ctx.get("query", default=None))
		print(f"Reranked nodes to {len(new_nodes)}")
		return RerankEvent(nodes=new_nodes)

	@step
	async def synthesize(self, ctx: Context, ev: RerankEvent) -> StopEvent:
		"""Return a streaming response using reranked nodes."""
		llm = Ollama(model="qwen3:4b")
		summarizer = CompactAndRefine(llm=llm, streaming=True, verbose=True)
		query = await ctx.get("query", default=None)
		response = await summarizer.asynthesize(query, nodes=ev.nodes)
		return StopEvent(result=response)

@mcp.tool(description="Search a query in the document database. The query is keywords, phrases or sentences to search.")
async def tool_search_documents(ctx : Context, queries : List[str], TOP_K_SEARCH : int = 5) -> List[DocumentResult]:
	try:
		document_mapping : Dict[str, DocumentResult] = {}
		for query in queries:
			result : AsyncStreamingResponse = await RAG_WORKFLOW.run(index=RAG_INDEX, query=query)
			# check for metadata
			metadata : Optional[Dict[str, Dict[str, Any]]] = result.metadata
			if metadata is None:
				print("Document embedding returned no metadata! Had to skip as no reference could be provided.")
				continue
			# get the output text
			text : str = ""
			async for chunk in result.async_response_gen():
				text += chunk
			# record documents used
			for embed_id, meta in metadata.items():
				file_path = meta.get("file_path", None)
				file_name = meta.get("file_name", None)
				# create a document result if not already listed
				if file_path not in document_mapping:
					document_mapping[file_path] = DocumentResult(file_path=file_path, file_name=file_name, text=text)
				# add the embedding id to it
				document_mapping[file_path].embed_ids.append(embed_id)
		return list(document_mapping.values())[:TOP_K_SEARCH]
	except Exception as e:
		print(f"An exception has occured! {e}")
		return []

@mcp.tool(description="Index a given directory into the document database. Skips any registered files.")
async def index_directory(ctx : Context, directory_path: str) -> None:
	global RAG_INDEX
	RAG_INDEX = await RAG_WORKFLOW.run(dirname=directory_path)
