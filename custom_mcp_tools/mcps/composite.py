
from fastmcp import FastMCP

import asyncio
import argparse

mcp = FastMCP(
	name="mcp_composite",
	debug=True,
	log_level="DEBUG"
)

async def get_server_details(server : FastMCP, depth : int = 1):
	"""Print information about mounted resources."""
	print(f"\n{' '*(depth-1)}Reading {server.name} details.")

	# Print available tools
	tools = server._tool_manager.list_tools()
	print(f"\n{'#'*depth} Available tools ({len(tools)}):")
	for tool in tools:
		print(f"{' '*depth}  - {tool.name}: {tool.description}")

	# Print available resources
	print(f"\n{'#'*depth} Available resources ({len(server._resource_manager._resources)}):")
	for uri in server._resource_manager._resources:
		print(f"{' '*depth} {uri}")

	print(f"\n{'#'*depth} Mounted subservers ({len(server._mounted_servers)}):")
	for server_name, subserver in server._mounted_servers.items():
		print(f"{' '*depth}- Found Mounted Server: {server_name}")

	for server_name, subserver in server._mounted_servers.items():
		await get_server_details(subserver.server, depth=depth+1)

async def main(args) -> None:
	if not args.disable_context:
		from context_mcp import mcp as context_mcp_mcp
		mcp.mount("context_", context_mcp_mcp)
	else:
		print("Disabled context server.")

	if not args.disable_deep_thinkers:
		from deep_thinkers import mcp as deep_thinkers_mcp
		mcp.mount("deepthinkers_", deep_thinkers_mcp)
	else:
		print("Disabled deep thinkers server.")

	if not args.disable_documents:
		from documents_mcp import mcp as documents_mcp_mcp, index_directory, set_rag_workflow, RAGWorkflow
		print("Indexing default directory:")
		set_rag_workflow(RAGWorkflow(timeout=120, num_concurrent_runs=6))
		await index_directory(None, args.documents_directory)
		mcp.mount("documents_", documents_mcp_mcp)
	else:
		print("Disabled documents server.")

	if not args.disable_miscellaneous:
		from misc_mcp import mcp as misc_mcp_mcp
		mcp.mount("misc_", misc_mcp_mcp)
	else:
		print("Disabled miscellaneous server.")

	if not args.disable_special:
		from special_mcp import mcp as special_mcp_mcp
		mcp.mount("special_", special_mcp_mcp)
	else:
		print("Disabled special server.")

	if not args.disable_thinkers:
		from thinkers_mcp import mcp as thinkers_mcp_mcp
		mcp.mount("thinkers_", thinkers_mcp_mcp)
	else:
		print("Disabled thinkers server.")

	if not args.disable_web_search:
		from web_search_mcp import mcp as web_search_mcp_mcp
		mcp.mount("web_search_", web_search_mcp_mcp)
	else:
		print("Disabled web search server.")

	await get_server_details(mcp)
	mcp.run(transport="sse")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--disable-context", action="store_true")
	parser.add_argument("--disable-deep-thinkers", action="store_true")
	parser.add_argument("--disable-documents", action="store_true")
	parser.add_argument("--disable-miscellaneous", action="store_true")
	parser.add_argument("--disable-special", action="store_true")
	parser.add_argument("--disable-thinkers", action="store_true")
	parser.add_argument("--disable-web-search", action="store_true")
	parser.add_argument("--documents-directory", default="./documents")

	args = parser.parse_args()

	asyncio.run(main(args))
