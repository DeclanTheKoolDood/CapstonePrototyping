

from typing import List
from fastmcp import FastMCP, Context

import os
import datetime

mcp = FastMCP(
	name="miscellaneous_tools",
	dependencies=[],
	debug=True,
	log_level="DEBUG"
)

@mcp.tool(description="Get the current UTC time.")
async def get_utc(ctx : Context) -> str:
	return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
