from typing import List, Optional, Dict
from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field
from threading import Lock

import os
import json
import uuid

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MEMORY_FILE = os.path.join(DATA_DIR, 'memory.json')
CONTEXT_FILE = os.path.join(DATA_DIR, 'context.json')
TASK_FILE = os.path.join(DATA_DIR, 'tasks.json')
MILESTONE_FILE = os.path.join(DATA_DIR, 'milestones.json')

# Thread-safe file access
file_locks = {MEMORY_FILE: Lock(), CONTEXT_FILE: Lock(), TASK_FILE: Lock(), MILESTONE_FILE: Lock(),}

def ensure_data_files():
	os.makedirs(DATA_DIR, exist_ok=True)
	for f in [MEMORY_FILE, CONTEXT_FILE, TASK_FILE, MILESTONE_FILE]:
		if not os.path.exists(f):
			with open(f, 'w') as fp:
				json.dump({}, fp)
ensure_data_files()

def read_json(file):
	with file_locks[file]:
		with open(file, 'r') as fp:
			data = json.load(fp)
			if not isinstance(data, dict):
				data = {}
			return data

def write_json(file, data):
	with file_locks[file]:
		with open(file, 'w') as fp:
			json.dump(data, fp, indent=2)

# Helper for conversation separation
def get_conversation_section(data, conversation_id):
	if conversation_id not in data:
		data[conversation_id] = {}
	return data[conversation_id]

mcp = FastMCP(
	name="context_tools",
	dependencies=[],
	debug=True,
	log_level="DEBUG"
)

# Contextual Memory
# - Save important information with tags
# - Retrieve information by tag or content
# - Link related memories
# - Sort memories by relevance
# - Forget obsolete information
class MemoryRecord(BaseModel):
	memory_id : str
	tags : List[str]
	content : str
	summary : Optional[str] = None
	links : List[str] = Field(default_factory=list)

@mcp.tool(description="Add a new memory record with tags and content.")
async def add_memory_record(ctx : Context, conversation_id : str, description : str, tags : List[str]) -> str:
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	memory_id = str(uuid.uuid4())
	record = MemoryRecord(memory_id=memory_id, tags=tags, content=description, summary=None, links=[])
	section[memory_id] = record.dict()
	data[conversation_id] = section
	write_json(MEMORY_FILE, data)
	return memory_id

@mcp.tool(description="Remove a memory record by ID.")
async def remove_memory_record(ctx : Context, conversation_id : str, memory_id : str) -> bool:
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	if memory_id in section:
		del section[memory_id]
		data[conversation_id] = section
		write_json(MEMORY_FILE, data)
		return True
	return False

@mcp.tool(description="Retrieve memories by IDs.")
async def select_memory_records_by_id(ctx : Context, conversation_id : str, memory_ids : List[str]) -> List[Optional[MemoryRecord]]:
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	return [section.get(mid) for mid in memory_ids]

@mcp.tool(description="Retrieve memories by tags.")
async def select_memory_records_by_tags(ctx : Context, conversation_id : str, tags : List[str], max_items : int) -> List[MemoryRecord]:
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	result = []
	for rec in section.values():
		if any(tag in rec['tags'] for tag in tags):
			result.append(rec)
		if len(result) >= max_items:
			break
	return result

@mcp.tool(description="Retrieve memories by content.")
async def select_memory_records_by_content(ctx : Context, conversation_id : str, content : List[str], distance_threshold : float) -> List[MemoryRecord]:
	# Simple substring match for demo; replace with embedding similarity for production
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	result = []
	for rec in section.values():
		if any(c.lower() in rec['content'].lower() for c in content):
			result.append(rec)
	return result

@mcp.tool(description="Link related memories.")
async def link_memory_records(ctx : Context, conversation_id : str, memory_ids : List[str]) -> bool:
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	for mid in memory_ids:
		if mid in section:
			links = set(section[mid].get('links', []))
			links.update([m for m in memory_ids if m != mid])
			section[mid]['links'] = list(links)
	data[conversation_id] = section
	write_json(MEMORY_FILE, data)
	return True

@mcp.tool(description="Sort memory IDs by relevance.")
async def sort_memory_ids_by_relevance(ctx : Context, conversation_id : str, content : str, memory_ids : List[str]) -> List[str]:
	data = read_json(MEMORY_FILE)
	section = get_conversation_section(data, conversation_id)
	def score(rec):
		return rec['content'].lower().count(content.lower())
	records = [section[mid] for mid in memory_ids if mid in section]
	records.sort(key=score, reverse=True)
	return [rec['memory_id'] for rec in records]

# Context Manager
# - Create context item
# - Bulk remove context items
# - Edit context item
# - List context items
# - Clear all context items
class ContextItem(BaseModel):
	context_id : str
	name : str
	content : str

@mcp.tool(description="Create a new context item.")
async def add_context_item(ctx : Context, context_id : str, name : str, content : str) -> str:
	data = read_json(CONTEXT_FILE)
	section = get_conversation_section(data, context_id)
	record_id = str(uuid.uuid4())
	item = ContextItem(context_id=record_id, name=name, content=content)
	section[record_id] = item.dict()
	data[context_id] = section
	write_json(CONTEXT_FILE, data)
	return record_id

@mcp.tool(description="Bulk remove context items.")
async def remove_context_items(ctx : Context, context_id : str, record_ids : List[str]) -> bool:
	data = read_json(CONTEXT_FILE)
	section = get_conversation_section(data, context_id)
	changed = False
	for rid in record_ids:
		if rid in section:
			del section[rid]
			changed = True
	if changed:
		data[context_id] = section
		write_json(CONTEXT_FILE, data)
	return changed

@mcp.tool(description="Edit an existing context item.")
async def edit_context_item(ctx : Context, context_id : str, record_id : str, name : str, content : str) -> bool:
	data = read_json(CONTEXT_FILE)
	section = get_conversation_section(data, context_id)
	if record_id in section:
		section[record_id]['name'] = name
		section[record_id]['content'] = content
		data[context_id] = section
		write_json(CONTEXT_FILE, data)
		return True
	return False

@mcp.tool(description="List all context items.")
async def list_context_items(ctx : Context, context_id : str) -> List[str]:
	data = read_json(CONTEXT_FILE)
	section = get_conversation_section(data, context_id)
	return list(section.keys())

@mcp.tool(description="Clear all context items.")
async def clear_all_context_items(ctx : Context, context_id : str) -> bool:
	data = read_json(CONTEXT_FILE)
	data[context_id] = {}
	write_json(CONTEXT_FILE, data)
	return True

# Self-Tasking
# - Create task with description and priority
# - List tasks
# - Mark task as complete
# - Mark task as priority
# - Mark task as normal priority
# - Edit task name and description
# - Clear all tasks
class TaskItem(BaseModel):
	task_id : str
	name : str
	content : str
	priority : str
	status : str

@mcp.tool(description="Create a new task.")
async def add_task(ctx : Context, conversation_id : str, task_id : str, name : str, content : str, priority : str) -> str:
	data = read_json(TASK_FILE)
	section = get_conversation_section(data, conversation_id)
	if not task_id:
		task_id = str(uuid.uuid4())
	item = TaskItem(task_id=task_id, name=name, content=content, priority=priority, status='pending')
	section[task_id] = item.dict()
	data[conversation_id] = section
	write_json(TASK_FILE, data)
	return task_id

@mcp.tool(description="List all tasks in priority order.")
async def list_tasks(ctx : Context, conversation_id : str) -> List[TaskItem]:
	data = read_json(TASK_FILE)
	section = get_conversation_section(data, conversation_id)
	# Priority: high > normal, Status: pending > complete
	def sort_key(x):
		p = 0 if x['priority'] == 'high' else 1
		s = 0 if x['status'] == 'pending' else 1
		return (p, s)
	return sorted([TaskItem(**v) for v in section.values()], key=sort_key)

@mcp.tool(description="Mark a task as complete.")
async def set_task_as_complete(ctx : Context, conversation_id : str, task_id : str) -> bool:
	data = read_json(TASK_FILE)
	section = get_conversation_section(data, conversation_id)
	if task_id in section:
		section[task_id]['status'] = 'complete'
		data[conversation_id] = section
		write_json(TASK_FILE, data)
		return True
	return False

@mcp.tool(description="Mark a task as priority.")
async def set_task_high_priority(ctx : Context, conversation_id : str, task_id : str) -> bool:
	data = read_json(TASK_FILE)
	section = get_conversation_section(data, conversation_id)
	if task_id in section:
		section[task_id]['priority'] = 'high'
		data[conversation_id] = section
		write_json(TASK_FILE, data)
		return True
	return False

@mcp.tool(description="Mark a task as normal priority.")
async def set_task_as_normal_priority(ctx : Context, conversation_id : str, task_id : str) -> bool:
	data = read_json(TASK_FILE)
	section = get_conversation_section(data, conversation_id)
	if task_id in section:
		section[task_id]['priority'] = 'normal'
		data[conversation_id] = section
		write_json(TASK_FILE, data)
		return True
	return False

@mcp.tool(description="Clear all tasks.")
async def clear_all_tasks(ctx : Context, conversation_id : str) -> bool:
	data = read_json(TASK_FILE)
	data[conversation_id] = {}
	write_json(TASK_FILE, data)
	return True

@mcp.tool(description="Edit an existing task.")
async def edit_task(ctx : Context, conversation_id : str, task_id : str, name : str, content : str) -> bool:
	data = read_json(TASK_FILE)
	section = get_conversation_section(data, conversation_id)
	if task_id in section:
		section[task_id]['name'] = name
		section[task_id]['content'] = content
		data[conversation_id] = section
		write_json(TASK_FILE, data)
		return True
	return False

# Milestone-Tasking
# - Create milestone with large description and priority
# - List milestones
# - Mark milestone as priority
# - Mark milestone as normal priority
# - Mark milestone as complete
# - Clear all milestones
class Milestone(BaseModel):
	milestone_id : str
	name : str
	description : str
	priority : str
	status : str

@mcp.tool(description="Create a new milestone.")
async def add_milestone(ctx : Context, conversation_id : str, milestone_id : str, name : str, description : str, priority : str) -> str:
	data = read_json(MILESTONE_FILE)
	section = get_conversation_section(data, conversation_id)
	if not milestone_id:
		milestone_id = str(uuid.uuid4())
	item = Milestone(milestone_id=milestone_id, description=description, priority=priority, status='pending')
	section[milestone_id] = item.dict()
	data[conversation_id] = section
	write_json(MILESTONE_FILE, data)
	return milestone_id

@mcp.tool(description="List all milestones in priority order.")
async def list_milestones(ctx : Context, conversation_id : str) -> list:
	data = read_json(MILESTONE_FILE)
	section = get_conversation_section(data, conversation_id)
	def sort_key(x):
		p = 0 if x['priority'] == 'high' else 1
		s = 0 if x['status'] == 'pending' else 1
		return (p, s)
	return sorted([Milestone(**v) for v in section.values()], key=sort_key)

@mcp.tool(description="Mark a milestone as priority.")
async def set_milestone_high_priority(ctx : Context, conversation_id : str, milestone_id : str) -> bool:
	data = read_json(MILESTONE_FILE)
	section = get_conversation_section(data, conversation_id)
	if milestone_id in section:
		section[milestone_id]['priority'] = 'high'
		data[conversation_id] = section
		write_json(MILESTONE_FILE, data)
		return True
	return False

@mcp.tool(description="Mark a milestone as normal priority.")
async def set_milestone_as_normal_priority(ctx : Context, conversation_id : str, milestone_id : str) -> bool:
	data = read_json(MILESTONE_FILE)
	section = get_conversation_section(data, conversation_id)
	if milestone_id in section:
		section[milestone_id]['priority'] = 'normal'
		data[conversation_id] = section
		write_json(MILESTONE_FILE, data)
		return True
	return False

@mcp.tool(description="Mark a milestone as complete.")
async def set_milestone_as_complete(ctx : Context, conversation_id : str, milestone_id : str) -> bool:
	data = read_json(MILESTONE_FILE)
	section = get_conversation_section(data, conversation_id)
	if milestone_id in section:
		section[milestone_id]['status'] = 'complete'
		data[conversation_id] = section
		write_json(MILESTONE_FILE, data)
		return True
	return False

@mcp.tool(description="Clear all milestones.")
async def clear_all_milestones(ctx : Context, conversation_id : str) -> bool:
	data = read_json(MILESTONE_FILE)
	data[conversation_id] = {}
	write_json(MILESTONE_FILE, data)
	return True

@mcp.tool(description="Edit an existing milestone.")
async def edit_milestone(ctx : Context, conversation_id : str, milestone_id : str, name : str, description : str) -> bool:
	data = read_json(MILESTONE_FILE)
	section = get_conversation_section(data, conversation_id)
	if milestone_id in section:
		section[milestone_id]['name'] = name
		section[milestone_id]['description'] = description
		data[conversation_id] = section
		write_json(MILESTONE_FILE, data)
		return True
	return False
