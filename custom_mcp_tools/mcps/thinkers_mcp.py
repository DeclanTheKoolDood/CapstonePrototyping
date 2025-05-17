
from typing import List
from fastmcp import FastMCP, Context

import os
import ollama

def split_by_think(text : str) -> str:
	"""Split the text by the <think> tag and return the last part."""
	parts = text.split("</think>")
	return parts[-1] if len(parts) > 1 else text

mcp = FastMCP(
	name="thinking_tools",
	dependencies=[],
	debug=True,
	log_level="DEBUG"
)

### General Cognition Thinking
# - What is being asked
# - Break into components
# - Answer each component
# - Compile answer and return
@mcp.tool(description="Break down a cognition prompt, answer each component, and compile the answer using deepseek-r1.")
async def general_cognition_think(ctx : Context, prompt: str) -> str:
	"""Use deepseek-r1 to break down and answer a general cognition prompt."""
	client = ollama.AsyncClient(host="http://localhost:11434")
	system_prompt = (
		"You are a world-class reasoning assistant.\n"
		"For any prompt, break down what is being asked, splitting it into components,"
		"then answer each component in detail and compile a cohesive answer."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return split_by_think(response.response)

### Problem Decomposition Thinking
# - Break complex problems into sub-problems
# - Identify independent components
# - Determine optimal solution order
# - Map dependencies between components
# - Create solution integration plan
# - Balance depth vs. breadth analysis
# - Maintain problem context during decomposition
@mcp.tool(description="Decompose a complex problem into sub-problems and create a solution plan using deepseek-r1.")
async def problem_decomposition_think(ctx : Context, problem: str) -> str:
	"""Use deepseek-r1 to decompose a complex problem and plan a solution."""
	client = ollama.AsyncClient(host="http://localhost:11434")
	system_prompt = (
		"You are a world-class problem decomposition assistant. "
		"For any problem, do the following:\n"
		"- Break complex problems into sub-problems.\n"
		"- Identify independent components.\n"
		"- Determine optimal solution order.\n"
		"- Map dependencies between components.\n"
		"- Create a solution integration plan.\n"
		"- Balance depth vs. breadth analysis.\n"
		"- Maintain problem context during decomposition.\n"
		"Present your reasoning and plan clearly."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=problem
	)
	return split_by_think(response.response)

### Tool Selection Thinking
# - Identify the best tools for a task
# - Evaluate tool capabilities and limitations
# - Consider task requirements and constraints
# - Assess tool compatibility and integration
# - Prioritize tools based on effectiveness and efficiency
# - Make informed decisions on tool selection
@mcp.tool(description="Select the best tools for a task.")
async def tool_selection_think(ctx : Context, task: str) -> str:
	"""Use deepseek-r1 to select the best tools for a task."""
	client = ollama.AsyncClient(host="http://localhost:11434")
	system_prompt = (
		"You are a world-class tool selection assistant. "
		"For any task, do the following:\n"
		"- Identify the best tools for the task.\n"
		"- Evaluate tool capabilities and limitations.\n"
		"- Consider task requirements and constraints.\n"
		"- Assess tool compatibility and integration.\n"
		"- Prioritize tools based on effectiveness and efficiency.\n"
		"- Make informed decisions on tool selection."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=task
	)
	return split_by_think(response.response)

### Reflection System Thinking
# - Review solution effectiveness
# - Evaluate tool selection accuracy
# - Assess thinking process quality
# - Identify improvement opportunities
# - Document lessons learned
# - Adjust future approach based on outcomes
# - Build meta-knowledge of tool effectiveness
@mcp.tool(description="Reflect on a solution and process, document lessons, and suggest improvements using deepseek-r1.")
async def solution_reflection_think(ctx : Context, solution: str) -> str:
	"""Use deepseek-r1 to review a solution, document lessons learned, and suggest improvements."""
	client = ollama.AsyncClient(host="http://localhost:11434")
	system_prompt = (
		"You are a world-class reflection assistant.\n"
		"Given a solution and its process, review its effectiveness,"
		"evaluate tool and process quality, document lessons learned,"
		"and suggest improvements and meta-knowledge for future"
		"approaches."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=solution
	)
	return response.response

### TODO: API Design Think
# - Identify use cases of the API
# - Define interface features
# - Design for extensibility
# - Design for bulk inputs and outputs
# - Balance simplicity and completeness
# - Determine appropriate abstractions
# - Document API behavior and examples
# - Validate design against requirements
# - Consider schemas for input and output data
@mcp.tool(description="Think about how to design and implement a target API idea.")
async def api_design_thinking(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	system_prompt = (
		"You are a world-class API design assistant.\n"
		"For any API, do the following:\n"
		"- Identify use cases of the API.\n"
		"- Define interface features.\n"
		"- Design for extensibility.\n"
		"- Design for bulk inputs and outputs.\n"
		"- Balance simplicity and completeness.\n"
		"- Determine appropriate abstractions.\n"
		"- Document API behavior and examples.\n"
		"- Validate design against requirements.\n"
		"- Consider schemas for input and output data."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### TODO: Trade-off Think
# - Identify competing concerns (time, space, simplicity, flexibility)
# - Evaluate short-term vs. long-term implications
# - Consider maintenance vs. performance
# - Assess development speed vs. code quality
# - Balance feature completeness vs. delivery timeline
# - Document trade-off decisions with rationale
# - Propose alternative approaches with pros/cons
@mcp.tool(description="Think about the tradeoffs of different implementations or provided ideas.")
async def trade_off_thinking(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	system_prompt = (
		"You are a world-class trade-off assistant.\n"
		"For any implementation or idea, do the following:\n"
		"- Identify competing concerns (time, space, simplicity, flexibility).\n"
		"- Evaluate short-term vs. long-term implications.\n"
		"- Consider maintenance vs. performance.\n"
		"- Assess development speed vs. code quality.\n"
		"- Balance feature completeness vs. delivery timeline.\n"
		"- Document trade-off decisions with rationale.\n"
		"- Propose alternative approaches with pros/cons."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response
