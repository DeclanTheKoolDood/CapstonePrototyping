

from typing import List
from fastmcp import FastMCP, Context

import os
import ollama

def split_by_think(text : str) -> str:
	"""Split the text by the <think> tag and return the last part."""
	parts = text.split("</think>")
	return parts[-1] if len(parts) > 1 else text

mcp = FastMCP(
	name="deep_thinker_tools",
	dependencies=[],
	debug=True,
	log_level="DEBUG"
)

### Algorithm Code Generation
# ITERATION 1:
	# - Identify the problem to be solved
	# - Define the input and output requirements
	# - Break down the problem into smaller sub-problems
# ITERATION 2:
	# - Design the algorithm step-by-step
	# - Skeleton code briefly describing the structure
# ITERATION 3:
	# - Critique output and ideas for improvement
	# - Pre-final skeleton code that has comments within functions that list the behavior of the code
# ITERATION 4:
	# - Final output generation
@mcp.tool(description="Implement a given algorithm by deep thinking about the problem. Specifically designed for implementing algorithms.")
async def algorithm_code_generation(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class algorithm code generation assistant.\n"
		"For any algorithm, do the following:\n"
		"- Identify the problem to be solved.\n"
		"- Define the input and output requirements.\n"
		"- Break down the problem into smaller sub-problems.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"You are a world-class algorithm code generation assistant.\n"
		"For any algorithm, do the following:\n"
		"- Design the algorithm step-by-step.\n"
		"- Skeleton code briefly describing the structure.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 3
	system_prompt = (
		"You are a world-class algorithm code generation assistant.\n"
		"For any algorithm, do the following:\n"
		"- Critique output and ideas for improvement.\n"
		"- Pre-final skeleton code that has comments within functions that list the behavior of the code"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 4
	system_prompt = (
		"You are a world-class algorithm code generation assistant.\n"
		"For any algorithm, do the following:\n"
		"- Final output generation.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### Deep Code Generation
# ITERATION 1:
	# - List all the essential features the code requires
	# - List all the features that are not essential but would be nice to have
	# - Break the code problem into components, outlines, and descriptions.
	# - Identify any edge cases or potential issues
# ITERATION 2:
	# - Skeleton code of what the solution looks like - do not implement individual functions.
	# - Critique output and what changes can be made.
# ITERATION 3:
	# - Final skeleton code of what it would look like - add comments of code behavior inside individual functions.
# ITERATION 4:
	# - Final output generation
@mcp.tool(description="Generate code by deep thinking about the problem and providing a iterative implementation. Specifically designed for large-scale code.")
async def deep_code_generation(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class deep code generation assistant.\n"
		"For any code, do the following:\n"
		"- List all the essential features the code requires.\n"
		"- List all the features that are not essential but would be nice to have.\n"
		"- Break the code problem into components, outlines, and descriptions.\n"
		"- Identify any edge cases or potential issues.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"You are a world-class deep code generation assistant.\n"
		"For any code, do the following:\n"
		"- Skeleton code of what the solution looks like - do not implement individual functions.\n"
		"- Critique output and what changes can be made.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 3
	system_prompt = (
		"You are a world-class deep code generation assistant.\n"
		"For any code, do the following:\n"
		"- Final skeleton code of what it would look like - add comments of code behavior inside individual functions but not actual code so pass the functions.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 4
	system_prompt = (
		"You are a world-class deep code generation assistant.\n"
		"For any code, do the following:\n"
		"- Final output generation.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### Fast Code Generation
# ITERATION 1
	# - List all the essential features the code requires
	# - List all the features that are not essential but would be nice to have
	# - Describe the code problem with components, outlines, and descriptions.
# ITERATION 2
	# - Final output generation
@mcp.tool(description="Generate code by fast thinking about the problem and providing a iterative implementation. Specifically designed for small-scale code.")
async def fast_code_generation(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class fast code generation assistant.\n"
		"For any code, do the following:\n"
		"- List all the essential features the code requires.\n"
		"- List all the features that are not essential but would be nice to have.\n"
		"- Describe the code problem with components, outlines, and descriptions.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"You are a world-class fast code generation assistant.\n"
		"For any code, do the following:\n"
		"- Final output generation.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### Code Analysis and Improvement Thinking
# ITERATION 1
	# - For each function, identify the purpose and functionality
	# - Structurally think of how the code communicates between each other
	# - Link any components together
	# - Identify any edge cases or potential issues
# ITERATION 2
	# - Identify any potential issues or bugs in the code
	# - Identify any potential performance issues
	# - Suggest improvements or optimizations
# ITERATION 3
	# - Provide a summary of the code's overall structure and functionality
	# - Provide a summary of the code's overall improvements and optimizations.
@mcp.tool(description="Analyze the code and provide suggestions for improvements and optimizations. Supports small-scale and large-scale code.")
async def code_analysis_and_improvement_thinking(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class code analysis and improvement assistant.\n"
		"For any code, do the following:\n"
		"- For each function, identify the purpose and functionality.\n"
		"- Structurally think of how the code communicates between each other.\n"
		"- Link any components together.\n"
		"- Identify any edge cases or potential issues.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"You are a world-class code analysis and improvement assistant.\n"
		"For any code, do the following:\n"
		"- Identify any potential issues or bugs in the code.\n"
		"- Identify any potential performance issues.\n"
		"- Suggest improvements or optimizations.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 3
	system_prompt = (
		"You are a world-class code analysis and improvement assistant.\n"
		"For any code, do the following:\n"
		"- Provide a summary of the code's overall structure and functionality.\n"
		"- Provide a summary of the code's overall improvements and optimizations.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### Security Analysis Thinking
# ITERATION 1:
	# - Structurally think of what operations are to happen on a high level
	# - Think of vulnerabilities in the code
	# - Think of methods to break the code
# ITERATION 2:
	# - Compile a list of potential security holes and return
	# - Categorize vulnerabilities by severity (Critical, High, Medium, Low)
	# - Provide remediation suggestions for each vulnerability
@mcp.tool(description="Analyze the code and provide information of security related to the code. Supports small-scale and large-scale code.")
async def security_analysis_thinking(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class security analysis assistant.\n"
		"For any code, do the following:\n"
		"- Structurally think of what operations are to happen on a high level.\n"
		"- Think of vulnerabilities in the code.\n"
		"- Think of methods to break the code.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"You are a world-class security analysis assistant.\n"
		"For any code, do the following:\n"
		"- Compile a list of potential security holes and return.\n"
		"- Categorize vulnerabilities by severity (Critical, High, Medium, Low).\n"
		"- Provide remediation suggestions for each vulnerability.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### System Design Thinking
# ITERATION 1:
	# - Identify system requirements
	# - Design component relationships
	# - List data flow between components
	# - Assess scalability requirements
# ITERATION 2:
	# - Consider architectural patterns
	# - Evaluate technology stack options
	# - Identify integration points
	# - Critique design for weaknesses
	# - Document system architecture
# ITERATION 3:
	# - Refine based on critique
	# - Provide additional ideas for improvement
	# - Provide a summary of the system design and architecture
@mcp.tool(description="Think about how to design and implement a target system idea. Supports small-scale and large-scale ideas.")
async def system_design_thinking(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class system design assistant.\n"
		"For any system, do the following:\n"
		"- Identify system requirements.\n"
		"- Design component relationships.\n"
		"- List data flow between components.\n"
		"- Assess scalability requirements.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"You are a world-class system design assistant.\n"
		"For any system, do the following:\n"
		"- Consider architectural patterns.\n"
		"- Evaluate technology stack options.\n"
		"- Identify integration points.\n"
		"- Critique design for weaknesses.\n"
		"- Document system architecture.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 3
	system_prompt = (
		"You are a world-class system design assistant.\n"
		"For any system, do the following:\n"
		"- Refine based on critique.\n"
		"- Provide additional ideas for improvement.\n"
		"- Provide a summary of the system design and architecture.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

### Backward Implementation Think
# ITERATION 1:
	# - Start with desired outcome
	# - Work backward to identify prerequisites
	# - Determine intermediate checkpoints
	# - Identify potential blockers and issues
	# - Map dependencies between steps
# ITERATION 2:
	# - Create reverse chronological plan
	# - Create a summary of forward implementation steps
	# - Provide a summary of the backward implementation steps.
@mcp.tool(description="Think about how to implement an idea starting from the desired outcome and reversing to individual components.")
async def backward_implementation_thinking(ctx : Context, prompt : str) -> str:
	client = ollama.AsyncClient(host="http://localhost:11434")
	# iteration 1
	system_prompt = (
		"You are a world-class backward implementation assistant.\n"
		"For any idea, do the following:\n"
		"- Start with desired outcome.\n"
		"- Work backward to identify prerequisites.\n"
		"- Determine intermediate checkpoints.\n"
		"- Identify potential blockers and issues.\n"
		"- Map dependencies between steps.\n"
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	prompt = response.response
	# iteration 2
	system_prompt = (
		"- Create reverse chronological plan.\n"
		"- Create a summary of forward implementation steps.\n"
		"- Provide a summary of the backward implementation steps."
	)
	response = await client.generate(
		model="qwen3:14b",
		system=system_prompt,
		prompt=prompt
	)
	return response.response

# TODO: create a UML diagram of the text and its components
# ITERATION 1:
	# - Identify the main components of the code
	# - Identify the relationships between the components
# ITERATION 2:
	# - Create a UML diagram of the code and its components
# @mcp.tool(description="Create a UML diagram of the code and its components.")
# async def create_code_UML_diagram(ctx : Context, prompt : str) -> str:
# 	raise NotImplementedError

# TODO: create a state transition diagram
# ITERATION 1:
	# - Identify the main components of the code
	# - Identify the relationships between the components
	# - Identify the states of the components
	# - Identify the transitions between the states
# ITERATION 2:
	# - Create a state transition diagram of the code and its components
# @mcp.tool(description="Create a state transition diagram of the code and its components.")
# async def create_code_state_transition_diagram(ctx : Context, prompt : str) -> str:
# 	raise NotImplementedError

# TODO: create a algorithm steps flow chart of the text
# ITERATION 1:
	# - Identify the main components of the code
	# - Identify the relationships between the components
	# - Identify the components that are to be analyzed
	# - Create a list of algorithm steps for each component
# ITERATION 2:
	# - Create a algorithm steps flow chart
# @mcp.tool(description="Create a algorithm steps flow chart of the code.")
# async def create_algorithm_step_by_step_diagram(ctx : Context, prompt : str) -> str:
# 	raise NotImplementedError

# TODO: create a timeline visualization
# ITERATION 1:
	# - Identify the main components of the code
	# - Identify the relationships between the components
	# - Identify the components that are to be analyzed
	# - Create a list of high-level operations for each component, for each step of the way.
# ITERATION 2:
	# - Create a timeline visualization of the code and its components
# @mcp.tool(description="Create a timeline visualization of the code and its components.")
# async def create_code_timeline_visualization(ctx : Context, prompt : str) -> str:
# 	raise NotImplementedError

# TODO: create a dependency graph of the text and its components
# ITERATION 1:
	# - Identify the main components of the code
	# - Identify the relationships between the components
	# - Identify the components that are to be analyzed
# ITERATION 2:
	# - Create a dependency graph of the code and its components
# @mcp.tool(description="Create a dependency graph of the code and its components.")
# async def create_code_dependency_graph_diagram(ctx : Context, prompt : str) -> str:
# 	raise NotImplementedError

# TODO: What-If Thinking - "what-if" scenarios for planning and decision-making.
# ITERATION 1:
	# - Identify important variables and parameters that can be altered
	# - Determine the range of values for each variable or parameter
# ITERATION 2:
	# - Explore alternative outcomes based on different inputs or decisions - short summaries
	# - Theorize potential consequences of changes - short summaries
	# - Assess risks and benefits of each scenario
