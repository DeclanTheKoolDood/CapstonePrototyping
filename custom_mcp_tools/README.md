
# CustomMCPTools

Custom MCP Tools

These tools allow different agentic AI platforms to utilize these tools to accomplish their tasks.
These tools were made using agentic AI coders and manual code work to show the capabilities of extension development that can be made to agentic frameworks with ease.

MCP is implemented using a Python library named FastMCP and includes the following tools:

# Features

#### a. **CONTEXT**
- Create and manage **memories**
- Create and manage **context**
- Create and manage **tasks**
- Create and manage **milestones**

#### b. **DEEP THINK**
- "Deep think" about implementing an algorithm
- "Deep think" about code generation
- "Fast think" about code generation
- "Deep think" about code analysis and improvement
- "Deep think" about code security analysis
- "Deep think" about system design
- "Deep think" about goal-to-start implementation

#### c. **DOCUMENTS**
- Query the document database and search for related text to the query
- Index a directory of documents and make them searchable

#### d. **MISC**
- Get the current time

#### e. **SPECIAL**
- Transcribe audio
- Transcribe video

#### f. **THINK**
- "Think" problem solving
- "Think" problem decomposition
- "Think" tool selection
- "Think" solution reflection
- "Think" API design
- "Think" trade-off thinking

#### g. **WEB SEARCH**
- Download webpage (given URL)
- Search Google
- Search Wikipedia
- Search research papers (various sources)
- Search GitHub repositories

#### **Uncompleted Tools**

Here is a list of uncompleted tools:
- create a UML diagram of the text and its components
- create a state transition diagram
- create a algorithm steps flow chart of the text
- create a timeline visualization
- create a dependency graph of the text and its components
- What-If Thinking - "what-if" scenarios for planning and decision-making.
- Interactive Shell / Command Prompt
- Sandboxed Code Environment (Jupyter Notebook?)
- Captioning Tools
- Web Browser Tools (headless, GUI)
- Application Controller Tools
- Image Generation Tools (comfyui workflows?)
- Video Generation Tools (comfyui workflows?)
- Structured Generation Tools (storyboards, math solver, logic/reasoning)
- Download YouTube Video
- Download Video
- Download Image

# Setup

1. Install requirements with `pip install -r requirements.txt`
2. Run MCP tools with `mcp_local.bat` or `py .\mcps\composite.py`
3. Connect MCP SSE server to tool SSE url.

# Prompt Tips and Tricks

Prompting the model to complete each tool was relatively easy.
I first created a list of the different tools that I wanted in the system, which is provided above, then I asked the agentic coder to create empty tool functions like
```
@mcp.tool(description="Think about how to implement an idea starting from the desired outcome and reversing to individual components.")
async def backward_implementation_thinking(ctx : Context, prompt : str) -> str:
	raise NotImplemented
```
to contain a basic format of each tool.

Then, for each individual tool, I created a comment and noted all the features of the tool and its step-by-step process like.
```
# TODO: create a UML diagram of the text and its components
# ITERATION 1:
	# - Identify the main components of the code
	# - Identify the relationships between the components
# ITERATION 2:
	# - Create a UML diagram of the code and its components
```
I did this by hand rather than with the tool. Finally, I got the tool to implement the actual tool, and if any mistakes were made or corrections were needed, I either did them by hand or with the agentic coder.
