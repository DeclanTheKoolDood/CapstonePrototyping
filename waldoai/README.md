
# Waldo AI

Waldo AI is a co-pilot and AUGMENT-like Visual Studio Code extension built specifically to use Ollama, a local LLM provider for any device, to agentically code for the user.
This implementation was not fully implemented, but was incredibly close to being implemented, with the conversation history and LLM in-out pipeline being the remaining part.

<img src="https://github.com/DeclanTheKoolDood/CapstonePrototyping/waldoai/docs/demonstration.PNG" alt="Image of the WaldoAI Widget" width="400"/></img>

# Features

The Visual Studio Code extension provides the following features:
- **Extension Widget:** The extension that contains a Co-Pilot-esk widget that allows conversation with LLMs and the toggeling of many tools the LLM can utilize.
- **MCP Tool Support:** The extension supports custom MCP tools, both STDIO and SSE (web-based and TCP-based).
- **Conversation Histories:** The extension allows multiple conversation histories that let continue where you left off.
- **Multiple Built-In Tools:** The extension has multiple built-in tools, read and write files, list directories, run commands (with user permission).
- **Expandable LLM Providers:** The extension has easy capabilities to improve the selection of LLM providers available to the user, done by abstraction. At the moment it supports Ollama and OpenAI, but can be extended upon easily.

# Missing Features

- **Working built-in tools:** The tools are listed in a comment but are not actually implemented.
- **LLM pipeline not completed:** The LLM in/out pipeline is not fully implemented, it cannot do conversations properly at the moment.
- **Tool Running:** Tools are sent to the LLM but responses with tool calls are not being triggered.
- **Built-In Tools:** Many of the clickable buttons on the extension widget don't function yet

# Setup

To setup, you must have `npm` or related tools installed.

1. Open the extension folder in VSCode (so the root directory is the folder with package.json).
2. Run `npm install` in the root directory
3. Press F5 to open a debug window of VSCode that will load the extension for testing.
4. In the new window that opened, go to the WaldoAI widget.
5. Done

# Prompt Tips and Tricks

Prompting the agentic model to code this was a bit trickier.

The initial setup required the agentic model to run a few commands and required some user initial setup. It utilized "yo code" to setup a initial repository for the agentic coder, which provides questions as listed below:
```
# ? What type of extension do you want to create? New Extension (TypeScript)
# ? What's the name of your extension? waldoai
### Press <Enter> to choose default for all options below ###

# ? What's the identifier of your extension? waldoai
# ? What's the description of your extension? WaldoAI is a companion agentic coder.
# ? Initialize a git repository? N
# ? Which bundler to use? unbundled
# ? Which package manager to use? npm
```

One the project was setup, I started off by doing the backend of the extention rather than the frontend first.
The backend included importing Python code from my Deep Researcher agent to abstract away the LLM providers so it was easier to support multiple LLMs.
Next, I did most of the conversation history device, MCP tools service, and the local tools service.
Next, I got the agentic coder to make the actual sidebar with HTML, CSS and JavaScript.
This section took the most time, as it had trouble editing the code many times and at one point I reviewed and refactored most of the code myself so it was easier for the agentic coder to edit, and for myself to edit.

Prompting was crucial, but it still had trouble because of the weird environment for the Visual Studio Code extension.
