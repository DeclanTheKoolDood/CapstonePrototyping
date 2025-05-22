import * as vscode from 'vscode';
import { getNonce } from '../utils';
import { ConversationService } from '../internal/conversationService';
import { LLMSelector, LLMService, OllamaService } from '../internal/llmService';

export class SidebarProvider implements vscode.WebviewViewProvider {
  _view?: vscode.WebviewView;
  _doc?: vscode.TextDocument;
  private mcpToolsCount: number = 0;

  constructor(private readonly _extensionUri: vscode.Uri) {}

  private async sendConversationMessage(data: any) {
    if (!data.value) return;
    const conversationService = ConversationService.getInstance();

    const conversationId: string = data.conversationId;
    const mode: "chat"|"agent" = data.mode || 'agent';

    let currentConversation = conversationService.getConversation(conversationId);
    if (currentConversation == undefined) {
      console.log("conversation does not exist! creating blank.");
      currentConversation = {mode: mode, messages: []};
      conversationService.setConversation(conversationId, currentConversation);
    };

    currentConversation.messages.push({
      mtype: "message", role: "user",
      text: data.value, isThink: false
    });

    let conversationMessages = currentConversation?.messages.filter(message => message.mtype == "message");
    let inputAIMessages = conversationService.castMessagesToChatMessages(conversationMessages);

    console.log("=== CHAT MESSAGES ===");
    console.log(inputAIMessages);
    console.log("=====================");

    const llmService = LLMSelector.getSelectedLLM();
    console.log(`Selected LLM Source: ${llmService.constructor.name}`);
    const tools = null;
    if (mode == "agent") {
      console.log("tools enabled! step-by-step as well.");
      // TODO
    }

    let total_usage_metrics = new Map<string, number>();
    const async_iter = await llmService.streamChat(inputAIMessages, null, tools, null, null);
    for await (const agent_step of async_iter) {
      console.log(JSON.stringify(agent_step));
      if (agent_step.finished) {
        break;
      }
      if (agent_step.invalid_tools.length > 0) {
        for (const tool of agent_step.invalid_tools) {
          const msg = `The tool ${tool.name || "unknown"} has errored!\n${tool.id || "unknown"}\n${tool.name || "unknown"}\n${tool.type || "unknown"}\n${tool.error || "unknown"}`;
          console.log(msg);
          currentConversation.messages.push({mtype: "tool", tool_id: tool.id || "unknown", text: msg})
          this._view?.webview.postMessage({
            type: 'receiveConversationTool',
            mtype: "tool",
            value: msg,
            conversationId: conversationId
          });
        }
      }
      if (agent_step.used_tools.length > 0) {
        for (const tool of agent_step.used_tools) {
          currentConversation.messages.push({mtype: "tool", tool_id: tool.tool_call_id, text: tool.text})
          this._view?.webview.postMessage({
            type: 'receiveConversationTool',
            mtype: "tool",
            value: `The tool ${tool.name} has successfully been called:\n${tool.text}`,
            conversationId: conversationId
          });
        }
      }
      if (agent_step.thoughts != null) {
        currentConversation.messages.push({
          mtype: "message",
          role: "assistant",
          text: agent_step.thoughts,
          isThink: true
        });
        this._view?.webview.postMessage({
          type: 'receiveConversationMessage',
          mtype: "message",
          value: "[THINK] " + agent_step.thoughts,
          role: "assistant",
          isThink: true,
          conversationId: conversationId
        });
      }
      if (agent_step.text != null) {
        currentConversation.messages.push({
          mtype: "message",
          role: "assistant",
          text: agent_step.text,
          isThink: false
        });
        this._view?.webview.postMessage({
          type: 'receiveConversationMessage',
          mtype: "message",
          value: agent_step.text,
          role: "assistant",
          conversationId: conversationId,
          isThink: false
        });
      }
      if (agent_step.usage_metrics) {
        for (const [index, value] of agent_step.usage_metrics.entries()) {
          const current = total_usage_metrics.get(index) || 0;
          total_usage_metrics.set(index, current + value);
        }
      }
    }
    console.log(JSON.stringify(total_usage_metrics));

    for (const [index, value] of total_usage_metrics.entries()) {
      this._view?.webview.postMessage({
        type: 'receiveConversationMessage',
        mtype: "message",
        value: `Total Usage Metric: ${index} = ${value}`,
        role: "assistant",
        conversationId: conversationId,
        isThink: false
      });
    }
  }

  public resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ) {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

    const conversationService = ConversationService.getInstance();

    webviewView.webview.onDidReceiveMessage(async (data) => {
      console.log(data);
      switch (data.type) {
        // CONVERSATION
        case 'sendConversationMessage': {
          await this.sendConversationMessage(data);
          break;
        }
        case 'saveConversation': {
          conversationService.setConversation(data.conversationId, data.conversation);
          vscode.window.showInformationMessage(`Conversation saved to ${data.conversationId}`);
          break;
        }
        case 'deleteConversation': {
          conversationService.deleteConversation(data.conversationId);
          vscode.window.showInformationMessage(`Conversation ${data.conversationId} deleted`);
          break;
        }
        case 'fetchAllConversations': {
          const conversations = conversationService.getAllConversations();
          this._view?.webview.postMessage({ type: 'fetchAllConversations', data: conversations });
          break;
        }
        case 'fetchConversation': {
          const conversation = conversationService.getConversation(data.conversationId);
          this._view?.webview.postMessage({ type: 'conversationData', data: conversation });
          break;
        }
        // MISCELLANEOUS
        case 'openConfig': {
          // Open extension settings
          vscode.commands.executeCommand('workbench.action.openSettings', 'waldoai');
          break;
        }
        case 'addAttachment': {
          // Handle attachment
          vscode.window.showOpenDialog({
            canSelectMany: false,
            openLabel: 'Select Attachment',
            filters: {
              'All Files': ['*']
            }
          }).then(fileUri => {
            if (fileUri && fileUri[0]) {
              vscode.window.showInformationMessage(`Selected file: ${fileUri[0].fsPath}`);
              // Here you would handle the attachment
            }
          });
          break;
        }
        case 'openMemories': {
          // Open global memories file
          vscode.commands.executeCommand('waldoai.openGlobalMemories');
          break;
        }
        case 'openRules': {
          // Open global rules file
          vscode.commands.executeCommand('waldoai.openGlobalRules');
          break;
        }
        case 'showMessage': {
          // Show message
          vscode.window.showInformationMessage(data.message);
          break;
        }
        case 'inputLabelAction': {
          // Handle input label button actions
          const action = data.action;
          switch (action) {
            case 'project':
              // Toggle project context inclusion
              // For now just show a message, but in a real implementation this would
              // toggle whether project context is included in AI requests
              vscode.window.showInformationMessage('Project context toggled');
              break;
            case 'file':
              // Toggle current file inclusion
              // For now just show a message, but in a real implementation this would
              // toggle whether the current file is included in AI requests
              vscode.window.showInformationMessage('Current file context toggled');
              break;
            case 'clipboard':
              // Toggle clipboard content inclusion
              vscode.env.clipboard.readText().then(text => {
                if (text) {
                  vscode.window.showInformationMessage(`Clipboard content ${text.length > 50 ? text.substring(0, 50) + '...' : text} will be included in context`);
                } else {
                  vscode.window.showInformationMessage('Clipboard is empty');
                }
              });
              break;
            case 'selection':
              // Toggle selection inclusion
              const editor = vscode.window.activeTextEditor;
              if (editor && !editor.selection.isEmpty) {
                const selectedText = editor.document.getText(editor.selection);
                vscode.window.showInformationMessage(`Selection (${selectedText.length} chars) will be included in context`);
              } else {
                vscode.window.showInformationMessage('No text is currently selected');
              }
              break;
            case 'mcpTools':
              // Toggle MCP Tools
              vscode.window.showInformationMessage('MCP Tools toggled');
              break;
            case 'webSearch':
              // Toggle Web Search
              vscode.window.showInformationMessage('Web Search toggled');
              break;
            case 'docSearch':
              // Toggle Document Search
              vscode.window.showInformationMessage('Document Search toggled');
              break;
          }
          break;
        }
      }
    });
  }

  /**
   * Update the MCP tools count displayed in the sidebar
   * @param count The number of available MCP tools
   */
  public updateMcpToolsCount(count: number): void {
    this.mcpToolsCount = count;

    if (this._view) {
      this._view.webview.postMessage({
        type: 'updateMcpToolsCount',
        count: count
      });
    }
  }

  /**
   * Update the input labels displayed above the input box
   * @param project The current project name
   * @param file The current file name
   * @param hasSelection Whether there is a text selection
   */
  public updateInputLabels(project: string, file: string, hasSelection: boolean): void {
    if (this._view) {
      this._view.webview.postMessage({
        type: 'updateInputLabels',
        data: {
          project,
          file,
          selection: hasSelection
        }
      });
    }
  }

  private _getHtmlForWebview(webview: vscode.Webview) {
    const styleResetUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, "media", "reset.css")
    );
    const styleVSCodeUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, "media", "vscode.css")
    );
    const styleMainUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, "media", "sidebar.css")
    );

    const nonce = getNonce();

    return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
      <link href="${styleResetUri}" rel="stylesheet">
      <link href="${styleVSCodeUri}" rel="stylesheet">
      <link href="${styleMainUri}" rel="stylesheet">
      <title>Waldo AI Chat</title>
    </head>
    <body>
      <div class="sidebar-container">
        <div class="agent-button-container">
          <button class="agent-button">
            <span class="agent-icon">🤖</span>
            <span>Waldo AI</span>
          </button>
          <button id="config-button" class="config-button" title="Settings">
            <span class="config-icon">⚙️</span>
          </button>
        </div>

        <div class="history-container">
          <select id="history-dropdown" class="history-dropdown" title="Select conversation history"></select>
          <div class="history-actions">
            <button id="new-conversation-button" class="history-button" title="New conversation">
              <span>+</span>
            </button>
            <button id="save-conversation-button" class="history-button" title="Save conversation">
              <span>💾</span>
            </button>
            <button id="delete-conversation-button" class="history-button" title="Delete conversation">
              <span>🗑️</span>
            </button>
          </div>
        </div>

        <div class="conversation-container" id="conversation">
          <!-- Conversation messages will be added here -->
          <div class="welcome-message">
            <h3>Welcome to Waldo AI</h3>
            <p>Your AI coding assistant. Ask me anything about your code!</p>
          </div>
        </div>

        <div class="input-container">
          <div class="input-labels">
            <div class="input-label-static" id="memories-label" title="Memories are always included. Click to open memories file.">
              <span class="input-label-icon">💭</span>
              <span class="input-label-text">Memories</span>
            </div>
            <div class="input-label-static" id="rules-label" title="Rules are always included. Click to open rules file.">
              <span class="input-label-icon">📝</span>
              <span class="input-label-text">Rules</span>
            </div>
            <button class="input-label-button input-label-active" id="project-button" title="Include project context">
              <span class="input-label-icon">📁</span>
              <span class="input-label-text">Project</span>
            </button>
            <button class="input-label-button input-label-active" id="file-button" title="Include current file">
              <span class="input-label-icon">📄</span>
              <span class="input-label-text">File</span>
            </button>
            <button class="input-label-button" id="clipboard-button" title="Include clipboard content">
              <span class="input-label-icon">📋</span>
              <span class="input-label-text">Clipboard</span>
            </button>
            <button class="input-label-button" id="selection-button" title="Include current selection">
              <span class="input-label-icon">🔍</span>
              <span class="input-label-text">Selection</span>
            </button>
            <button class="input-label-button" id="mcp-tools-button" title="Use MCP Tools">
              <span class="input-label-icon">🛠️</span>
              <span class="input-label-text">MCP Tools</span>
            </button>
            <button class="input-label-button" id="web-search-button" title="Enable web search">
              <span class="input-label-icon">🌐</span>
              <span class="input-label-text">Web Search</span>
            </button>
            <button class="input-label-button" id="doc-search-button" title="Enable document search">
              <span class="input-label-icon">📚</span>
              <span class="input-label-text">Doc Search</span>
            </button>
          </div>
          <textarea id="message-input" placeholder="Ask a question..." rows="3"></textarea>
          <button id="send-button" class="send-button">
            <span>Send</span>
            <span class="send-icon">↑</span>
          </button>
        </div>

        <div class="mode-container">
          <div class="mode-section">
            <div class="mode-buttons">
              <button id="agent-mode-button" class="mode-button mode-button-active">
                <span class="mode-icon">🤖</span>
                <span>Agent</span>
              </button>
              <button id="chat-mode-button" class="mode-button">
                <span class="mode-icon">💬</span>
                <span>Chat</span>
              </button>
            </div>
            <div class="mcp-tools-label">
              <span>Available MCP Tools: </span>
              <span id="mcp-tools-count">0</span>
            </div>
          </div>
          <button id="attachment-button" class="attachment-button" title="Add attachment">
            <span class="attachment-icon">📄</span>
          </button>
        </div>
      </div>

      <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        const conversationContainer = document.getElementById('conversation');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');
        const configButton = document.getElementById('config-button');
        const agentModeButton = document.getElementById('agent-mode-button');
        const chatModeButton = document.getElementById('chat-mode-button');
        const attachmentButton = document.getElementById('attachment-button');
        const mcpToolsCount = document.getElementById('mcp-tools-count');

        const historyDropdown = document.getElementById('history-dropdown');
        const newConversationButton = document.getElementById('new-conversation-button');
        const saveConversationButton = document.getElementById('save-conversation-button');
        const deleteConversationButton = document.getElementById('delete-conversation-button');

        const memoriesLabel = document.getElementById('memories-label');
        const rulesLabel = document.getElementById('rules-label');

        const projectButton = document.getElementById('project-button');
        const fileButton = document.getElementById('file-button');
        const clipboardButton = document.getElementById('clipboard-button');
        const selectionButton = document.getElementById('selection-button');
        const mcpToolsButton = document.getElementById('mcp-tools-button');
        const webSearchButton = document.getElementById('web-search-button');
        const docSearchButton = document.getElementById('doc-search-button');

        const projectLabel = projectButton.querySelector('.input-label-text');
        const fileLabel = fileButton.querySelector('.input-label-text');
        const selectionLabel = selectionButton.querySelector('.input-label-text');

        let storedConversations = {};
        let currentConversationId = null;
        let currentConversation = { messages :[], mode: "agent" };
        let currentMode = 'agent'; // 'agent' or 'chat'
        let availableMcpTools = 0;
        let awaitingMessage = false;

        function updateInputLabels(data) {
          if (data.project) {
            projectLabel.textContent = data.project;
          }

          if (data.file) {
            fileLabel.textContent = data.file;
          }

          if (data.selection !== undefined) {
            selectionLabel.textContent = data.selection ? 'Selection' : 'No Selection';
          }
        }

        function updateMcpToolsCount(count) {
          availableMcpTools = count;
          mcpToolsCount.textContent = count.toString();

          vscode.setState({
            conversation: currentConversation,
            conversationId: currentConversationId,
            mode: currentMode,
            mcpToolsCount: availableMcpTools
          });
        }

        function loadConversation(targetId) {
          let conversationData = storedConversations[targetId];
          if (conversationData == null) {
            const newConversationId = Date.now();
            vscode.setState({
              conversationId: newConversationId,
              conversation: {messages: [], mode: currentMode},
              mode: currentMode,
              mcpToolsCount: availableMcpTools
            });
            return;
          }
          currentConversationId = targetId;
          currentConversation = conversationData;
          currentMode = conversationData.mode;

          if (currentMode === 'agent') {
            agentModeButton.classList.add('mode-button-active');
            chatModeButton.classList.remove('mode-button-active');
            messageInput.placeholder = 'Ask a question...';
          } else {
            chatModeButton.classList.add('mode-button-active');
            agentModeButton.classList.remove('mode-button-active');
            messageInput.placeholder = 'Chat with Waldo...';
          }

          renderConversation();

          vscode.setState({
            conversationId: currentConversationId,
            conversation: currentConversation,
            mode: currentMode,
            mcpToolsCount: availableMcpTools
          });
        }

        function updateConversationOptions() {
          historyDropdown.innerHTML = '';

          Object.keys(storedConversations).forEach(function(conversationId) {
            const option = document.createElement('option');
            option.value = conversationId;
            option.textContent = 'Conversation ' + conversationId;
            historyDropdown.appendChild(option);
          });
        }

        updateConversationOptions();

        const previousState = vscode.getState();
        if (previousState) {
          if (previousState.conversation) {
            currentConversation = previousState.conversation;
            currentConversationId = previousState.conversationId;
            renderConversation();
          }

          if (previousState.conversationId) {
            historyDropdown.value = previousState.conversationId;
          }

          if (previousState.mode) {
            currentMode = previousState.mode;
            if (currentMode === 'agent') {
              agentModeButton.classList.add('mode-button-active');
              chatModeButton.classList.remove('mode-button-active');
              messageInput.placeholder = 'Ask a question...';
            } else {
              chatModeButton.classList.add('mode-button-active');
              agentModeButton.classList.remove('mode-button-active');
              messageInput.placeholder = 'Chat with Waldo...';
            }
          }

          if (previousState.mcpToolsCount !== undefined) {
            updateMcpToolsCount(previousState.mcpToolsCount);
          }
        }

        function sendMessage() {
          const message = messageInput.value.trim();
          if (message) {
            currentConversation.messages.push({
              mtype: "message",
              text: message,
              role: "user",
              isThink: false
            });

            messageInput.value = '';

            renderConversation();

            vscode.setState({
              conversationId: currentConversationId,
              conversation: currentConversation,
              mode: currentMode,
              mcpToolsCount: availableMcpTools
            });

            vscode.postMessage({
              type: 'sendConversationMessage',
              value: message,
              mode: currentMode,
              conversationId: currentConversationId
            });

            awaitingMessage = true;
          }
        }

        function loadAllConversations() {
          vscode.postMessage({type: 'fetchAllConversations'})
        }

        function renderConversation() {
          const welcomeMessage = conversationContainer.querySelector('.welcome-message');
          conversationContainer.innerHTML = '';

          if (currentConversation.messages.length === 0 && welcomeMessage) {
            conversationContainer.appendChild(welcomeMessage);
            return;
          }

          currentConversation.messages.forEach(message => {
            console.log(message);
            const messageElement = document.createElement('div');
            if (message.mtype == "message") {
                messageElement.className = (message.role == "user") ? 'message user-message' : 'message ai-message';

                const textElement = document.createElement('div');
                textElement.className = 'message-text';
                textElement.textContent = message.text;

                messageElement.appendChild(textElement);
                conversationContainer.appendChild(messageElement);
            } else if (message.mtype == "tool") {
              messageElement.className = 'message ai-tool';

              const textElement = document.createElement('div');
              textElement.className = 'message-text';
              textElement.textContent = "[TOOL] " + message.text;

              messageElement.appendChild(textElement);
              conversationContainer.appendChild(messageElement);
            } else {
              vscode.postMessage({ type: 'showMessage', message: "unsupported tool type:" + JSON.stringify(message) });
              console.log("unsupported tool type:" + JSON.stringify(message));
            }
          });

          conversationContainer.scrollTop = conversationContainer.scrollHeight;
        }

        sendButton.addEventListener('click', sendMessage);

        messageInput.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
          }
        });

        configButton.addEventListener('click', () => {
          vscode.postMessage({
            type: 'openConfig'
          });
        });

        historyDropdown.addEventListener('change', () => {
          loadConversation(historyDropdown.value);
        });

        function newConversation() {
          const newConversationId = Date.now();
          storedConversations[newConversationId] = { messages: [], mode: 'agent' };
          updateConversationOptions();
          historyDropdown.value = newConversationId;
          currentConversation = {messages: [], mode: currentMode};
          renderConversation();
          vscode.setState({
            conversationId: currentConversationId,
            conversation: currentConversation,
            mode: currentMode,
            mcpToolsCount: availableMcpTools
          });
          vscode.postMessage({ type: 'showMessage', message: 'New conversation created' });
        }

        newConversationButton.addEventListener('click', async function() {
          newConversation();
        });

        saveConversationButton.addEventListener('click', function() {
          if (currentConversation.messages.length > 0) {
            const conversationId = historyDropdown.value;
            storedConversations[conversationId] = {
              messages: [...currentConversation.messages],
              mode: currentMode
            };
            updateConversationOptions();
            vscode.postMessage({ type: 'saveConversation', conversationId: conversationId, conversation: currentConversation });
          } else {
            vscode.postMessage({ type: 'showMessage', message: 'No conversation to save...' });
          }
        });

        deleteConversationButton.addEventListener('click', function() {
          const conversationId = historyDropdown.value;
          delete storedConversations[conversationId];
          currentConversation = {messages: [], mode: currentMode};
          vscode.postMessage({ type: 'deleteConversation', conversationId: conversationId });
          loadAllConversations();
          if (Object.keys(storedConversations).length == 0) {
            newConversation();
          } else {
            const targetId = Object.keys(storedConversations)[0];
            loadConversation(targetId);
          }
          updateConversationOptions();
          renderConversation();
        });

        agentModeButton.addEventListener('click', () => {
          if (currentMode !== 'agent') {
            currentMode = 'agent';
            agentModeButton.classList.add('mode-button-active');
            chatModeButton.classList.remove('mode-button-active');
            messageInput.placeholder = 'Ask a question...';

            vscode.setState({
              conversationId: currentConversationId,
              conversation: currentConversation,
              mode: currentMode,
              mcpToolsCount: availableMcpTools
            });
          }
        });

        chatModeButton.addEventListener('click', () => {
          if (currentMode !== 'chat') {
            currentMode = 'chat';
            chatModeButton.classList.add('mode-button-active');
            agentModeButton.classList.remove('mode-button-active');
            messageInput.placeholder = 'Chat with Waldo...';

            vscode.setState({
              conversationId: currentConversationId,
              conversation: currentConversation,
              mode: currentMode,
              mcpToolsCount: availableMcpTools
            });
          }
        });

        attachmentButton.addEventListener('click', () => {
          vscode.postMessage({type: 'addAttachment'});
        });

        memoriesLabel.addEventListener('click', () => {
          vscode.postMessage({type: 'openMemories'});

          memoriesLabel.style.transform = 'scale(1.05)';
          setTimeout(() => {
            memoriesLabel.style.transform = '';
          }, 200);
        });

        rulesLabel.addEventListener('click', () => {
          vscode.postMessage({type: 'openRules'});

          rulesLabel.style.transform = 'scale(1.05)';
          setTimeout(() => {
            rulesLabel.style.transform = '';
          }, 200);
        });

        projectButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction',action: 'project'});
          toggleButtonActive(projectButton);
        });

        fileButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction', action: 'file'});
          toggleButtonActive(fileButton);
        });

        clipboardButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction', action: 'clipboard'});
          toggleButtonActive(clipboardButton);
        });

        selectionButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction', action: 'selection'});
          toggleButtonActive(selectionButton);
        });

        mcpToolsButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction', action: 'mcpTools'});
          toggleButtonActive(mcpToolsButton);
        });

        webSearchButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction', action: 'webSearch'});
          toggleButtonActive(webSearchButton);
        });

        docSearchButton.addEventListener('click', () => {
          vscode.postMessage({type: 'inputLabelAction', action: 'docSearch'});
          toggleButtonActive(docSearchButton);
        });

        function toggleButtonActive(button) {
          button.classList.toggle('input-label-active');

          button.style.transform = 'scale(1.05)';
          setTimeout(() => {
            button.style.transform = '';
          }, 200);
        }

        window.addEventListener('message', event => {
          const message = event.data;

          switch (message.type) {
            case 'receiveConversationMessage':
              awaitingMessage = false;
              let conversation = storedConversations[message.conversationId];
              conversation.messages.push({
                mtype: "message",
                text: message.value,
                role: message.role || "user",
                isThink: message.isThink || false
              });
              renderConversation();
              vscode.setState({
                conversationId: currentConversationId,
                conversation: currentConversation,
                mode: currentMode,
                mcpToolsCount: availableMcpTools
              });
              break;
            case 'receiveConversationTool':
              currentConversation.messages.push({
                mtype: "tool",
                tool_id: message.tool_id,
                text: message.value,
                conversationId: message.conversationId
              });
              renderConversation();
              vscode.setState({
                conversationId: currentConversationId,
                conversation: currentConversation,
                mode: currentMode,
                mcpToolsCount: availableMcpTools
              });
              break;
            case 'fetchAllConversations':
              storedConversations = message.data;
              if (!(currentConversationId in storedConversations)) {
                if (Object.keys(storedConversations).length == 0) {
                  newConversation();
                } else {
                  const targetId = Object.keys(storedConversations)[0];
                  loadConversation(targetId);
                }
              }
              updateConversationOptions();
              break;

            case 'clearChat':
              const newConversationId = Date.now();
              currentConversationId = newConversationId;
              currentConversation = { messages :[], mode: "agent" };
              renderConversation();
              vscode.setState({
                conversationId: currentConversationId,
                conversation: currentConversation,
                mode: currentMode,
                mcpToolsCount: availableMcpTools
              });
              break;

            case 'updateMcpToolsCount':
              updateMcpToolsCount(message.count);
              break;

            case 'updateInputLabels':
              updateInputLabels(message.data);
              break;
          }
        });

        loadAllConversations();
        console.log(storedConversations);
        if (Object.keys(storedConversations).length == 0) {
          newConversation();
        } else {
          const targetId = Object.keys(storedConversations)[0];
          loadConversation(targetId);
        }
      </script>
    </body>
    </html>`;
  }
}
