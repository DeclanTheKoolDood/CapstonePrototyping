// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import { SidebarProvider } from './sidebar/sidebarProvider';
import { ConfigService } from './config/configService';
import { ConfigFilesManager } from './config/configFiles';
import { McpService } from './internal/mcpService';
import { LLMService, OllamaService, OpenAIService } from './internal/llmService';
import { ToolsService } from './internal/toolsService';
import { ConversationService } from './internal/conversationService';

// Helper function to update input labels based on the current context
function updateInputLabelsFromContext(sidebarProvider: SidebarProvider): void {
	// Get the current workspace folder name
	const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
	const projectName = workspaceFolder ? workspaceFolder.name : 'No Project';

	// Get the current file name
	const activeEditor = vscode.window.activeTextEditor;
	const fileName = activeEditor ? activeEditor.document.fileName.split(/[\\/]/).pop() || 'No File' : 'No File';

	// Check if there's a selection
	const hasSelection = activeEditor ? !activeEditor.selection.isEmpty : false;

	// Update the input labels
	sidebarProvider.updateInputLabels(projectName, fileName, hasSelection);
}

export function activate(context: vscode.ExtensionContext) {
	console.log('Extension "waldoai" is now active!');

	// Initialize the config service
	ConfigService.getInstance();
	McpService.getInstance();
	ToolsService.getInstance();
	LLMService.getInstance();
	OllamaService.getInstance();
	OpenAIService.getInstance();
	ConversationService.getInstance();

	// Initialize the config files manager
	const configFilesManager = ConfigFilesManager.getInstance(context);

	// Register the sidebar provider
	const sidebarProvider = new SidebarProvider(context.extensionUri);

	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider("waldoai.sidebar", sidebarProvider)
	);

	// Initial update of MCP tools count
	(async () => {
		let tools = await McpService.getInstance().getRegisteredMCPTools();
		let count = tools.length;
		sidebarProvider.updateMcpToolsCount(count);
	})();

	// Initial update of input labels
	updateInputLabelsFromContext(sidebarProvider);

	// Example: Update MCP tools count when extension settings change
	context.subscriptions.push(
		vscode.workspace.onDidChangeConfiguration(e => {
			if (e.affectsConfiguration('waldoai.mcpServersSTDIO') || e.affectsConfiguration('waldoai.mcpServersSSE')) {
				(async () => {
					let tools = await McpService.getInstance().getRegisteredMCPTools();
					let count = tools.length;
					sidebarProvider.updateMcpToolsCount(count);
				})();
			}
		})
	);

	// Update input labels when the active editor changes
	context.subscriptions.push(
		vscode.window.onDidChangeActiveTextEditor(() => {
			updateInputLabelsFromContext(sidebarProvider);
		})
	);

	// Update input labels when the selection changes
	context.subscriptions.push(
		vscode.window.onDidChangeTextEditorSelection(() => {
			updateInputLabelsFromContext(sidebarProvider);
		})
	);

	// Register the start chat command
	const startChatDisposable = vscode.commands.registerCommand('waldoai.startChat', () => {
		vscode.commands.executeCommand('waldoai.sidebar.focus');
	});

	// Register the clear chat command
	const clearChatDisposable = vscode.commands.registerCommand('waldoai.clearChat', () => {
		sidebarProvider._view?.webview.postMessage({ type: 'clearChat' });
	});

	// Register the open settings command
	const openSettingsDisposable = vscode.commands.registerCommand('waldoai.openSettings', () => {
		vscode.commands.executeCommand('workbench.action.openSettings', 'waldoai');
	});

	// Register commands for opening configuration files
	const openGlobalRulesDisposable = vscode.commands.registerCommand('waldoai.openGlobalRules', () => {
		configFilesManager.openGlobalRules();
	});

	const openLocalRulesDisposable = vscode.commands.registerCommand('waldoai.openLocalRules', () => {
		configFilesManager.openLocalRules();
	});

	const openGlobalMemoriesDisposable = vscode.commands.registerCommand('waldoai.openGlobalMemories', () => {
		configFilesManager.openGlobalMemories();
	});

	const openLocalMemoriesDisposable = vscode.commands.registerCommand('waldoai.openLocalMemories', () => {
		configFilesManager.openLocalMemories();
	});

	const keybindTestDisposable = vscode.commands.registerCommand('waldoai.keybindTestOllama', () => {
		OllamaService.getInstance().keybindTest();
	});

	const keybindTestStreamDisposable = vscode.commands.registerCommand('waldoai.keybindTestOllamaStream', () => {
		OllamaService.getInstance().keybindTestStream();
	});

	const keybindTestToolsDisposable = vscode.commands.registerCommand('waldoai.keybindTestTools', () => {
		ToolsService.getInstance().keybindGetToolsTest();
	});

	const keybindTestMCPDisposable = vscode.commands.registerCommand('waldoai.keybindTestMCP', () => {
		McpService.getInstance().keybindGetToolsTest();
	});

	context.subscriptions.push(
		startChatDisposable,
		clearChatDisposable,
		openSettingsDisposable,
		openGlobalRulesDisposable,
		openLocalRulesDisposable,
		openGlobalMemoriesDisposable,
		openLocalMemoriesDisposable,
		keybindTestDisposable,
		keybindTestStreamDisposable,
		keybindTestToolsDisposable,
		keybindTestMCPDisposable
	);
}

// This method is called when your extension is deactivated
export function deactivate() {}
