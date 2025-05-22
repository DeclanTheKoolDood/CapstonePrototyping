import {DynamicTool, DynamicStructuredTool, tool} from "@langchain/core/tools";
import {MultiServerMCPClient} from "@langchain/mcp-adapters";
import * as vscode from "vscode";

// https://github.com/langchain-ai/langchainjs/tree/main/libs/langchain-mcp-adapters#readme

export interface StdioServerConfig {
	enabled: boolean;
	name: string;
	command: string;
	args: Array<string>;
	env: Map<string, any>;
}

export interface SseServerConfig {
	enabled: boolean;
	name: string;
	url: string;
	apiKey: string;
	headers: Map<string, any>;
	cookies: Map<string, any>;
}

export type MCPServerConfig = StdioServerConfig | SseServerConfig;

export class McpService {
	private static instance: McpService;
	private multi_mcp_client: MultiServerMCPClient | null;

	private constructor() {
		this.multi_mcp_client = null;
	}

	public static getInstance(): McpService {
		if (!McpService.instance) {
			McpService.instance = new McpService();
		}
		return McpService.instance;
	}

	public getSSEMCPServers(): SseServerConfig[] {
		const config = vscode.workspace.getConfiguration("waldoai");
		const serversJson = config.get<string>("mcpServersSSE", "[]");

		try {
			return JSON.parse(serversJson) as SseServerConfig[];
		} catch (error) {
			console.error("Failed to parse MCP servers configuration:", error);
			// Return default configuration if parsing fails
			return [
				{
					enabled: false,
					name: "Default SSE",
					url: "http://localhost:8000",
					apiKey: "",
					headers: new Map<string, any>(),
					cookies: new Map<string, any>(),
				},
			];
		}
	}

	public getSTDIOMCPServers(): StdioServerConfig[] {
		const config = vscode.workspace.getConfiguration("waldoai");
		const serversJson = config.get<string>("mcpServersSTDIO", "[]");

		try {
			return JSON.parse(serversJson) as StdioServerConfig[];
		} catch (error) {
			console.error("Failed to parse MCP servers configuration:", error);
			// Return default configuration if parsing fails
			return [
				{
					enabled: false,
					name: "Default SSE",
					command: "npx",
					args: ["-y", "@modelcontextprotocol/server-memory"],
					env: new Map<string, any>(),
				},
			];
		}
	}

	public async updateSseMCPServers(servers: SseServerConfig[]): Promise<void> {
		const config = vscode.workspace.getConfiguration("waldoai");
		try {
			const serversJson = JSON.stringify(servers, null, 2);
			await config.update(
				"mcpServersSSE",
				serversJson,
				vscode.ConfigurationTarget.Global
			);
		} catch (error) {
			console.error(
				"Failed to update SSE MCP servers configuration:",
				error
			);
			throw new Error(`Failed to update SSE MCP servers: ${error}`);
		}
	}

	public async updateStdioMCPServers(servers: StdioServerConfig[]): Promise<void> {
		const config = vscode.workspace.getConfiguration("waldoai");
		try {
			const serversJson = JSON.stringify(servers, null, 2);
			await config.update(
				"mcpServersSTDIO",
				serversJson,
				vscode.ConfigurationTarget.Global
			);
		} catch (error) {
			console.error(
				"Failed to update STDIO MCP servers configuration:",
				error
			);
			throw new Error(`Failed to update STDIO MCP servers: ${error}`);
		}
	}

	public async addSSEMCPServer(server: SseServerConfig): Promise<void> {
		const servers = this.getSSEMCPServers();
		servers.push(server);
		await this.updateSseMCPServers(servers);
	}

	public async addSTDIOMCPServer(server: StdioServerConfig): Promise<void> {
		const servers = this.getSTDIOMCPServers();
		servers.push(server);
		await this.updateStdioMCPServers(servers);
	}

	public async removeSSEMCPServer(serverName: string): Promise<void> {
		const servers = this.getSSEMCPServers();
		const filteredServers = servers.filter(
			(server) => server.name !== serverName
		);
		if (filteredServers.length === servers.length) {
			throw new Error(`Server with name "${serverName}" not found`);
		}
		await this.updateSseMCPServers(filteredServers);
	}

	public async removeSTDIOMCPServer(serverName: string): Promise<void> {
		const servers = this.getSTDIOMCPServers();
		const filteredServers = servers.filter(
			(server) => server.name !== serverName
		);

		if (filteredServers.length === servers.length) {
			throw new Error(`Server with name "${serverName}" not found`);
		}

		await this.updateStdioMCPServers(filteredServers);
	}

	public async updateSSEMCPServer(
		serverName: string,
		updatedServer: SseServerConfig
	): Promise<void> {
		const servers = this.getSSEMCPServers();
		const serverIndex = servers.findIndex(
			(server) => server.name === serverName
		);

		if (serverIndex === -1) {
			throw new Error(`Server with name "${serverName}" not found`);
		}

		servers[serverIndex] = updatedServer;
		await this.updateSseMCPServers(servers);
	}

	public async updateSTDIOMCPServer(
		serverName: string,
		updatedServer: StdioServerConfig
	): Promise<void> {
		const servers = this.getSTDIOMCPServers();
		const serverIndex = servers.findIndex(
			(server) => server.name === serverName
		);

		if (serverIndex === -1) {
			throw new Error(`Server with name "${serverName}" not found`);
		}

		servers[serverIndex] = updatedServer;
		await this.updateStdioMCPServers(servers);
	}

	public async updateMultiMCPClient() {
		const sse_servers = this.getSSEMCPServers();
		const stdio_servers = this.getSTDIOMCPServers();

		let mcpServers = {} as Record<string, any>;
		let total_servers = 0;

		for (const server of sse_servers) {
			if (server.enabled) {
				mcpServers[server.name] = {
					transport: "sse",
					url: server.url,
					headers: server.headers,
					cookies: server.cookies,

					automaticSSEFallback: false,
					reconnect: {
						enabled: true,
						maxAttempts: 2,
						delayMs: 2000,
					},
				};
				total_servers += 1;
			}
		}

		for (const server of stdio_servers) {
			if (server.enabled) {
				mcpServers[server.name] = {
					transport: "stdio",
					command: server.command,
					args: server.args,
					env: server.env,

					restart: {
						enabled: true,
						maxAttempts: 2,
						delayMs: 2000,
					},
				};
				total_servers += 1;
			}
		}

		if (this.multi_mcp_client) {
			this.multi_mcp_client.close();
		}

		if (this.multi_mcp_client) {
			this.multi_mcp_client.close();
		}

		if (total_servers == 0) {
			this.multi_mcp_client = null;
			return;
		}

		this.multi_mcp_client = new MultiServerMCPClient({
			throwOnLoadError: true,
			prefixToolNameWithServerName: true,
			additionalToolNamePrefix: "mcp",

			mcpServers: mcpServers
		});
	}

	public async getRegisteredMCPTools() : Promise<Array<DynamicTool>> {
		if (this.multi_mcp_client == null) {
			return new Array<DynamicTool>();
		}
		return await this.multi_mcp_client.getTools() as Array<DynamicTool>;
	}

	public async keybindGetToolsTest() {
		await this.updateMultiMCPClient();
		console.log("Get MCP tools");
		const tools = await this.getRegisteredMCPTools();
		console.log(tools);
	}
}
