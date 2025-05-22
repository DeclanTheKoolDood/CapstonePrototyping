
import {DynamicTool} from "@langchain/core/tools";
import { McpService } from "./mcpService";

// https://gist.github.com/x51xxx/4d61e8c675681d165f012a7231d06976

export class ToolsService {
	private static instance: ToolsService;

	private constructor() {}

	public static getInstance(): ToolsService {
		if (!ToolsService.instance) {
			ToolsService.instance = new ToolsService();
		}
		return ToolsService.instance;
	}

	public async getBuiltInTools() : Promise<Array<DynamicTool>> {
		return new Array<DynamicTool>();
	}

	public async getTools() : Promise<Array<DynamicTool>> {
		let built_in_tools = await this.getBuiltInTools();
		// TODO
		// list descendants (ignore .git, node_modules, etc.)
		// list directory
		// read file (chunked)
		// write file (chunked)
		// run command in terminal + get output
		// web search
		// add document to document database
		// query document database
		let mcp_tools = await McpService.getInstance().getRegisteredMCPTools();
		return built_in_tools.concat(mcp_tools);
	}

	public async keybindGetToolsTest() {
		console.log("Get Local Tools");
		const tools = await this.getTools();
		console.log(tools);
	}
}
