// https://gist.github.com/x51xxx/4d61e8c675681d165f012a7231d06976
// https://v03.api.js.langchain.com/functions/langchain.agents.createReactAgent.html
// https://smith.langchain.com/hub/hwchase17/react
// https://github.com/langchain-ai/langchainjs/blob/d4a8e297c264f1b8e434cc970d0af474fd5f9446/langchain/src/agents/react/index.ts#L77

// TODO: 'system' prompts for agent and chat mode (for each LLM provider)

import { BaseLanguageModelInterface } from "@langchain/core/language_models/base";
import { ChatOllama } from "@langchain/ollama";
import { ChatOpenAI } from "@langchain/openai";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, BaseMessage, ChatMessage, InvalidToolCall, ToolMessage } from "@langchain/core/messages";
import { DynamicTool, StructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import { AgentExecutor, createReactAgent } from "langchain/agents";
import { PromptTemplate } from "@langchain/core/prompts";
import { Runnable } from "@langchain/core/runnables";
import { BaseLanguageModelInput } from "@langchain/core/language_models/base";
import { pull } from "langchain/hub";

export type ChatHistory = ChatMessage[];

export interface LLMResponse {
	raw: string;
	thinking: string;
	text: string;
	formatted: JSON | null;

	used_tools: ToolMessage[];
	invalid_tools: InvalidToolCall[];
	usage_metrics: Map<string, number>;
}

export interface StreamingAgentStep {
	finished: boolean,
	thoughts : string | null,
	text: string | null,
	used_tools: ToolMessage[],
	invalid_tools: InvalidToolCall[],
	usage_metrics: Map<string, number>,
	intermediate_steps: any[],
	chunk : any | null
}

export interface AbstractLLMService {
	postChat(
		messages: ChatHistory,
		llm: BaseChatModel,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<LLMResponse>;
	streamChat(
		messages: ChatHistory,
		llm: BaseChatModel,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): AsyncGenerator<StreamingAgentStep>;
}

export interface FrontLLMService {
	postChat(
		messages: ChatHistory,
		model: string | null,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<LLMResponse>;
	streamChat(
		messages: ChatHistory,
		model: string | null,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<AsyncGenerator<StreamingAgentStep>>;
}

export type LLMChatModel =
	| BaseChatModel
	| Runnable<BaseLanguageModelInput, any>
	| BaseLanguageModelInterface<any, any>;

class CustomLLMWrapper extends BaseChatModel {
	private llm: BaseChatModel;
	private lastThought: string = ""; // Store last <think> content

	constructor(llm: BaseChatModel) {
		super({}); // Pass empty options or configure as needed
		this.llm = llm;
	}

	// Get the last captured thought
	public getLastThought(): string {
		return this.lastThought;
	}

	async generate(
		messages: BaseMessage[][],
		options?: Record<string, any>
	): Promise<any> {
		// Call the underlying LLM
		const response = await this.llm.generate(messages, options);

		// Process generations to extract <think> content
		this.lastThought = ""; // Reset before processing
		for (const generation of response.generations) {
			for (const gen of generation) {
				if (!gen.text || !gen.text.includes("<think>")) {
					continue;
				}
				// Extract all <think> content
				const thinkMatches = gen.text.matchAll(/<think>([\s\S]*?)(<\/think>|(?=<think>|$)|\n\n)/g);
				let processedText = gen.text;
				let thinkContent = "";
				for (const match of thinkMatches) {
					const content = match[1].trim();
					if (!content) {
						continue;
					}
					thinkContent += content + "\n";
					processedText = processedText.replace(
						match[0],
						`[Thought]\n${content}\n\n`
					);
				}
				this.lastThought += thinkContent;
				gen.text = processedText;
			}
		}

		return response;
	}

	// Implement required abstract methods
	_llmType(): string {
		return "custom_llm_wrapper";
	}

	async _generate(
		messages: BaseMessage[],
		options?: Record<string, any>
	): Promise<any> {
		return this.generate([messages], options);
	}
}

export class LLMService implements AbstractLLMService {
	private static instance: LLMService;

	private constructor() {}

	public static getInstance(): LLMService {
		if (!LLMService.instance) {
			LLMService.instance = new LLMService();
		}
		return LLMService.instance;
	}

	public async postChat(
		messages: ChatHistory,
		llm: LLMChatModel,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<LLMResponse> {
		// Enhanced input validation
		if (!messages || messages.length === 0) {
			console.error("Error: Messages array is empty or undefined", {
				messages,
			});
			throw new Error("No messages provided");
		}

		// Find the last non-empty message
		let lastMessage: string | undefined;
		let lastMessageIndex = -1;
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].content && messages[i].content.toString().trim() !== "") {
				lastMessage = messages[i].content.toString();
				lastMessageIndex = i;
				break;
			}
		}

		if (!lastMessage) {
			console.error("Error: No non-empty message found", { messages });
			throw new Error("No valid input question provided in messages");
		}

		let structured_llm = llm;
		if (format) {
			console.log("Structured output!");
			structured_llm = (llm as BaseChatModel).withStructuredOutput(
				format,
				{
					name: "output_formatter",
				}
			);
		}

		// Wrap the LLM
		const wrapped_llm = new CustomLLMWrapper(
			structured_llm as BaseChatModel
		);

		const default_react = await pull<PromptTemplate>("hwchase17/react");

		let agent = await createReactAgent({
			llm: wrapped_llm as BaseLanguageModelInterface<any, any>,
			tools: tools || [],
			prompt: default_react,
			streamRunnable: true,
		});

		const agentExecutor = AgentExecutor.fromAgentAndTools({
			agent,
			tools: tools || [],
			returnIntermediateSteps: true,
			maxIterations: max_iterations || 10,
		});

		// Stream the agent's output
		let finalOutput = "";
		const intermediateSteps: any[] = [];
		let used_tools_calls: any[] = [];
		let invalid_tool_calls: any[] = [];
		let usage_metrics: any = {};

		const stream_params = {
			input: lastMessage,
			chat_history: messages
				.slice(0, lastMessageIndex)
				.map((m) => `${m.role}: ${m.content}`)
				.join("\n"),
		};

		for await (const chunk of await agentExecutor.stream(stream_params)) {
			// Log chunk for debugging
			console.log("Stream chunk:", JSON.stringify(chunk, null, 2));

			// Aggregate output
			if (chunk.output) {
				finalOutput += chunk.output;
			}

			// Collect intermediate steps
			if (chunk.intermediateSteps) {
				intermediateSteps.push(...chunk.intermediateSteps);
				used_tools_calls.push(
					...chunk.intermediateSteps
						.filter((step: any) => step.action && step.action.tool)
						.map((step: any) => step.action)
				);
			}
		}

		// Extract usage metrics if available
		if (intermediateSteps.length > 0) {
			const lastStep = intermediateSteps[intermediateSteps.length - 1];
			if (lastStep?.message?.kwargs?.usage_metadata) {
				usage_metrics = lastStep.message.kwargs.usage_metadata;
			}
		}

		// Handle invalid tool calls (if any)
		invalid_tool_calls = intermediateSteps
			.filter((step: any) => step.action && step.action.invalid)
			.map((step: any) => step.action);

		// Get thinking from wrapped LLM
		const thinking = wrapped_llm.getLastThought();

		// Format the final response
		let output = finalOutput;
		if (format) {
			console.log("Formatted output!");
			return {
				raw: finalOutput,
				thinking,
				text: "",
				formatted: format.parse(finalOutput),
				used_tools: used_tools_calls,
				invalid_tools: invalid_tool_calls,
				usage_metrics,
			};
		}

		if (!thinking) {
			console.log("No thinking text!");
			return {
				raw: finalOutput,
				thinking: "",
				text: finalOutput,
				formatted: null,
				used_tools: used_tools_calls,
				invalid_tools: invalid_tool_calls,
				usage_metrics,
			};
		}

		console.log("Thinking text!");
		// Remove [Thought] section from text if present
		output = finalOutput.replace(/\[Thought\][\s\S]*?\n\n/g, "").trim();

		return {
			raw: finalOutput,
			thinking,
			text: output,
			formatted: null,
			used_tools: used_tools_calls,
			invalid_tools: invalid_tool_calls,
			usage_metrics,
		};
	}

	public async* streamChat(
		messages: ChatHistory,
		llm: LLMChatModel,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): AsyncGenerator<StreamingAgentStep> {
		// Enhanced input validation
		if (!messages || messages.length === 0) {
			console.error("Error: Messages array is empty or undefined", {
				messages,
			});
			throw new Error("No messages provided");
		}

		// Find the last non-empty message
		let lastMessage: string | undefined;
		let lastMessageIndex = -1;
		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].content && messages[i].content.toString().trim() !== "") {
				lastMessage = messages[i].content.toString();
				lastMessageIndex = i;
				break;
			}
		}

		if (!lastMessage) {
			console.error("Error: No non-empty message found", { messages });
			throw new Error("No valid input question provided in messages");
		}

		let structured_llm = llm;
		if (format) {
			console.log("Structured output!");
			structured_llm = (llm as BaseChatModel).withStructuredOutput(
				format, { name: "output_formatter", }
			);
		}

		// Wrap the LLM
		const wrapped_llm = new CustomLLMWrapper(structured_llm as BaseChatModel);
		const default_react = await pull<PromptTemplate>("hwchase17/react");

		let agent = await createReactAgent({
			llm: wrapped_llm as BaseLanguageModelInterface<any, any>,
			tools: tools || [],
			prompt: default_react,
			streamRunnable: true,
		});

		const agentExecutor = AgentExecutor.fromAgentAndTools({
			agent,
			tools: tools || [],
			returnIntermediateSteps: true,
			maxIterations: max_iterations || 10,
		});

		const stream_params = {
			input: lastMessage,
			chat_history: messages
				.slice(0, lastMessageIndex)
				.map((m) => `${m.role}: ${m.content}`)
				.join("\n"),
		};

		for await (const chunk of await agentExecutor.stream(stream_params)) {
			// Log chunk for debugging
			console.log("Stream chunk:", JSON.stringify(chunk, null, 2));

			let agent_step = {
				finished: false,
				thoughts: null,
				text: null,
				used_tools: [],
				invalid_tools: [],
				usage_metrics: new Map<string, any>(),
				intermediate_steps: [],
				chunk: chunk,
			} as StreamingAgentStep;

			// Collect output if available
			if (chunk.output) {
				agent_step.text = chunk.output;
			}

			// Collect intermediate step information if available
			if (chunk.intermediateSteps) {
				agent_step.thoughts = wrapped_llm.getLastThought();

				agent_step.intermediate_steps = chunk.intermediateSteps;

				agent_step.used_tools = chunk.intermediateSteps
					.filter((step: any) => step.action && step.action.tool)
					.map((step: any) => step.action);

				agent_step.usage_metrics = chunk.intermediateSteps[chunk.intermediateSteps.length - 1]?.message?.kwargs?.usage_metadata;

				agent_step.invalid_tools = chunk.intermediateSteps
					.filter((step: any) => step.action && step.action.invalid)
					.map((step: any) => step.action);
			}

			// yield the chunk containing the information of StreamingAgentStep
			yield agent_step;
		}

		yield {
			finished: true,
			thoughts: null,
			text: null,
			used_tools: [],
			invalid_tools: [],
			usage_metrics: new Map<string, any>(),
			intermediate_steps: [],
			chunk: null,
		} as StreamingAgentStep;

	}
}

export class OllamaService implements FrontLLMService {
	private static instance: OllamaService;

	private constructor() {}

	public static getInstance(): OllamaService {
		if (!OllamaService.instance) {
			OllamaService.instance = new OllamaService();
		}
		return OllamaService.instance;
	}

	public async postChat(
		messages: ChatHistory,
		model: string | null,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<LLMResponse> {
		if (model == null) model = "qwen3:8b";
		const llm = new ChatOllama({
			baseUrl: "http://localhost:11434",
			model,
			verbose: true,
		});
		return LLMService.getInstance().postChat(
			messages,
			llm,
			tools,
			format,
			max_iterations
		);
	}

	public async streamChat(
		messages: ChatHistory,
		model: string | null,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<AsyncGenerator<StreamingAgentStep>> {
		if (model == null) model = "qwen3:8b";
		const llm = new ChatOllama({
			baseUrl: "http://localhost:11434",
			model,
			verbose: true,
		});
		return new Promise((resolve, reject) => {
			const iter = LLMService.getInstance().streamChat(
				messages,
				llm,
				tools,
				format,
				max_iterations
			);
			resolve(iter);
		});
	}

	public async keybindTest() {
		console.log("Ollama Test");
		const response = (await this.postChat(
			[
				new ChatMessage(
					"What is a fractal? Can you think then explain?",
					"user"
				),
			],
			"qwen3:8b",
			null,
			null,
			null
		)) as LLMResponse;
		console.log(response);
	}

	public async keybindTestStream() {
		console.log("Ollama Stream Test");
		try {
			const messages: ChatHistory = [
				{
					content: "What is a fractal? Can you think then explain?",
					role: "user"
				} as ChatMessage
			];
			console.log("Messages:", JSON.stringify(messages, null, 2));

			const iter = await this.streamChat(messages, "qwen3:8b", null, null, null);
			console.log("Starting stream...");

			let counter = 1;
			for await (const chunk of iter) {
				console.log(`Chunk ${counter++}`);
				console.log(JSON.stringify(chunk, null, 2));
			}
			console.log("Stream completed.");
		} catch (error) {
			console.error("Error in keybindTestStream:", error);
			throw error;
		}
	}
}

export class OpenAIService implements FrontLLMService {
	private static instance: OpenAIService;

	private constructor() {}

	public static getInstance(): OpenAIService {
		if (!OpenAIService.instance) {
			OpenAIService.instance = new OpenAIService();
		}
		return OpenAIService.instance;
	}

	public async postChat(
		messages: ChatHistory,
		model: string | null,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<LLMResponse> {
		if (model == null) model = "o4-mini";
		const llm = new ChatOpenAI({
			model,
			verbose: true,
		});
		return LLMService.getInstance().postChat(
			messages,
			llm,
			tools,
			format,
			max_iterations
		);
	}

	public async streamChat(
		messages: ChatHistory,
		model: string | null,
		tools: Array<DynamicTool> | null,
		format: z.ZodTypeAny | null,
		max_iterations: number | null
	): Promise<AsyncGenerator<StreamingAgentStep>> {
		if (model == null) model = "o4-mini";
		const llm = new ChatOpenAI({
			model,
			verbose: true,
		});
		return new Promise((resolve, reject) => {
			const iter = LLMService.getInstance().streamChat(
				messages,
				llm,
				tools,
				format,
				max_iterations
			);
			resolve(iter);
		});
	}
}

export class LLMSelector {

	// TODO read options
	public static getSelectedLLM(): FrontLLMService {
		return OllamaService.getInstance();
	}

}
