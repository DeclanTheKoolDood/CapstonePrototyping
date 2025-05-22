import * as vscode from "vscode";

export type Provider = "openai" | "ollama";

export interface OpenAIConfig {
	apiKey: string;
	baseUrl: string;
	models: ModelsConfig;
}

export interface OllamaConfig {
	url: string;
	models: ModelsConfig;
}

export interface ModelsConfig {
	plan: string;
	think: string;
	code: string;
}

export interface TimeoutConfig {
	request: number;
	stream: number;
}

export class ConfigService {
	private static instance: ConfigService;

	private constructor() {}

	public static getInstance(): ConfigService {
		if (!ConfigService.instance) {
			ConfigService.instance = new ConfigService();
		}
		return ConfigService.instance;
	}

	public getProvider(): Provider {
		const config = vscode.workspace.getConfiguration("waldoai");
		return config.get<Provider>("provider", "openai");
	}

	public getOpenAIConfig(): OpenAIConfig {
		const config = vscode.workspace.getConfiguration("waldoai");
		return {
			apiKey: config.get<string>("openai.apiKey", ""),
			baseUrl: config.get<string>(
				"openai.baseUrl",
				"https://api.openai.com/v1"
			),
			models: {
				plan: config.get<string>("openai.models.plan", "o4-mini"),
				think: config.get<string>("openai.models.think", "o4-mini"),
				code: config.get<string>("openai.models.code", "o4-mini"),
			},
		};
	}

	public getOllamaConfig(): OllamaConfig {
		const config = vscode.workspace.getConfiguration("waldoai");
		return {
			url: config.get<string>("ollama.url", "http://localhost:11434"),
			models: {
				plan: config.get<string>("ollama.models.plan", "qwen3:8b"),
				think: config.get<string>("ollama.models.think", "qwen3:8b"),
				code: config.get<string>("ollama.models.code", "qwen3:8b"),
			},
		};
	}

	public getModelsConfig(): ModelsConfig {
		const provider = this.getProvider();

		if (provider === "openai") {
			return this.getOpenAIConfig().models;
		} else {
			return this.getOllamaConfig().models;
		}
	}

	public getTimeoutConfig(): TimeoutConfig {
		const config = vscode.workspace.getConfiguration("waldoai");
		return {
			request: config.get<number>("timeout.request", 60000),
			stream: config.get<number>("timeout.stream", 300000),
		};
	}

	public async updateModelsConfig(
		modelsConfig: Partial<ModelsConfig>
	): Promise<void> {
		const provider = this.getProvider();
		const prefix =
			provider === "openai" ? "openai.models" : "ollama.models";
		const config = vscode.workspace.getConfiguration("waldoai");

		if (modelsConfig.plan !== undefined) {
			await config.update(
				`${prefix}.plan`,
				modelsConfig.plan,
				vscode.ConfigurationTarget.Global
			);
		}

		if (modelsConfig.think !== undefined) {
			await config.update(
				`${prefix}.think`,
				modelsConfig.think,
				vscode.ConfigurationTarget.Global
			);
		}

		if (modelsConfig.code !== undefined) {
			await config.update(
				`${prefix}.code`,
				modelsConfig.code,
				vscode.ConfigurationTarget.Global
			);
		}
	}

	public async updateTimeoutConfig(
		timeoutConfig: Partial<TimeoutConfig>
	): Promise<void> {
		const config = vscode.workspace.getConfiguration("waldoai");

		if (timeoutConfig.request !== undefined) {
			await config.update(
				"timeout.request",
				timeoutConfig.request,
				vscode.ConfigurationTarget.Global
			);
		}

		if (timeoutConfig.stream !== undefined) {
			await config.update(
				"timeout.stream",
				timeoutConfig.stream,
				vscode.ConfigurationTarget.Global
			);
		}
	}

	public async updateProvider(provider: Provider): Promise<void> {
		const config = vscode.workspace.getConfiguration("waldoai");
		await config.update(
			"provider",
			provider,
			vscode.ConfigurationTarget.Global
		);
	}

	public async updateOpenAIConfig(
		openAIConfig: Partial<OpenAIConfig>
	): Promise<void> {
		const config = vscode.workspace.getConfiguration("waldoai");

		if (openAIConfig.apiKey !== undefined) {
			await config.update(
				"openai.apiKey",
				openAIConfig.apiKey,
				vscode.ConfigurationTarget.Global
			);
		}

		if (openAIConfig.baseUrl !== undefined) {
			await config.update(
				"openai.baseUrl",
				openAIConfig.baseUrl,
				vscode.ConfigurationTarget.Global
			);
		}

		if (openAIConfig.models) {
			if (openAIConfig.models.plan !== undefined) {
				await config.update(
					"openai.models.plan",
					openAIConfig.models.plan,
					vscode.ConfigurationTarget.Global
				);
			}

			if (openAIConfig.models.think !== undefined) {
				await config.update(
					"openai.models.think",
					openAIConfig.models.think,
					vscode.ConfigurationTarget.Global
				);
			}

			if (openAIConfig.models.code !== undefined) {
				await config.update(
					"openai.models.code",
					openAIConfig.models.code,
					vscode.ConfigurationTarget.Global
				);
			}
		}
	}

	public async updateOllamaConfig(
		ollamaConfig: Partial<OllamaConfig>
	): Promise<void> {
		const config = vscode.workspace.getConfiguration("waldoai");

		if (ollamaConfig.url !== undefined) {
			await config.update(
				"ollama.url",
				ollamaConfig.url,
				vscode.ConfigurationTarget.Global
			);
		}

		if (ollamaConfig.models) {
			if (ollamaConfig.models.plan !== undefined) {
				await config.update(
					"ollama.models.plan",
					ollamaConfig.models.plan,
					vscode.ConfigurationTarget.Global
				);
			}

			if (ollamaConfig.models.think !== undefined) {
				await config.update(
					"ollama.models.think",
					ollamaConfig.models.think,
					vscode.ConfigurationTarget.Global
				);
			}

			if (ollamaConfig.models.code !== undefined) {
				await config.update(
					"ollama.models.code",
					ollamaConfig.models.code,
					vscode.ConfigurationTarget.Global
				);
			}
		}
	}
}
