import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";

/**
 * Manages configuration files for rules and memories
 */
export class ConfigFilesManager {
	private static instance: ConfigFilesManager;
	private globalStoragePath: string;
	private workspacePath: string | undefined;

	private constructor(context: vscode.ExtensionContext) {
		this.globalStoragePath = context.globalStoragePath;
		this.workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

		// Create necessary directories
		this.ensureDirectoriesExist();
	}

	public static getInstance(
		context: vscode.ExtensionContext
	): ConfigFilesManager {
		if (!ConfigFilesManager.instance) {
			ConfigFilesManager.instance = new ConfigFilesManager(context);
		}
		return ConfigFilesManager.instance;
	}

	/**
	 * Ensure all necessary directories exist
	 */
	private ensureDirectoriesExist(): void {
		// Global directories
		const globalRulesDir = path.join(this.globalStoragePath, "rules");
		const globalMemoriesDir = path.join(this.globalStoragePath, "memories");

		if (!fs.existsSync(this.globalStoragePath)) {
			fs.mkdirSync(this.globalStoragePath, { recursive: true });
		}

		if (!fs.existsSync(globalRulesDir)) {
			fs.mkdirSync(globalRulesDir, { recursive: true });
		}

		if (!fs.existsSync(globalMemoriesDir)) {
			fs.mkdirSync(globalMemoriesDir, { recursive: true });
		}

		// Local directories (if workspace exists)
		if (this.workspacePath) {
			const localRulesDir = path.join(
				this.workspacePath,
				".waldoai",
				"rules"
			);
			const localMemoriesDir = path.join(
				this.workspacePath,
				".waldoai",
				"memories"
			);

			if (!fs.existsSync(path.join(this.workspacePath, ".waldoai"))) {
				fs.mkdirSync(path.join(this.workspacePath, ".waldoai"), {
					recursive: true,
				});
			}

			if (!fs.existsSync(localRulesDir)) {
				fs.mkdirSync(localRulesDir, { recursive: true });
			}

			if (!fs.existsSync(localMemoriesDir)) {
				fs.mkdirSync(localMemoriesDir, { recursive: true });
			}
		}
	}

	/**
	 * Get the path to the global rules file
	 */
	public getGlobalRulesFilePath(): string {
		return path.join(this.globalStoragePath, "rules", "global-rules.md");
	}

	/**
	 * Get the path to the local rules file
	 */
	public getLocalRulesFilePath(): string {
		if (!this.workspacePath) {
			throw new Error("No workspace folder is open");
		}
		return path.join(
			this.workspacePath,
			".waldoai",
			"rules",
			"local-rules.md"
		);
	}

	/**
	 * Get the path to the global memories file
	 */
	public getGlobalMemoriesFilePath(): string {
		return path.join(
			this.globalStoragePath,
			"memories",
			"global-memories.md"
		);
	}

	/**
	 * Get the path to the local memories file
	 */
	public getLocalMemoriesFilePath(): string {
		if (!this.workspacePath) {
			throw new Error("No workspace folder is open");
		}
		return path.join(
			this.workspacePath,
			".waldoai",
			"memories",
			"local-memories.md"
		);
	}

	/**
	 * Open the global rules file
	 */
	public async openGlobalRules(): Promise<void> {
		const filePath = this.getGlobalRulesFilePath();
		await this.ensureFileExists(
			filePath,
			this.getDefaultGlobalRulesContent()
		);
		const document = await vscode.workspace.openTextDocument(filePath);
		await vscode.window.showTextDocument(document);
	}

	/**
	 * Open the local rules file
	 */
	public async openLocalRules(): Promise<void> {
		if (!this.workspacePath) {
			vscode.window.showErrorMessage(
				"No workspace folder is open. Local rules require an open workspace."
			);
			return;
		}

		const filePath = this.getLocalRulesFilePath();
		await this.ensureFileExists(
			filePath,
			this.getDefaultLocalRulesContent()
		);
		const document = await vscode.workspace.openTextDocument(filePath);
		await vscode.window.showTextDocument(document);
	}

	/**
	 * Open the global memories file
	 */
	public async openGlobalMemories(): Promise<void> {
		const filePath = this.getGlobalMemoriesFilePath();
		await this.ensureFileExists(
			filePath,
			this.getDefaultGlobalMemoriesContent()
		);
		const document = await vscode.workspace.openTextDocument(filePath);
		await vscode.window.showTextDocument(document);
	}

	/**
	 * Open the local memories file
	 */
	public async openLocalMemories(): Promise<void> {
		if (!this.workspacePath) {
			vscode.window.showErrorMessage(
				"No workspace folder is open. Local memories require an open workspace."
			);
			return;
		}

		const filePath = this.getLocalMemoriesFilePath();
		await this.ensureFileExists(
			filePath,
			this.getDefaultLocalMemoriesContent()
		);
		const document = await vscode.workspace.openTextDocument(filePath);
		await vscode.window.showTextDocument(document);
	}

	/**
	 * Ensure a file exists, creating it with default content if it doesn't
	 */
	private async ensureFileExists(
		filePath: string,
		defaultContent: string
	): Promise<void> {
		if (!fs.existsSync(filePath)) {
			fs.writeFileSync(filePath, defaultContent, "utf8");
		}
	}

	/**
	 * Get default content for the global rules file
	 */
	private getDefaultGlobalRulesContent(): string {
		return `# Global Rules for Waldo AI

These rules apply to all projects and conversations with Waldo AI.

## General Rules

- Rule 1: Example global rule
- Rule 2: Another example global rule

## Coding Style

- Style 1: Example coding style rule
- Style 2: Another example coding style rule

## Documentation

- Doc 1: Example documentation rule
- Doc 2: Another example documentation rule

---

You can edit this file to add your own global rules for Waldo AI.
`;
	}

	/**
	 * Get default content for the local rules file
	 */
	private getDefaultLocalRulesContent(): string {
		const projectName = this.workspacePath
			? path.basename(this.workspacePath)
			: "Current Project";

		return `# Local Rules for ${projectName}

These rules apply only to this specific project when working with Waldo AI.

## Project-Specific Rules

- Rule 1: Example project-specific rule
- Rule 2: Another example project-specific rule

## Coding Style

- Style 1: Example project-specific coding style rule
- Style 2: Another example project-specific coding style rule

## Documentation

- Doc 1: Example project-specific documentation rule
- Doc 2: Another example project-specific documentation rule

---

You can edit this file to add your own local rules for this project.
`;
	}

	/**
	 * Get default content for the global memories file
	 */
	private getDefaultGlobalMemoriesContent(): string {
		return `# Global Memories for Waldo AI

These memories are available to Waldo AI across all projects.

## User Preferences

- Preference 1: Example user preference
- Preference 2: Another example user preference

## Common Tasks

- Task 1: Example common task
- Task 2: Another example common task

---

You can edit this file to add your own global memories for Waldo AI.
`;
	}

	/**
	 * Get default content for the local memories file
	 */
	private getDefaultLocalMemoriesContent(): string {
		const projectName = this.workspacePath
			? path.basename(this.workspacePath)
			: "Current Project";

		return `# Local Memories for ${projectName}

These memories are specific to this project when working with Waldo AI.

## Project Context

- Context 1: Example project context
- Context 2: Another example project context

## Project-Specific Tasks

- Task 1: Example project-specific task
- Task 2: Another example project-specific task

---

You can edit this file to add your own local memories for this project.
`;
	}
}
