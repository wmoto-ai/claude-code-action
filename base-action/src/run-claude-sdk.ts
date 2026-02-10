import * as core from "@actions/core";
import { readFile, writeFile, access } from "fs/promises";
import { dirname, join } from "path";
import { query } from "@anthropic-ai/claude-agent-sdk";
import type {
  SDKMessage,
  SDKResultMessage,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { ParsedSdkOptions } from "./parse-sdk-options";

export type ClaudeRunResult = {
  executionFile?: string;
  sessionId?: string;
  conclusion: "success" | "failure";
  structuredOutput?: string;
};

const EXECUTION_FILE = `${process.env.RUNNER_TEMP}/claude-execution-output.json`;

/** Filename for the user request file, written by prompt generation */
const USER_REQUEST_FILENAME = "claude-user-request.txt";

/**
 * Check if a file exists
 */
async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Creates a prompt configuration for the SDK.
 * If a user request file exists alongside the prompt file, returns a multi-block
 * SDKUserMessage that enables slash command processing in the CLI.
 * Otherwise, returns the prompt as a simple string.
 */
async function createPromptConfig(
  promptPath: string,
  showFullOutput: boolean,
): Promise<string | AsyncIterable<SDKUserMessage>> {
  const promptContent = await readFile(promptPath, "utf-8");

  // Check for user request file in the same directory
  const userRequestPath = join(dirname(promptPath), USER_REQUEST_FILENAME);
  const hasUserRequest = await fileExists(userRequestPath);

  if (!hasUserRequest) {
    // No user request file - use simple string prompt
    return promptContent;
  }

  // User request file exists - create multi-block message
  const userRequest = await readFile(userRequestPath, "utf-8");
  if (showFullOutput) {
    console.log("Using multi-block message with user request:", userRequest);
  } else {
    console.log("Using multi-block message with user request (content hidden)");
  }

  // Create an async generator that yields a single multi-block message
  // The context/instructions go first, then the user's actual request last
  // This allows the CLI to detect and process slash commands in the user request
  async function* createMultiBlockMessage(): AsyncGenerator<SDKUserMessage> {
    yield {
      type: "user",
      session_id: "",
      message: {
        role: "user",
        content: [
          { type: "text", text: promptContent }, // Instructions + GitHub context
          { type: "text", text: userRequest }, // User's request (may be a slash command)
        ],
      },
      parent_tool_use_id: null,
    };
  }

  return createMultiBlockMessage();
}

/**
 * Truncate a string to maxLen characters, appending "..." if truncated.
 */
function truncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen) + "...";
}

/**
 * Format a compact one-line summary of a tool_use call (OpenCode style).
 * Shows tool name and a brief summary of its parameters.
 */
function formatCompactToolUse(item: {
  name?: string;
  input?: Record<string, any>;
}): string {
  const toolName = (item.name || "unknown").padEnd(16);
  const input = item.input || {};

  // Build a brief parameter summary depending on tool type
  let paramSummary: string;
  const name = item.name || "";

  if (name === "Read" || name === "read_file") {
    paramSummary = input.file_path || input.filePath || "";
  } else if (name === "Write" || name === "write_file") {
    paramSummary = input.file_path || input.filePath || "";
  } else if (name === "Edit" || name === "edit_file") {
    paramSummary = input.file_path || input.filePath || "";
  } else if (name === "Bash" || name === "bash") {
    paramSummary = truncate(
      String(input.command || input.cmd || ""),
      200,
    );
  } else if (name === "Glob" || name === "glob") {
    paramSummary = input.pattern || "";
  } else if (name === "Grep" || name === "grep") {
    paramSummary = `pattern=${input.pattern || ""} path=${input.path || ""}`;
  } else if (name === "Task" || name === "task") {
    paramSummary = `[${input.subagent_type || "agent"}] ${input.description || ""}`;
  } else if (name === "TodoWrite") {
    const todos = input.todos;
    if (Array.isArray(todos)) {
      const inProgress = todos.find(
        (t: any) => t.status === "in_progress",
      );
      paramSummary = inProgress
        ? inProgress.activeForm || inProgress.content || ""
        : `${todos.length} items`;
    } else {
      paramSummary = "";
    }
  } else if (name === "WebFetch" || name === "WebSearch") {
    paramSummary = input.url || input.query || "";
  } else if (name === "apply_patch") {
    const patch = String(input.patchText || input.patch || "");
    // Extract file names from patch
    const files = patch.match(
      /\*\*\* (?:Add|Update|Delete) File: (.+)/g,
    );
    paramSummary = files
      ? files.map((f: string) => f.replace(/\*\*\* \w+ File: /, "")).join(", ")
      : truncate(patch, 100);
  } else {
    // Generic: show compact JSON of input
    paramSummary = truncate(JSON.stringify(input), 200);
  }

  return `| ${toolName} ${paramSummary}`;
}

/**
 * Sanitizes SDK output to match CLI sanitization behavior.
 * When showFullOutput is false, outputs compact OpenCode-style tool call logs.
 */
function sanitizeSdkOutput(
  message: SDKMessage,
  showFullOutput: boolean,
): string | null {
  if (showFullOutput) {
    return JSON.stringify(message, null, 2);
  }

  // System initialization - safe to show
  if (message.type === "system" && message.subtype === "init") {
    return JSON.stringify(
      {
        type: "system",
        subtype: "init",
        message: "Claude Code initialized",
        model: "model" in message ? message.model : "unknown",
      },
      null,
      2,
    );
  }

  // Result messages - show sanitized summary
  if (message.type === "result") {
    const resultMsg = message as SDKResultMessage;
    return JSON.stringify(
      {
        type: "result",
        subtype: resultMsg.subtype,
        is_error: resultMsg.is_error,
        duration_ms: resultMsg.duration_ms,
        num_turns: resultMsg.num_turns,
        total_cost_usd: resultMsg.total_cost_usd,
        permission_denials: resultMsg.permission_denials,
      },
      null,
      2,
    );
  }

  // Assistant messages - show compact tool calls and text (OpenCode style)
  if (message.type === "assistant" && "message" in message) {
    const msg = message as any;
    const content = msg.message?.content;
    if (!Array.isArray(content)) return null;

    const lines: string[] = [];

    for (const item of content) {
      if (item.type === "tool_use") {
        lines.push(formatCompactToolUse(item));
      } else if (item.type === "text" && item.text?.trim()) {
        // Show assistant text (truncated) - useful to see Claude's reasoning
        const text = truncate(item.text.trim().replace(/\n/g, " "), 300);
        lines.push(`[Claude] ${text}`);
      }
    }

    return lines.length > 0 ? lines.join("\n") : null;
  }

  // Suppress other message types in non-full-output mode
  return null;
}

/**
 * Run Claude using the Agent SDK
 */
export async function runClaudeWithSdk(
  promptPath: string,
  { sdkOptions, showFullOutput, hasJsonSchema }: ParsedSdkOptions,
): Promise<ClaudeRunResult> {
  // Create prompt configuration - may be a string or multi-block message
  const prompt = await createPromptConfig(promptPath, showFullOutput);

  if (!showFullOutput) {
    console.log(
      "Running Claude Code via SDK (full output hidden for security)...",
    );
    console.log(
      "Rerun in debug mode or enable `show_full_output: true` in your workflow file for full output.",
    );
  }

  console.log(`Running Claude with prompt from file: ${promptPath}`);
  // Log SDK options without env (which could contain sensitive data)
  const { env, ...optionsToLog } = sdkOptions;
  console.log("SDK options:", JSON.stringify(optionsToLog, null, 2));

  const messages: SDKMessage[] = [];
  let resultMessage: SDKResultMessage | undefined;

  try {
    for await (const message of query({ prompt, options: sdkOptions })) {
      messages.push(message);

      const sanitized = sanitizeSdkOutput(message, showFullOutput);
      if (sanitized) {
        console.log(sanitized);
      }

      if (message.type === "result") {
        resultMessage = message as SDKResultMessage;
      }
    }
  } catch (error) {
    console.error("SDK execution error:", error);
    throw new Error(`SDK execution error: ${error}`);
  }

  const result: ClaudeRunResult = {
    conclusion: "failure",
  };

  // Write execution file
  try {
    await writeFile(EXECUTION_FILE, JSON.stringify(messages, null, 2));
    console.log(`Log saved to ${EXECUTION_FILE}`);
    result.executionFile = EXECUTION_FILE;
  } catch (error) {
    core.warning(`Failed to write execution file: ${error}`);
  }

  // Extract session_id from system.init message
  const initMessage = messages.find(
    (m) => m.type === "system" && "subtype" in m && m.subtype === "init",
  );
  if (initMessage && "session_id" in initMessage && initMessage.session_id) {
    result.sessionId = initMessage.session_id as string;
    core.info(`Set session_id: ${result.sessionId}`);
  }

  if (!resultMessage) {
    core.error("No result message received from Claude");
    throw new Error("No result message received from Claude");
  }

  const isSuccess = resultMessage.subtype === "success";
  result.conclusion = isSuccess ? "success" : "failure";

  // Handle structured output
  if (hasJsonSchema) {
    if (
      isSuccess &&
      "structured_output" in resultMessage &&
      resultMessage.structured_output
    ) {
      result.structuredOutput = JSON.stringify(resultMessage.structured_output);
      core.info(
        `Set structured_output with ${Object.keys(resultMessage.structured_output as object).length} field(s)`,
      );
    } else {
      core.setFailed(
        `--json-schema was provided but Claude did not return structured_output. Result subtype: ${resultMessage.subtype}`,
      );
      result.conclusion = "failure";
      throw new Error(
        `--json-schema was provided but Claude did not return structured_output. Result subtype: ${resultMessage.subtype}`,
      );
    }
  }

  if (!isSuccess) {
    if ("errors" in resultMessage && resultMessage.errors) {
      core.error(`Execution failed: ${resultMessage.errors.join(", ")}`);
    }
    throw new Error(
      `Claude execution failed: ${
        "errors" in resultMessage && resultMessage.errors
          ? resultMessage.errors.join(", ")
          : "unknown error"
      }`,
    );
  }

  return result;
}
