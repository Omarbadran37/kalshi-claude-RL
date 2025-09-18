TypeScript SDK reference

\> Complete API reference for the Claude Code TypeScript SDK, including all functions, types, and interfaces.

\<script src="/components/typescript-sdk-type-links.js" defer /\>

\#\# Functions

\#\#\# \`query()\`

The primary function for interacting with Claude Code. Creates an async generator that streams messages as they arrive.

\`\`\`ts  
function query({  
  prompt,  
  options  
}: {  
  prompt: string | AsyncIterable\<SDKUserMessage\>;  
  options?: Options;  
}): Query  
\`\`\`

\#\#\#\# Parameters

| Parameter | Type                                                             | Description                                                       |  
| :-------- | :--------------------------------------------------------------- | :---------------------------------------------------------------- |  
| \`prompt\`  | \`string \\| AsyncIterable\<\`\[\`SDKUserMessage\`\](\#sdkusermessage)\`\>\` | The input prompt as a string or async iterable for streaming mode |  
| \`options\` | \[\`Options\`\](\#options)                                            | Optional configuration object (see Options type below)            |

\#\#\#\# Returns

Returns a \[\`Query\`\](\#query-1) object that extends \`AsyncGenerator\<\`\[\`SDKMessage\`\](\#sdkmessage)\`, void\>\` with additional methods.

\#\#\# \`tool()\`

Creates a type-safe MCP tool definition for use with SDK MCP servers.

\`\`\`ts  
function tool\<Schema extends ZodRawShape\>(  
  name: string,  
  description: string,  
  inputSchema: Schema,  
  handler: (args: z.infer\<ZodObject\<Schema\>\>, extra: unknown) \=\> Promise\<CallToolResult\>  
): SdkMcpToolDefinition\<Schema\>  
\`\`\`

\#\#\#\# Parameters

| Parameter     | Type                                                              | Description                                     |  
| :------------ | :---------------------------------------------------------------- | :---------------------------------------------- |  
| \`name\`        | \`string\`                                                          | The name of the tool                            |  
| \`description\` | \`string\`                                                          | A description of what the tool does             |  
| \`inputSchema\` | \`Schema extends ZodRawShape\`                                      | Zod schema defining the tool's input parameters |  
| \`handler\`     | \`(args, extra) \=\> Promise\<\`\[\`CallToolResult\`\](\#calltoolresult)\`\>\` | Async function that executes the tool logic     |

\#\#\# \`createSdkMcpServer()\`

Creates an MCP server instance that runs in the same process as your application.

\`\`\`ts  
function createSdkMcpServer(options: {  
  name: string;  
  version?: string;  
  tools?: Array\<SdkMcpToolDefinition\<any\>\>;  
}): McpSdkServerConfigWithInstance  
\`\`\`

\#\#\#\# Parameters

| Parameter         | Type                          | Description                                              |  
| :---------------- | :---------------------------- | :------------------------------------------------------- |  
| \`options.name\`    | \`string\`                      | The name of the MCP server                               |  
| \`options.version\` | \`string\`                      | Optional version string                                  |  
| \`options.tools\`   | \`Array\<SdkMcpToolDefinition\>\` | Array of tool definitions created with \[\`tool()\`\](\#tool) |

\#\# Types

\#\#\# \`Options\`

Configuration object for the \`query()\` function.

| Property                     | Type                                                                                              | Default                 | Description                                 |  
| :--------------------------- | :------------------------------------------------------------------------------------------------ | :---------------------- | :------------------------------------------ |  
| \`abortController\`            | \`AbortController\`                                                                                 | \`new AbortController()\` | Controller for cancelling operations        |  
| \`additionalDirectories\`      | \`string\[\]\`                                                                                        | \`\[\]\`                    | Additional directories Claude can access    |  
| \`allowedTools\`               | \`string\[\]\`                                                                                        | All tools               | List of allowed tool names                  |  
| \`appendSystemPrompt\`         | \`string\`                                                                                          | \`undefined\`             | Text to append to the default system prompt |  
| \`canUseTool\`                 | \[\`CanUseTool\`\](\#canusetool)                                                                       | \`undefined\`             | Custom permission function for tool usage   |  
| \`continue\`                   | \`boolean\`                                                                                         | \`false\`                 | Continue the most recent conversation       |  
| \`customSystemPrompt\`         | \`string\`                                                                                          | \`undefined\`             | Replace the default system prompt entirely  |  
| \`cwd\`                        | \`string\`                                                                                          | \`process.cwd()\`         | Current working directory                   |  
| \`disallowedTools\`            | \`string\[\]\`                                                                                        | \`\[\]\`                    | List of disallowed tool names               |  
| \`env\`                        | \`Dict\<string\>\`                                                                                    | \`process.env\`           | Environment variables                       |  
| \`executable\`                 | \`'bun' \\| 'deno' \\| 'node'\`                                                                       | Auto-detected           | JavaScript runtime to use                   |  
| \`executableArgs\`             | \`string\[\]\`                                                                                        | \`\[\]\`                    | Arguments to pass to the executable         |  
| \`extraArgs\`                  | \`Record\<string, string \\| null\>\`                                                                  | \`{}\`                    | Additional arguments                        |  
| \`fallbackModel\`              | \`string\`                                                                                          | \`undefined\`             | Model to use if primary fails               |  
| \`hooks\`                      | \`Partial\<Record\<\`\[\`HookEvent\`\](\#hookevent)\`, \`\[\`HookCallbackMatcher\`\](\#hookcallbackmatcher)\`\[\]\>\>\` | \`{}\`                    | Hook callbacks for events                   |  
| \`includePartialMessages\`     | \`boolean\`                                                                                         | \`false\`                 | Include partial message events              |  
| \`maxThinkingTokens\`          | \`number\`                                                                                          | \`undefined\`             | Maximum tokens for thinking process         |  
| \`maxTurns\`                   | \`number\`                                                                                          | \`undefined\`             | Maximum conversation turns                  |  
| \`mcpServers\`                 | \`Record\<string, \`\[\`McpServerConfig\`\](\#mcpserverconfig)\`\>\`                                         | \`{}\`                    | MCP server configurations                   |  
| \`model\`                      | \`string\`                                                                                          | Default from CLI        | Claude model to use                         |  
| \`pathToClaudeCodeExecutable\` | \`string\`                                                                                          | Auto-detected           | Path to Claude Code executable              |  
| \`permissionMode\`             | \[\`PermissionMode\`\](\#permissionmode)                                                               | \`'default'\`             | Permission mode for the session             |  
| \`permissionPromptToolName\`   | \`string\`                                                                                          | \`undefined\`             | MCP tool name for permission prompts        |  
| \`resume\`                     | \`string\`                                                                                          | \`undefined\`             | Session ID to resume                        |  
| \`stderr\`                     | \`(data: string) \=\> void\`                                                                          | \`undefined\`             | Callback for stderr output                  |  
| \`strictMcpConfig\`            | \`boolean\`                                                                                         | \`false\`                 | Enforce strict MCP validation               |

\#\#\# \`Query\`

Interface returned by the \`query()\` function.

\`\`\`ts  
interface Query extends AsyncGenerator\<SDKMessage, void\> {  
  interrupt(): Promise\<void\>;  
  setPermissionMode(mode: PermissionMode): Promise\<void\>;  
}  
\`\`\`

\#\#\#\# Methods

| Method                | Description                                                          |  
| :-------------------- | :------------------------------------------------------------------- |  
| \`interrupt()\`         | Interrupts the query (only available in streaming input mode)        |  
| \`setPermissionMode()\` | Changes the permission mode (only available in streaming input mode) |

\#\#\# \`PermissionMode\`

\`\`\`ts  
type PermissionMode \=   
  | 'default'           // Standard permission behavior  
  | 'acceptEdits'       // Auto-accept file edits  
  | 'bypassPermissions' // Bypass all permission checks  
  | 'plan'              // Planning mode \- no execution  
\`\`\`

\#\#\# \`CanUseTool\`

Custom permission function type for controlling tool usage.

\`\`\`ts  
type CanUseTool \= (  
  toolName: string,  
  input: ToolInput,  
  options: {  
    signal: AbortSignal;  
    suggestions?: PermissionUpdate\[\];  
  }  
) \=\> Promise\<PermissionResult\>;  
\`\`\`

\#\#\# \`PermissionResult\`

Result of a permission check.

\`\`\`ts  
type PermissionResult \=   
  | {  
      behavior: 'allow';  
      updatedInput: ToolInput;  
      updatedPermissions?: PermissionUpdate\[\];  
    }  
  | {  
      behavior: 'deny';  
      message: string;  
      interrupt?: boolean;  
    }  
\`\`\`

\#\#\# \`McpServerConfig\`

Configuration for MCP servers.

\`\`\`ts  
type McpServerConfig \=   
  | McpStdioServerConfig  
  | McpSSEServerConfig  
  | McpHttpServerConfig  
  | McpSdkServerConfigWithInstance;  
\`\`\`

\#\#\#\# \`McpStdioServerConfig\`

\`\`\`ts  
type McpStdioServerConfig \= {  
  type?: 'stdio';  
  command: string;  
  args?: string\[\];  
  env?: Record\<string, string\>;  
}  
\`\`\`

\#\#\#\# \`McpSSEServerConfig\`

\`\`\`ts  
type McpSSEServerConfig \= {  
  type: 'sse';  
  url: string;  
  headers?: Record\<string, string\>;  
}  
\`\`\`

\#\#\#\# \`McpHttpServerConfig\`

\`\`\`ts  
type McpHttpServerConfig \= {  
  type: 'http';  
  url: string;  
  headers?: Record\<string, string\>;  
}  
\`\`\`

\#\#\#\# \`McpSdkServerConfigWithInstance\`

\`\`\`ts  
type McpSdkServerConfigWithInstance \= {  
  type: 'sdk';  
  name: string;  
  instance: McpServer;  
}  
\`\`\`

\#\# Message Types

\#\#\# \`SDKMessage\`

Union type of all possible messages returned by the query.

\`\`\`ts  
type SDKMessage \=   
  | SDKAssistantMessage  
  | SDKUserMessage  
  | SDKUserMessageReplay  
  | SDKResultMessage  
  | SDKSystemMessage  
  | SDKPartialAssistantMessage  
  | SDKCompactBoundaryMessage;  
\`\`\`

\#\#\# \`SDKAssistantMessage\`

Assistant response message.

\`\`\`ts  
type SDKAssistantMessage \= {  
  type: 'assistant';  
  uuid: UUID;  
  session\_id: string;  
  message: APIAssistantMessage; // From Anthropic SDK  
  parent\_tool\_use\_id: string | null;  
}  
\`\`\`

\#\#\# \`SDKUserMessage\`

User input message.

\`\`\`ts  
type SDKUserMessage \= {  
  type: 'user';  
  uuid?: UUID;  
  session\_id: string;  
  message: APIUserMessage; // From Anthropic SDK  
  parent\_tool\_use\_id: string | null;  
}  
\`\`\`

\#\#\# \`SDKUserMessageReplay\`

Replayed user message with required UUID.

\`\`\`ts  
type SDKUserMessageReplay \= {  
  type: 'user';  
  uuid: UUID;  
  session\_id: string;  
  message: APIUserMessage;  
  parent\_tool\_use\_id: string | null;  
}  
\`\`\`

\#\#\# \`SDKResultMessage\`

Final result message.

\`\`\`ts  
type SDKResultMessage \=   
  | {  
      type: 'result';  
      subtype: 'success';  
      uuid: UUID;  
      session\_id: string;  
      duration\_ms: number;  
      duration\_api\_ms: number;  
      is\_error: boolean;  
      num\_turns: number;  
      result: string;  
      total\_cost\_usd: number;  
      usage: NonNullableUsage;  
      permission\_denials: SDKPermissionDenial\[\];  
    }  
  | {  
      type: 'result';  
      subtype: 'error\_max\_turns' | 'error\_during\_execution';  
      uuid: UUID;  
      session\_id: string;  
      duration\_ms: number;  
      duration\_api\_ms: number;  
      is\_error: boolean;  
      num\_turns: number;  
      total\_cost\_usd: number;  
      usage: NonNullableUsage;  
      permission\_denials: SDKPermissionDenial\[\];  
    }  
\`\`\`

\#\#\# \`SDKSystemMessage\`

System initialization message.

\`\`\`ts  
type SDKSystemMessage \= {  
  type: 'system';  
  subtype: 'init';  
  uuid: UUID;  
  session\_id: string;  
  apiKeySource: ApiKeySource;  
  cwd: string;  
  tools: string\[\];  
  mcp\_servers: {  
    name: string;  
    status: string;  
  }\[\];  
  model: string;  
  permissionMode: PermissionMode;  
  slash\_commands: string\[\];  
  output\_style: string;  
}  
\`\`\`

\#\#\# \`SDKPartialAssistantMessage\`

Streaming partial message (only when \`includePartialMessages\` is true).

\`\`\`ts  
type SDKPartialAssistantMessage \= {  
  type: 'stream\_event';  
  event: RawMessageStreamEvent; // From Anthropic SDK  
  parent\_tool\_use\_id: string | null;  
  uuid: UUID;  
  session\_id: string;  
}  
\`\`\`

\#\#\# \`SDKCompactBoundaryMessage\`

Message indicating a conversation compaction boundary.

\`\`\`ts  
type SDKCompactBoundaryMessage \= {  
  type: 'system';  
  subtype: 'compact\_boundary';  
  uuid: UUID;  
  session\_id: string;  
  compact\_metadata: {  
    trigger: 'manual' | 'auto';  
    pre\_tokens: number;  
  };  
}  
\`\`\`

\#\#\# \`SDKPermissionDenial\`

Information about a denied tool use.

\`\`\`ts  
type SDKPermissionDenial \= {  
  tool\_name: string;  
  tool\_use\_id: string;  
  tool\_input: ToolInput;  
}  
\`\`\`

\#\# Hook Types

\#\#\# \`HookEvent\`

Available hook events.

\`\`\`ts  
type HookEvent \=   
  | 'PreToolUse'  
  | 'PostToolUse'  
  | 'Notification'  
  | 'UserPromptSubmit'  
  | 'SessionStart'  
  | 'SessionEnd'  
  | 'Stop'  
  | 'SubagentStop'  
  | 'PreCompact';  
\`\`\`

\#\#\# \`HookCallback\`

Hook callback function type.

\`\`\`ts  
type HookCallback \= (  
  input: HookInput, // Union of all hook input types  
  toolUseID: string | undefined,  
  options: { signal: AbortSignal }  
) \=\> Promise\<HookJSONOutput\>;  
\`\`\`

\#\#\# \`HookCallbackMatcher\`

Hook configuration with optional matcher.

\`\`\`ts  
interface HookCallbackMatcher {  
  matcher?: string;  
  hooks: HookCallback\[\];  
}  
\`\`\`

\#\#\# \`HookInput\`

Union type of all hook input types.

\`\`\`ts  
type HookInput \=   
  | PreToolUseHookInput  
  | PostToolUseHookInput  
  | NotificationHookInput  
  | UserPromptSubmitHookInput  
  | SessionStartHookInput  
  | SessionEndHookInput  
  | StopHookInput  
  | SubagentStopHookInput  
  | PreCompactHookInput;  
\`\`\`

\#\#\# \`BaseHookInput\`

Base interface that all hook input types extend.

\`\`\`ts  
type BaseHookInput \= {  
  session\_id: string;  
  transcript\_path: string;  
  cwd: string;  
  permission\_mode?: string;  
}  
\`\`\`

\#\#\#\# \`PreToolUseHookInput\`

\`\`\`ts  
type PreToolUseHookInput \= BaseHookInput & {  
  hook\_event\_name: 'PreToolUse';  
  tool\_name: string;  
  tool\_input: ToolInput;  
}  
\`\`\`

\#\#\#\# \`PostToolUseHookInput\`

\`\`\`ts  
type PostToolUseHookInput \= BaseHookInput & {  
  hook\_event\_name: 'PostToolUse';  
  tool\_name: string;  
  tool\_input: ToolInput;  
  tool\_response: ToolOutput;  
}  
\`\`\`

\#\#\#\# \`NotificationHookInput\`

\`\`\`ts  
type NotificationHookInput \= BaseHookInput & {  
  hook\_event\_name: 'Notification';  
  message: string;  
  title?: string;  
}  
\`\`\`

\#\#\#\# \`UserPromptSubmitHookInput\`

\`\`\`ts  
type UserPromptSubmitHookInput \= BaseHookInput & {  
  hook\_event\_name: 'UserPromptSubmit';  
  prompt: string;  
}  
\`\`\`

\#\#\#\# \`SessionStartHookInput\`

\`\`\`ts  
type SessionStartHookInput \= BaseHookInput & {  
  hook\_event\_name: 'SessionStart';  
  source: 'startup' | 'resume' | 'clear' | 'compact';  
}  
\`\`\`

\#\#\#\# \`SessionEndHookInput\`

\`\`\`ts  
type SessionEndHookInput \= BaseHookInput & {  
  hook\_event\_name: 'SessionEnd';  
  reason: 'clear' | 'logout' | 'prompt\_input\_exit' | 'other';  
}  
\`\`\`

\#\#\#\# \`StopHookInput\`

\`\`\`ts  
type StopHookInput \= BaseHookInput & {  
  hook\_event\_name: 'Stop';  
  stop\_hook\_active: boolean;  
}  
\`\`\`

\#\#\#\# \`SubagentStopHookInput\`

\`\`\`ts  
type SubagentStopHookInput \= BaseHookInput & {  
  hook\_event\_name: 'SubagentStop';  
  stop\_hook\_active: boolean;  
}  
\`\`\`

\#\#\#\# \`PreCompactHookInput\`

\`\`\`ts  
type PreCompactHookInput \= BaseHookInput & {  
  hook\_event\_name: 'PreCompact';  
  trigger: 'manual' | 'auto';  
  custom\_instructions: string | null;  
}  
\`\`\`

\#\#\# \`HookJSONOutput\`

Hook return value.

\`\`\`ts  
type HookJSONOutput \= AsyncHookJSONOutput | SyncHookJSONOutput;  
\`\`\`

\#\#\#\# \`AsyncHookJSONOutput\`

\`\`\`ts  
type AsyncHookJSONOutput \= {  
  async: true;  
  asyncTimeout?: number;  
}  
\`\`\`

\#\#\#\# \`SyncHookJSONOutput\`

\`\`\`ts  
type SyncHookJSONOutput \= {  
  continue?: boolean;  
  suppressOutput?: boolean;  
  stopReason?: string;  
  decision?: 'approve' | 'block';  
  systemMessage?: string;  
  reason?: string;  
  hookSpecificOutput?:  
    | {  
        hookEventName: 'PreToolUse';  
        permissionDecision?: 'allow' | 'deny' | 'ask';  
        permissionDecisionReason?: string;  
      }  
    | {  
        hookEventName: 'UserPromptSubmit';  
        additionalContext?: string;  
      }  
    | {  
        hookEventName: 'SessionStart';  
        additionalContext?: string;  
      }  
    | {  
        hookEventName: 'PostToolUse';  
        additionalContext?: string;  
      };  
}  
\`\`\`

\#\# Tool Input Types

Documentation of input schemas for all built-in Claude Code tools. These types are exported from \`@anthropic/claude-code-sdk\` and can be used for type-safe tool interactions.

\#\#\# \`ToolInput\`

\*\*Note:\*\* This is a documentation-only type for clarity. It represents the union of all tool input types.

\`\`\`ts  
type ToolInput \=   
  | AgentInput  
  | BashInput  
  | BashOutputInput  
  | FileEditInput  
  | FileMultiEditInput  
  | FileReadInput  
  | FileWriteInput  
  | GlobInput  
  | GrepInput  
  | KillShellInput  
  | NotebookEditInput  
  | WebFetchInput  
  | WebSearchInput  
  | TodoWriteInput  
  | ExitPlanModeInput  
  | ListMcpResourcesInput  
  | ReadMcpResourceInput;  
\`\`\`

\#\#\# Task

\*\*Tool name:\*\* \`Task\`

\`\`\`ts  
interface AgentInput {  
  /\*\*  
   \* A short (3-5 word) description of the task  
   \*/  
  description: string;  
  /\*\*  
   \* The task for the agent to perform  
   \*/  
  prompt: string;  
  /\*\*  
   \* The type of specialized agent to use for this task  
   \*/  
  subagent\_type: string;  
}  
\`\`\`

Launches a new agent to handle complex, multi-step tasks autonomously.

\#\#\# Bash

\*\*Tool name:\*\* \`Bash\`

\`\`\`ts  
interface BashInput {  
  /\*\*  
   \* The command to execute  
   \*/  
  command: string;  
  /\*\*  
   \* Optional timeout in milliseconds (max 600000\)  
   \*/  
  timeout?: number;  
  /\*\*  
   \* Clear, concise description of what this command does in 5-10 words  
   \*/  
  description?: string;  
  /\*\*  
   \* Set to true to run this command in the background  
   \*/  
  run\_in\_background?: boolean;  
}  
\`\`\`

Executes bash commands in a persistent shell session with optional timeout and background execution.

\#\#\# BashOutput

\*\*Tool name:\*\* \`BashOutput\`

\`\`\`ts  
interface BashOutputInput {  
  /\*\*  
   \* The ID of the background shell to retrieve output from  
   \*/  
  bash\_id: string;  
  /\*\*  
   \* Optional regex to filter output lines  
   \*/  
  filter?: string;  
}  
\`\`\`

Retrieves output from a running or completed background bash shell.

\#\#\# Edit

\*\*Tool name:\*\* \`Edit\`

\`\`\`ts  
interface FileEditInput {  
  /\*\*  
   \* The absolute path to the file to modify  
   \*/  
  file\_path: string;  
  /\*\*  
   \* The text to replace  
   \*/  
  old\_string: string;  
  /\*\*  
   \* The text to replace it with (must be different from old\_string)  
   \*/  
  new\_string: string;  
  /\*\*  
   \* Replace all occurrences of old\_string (default false)  
   \*/  
  replace\_all?: boolean;  
}  
\`\`\`

Performs exact string replacements in files.

\#\#\# MultiEdit

\*\*Tool name:\*\* \`MultiEdit\`

\`\`\`ts  
interface FileMultiEditInput {  
  /\*\*  
   \* The absolute path to the file to modify  
   \*/  
  file\_path: string;  
  /\*\*  
   \* Array of edit operations to perform sequentially  
   \*/  
  edits: Array\<{  
    /\*\*  
     \* The text to replace  
     \*/  
    old\_string: string;  
    /\*\*  
     \* The text to replace it with  
     \*/  
    new\_string: string;  
    /\*\*  
     \* Replace all occurrences (default false)  
     \*/  
    replace\_all?: boolean;  
  }\>;  
}  
\`\`\`

Makes multiple edits to a single file in one operation.

\#\#\# Read

\*\*Tool name:\*\* \`Read\`

\`\`\`ts  
interface FileReadInput {  
  /\*\*  
   \* The absolute path to the file to read  
   \*/  
  file\_path: string;  
  /\*\*  
   \* The line number to start reading from  
   \*/  
  offset?: number;  
  /\*\*  
   \* The number of lines to read  
   \*/  
  limit?: number;  
}  
\`\`\`

Reads files from the local filesystem, including text, images, PDFs, and Jupyter notebooks.

\#\#\# Write

\*\*Tool name:\*\* \`Write\`

\`\`\`ts  
interface FileWriteInput {  
  /\*\*  
   \* The absolute path to the file to write  
   \*/  
  file\_path: string;  
  /\*\*  
   \* The content to write to the file  
   \*/  
  content: string;  
}  
\`\`\`

Writes a file to the local filesystem, overwriting if it exists.

\#\#\# Glob

\*\*Tool name:\*\* \`Glob\`

\`\`\`ts  
interface GlobInput {  
  /\*\*  
   \* The glob pattern to match files against  
   \*/  
  pattern: string;  
  /\*\*  
   \* The directory to search in (defaults to cwd)  
   \*/  
  path?: string;  
}  
\`\`\`

Fast file pattern matching that works with any codebase size.

\#\#\# Grep

\*\*Tool name:\*\* \`Grep\`

\`\`\`ts  
interface GrepInput {  
  /\*\*  
   \* The regular expression pattern to search for  
   \*/  
  pattern: string;  
  /\*\*  
   \* File or directory to search in (defaults to cwd)  
   \*/  
  path?: string;  
  /\*\*  
   \* Glob pattern to filter files (e.g. "\*.js")  
   \*/  
  glob?: string;  
  /\*\*  
   \* File type to search (e.g. "js", "py", "rust")  
   \*/  
  type?: string;  
  /\*\*  
   \* Output mode: "content", "files\_with\_matches", or "count"  
   \*/  
  output\_mode?: 'content' | 'files\_with\_matches' | 'count';  
  /\*\*  
   \* Case insensitive search  
   \*/  
  '-i'?: boolean;  
  /\*\*  
   \* Show line numbers (for content mode)  
   \*/  
  '-n'?: boolean;  
  /\*\*  
   \* Lines to show before each match  
   \*/  
  '-B'?: number;  
  /\*\*  
   \* Lines to show after each match  
   \*/  
  '-A'?: number;  
  /\*\*  
   \* Lines to show before and after each match  
   \*/  
  '-C'?: number;  
  /\*\*  
   \* Limit output to first N lines/entries  
   \*/  
  head\_limit?: number;  
  /\*\*  
   \* Enable multiline mode  
   \*/  
  multiline?: boolean;  
}  
\`\`\`

Powerful search tool built on ripgrep with regex support.

\#\#\# KillBash

\*\*Tool name:\*\* \`KillBash\`

\`\`\`ts  
interface KillShellInput {  
  /\*\*  
   \* The ID of the background shell to kill  
   \*/  
  shell\_id: string;  
}  
\`\`\`

Kills a running background bash shell by its ID.

\#\#\# NotebookEdit

\*\*Tool name:\*\* \`NotebookEdit\`

\`\`\`ts  
interface NotebookEditInput {  
  /\*\*  
   \* The absolute path to the Jupyter notebook file  
   \*/  
  notebook\_path: string;  
  /\*\*  
   \* The ID of the cell to edit  
   \*/  
  cell\_id?: string;  
  /\*\*  
   \* The new source for the cell  
   \*/  
  new\_source: string;  
  /\*\*  
   \* The type of the cell (code or markdown)  
   \*/  
  cell\_type?: 'code' | 'markdown';  
  /\*\*  
   \* The type of edit (replace, insert, delete)  
   \*/  
  edit\_mode?: 'replace' | 'insert' | 'delete';  
}  
\`\`\`

Edits cells in Jupyter notebook files.

\#\#\# WebFetch

\*\*Tool name:\*\* \`WebFetch\`

\`\`\`ts  
interface WebFetchInput {  
  /\*\*  
   \* The URL to fetch content from  
   \*/  
  url: string;  
  /\*\*  
   \* The prompt to run on the fetched content  
   \*/  
  prompt: string;  
}  
\`\`\`

Fetches content from a URL and processes it with an AI model.

\#\#\# WebSearch

\*\*Tool name:\*\* \`WebSearch\`

\`\`\`ts  
interface WebSearchInput {  
  /\*\*  
   \* The search query to use  
   \*/  
  query: string;  
  /\*\*  
   \* Only include results from these domains  
   \*/  
  allowed\_domains?: string\[\];  
  /\*\*  
   \* Never include results from these domains  
   \*/  
  blocked\_domains?: string\[\];  
}  
\`\`\`

Searches the web and returns formatted results.

\#\#\# TodoWrite

\*\*Tool name:\*\* \`TodoWrite\`

\`\`\`ts  
interface TodoWriteInput {  
  /\*\*  
   \* The updated todo list  
   \*/  
  todos: Array\<{  
    /\*\*  
     \* The task description  
     \*/  
    content: string;  
    /\*\*  
     \* The task status  
     \*/  
    status: 'pending' | 'in\_progress' | 'completed';  
    /\*\*  
     \* Active form of the task description  
     \*/  
    activeForm: string;  
  }\>;  
}  
\`\`\`

Creates and manages a structured task list for tracking progress.

\#\#\# ExitPlanMode

\*\*Tool name:\*\* \`ExitPlanMode\`

\`\`\`ts  
interface ExitPlanModeInput {  
  /\*\*  
   \* The plan to run by the user for approval  
   \*/  
  plan: string;  
}  
\`\`\`

Exits planning mode and prompts the user to approve the plan.

\#\#\# ListMcpResources

\*\*Tool name:\*\* \`ListMcpResources\`

\`\`\`ts  
interface ListMcpResourcesInput {  
  /\*\*  
   \* Optional server name to filter resources by  
   \*/  
  server?: string;  
}  
\`\`\`

Lists available MCP resources from connected servers.

\#\#\# ReadMcpResource

\*\*Tool name:\*\* \`ReadMcpResource\`

\`\`\`ts  
interface ReadMcpResourceInput {  
  /\*\*  
   \* The MCP server name  
   \*/  
  server: string;  
  /\*\*  
   \* The resource URI to read  
   \*/  
  uri: string;  
}  
\`\`\`

Reads a specific MCP resource from a server.

\#\# Tool Output Types

Documentation of output schemas for all built-in Claude Code tools. These types represent the actual response data returned by each tool.

\#\#\# \`ToolOutput\`

\*\*Note:\*\* This is a documentation-only type for clarity. It represents the union of all tool output types.

\`\`\`ts  
type ToolOutput \=   
  | TaskOutput  
  | BashOutput  
  | BashOutputToolOutput  
  | EditOutput  
  | MultiEditOutput  
  | ReadOutput  
  | WriteOutput  
  | GlobOutput  
  | GrepOutput  
  | KillBashOutput  
  | NotebookEditOutput  
  | WebFetchOutput  
  | WebSearchOutput  
  | TodoWriteOutput  
  | ExitPlanModeOutput  
  | ListMcpResourcesOutput  
  | ReadMcpResourceOutput;  
\`\`\`

\#\#\# Task

\*\*Tool name:\*\* \`Task\`

\`\`\`ts  
interface TaskOutput {  
  /\*\*  
   \* Final result message from the subagent  
   \*/  
  result: string;  
  /\*\*  
   \* Token usage statistics  
   \*/  
  usage?: {  
    input\_tokens: number;  
    output\_tokens: number;  
    cache\_creation\_input\_tokens?: number;  
    cache\_read\_input\_tokens?: number;  
  };  
  /\*\*  
   \* Total cost in USD  
   \*/  
  total\_cost\_usd?: number;  
  /\*\*  
   \* Execution duration in milliseconds  
   \*/  
  duration\_ms?: number;  
}  
\`\`\`

Returns the final result from the subagent after completing the delegated task.

\#\#\# Bash

\*\*Tool name:\*\* \`Bash\`

\`\`\`ts  
interface BashOutput {  
  /\*\*  
   \* Combined stdout and stderr output  
   \*/  
  output: string;  
  /\*\*  
   \* Exit code of the command  
   \*/  
  exitCode: number;  
  /\*\*  
   \* Whether the command was killed due to timeout  
   \*/  
  killed?: boolean;  
  /\*\*  
   \* Shell ID for background processes  
   \*/  
  shellId?: string;  
}  
\`\`\`

Returns command output with exit status. Background commands return immediately with a shellId.

\#\#\# BashOutput

\*\*Tool name:\*\* \`BashOutput\`

\`\`\`ts  
interface BashOutputToolOutput {  
  /\*\*  
   \* New output since last check  
   \*/  
  output: string;  
  /\*\*  
   \* Current shell status  
   \*/  
  status: 'running' | 'completed' | 'failed';  
  /\*\*  
   \* Exit code (when completed)  
   \*/  
  exitCode?: number;  
}  
\`\`\`

Returns incremental output from background shells.

\#\#\# Edit

\*\*Tool name:\*\* \`Edit\`

\`\`\`ts  
interface EditOutput {  
  /\*\*  
   \* Confirmation message  
   \*/  
  message: string;  
  /\*\*  
   \* Number of replacements made  
   \*/  
  replacements: number;  
  /\*\*  
   \* File path that was edited  
   \*/  
  file\_path: string;  
}  
\`\`\`

Returns confirmation of successful edits with replacement count.

\#\#\# MultiEdit

\*\*Tool name:\*\* \`MultiEdit\`

\`\`\`ts  
interface MultiEditOutput {  
  /\*\*  
   \* Success message  
   \*/  
  message: string;  
  /\*\*  
   \* Total number of edits applied  
   \*/  
  edits\_applied: number;  
  /\*\*  
   \* File path that was edited  
   \*/  
  file\_path: string;  
}  
\`\`\`

Returns confirmation after applying all edits sequentially.

\#\#\# Read

\*\*Tool name:\*\* \`Read\`

\`\`\`ts  
type ReadOutput \=   
  | TextFileOutput  
  | ImageFileOutput  
  | PDFFileOutput  
  | NotebookFileOutput;

interface TextFileOutput {  
  /\*\*  
   \* File contents with line numbers  
   \*/  
  content: string;  
  /\*\*  
   \* Total number of lines in file  
   \*/  
  total\_lines: number;  
  /\*\*  
   \* Lines actually returned  
   \*/  
  lines\_returned: number;  
}

interface ImageFileOutput {  
  /\*\*  
   \* Base64 encoded image data  
   \*/  
  image: string;  
  /\*\*  
   \* Image MIME type  
   \*/  
  mime\_type: string;  
  /\*\*  
   \* File size in bytes  
   \*/  
  file\_size: number;  
}

interface PDFFileOutput {  
  /\*\*  
   \* Array of page contents  
   \*/  
  pages: Array\<{  
    page\_number: number;  
    text?: string;  
    images?: Array\<{  
      image: string;  
      mime\_type: string;  
    }\>;  
  }\>;  
  /\*\*  
   \* Total number of pages  
   \*/  
  total\_pages: number;  
}

interface NotebookFileOutput {  
  /\*\*  
   \* Jupyter notebook cells  
   \*/  
  cells: Array\<{  
    cell\_type: 'code' | 'markdown';  
    source: string;  
    outputs?: any\[\];  
    execution\_count?: number;  
  }\>;  
  /\*\*  
   \* Notebook metadata  
   \*/  
  metadata?: Record\<string, any\>;  
}  
\`\`\`

Returns file contents in format appropriate to file type.

\#\#\# Write

\*\*Tool name:\*\* \`Write\`

\`\`\`ts  
interface WriteOutput {  
  /\*\*  
   \* Success message  
   \*/  
  message: string;  
  /\*\*  
   \* Number of bytes written  
   \*/  
  bytes\_written: number;  
  /\*\*  
   \* File path that was written  
   \*/  
  file\_path: string;  
}  
\`\`\`

Returns confirmation after successfully writing the file.

\#\#\# Glob

\*\*Tool name:\*\* \`Glob\`

\`\`\`ts  
interface GlobOutput {  
  /\*\*  
   \* Array of matching file paths  
   \*/  
  matches: string\[\];  
  /\*\*  
   \* Number of matches found  
   \*/  
  count: number;  
  /\*\*  
   \* Search directory used  
   \*/  
  search\_path: string;  
}  
\`\`\`

Returns file paths matching the glob pattern, sorted by modification time.

\#\#\# Grep

\*\*Tool name:\*\* \`Grep\`

\`\`\`ts  
type GrepOutput \=   
  | GrepContentOutput  
  | GrepFilesOutput  
  | GrepCountOutput;

interface GrepContentOutput {  
  /\*\*  
   \* Matching lines with context  
   \*/  
  matches: Array\<{  
    file: string;  
    line\_number?: number;  
    line: string;  
    before\_context?: string\[\];  
    after\_context?: string\[\];  
  }\>;  
  /\*\*  
   \* Total number of matches  
   \*/  
  total\_matches: number;  
}

interface GrepFilesOutput {  
  /\*\*  
   \* Files containing matches  
   \*/  
  files: string\[\];  
  /\*\*  
   \* Number of files with matches  
   \*/  
  count: number;  
}

interface GrepCountOutput {  
  /\*\*  
   \* Match counts per file  
   \*/  
  counts: Array\<{  
    file: string;  
    count: number;  
  }\>;  
  /\*\*  
   \* Total matches across all files  
   \*/  
  total: number;  
}  
\`\`\`

Returns search results in the format specified by output\\\_mode.

\#\#\# KillBash

\*\*Tool name:\*\* \`KillBash\`

\`\`\`ts  
interface KillBashOutput {  
  /\*\*  
   \* Success message  
   \*/  
  message: string;  
  /\*\*  
   \* ID of the killed shell  
   \*/  
  shell\_id: string;  
}  
\`\`\`

Returns confirmation after terminating the background shell.

\#\#\# NotebookEdit

\*\*Tool name:\*\* \`NotebookEdit\`

\`\`\`ts  
interface NotebookEditOutput {  
  /\*\*  
   \* Success message  
   \*/  
  message: string;  
  /\*\*  
   \* Type of edit performed  
   \*/  
  edit\_type: 'replaced' | 'inserted' | 'deleted';  
  /\*\*  
   \* Cell ID that was affected  
   \*/  
  cell\_id?: string;  
  /\*\*  
   \* Total cells in notebook after edit  
   \*/  
  total\_cells: number;  
}  
\`\`\`

Returns confirmation after modifying the Jupyter notebook.

\#\#\# WebFetch

\*\*Tool name:\*\* \`WebFetch\`

\`\`\`ts  
interface WebFetchOutput {  
  /\*\*  
   \* AI model's response to the prompt  
   \*/  
  response: string;  
  /\*\*  
   \* URL that was fetched  
   \*/  
  url: string;  
  /\*\*  
   \* Final URL after redirects  
   \*/  
  final\_url?: string;  
  /\*\*  
   \* HTTP status code  
   \*/  
  status\_code?: number;  
}  
\`\`\`

Returns the AI's analysis of the fetched web content.

\#\#\# WebSearch

\*\*Tool name:\*\* \`WebSearch\`

\`\`\`ts  
interface WebSearchOutput {  
  /\*\*  
   \* Search results  
   \*/  
  results: Array\<{  
    title: string;  
    url: string;  
    snippet: string;  
    /\*\*  
     \* Additional metadata if available  
     \*/  
    metadata?: Record\<string, any\>;  
  }\>;  
  /\*\*  
   \* Total number of results  
   \*/  
  total\_results: number;  
  /\*\*  
   \* The query that was searched  
   \*/  
  query: string;  
}  
\`\`\`

Returns formatted search results from the web.

\#\#\# TodoWrite

\*\*Tool name:\*\* \`TodoWrite\`

\`\`\`ts  
interface TodoWriteOutput {  
  /\*\*  
   \* Success message  
   \*/  
  message: string;  
  /\*\*  
   \* Current todo statistics  
   \*/  
  stats: {  
    total: number;  
    pending: number;  
    in\_progress: number;  
    completed: number;  
  };  
}  
\`\`\`

Returns confirmation with current task statistics.

\#\#\# ExitPlanMode

\*\*Tool name:\*\* \`ExitPlanMode\`

\`\`\`ts  
interface ExitPlanModeOutput {  
  /\*\*  
   \* Confirmation message  
   \*/  
  message: string;  
  /\*\*  
   \* Whether user approved the plan  
   \*/  
  approved?: boolean;  
}  
\`\`\`

Returns confirmation after exiting plan mode.

\#\#\# ListMcpResources

\*\*Tool name:\*\* \`ListMcpResources\`

\`\`\`ts  
interface ListMcpResourcesOutput {  
  /\*\*  
   \* Available resources  
   \*/  
  resources: Array\<{  
    uri: string;  
    name: string;  
    description?: string;  
    mimeType?: string;  
    server: string;  
  }\>;  
  /\*\*  
   \* Total number of resources  
   \*/  
  total: number;  
}  
\`\`\`

Returns list of available MCP resources.

\#\#\# ReadMcpResource

\*\*Tool name:\*\* \`ReadMcpResource\`

\`\`\`ts  
interface ReadMcpResourceOutput {  
  /\*\*  
   \* Resource contents  
   \*/  
  contents: Array\<{  
    uri: string;  
    mimeType?: string;  
    text?: string;  
    blob?: string;  
  }\>;  
  /\*\*  
   \* Server that provided the resource  
   \*/  
  server: string;  
}  
\`\`\`

Returns the contents of the requested MCP resource.

\#\# Permission Types

\#\#\# \`PermissionUpdate\`

Operations for updating permissions.

\`\`\`ts  
type PermissionUpdate \=   
  | {  
      type: 'addRules';  
      rules: PermissionRuleValue\[\];  
      behavior: PermissionBehavior;  
      destination: PermissionUpdateDestination;  
    }  
  | {  
      type: 'replaceRules';  
      rules: PermissionRuleValue\[\];  
      behavior: PermissionBehavior;  
      destination: PermissionUpdateDestination;  
    }  
  | {  
      type: 'removeRules';  
      rules: PermissionRuleValue\[\];  
      behavior: PermissionBehavior;  
      destination: PermissionUpdateDestination;  
    }  
  | {  
      type: 'setMode';  
      mode: PermissionMode;  
      destination: PermissionUpdateDestination;  
    }  
  | {  
      type: 'addDirectories';  
      directories: string\[\];  
      destination: PermissionUpdateDestination;  
    }  
  | {  
      type: 'removeDirectories';  
      directories: string\[\];  
      destination: PermissionUpdateDestination;  
    }  
\`\`\`

\#\#\# \`PermissionBehavior\`

\`\`\`ts  
type PermissionBehavior \= 'allow' | 'deny' | 'ask';  
\`\`\`

\#\#\# \`PermissionUpdateDestination\`

\`\`\`ts  
type PermissionUpdateDestination \=   
  | 'userSettings'     // Global user settings  
  | 'projectSettings'  // Per-directory project settings  
  | 'localSettings'    // Gitignored local settings  
  | 'session'          // Current session only  
\`\`\`

\#\#\# \`PermissionRuleValue\`

\`\`\`ts  
type PermissionRuleValue \= {  
  toolName: string;  
  ruleContent?: string;  
}  
\`\`\`

\#\# Other Types

\#\#\# \`ApiKeySource\`

\`\`\`ts  
type ApiKeySource \= 'user' | 'project' | 'org' | 'temporary';  
\`\`\`

\#\#\# \`ConfigScope\`

\`\`\`ts  
type ConfigScope \= 'local' | 'user' | 'project';  
\`\`\`

\#\#\# \`NonNullableUsage\`

A version of \[\`Usage\`\](\#usage) with all nullable fields made non-nullable.

\`\`\`ts  
type NonNullableUsage \= {  
  \[K in keyof Usage\]: NonNullable\<Usage\[K\]\>;  
}  
\`\`\`

\#\#\# \`Usage\`

Token usage statistics (from \`@anthropic-ai/sdk\`).

\`\`\`ts  
type Usage \= {  
  input\_tokens: number | null;  
  output\_tokens: number | null;  
  cache\_creation\_input\_tokens?: number | null;  
  cache\_read\_input\_tokens?: number | null;  
}  
\`\`\`

\#\#\# \`CallToolResult\`

MCP tool result type (from \`@modelcontextprotocol/sdk/types.js\`).

\`\`\`ts  
type CallToolResult \= {  
  content: Array\<{  
    type: 'text' | 'image' | 'resource';  
    // Additional fields vary by type  
  }\>;  
  isError?: boolean;  
}  
\`\`\`

\#\#\# \`AbortError\`

Custom error class for abort operations.

\`\`\`ts  
class AbortError extends Error {}  
\`\`\`

\#\# See also

\* \[TypeScript SDK guide\](/en/docs/claude-code/sdk/sdk-typescript) \- Tutorial and examples  
\* \[SDK overview\](/en/docs/claude-code/sdk/sdk-overview) \- General SDK concepts  
\* \[Python SDK reference\](/en/docs/claude-code/sdk/sdk-python) \- Python SDK documentation  
\* \[CLI reference\](/en/docs/claude-code/cli-reference) \- Command-line interface  
\* \[Common workflows\](/en/docs/claude-code/common-workflows) \- Step-by-step guides  
