Python SDK reference

\> Complete API reference for the Claude Code Python SDK, including all functions, types, and classes.

\#\# Functions

\#\#\# \`query()\`

The primary async function for interacting with Claude Code. Returns an async iterator that yields messages as they arrive.

\`\`\`python  
async def query(  
    \*,  
    prompt: str | AsyncIterable\[dict\[str, Any\]\],  
    options: ClaudeCodeOptions | None \= None  
) \-\> AsyncIterator\[Message\]  
\`\`\`

\#\#\#\# Parameters

| Parameter | Type                         | Description                                                               |  
| :-------- | :--------------------------- | :------------------------------------------------------------------------ |  
| \`prompt\`  | \`str \\| AsyncIterable\[dict\]\` | The input prompt as a string or async iterable for streaming mode         |  
| \`options\` | \`ClaudeCodeOptions \\| None\`  | Optional configuration object (defaults to \`ClaudeCodeOptions()\` if None) |

\#\#\#\# Returns

Returns an \`AsyncIterator\[Message\]\` that yields messages from the conversation.

\#\#\#\# Example \- Simple query

\`\`\`python  
import asyncio  
from claude\_code\_sdk import query

async def main():  
    async for message in query(prompt="What is 2+2?"):  
        print(message)

asyncio.run(main())  
\`\`\`

\#\#\#\# Example \- With options

\`\`\`python

import asyncio  
from claude\_code\_sdk import query, ClaudeCodeOptions

async def main():  
    options \= ClaudeCodeOptions(  
        system\_prompt="You are an expert Python developer",  
        permission\_mode='acceptEdits',  
        cwd="/home/user/project"  
    )

    async for message in query(  
        prompt="Create a Python web server",  
        options=options  
    ):  
        print(message)

asyncio.run(main())  
\`\`\`

\#\#\# \`tool()\`

Decorator for defining MCP tools with type safety.

\`\`\`python  
def tool(  
    name: str,  
    description: str,  
    input\_schema: type | dict\[str, Any\]  
) \-\> Callable\[\[Callable\[\[Any\], Awaitable\[dict\[str, Any\]\]\]\], SdkMcpTool\[Any\]\]  
\`\`\`

\#\#\#\# Parameters

| Parameter      | Type                     | Description                                             |  
| :------------- | :----------------------- | :------------------------------------------------------ |  
| \`name\`         | \`str\`                    | Unique identifier for the tool                          |  
| \`description\`  | \`str\`                    | Human-readable description of what the tool does        |  
| \`input\_schema\` | \`type \\| dict\[str, Any\]\` | Schema defining the tool's input parameters (see below) |

\#\#\#\# Input Schema Options

1\. \*\*Simple type mapping\*\* (recommended):  
   \`\`\`python  
   {"text": str, "count": int, "enabled": bool}  
   \`\`\`

2\. \*\*JSON Schema format\*\* (for complex validation):  
   \`\`\`python  
   {  
       "type": "object",  
       "properties": {  
           "text": {"type": "string"},  
           "count": {"type": "integer", "minimum": 0}  
       },  
       "required": \["text"\]  
   }  
   \`\`\`

\#\#\#\# Returns

A decorator function that wraps the tool implementation and returns an \`SdkMcpTool\` instance.

\#\#\#\# Example

\`\`\`python  
from claude\_code\_sdk import tool  
from typing import Any

@tool("greet", "Greet a user", {"name": str})  
async def greet(args: dict\[str, Any\]) \-\> dict\[str, Any\]:  
    return {  
        "content": \[{  
            "type": "text",  
            "text": f"Hello, {args\['name'\]}\!"  
        }\]  
    }  
\`\`\`

\#\#\# \`create\_sdk\_mcp\_server()\`

Create an in-process MCP server that runs within your Python application.

\`\`\`python  
def create\_sdk\_mcp\_server(  
    name: str,  
    version: str \= "1.0.0",  
    tools: list\[SdkMcpTool\[Any\]\] | None \= None  
) \-\> McpSdkServerConfig  
\`\`\`

\#\#\#\# Parameters

| Parameter | Type                            | Default   | Description                                           |  
| :-------- | :------------------------------ | :-------- | :---------------------------------------------------- |  
| \`name\`    | \`str\`                           | \-         | Unique identifier for the server                      |  
| \`version\` | \`str\`                           | \`"1.0.0"\` | Server version string                                 |  
| \`tools\`   | \`list\[SdkMcpTool\[Any\]\] \\| None\` | \`None\`    | List of tool functions created with \`@tool\` decorator |

\#\#\#\# Returns

Returns an \`McpSdkServerConfig\` object that can be passed to \`ClaudeCodeOptions.mcp\_servers\`.

\#\#\#\# Example

\`\`\`python  
from claude\_code\_sdk import tool, create\_sdk\_mcp\_server

@tool("add", "Add two numbers", {"a": float, "b": float})  
async def add(args):  
    return {  
        "content": \[{  
            "type": "text",  
            "text": f"Sum: {args\['a'\] \+ args\['b'\]}"  
        }\]  
    }

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})  
async def multiply(args):  
    return {  
        "content": \[{  
            "type": "text",  
            "text": f"Product: {args\['a'\] \* args\['b'\]}"  
        }\]  
    }

calculator \= create\_sdk\_mcp\_server(  
    name="calculator",  
    version="2.0.0",  
    tools=\[add, multiply\]  \# Pass decorated functions  
)

\# Use with Claude  
options \= ClaudeCodeOptions(  
    mcp\_servers={"calc": calculator},  
    allowed\_tools=\["mcp\_\_calc\_\_add", "mcp\_\_calc\_\_multiply"\]  
)  
\`\`\`

\#\# Classes

\#\#\# \`ClaudeSDKClient\`

Client for bidirectional, interactive conversations with Claude Code. Provides full control over conversation flow with support for streaming, interrupts, and dynamic message sending.

\`\`\`python  
class ClaudeSDKClient:  
    def \_\_init\_\_(self, options: ClaudeCodeOptions | None \= None)  
    async def connect(self, prompt: str | AsyncIterable\[dict\] | None \= None) \-\> None  
    async def query(self, prompt: str | AsyncIterable\[dict\], session\_id: str \= "default") \-\> None  
    async def receive\_messages(self) \-\> AsyncIterator\[Message\]  
    async def receive\_response(self) \-\> AsyncIterator\[Message\]  
    async def interrupt(self) \-\> None  
    async def disconnect(self) \-\> None  
\`\`\`

\#\#\#\# Methods

| Method                      | Description                                                         |  
| :-------------------------- | :------------------------------------------------------------------ |  
| \`\_\_init\_\_(options)\`         | Initialize the client with optional configuration                   |  
| \`connect(prompt)\`           | Connect to Claude with an optional initial prompt or message stream |  
| \`query(prompt, session\_id)\` | Send a new request in streaming mode                                |  
| \`receive\_messages()\`        | Receive all messages from Claude as an async iterator               |  
| \`receive\_response()\`        | Receive messages until and including a ResultMessage                |  
| \`interrupt()\`               | Send interrupt signal (only works in streaming mode)                |  
| \`disconnect()\`              | Disconnect from Claude                                              |

\#\#\#\# Context Manager Support

The client can be used as an async context manager for automatic connection management:

\`\`\`python  
async with ClaudeSDKClient() as client:  
    await client.query("Hello Claude")  
    async for message in client.receive\_response():  
        print(message)  
\`\`\`

\> \*\*Important:\*\* When iterating over messages, avoid using \`break\` to exit early as this can cause asyncio cleanup issues. Instead, let the iteration complete naturally or use flags to track when you've found what you need.

\#\#\#\# Example \- Interactive conversation

\`\`\`python  
import asyncio  
from claude\_code\_sdk import ClaudeSDKClient, AssistantMessage, TextBlock, ResultMessage

async def main():  
    async with ClaudeSDKClient() as client:  
        \# Send initial message  
        await client.query("Let's solve a math problem step by step")  
          
        \# Receive and process response  
        async for message in client.receive\_response():  
            if isinstance(message, AssistantMessage):  
                for block in message.content:  
                    if isinstance(block, TextBlock):  
                        print(f"Assistant: {block.text\[:100\]}...")  
            elif isinstance(message, ResultMessage):  
                print("Response complete")  
          
        \# Send follow-up based on response  
        await client.query("What's 15% of 80?")

asyncio.run(main())  
\`\`\`

\#\# Types

\#\#\# \`SdkMcpTool\`

Definition for an SDK MCP tool created with the \`@tool\` decorator.

\`\`\`python  
@dataclass  
class SdkMcpTool(Generic\[T\]):  
    name: str  
    description: str  
    input\_schema: type\[T\] | dict\[str, Any\]  
    handler: Callable\[\[T\], Awaitable\[dict\[str, Any\]\]\]  
\`\`\`

| Property       | Type                                       | Description                                |  
| :------------- | :----------------------------------------- | :----------------------------------------- |  
| \`name\`         | \`str\`                                      | Unique identifier for the tool             |  
| \`description\`  | \`str\`                                      | Human-readable description                 |  
| \`input\_schema\` | \`type\[T\] \\| dict\[str, Any\]\`                | Schema for input validation                |  
| \`handler\`      | \`Callable\[\[T\], Awaitable\[dict\[str, Any\]\]\]\` | Async function that handles tool execution |

\#\#\# \`ClaudeCodeOptions\`

Configuration dataclass for Claude Code queries.

\`\`\`python  
@dataclass  
class ClaudeCodeOptions:  
    allowed\_tools: list\[str\] \= field(default\_factory=list)  
    max\_thinking\_tokens: int \= 8000  
    system\_prompt: str | None \= None  
    append\_system\_prompt: str | None \= None  
    mcp\_servers: dict\[str, McpServerConfig\] | str | Path \= field(default\_factory=dict)  
    permission\_mode: PermissionMode | None \= None  
    continue\_conversation: bool \= False  
    resume: str | None \= None  
    max\_turns: int | None \= None  
    disallowed\_tools: list\[str\] \= field(default\_factory=list)  
    model: str | None \= None  
    permission\_prompt\_tool\_name: str | None \= None  
    cwd: str | Path | None \= None  
    settings: str | None \= None  
    add\_dirs: list\[str | Path\] \= field(default\_factory=list)  
    env: dict\[str, str\] \= field(default\_factory=dict)  
    extra\_args: dict\[str, str | None\] \= field(default\_factory=dict)  
\`\`\`

| Property                      | Type                                         | Default | Description                                          |  
| :---------------------------- | :------------------------------------------- | :------ | :--------------------------------------------------- |  
| \`allowed\_tools\`               | \`list\[str\]\`                                  | \`\[\]\`    | List of allowed tool names                           |  
| \`max\_thinking\_tokens\`         | \`int\`                                        | \`8000\`  | Maximum tokens for thinking process                  |  
| \`system\_prompt\`               | \`str \\| None\`                                | \`None\`  | Replace the default system prompt entirely           |  
| \`append\_system\_prompt\`        | \`str \\| None\`                                | \`None\`  | Text to append to the default system prompt          |  
| \`mcp\_servers\`                 | \`dict\[str, McpServerConfig\] \\| str \\| Path\`  | \`{}\`    | MCP server configurations or path to config file     |  
| \`permission\_mode\`             | \`PermissionMode \\| None\`                     | \`None\`  | Permission mode for tool usage                       |  
| \`continue\_conversation\`       | \`bool\`                                       | \`False\` | Continue the most recent conversation                |  
| \`resume\`                      | \`str \\| None\`                                | \`None\`  | Session ID to resume                                 |  
| \`max\_turns\`                   | \`int \\| None\`                                | \`None\`  | Maximum conversation turns                           |  
| \`disallowed\_tools\`            | \`list\[str\]\`                                  | \`\[\]\`    | List of disallowed tool names                        |  
| \`model\`                       | \`str \\| None\`                                | \`None\`  | Claude model to use                                  |  
| \`permission\_prompt\_tool\_name\` | \`str \\| None\`                                | \`None\`  | MCP tool name for permission prompts                 |  
| \`cwd\`                         | \`str \\| Path \\| None\`                        | \`None\`  | Current working directory                            |  
| \`settings\`                    | \`str \\| None\`                                | \`None\`  | Path to settings file                                |  
| \`add\_dirs\`                    | \`list\[str \\| Path\]\`                          | \`\[\]\`    | Additional directories Claude can access             |  
| \`extra\_args\`                  | \`dict\[str, str \\| None\]\`                     | \`{}\`    | Additional CLI arguments to pass directly to the CLI |  
| \`can\_use\_tool\`                | \`CanUseTool \\| None\`                         | \`None\`  | Tool permission callback function                    |  
| \`hooks\`                       | \`dict\[HookEvent, list\[HookMatcher\]\] \\| None\` | \`None\`  | Hook configurations for intercepting events          |

\#\#\# \`PermissionMode\`

Permission modes for controlling tool execution.

\`\`\`python  
PermissionMode \= Literal\[  
    "default",           \# Standard permission behavior  
    "acceptEdits",       \# Auto-accept file edits  
    "plan",              \# Planning mode \- no execution  
    "bypassPermissions"  \# Bypass all permission checks (use with caution)  
\]  
\`\`\`

\#\#\# \`McpSdkServerConfig\`

Configuration for SDK MCP servers created with \`create\_sdk\_mcp\_server()\`.

\`\`\`python  
class McpSdkServerConfig(TypedDict):  
    type: Literal\["sdk"\]  
    name: str  
    instance: Any  \# MCP Server instance  
\`\`\`

\#\#\# \`McpServerConfig\`

Union type for MCP server configurations.

\`\`\`python  
McpServerConfig \= McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig | McpSdkServerConfig  
\`\`\`

\#\#\#\# \`McpStdioServerConfig\`

\`\`\`python  
class McpStdioServerConfig(TypedDict):  
    type: NotRequired\[Literal\["stdio"\]\]  \# Optional for backwards compatibility  
    command: str  
    args: NotRequired\[list\[str\]\]  
    env: NotRequired\[dict\[str, str\]\]  
\`\`\`

\#\#\#\# \`McpSSEServerConfig\`

\`\`\`python  
class McpSSEServerConfig(TypedDict):  
    type: Literal\["sse"\]  
    url: str  
    headers: NotRequired\[dict\[str, str\]\]  
\`\`\`

\#\#\#\# \`McpHttpServerConfig\`

\`\`\`python  
class McpHttpServerConfig(TypedDict):  
    type: Literal\["http"\]  
    url: str  
    headers: NotRequired\[dict\[str, str\]\]  
\`\`\`

\#\# Message Types

\#\#\# \`Message\`

Union type of all possible messages.

\`\`\`python  
Message \= UserMessage | AssistantMessage | SystemMessage | ResultMessage  
\`\`\`

\#\#\# \`UserMessage\`

User input message.

\`\`\`python  
@dataclass  
class UserMessage:  
    content: str | list\[ContentBlock\]  
\`\`\`

\#\#\# \`AssistantMessage\`

Assistant response message with content blocks.

\`\`\`python  
@dataclass  
class AssistantMessage:  
    content: list\[ContentBlock\]  
    model: str  
\`\`\`

\#\#\# \`SystemMessage\`

System message with metadata.

\`\`\`python  
@dataclass  
class SystemMessage:  
    subtype: str  
    data: dict\[str, Any\]  
\`\`\`

\#\#\# \`ResultMessage\`

Final result message with cost and usage information.

\`\`\`python  
@dataclass  
class ResultMessage:  
    subtype: str  
    duration\_ms: int  
    duration\_api\_ms: int  
    is\_error: bool  
    num\_turns: int  
    session\_id: str  
    total\_cost\_usd: float | None \= None  
    usage: dict\[str, Any\] | None \= None  
    result: str | None \= None  
\`\`\`

\#\# Content Block Types

\#\#\# \`ContentBlock\`

Union type of all content blocks.

\`\`\`python  
ContentBlock \= TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock  
\`\`\`

\#\#\# \`TextBlock\`

Text content block.

\`\`\`python  
@dataclass  
class TextBlock:  
    text: str  
\`\`\`

\#\#\# \`ThinkingBlock\`

Thinking content block (for models with thinking capability).

\`\`\`python  
@dataclass  
class ThinkingBlock:  
    thinking: str  
    signature: str  
\`\`\`

\#\#\# \`ToolUseBlock\`

Tool use request block.

\`\`\`python  
@dataclass  
class ToolUseBlock:  
    id: str  
    name: str  
    input: dict\[str, Any\]  
\`\`\`

\#\#\# \`ToolResultBlock\`

Tool execution result block.

\`\`\`python  
@dataclass  
class ToolResultBlock:  
    tool\_use\_id: str  
    content: str | list\[dict\[str, Any\]\] | None \= None  
    is\_error: bool | None \= None  
\`\`\`

\#\# Error Types

\#\#\# \`ClaudeSDKError\`

Base exception class for all SDK errors.

\`\`\`python  
class ClaudeSDKError(Exception):  
    """Base error for Claude SDK."""  
\`\`\`

\#\#\# \`CLINotFoundError\`

Raised when Claude Code CLI is not installed or not found.

\`\`\`python  
class CLINotFoundError(CLIConnectionError):  
    def \_\_init\_\_(self, message: str \= "Claude Code not found", cli\_path: str | None \= None):  
        """  
        Args:  
            message: Error message (default: "Claude Code not found")  
            cli\_path: Optional path to the CLI that was not found  
        """  
\`\`\`

\#\#\# \`CLIConnectionError\`

Raised when connection to Claude Code fails.

\`\`\`python  
class CLIConnectionError(ClaudeSDKError):  
    """Failed to connect to Claude Code."""  
\`\`\`

\#\#\# \`ProcessError\`

Raised when the Claude Code process fails.

\`\`\`python  
class ProcessError(ClaudeSDKError):  
    def \_\_init\_\_(self, message: str, exit\_code: int | None \= None, stderr: str | None \= None):  
        self.exit\_code \= exit\_code  
        self.stderr \= stderr  
\`\`\`

\#\#\# \`CLIJSONDecodeError\`

Raised when JSON parsing fails.

\`\`\`python  
class CLIJSONDecodeError(ClaudeSDKError):  
    def \_\_init\_\_(self, line: str, original\_error: Exception):  
        """  
        Args:  
            line: The line that failed to parse  
            original\_error: The original JSON decode exception  
        """  
        self.line \= line  
        self.original\_error \= original\_error  
\`\`\`

\#\# Hook Types

\#\#\# \`HookEvent\`

Supported hook event types. Note that due to setup limitations, the Python SDK does not support SessionStart, SessionEnd, and Notification hooks.

\`\`\`python  
HookEvent \= Literal\[  
    "PreToolUse",      \# Called before tool execution  
    "PostToolUse",     \# Called after tool execution  
    "UserPromptSubmit", \# Called when user submits a prompt  
    "Stop",            \# Called when stopping execution  
    "SubagentStop",    \# Called when a subagent stops  
    "PreCompact"       \# Called before message compaction  
\]  
\`\`\`

\#\#\# \`HookCallback\`

Type definition for hook callback functions.

\`\`\`python  
HookCallback \= Callable\[  
    \[dict\[str, Any\], str | None, HookContext\],  
    Awaitable\[dict\[str, Any\]\]  
\]  
\`\`\`

Parameters:

\* \`input\_data\`: Hook-specific input data (see \[hook documentation\](https://docs.anthropic.com/en/docs/claude-code/hooks\#hook-input))  
\* \`tool\_use\_id\`: Optional tool use identifier (for tool-related hooks)  
\* \`context\`: Hook context with additional information

Returns a dictionary that may contain:

\* \`decision\`: \`"block"\` to block the action  
\* \`systemMessage\`: System message to add to the transcript  
\* \`hookSpecificOutput\`: Hook-specific output data

\#\#\# \`HookContext\`

Context information passed to hook callbacks.

\`\`\`python  
@dataclass  
class HookContext:  
    signal: Any | None \= None  \# Future: abort signal support  
\`\`\`

\#\#\# \`HookMatcher\`

Configuration for matching hooks to specific events or tools.

\`\`\`python  
@dataclass  
class HookMatcher:  
    matcher: str | None \= None        \# Tool name or pattern to match (e.g., "Bash", "Write|Edit")  
    hooks: list\[HookCallback\] \= field(default\_factory=list)  \# List of callbacks to execute  
\`\`\`

\#\#\# Hook Usage Example

\`\`\`python  
from claude\_code\_sdk import query, ClaudeCodeOptions, HookMatcher, HookContext  
from typing import Any

async def validate\_bash\_command(  
    input\_data: dict\[str, Any\],  
    tool\_use\_id: str | None,  
    context: HookContext  
) \-\> dict\[str, Any\]:  
    """Validate and potentially block dangerous bash commands."""  
    if input\_data\['tool\_name'\] \== 'Bash':  
        command \= input\_data\['tool\_input'\].get('command', '')  
        if 'rm \-rf /' in command:  
            return {  
                'hookSpecificOutput': {  
                    'hookEventName': 'PreToolUse',  
                    'permissionDecision': 'deny',  
                    'permissionDecisionReason': 'Dangerous command blocked'  
                }  
            }  
    return {}

async def log\_tool\_use(  
    input\_data: dict\[str, Any\],  
    tool\_use\_id: str | None,  
    context: HookContext  
) \-\> dict\[str, Any\]:  
    """Log all tool usage for auditing."""  
    print(f"Tool used: {input\_data.get('tool\_name')}")  
    return {}

options \= ClaudeCodeOptions(  
    hooks={  
        'PreToolUse': \[  
            HookMatcher(matcher='Bash', hooks=\[validate\_bash\_command\]),  
            HookMatcher(hooks=\[log\_tool\_use\])  \# Applies to all tools  
        \],  
        'PostToolUse': \[  
            HookMatcher(hooks=\[log\_tool\_use\])  
        \]  
    }  
)

async for message in query(  
    prompt="Analyze this codebase",  
    options=options  
):  
    print(message)  
\`\`\`

\#\# Tool Input/Output Types

Documentation of input/output schemas for all built-in Claude Code tools. While the Python SDK doesn't export these as types, they represent the structure of tool inputs and outputs in messages.

\#\#\# Task

\*\*Tool name:\*\* \`Task\`

\*\*Input:\*\*

\`\`\`python  
{  
    "description": str,      \# A short (3-5 word) description of the task  
    "prompt": str,           \# The task for the agent to perform  
    "subagent\_type": str     \# The type of specialized agent to use  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "result": str,                    \# Final result from the subagent  
    "usage": dict | None,             \# Token usage statistics  
    "total\_cost\_usd": float | None,  \# Total cost in USD  
    "duration\_ms": int | None         \# Execution duration in milliseconds  
}  
\`\`\`

\#\#\# Bash

\*\*Tool name:\*\* \`Bash\`

\*\*Input:\*\*

\`\`\`python  
{  
    "command": str,                  \# The command to execute  
    "timeout": int | None,           \# Optional timeout in milliseconds (max 600000\)  
    "description": str | None,       \# Clear, concise description (5-10 words)  
    "run\_in\_background": bool | None \# Set to true to run in background  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "output": str,              \# Combined stdout and stderr output  
    "exitCode": int,            \# Exit code of the command  
    "killed": bool | None,      \# Whether command was killed due to timeout  
    "shellId": str | None       \# Shell ID for background processes  
}  
\`\`\`

\#\#\# Edit

\*\*Tool name:\*\* \`Edit\`

\*\*Input:\*\*

\`\`\`python  
{  
    "file\_path": str,           \# The absolute path to the file to modify  
    "old\_string": str,          \# The text to replace  
    "new\_string": str,          \# The text to replace it with  
    "replace\_all": bool | None  \# Replace all occurrences (default False)  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,      \# Confirmation message  
    "replacements": int, \# Number of replacements made  
    "file\_path": str     \# File path that was edited  
}  
\`\`\`

\#\#\# MultiEdit

\*\*Tool name:\*\* \`MultiEdit\`

\*\*Input:\*\*

\`\`\`python  
{  
    "file\_path": str,     \# The absolute path to the file to modify  
    "edits": \[            \# Array of edit operations  
        {  
            "old\_string": str,          \# The text to replace  
            "new\_string": str,          \# The text to replace it with  
            "replace\_all": bool | None  \# Replace all occurrences  
        }  
    \]  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,       \# Success message  
    "edits\_applied": int, \# Total number of edits applied  
    "file\_path": str      \# File path that was edited  
}  
\`\`\`

\#\#\# Read

\*\*Tool name:\*\* \`Read\`

\*\*Input:\*\*

\`\`\`python  
{  
    "file\_path": str,       \# The absolute path to the file to read  
    "offset": int | None,   \# The line number to start reading from  
    "limit": int | None     \# The number of lines to read  
}  
\`\`\`

\*\*Output (Text files):\*\*

\`\`\`python  
{  
    "content": str,         \# File contents with line numbers  
    "total\_lines": int,     \# Total number of lines in file  
    "lines\_returned": int   \# Lines actually returned  
}  
\`\`\`

\*\*Output (Images):\*\*

\`\`\`python  
{  
    "image": str,       \# Base64 encoded image data  
    "mime\_type": str,   \# Image MIME type  
    "file\_size": int    \# File size in bytes  
}  
\`\`\`

\#\#\# Write

\*\*Tool name:\*\* \`Write\`

\*\*Input:\*\*

\`\`\`python  
{  
    "file\_path": str,  \# The absolute path to the file to write  
    "content": str     \# The content to write to the file  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,        \# Success message  
    "bytes\_written": int,  \# Number of bytes written  
    "file\_path": str       \# File path that was written  
}  
\`\`\`

\#\#\# Glob

\*\*Tool name:\*\* \`Glob\`

\*\*Input:\*\*

\`\`\`python  
{  
    "pattern": str,       \# The glob pattern to match files against  
    "path": str | None    \# The directory to search in (defaults to cwd)  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "matches": list\[str\],  \# Array of matching file paths  
    "count": int,          \# Number of matches found  
    "search\_path": str     \# Search directory used  
}  
\`\`\`

\#\#\# Grep

\*\*Tool name:\*\* \`Grep\`

\*\*Input:\*\*

\`\`\`python  
{  
    "pattern": str,                    \# The regular expression pattern  
    "path": str | None,                \# File or directory to search in  
    "glob": str | None,                \# Glob pattern to filter files  
    "type": str | None,                \# File type to search  
    "output\_mode": str | None,         \# "content", "files\_with\_matches", or "count"  
    "-i": bool | None,                 \# Case insensitive search  
    "-n": bool | None,                 \# Show line numbers  
    "-B": int | None,                  \# Lines to show before each match  
    "-A": int | None,                  \# Lines to show after each match  
    "-C": int | None,                  \# Lines to show before and after  
    "head\_limit": int | None,          \# Limit output to first N lines/entries  
    "multiline": bool | None           \# Enable multiline mode  
}  
\`\`\`

\*\*Output (content mode):\*\*

\`\`\`python  
{  
    "matches": \[  
        {  
            "file": str,  
            "line\_number": int | None,  
            "line": str,  
            "before\_context": list\[str\] | None,  
            "after\_context": list\[str\] | None  
        }  
    \],  
    "total\_matches": int  
}  
\`\`\`

\*\*Output (files\\\_with\\\_matches mode):\*\*

\`\`\`python  
{  
    "files": list\[str\],  \# Files containing matches  
    "count": int         \# Number of files with matches  
}  
\`\`\`

\#\#\# NotebookEdit

\*\*Tool name:\*\* \`NotebookEdit\`

\*\*Input:\*\*

\`\`\`python  
{  
    "notebook\_path": str,                     \# Absolute path to the Jupyter notebook  
    "cell\_id": str | None,                    \# The ID of the cell to edit  
    "new\_source": str,                        \# The new source for the cell  
    "cell\_type": "code" | "markdown" | None,  \# The type of the cell  
    "edit\_mode": "replace" | "insert" | "delete" | None  \# Edit operation type  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,                              \# Success message  
    "edit\_type": "replaced" | "inserted" | "deleted",  \# Type of edit performed  
    "cell\_id": str | None,                       \# Cell ID that was affected  
    "total\_cells": int                           \# Total cells in notebook after edit  
}  
\`\`\`

\#\#\# WebFetch

\*\*Tool name:\*\* \`WebFetch\`

\*\*Input:\*\*

\`\`\`python  
{  
    "url": str,     \# The URL to fetch content from  
    "prompt": str   \# The prompt to run on the fetched content  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "response": str,           \# AI model's response to the prompt  
    "url": str,                \# URL that was fetched  
    "final\_url": str | None,   \# Final URL after redirects  
    "status\_code": int | None  \# HTTP status code  
}  
\`\`\`

\#\#\# WebSearch

\*\*Tool name:\*\* \`WebSearch\`

\*\*Input:\*\*

\`\`\`python  
{  
    "query": str,                        \# The search query to use  
    "allowed\_domains": list\[str\] | None, \# Only include results from these domains  
    "blocked\_domains": list\[str\] | None  \# Never include results from these domains  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "results": \[  
        {  
            "title": str,  
            "url": str,  
            "snippet": str,  
            "metadata": dict | None  
        }  
    \],  
    "total\_results": int,  
    "query": str  
}  
\`\`\`

\#\#\# TodoWrite

\*\*Tool name:\*\* \`TodoWrite\`

\*\*Input:\*\*

\`\`\`python  
{  
    "todos": \[  
        {  
            "content": str,                              \# The task description  
            "status": "pending" | "in\_progress" | "completed",  \# Task status  
            "activeForm": str                            \# Active form of the description  
        }  
    \]  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,  \# Success message  
    "stats": {  
        "total": int,  
        "pending": int,  
        "in\_progress": int,  
        "completed": int  
    }  
}  
\`\`\`

\#\#\# BashOutput

\*\*Tool name:\*\* \`BashOutput\`

\*\*Input:\*\*

\`\`\`python  
{  
    "bash\_id": str,       \# The ID of the background shell  
    "filter": str | None  \# Optional regex to filter output lines  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "output": str,                                      \# New output since last check  
    "status": "running" | "completed" | "failed",       \# Current shell status  
    "exitCode": int | None                              \# Exit code when completed  
}  
\`\`\`

\#\#\# KillBash

\*\*Tool name:\*\* \`KillBash\`

\*\*Input:\*\*

\`\`\`python  
{  
    "shell\_id": str  \# The ID of the background shell to kill  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,  \# Success message  
    "shell\_id": str  \# ID of the killed shell  
}  
\`\`\`

\#\#\# ExitPlanMode

\*\*Tool name:\*\* \`ExitPlanMode\`

\*\*Input:\*\*

\`\`\`python  
{  
    "plan": str  \# The plan to run by the user for approval  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "message": str,          \# Confirmation message  
    "approved": bool | None  \# Whether user approved the plan  
}  
\`\`\`

\#\#\# ListMcpResources

\*\*Tool name:\*\* \`ListMcpResources\`

\*\*Input:\*\*

\`\`\`python  
{  
    "server": str | None  \# Optional server name to filter resources by  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "resources": \[  
        {  
            "uri": str,  
            "name": str,  
            "description": str | None,  
            "mimeType": str | None,  
            "server": str  
        }  
    \],  
    "total": int  
}  
\`\`\`

\#\#\# ReadMcpResource

\*\*Tool name:\*\* \`ReadMcpResource\`

\*\*Input:\*\*

\`\`\`python  
{  
    "server": str,  \# The MCP server name  
    "uri": str      \# The resource URI to read  
}  
\`\`\`

\*\*Output:\*\*

\`\`\`python  
{  
    "contents": \[  
        {  
            "uri": str,  
            "mimeType": str | None,  
            "text": str | None,  
            "blob": str | None  
        }  
    \],  
    "server": str  
}  
\`\`\`

\#\# Example Usage

\#\#\# Basic file operations

\`\`\`python  
from claude\_code\_sdk import query, ClaudeCodeOptions, AssistantMessage, ToolUseBlock  
import asyncio

async def create\_project():  
    options \= ClaudeCodeOptions(  
        allowed\_tools=\["Read", "Write", "Bash"\],  
        permission\_mode='acceptEdits',  
        cwd="/home/user/project"  
    )  
      
    async for message in query(  
        prompt="Create a Python project structure with setup.py",  
        options=options  
    ):  
        if isinstance(message, AssistantMessage):  
            for block in message.content:  
                if isinstance(block, ToolUseBlock):  
                    print(f"Using tool: {block.name}")

asyncio.run(create\_project())  
\`\`\`

\#\#\# Error handling

\`\`\`python  
from claude\_code\_sdk import (  
    query,  
    CLINotFoundError,  
    ProcessError,  
    CLIJSONDecodeError  
)

try:  
    async for message in query(prompt="Hello"):  
        print(message)  
except CLINotFoundError:  
    print("Please install Claude Code: npm install \-g @anthropic-ai/claude-code")  
except ProcessError as e:  
    print(f"Process failed with exit code: {e.exit\_code}")  
except CLIJSONDecodeError as e:  
    print(f"Failed to parse response: {e}")  
\`\`\`

\#\#\# Streaming mode with client

\`\`\`python  
from claude\_code\_sdk import ClaudeSDKClient  
import asyncio

async def interactive\_session():  
    async with ClaudeSDKClient() as client:  
        \# Send initial message  
        await client.query("What's the weather like?")  
          
        \# Process responses  
        async for msg in client.receive\_response():  
            print(msg)  
          
        \# Send follow-up  
        await client.query("Tell me more about that")  
          
        \# Process follow-up response  
        async for msg in client.receive\_response():  
            print(msg)

asyncio.run(interactive\_session())  
\`\`\`

\#\#\# Using custom tools

\`\`\`python  
from claude\_code\_sdk import (  
    query,  
    ClaudeCodeOptions,  
    tool,  
    create\_sdk\_mcp\_server,  
    AssistantMessage,  
    TextBlock  
)  
import asyncio  
from typing import Any

\# Define custom tools with @tool decorator  
@tool("calculate", "Perform mathematical calculations", {"expression": str})  
async def calculate(args: dict\[str, Any\]) \-\> dict\[str, Any\]:  
    try:  
        result \= eval(args\["expression"\], {"\_\_builtins\_\_": {}})  
        return {  
            "content": \[{  
                "type": "text",  
                "text": f"Result: {result}"  
            }\]  
        }  
    except Exception as e:  
        return {  
            "content": \[{  
                "type": "text",  
                "text": f"Error: {str(e)}"  
            }\],  
            "is\_error": True  
        }

@tool("get\_time", "Get current time", {})  
async def get\_time(args: dict\[str, Any\]) \-\> dict\[str, Any\]:  
    from datetime import datetime  
    current\_time \= datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
    return {  
        "content": \[{  
            "type": "text",  
            "text": f"Current time: {current\_time}"  
        }\]  
    }

async def main():  
    \# Create SDK MCP server with custom tools  
    my\_server \= create\_sdk\_mcp\_server(  
        name="utilities",  
        version="1.0.0",  
        tools=\[calculate, get\_time\]  
    )  
      
    \# Configure options with the server  
    options \= ClaudeCodeOptions(  
        mcp\_servers={"utils": my\_server},  
        allowed\_tools=\[  
            "mcp\_\_utils\_\_calculate",  
            "mcp\_\_utils\_\_get\_time"  
        \]  
    )  
      
    \# Query Claude with custom tools available  
    async for message in query(  
        prompt="What's 123 \* 456 and what time is it?",  
        options=options  
    ):  
        if isinstance(message, AssistantMessage):  
            for block in message.content:  
                if isinstance(block, TextBlock):  
                    print(block.text)

asyncio.run(main())  
\`\`\`

\#\# See also

\* \[Python SDK guide\](/en/docs/claude-code/sdk/sdk-python) \- Tutorial and examples  
\* \[SDK overview\](/en/docs/claude-code/sdk/sdk-overview) \- General SDK concepts  
\* \[TypeScript SDK reference\](/en/docs/claude-code/typescript-sdk-reference) \- TypeScript SDK documentation  
\* \[CLI reference\](/en/docs/claude-code/cli-reference) \- Command-line interface  
\* \[Common workflows\](/en/docs/claude-code/common-workflows) \- Step-by-step guides

