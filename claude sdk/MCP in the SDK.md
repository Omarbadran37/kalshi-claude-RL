\# MCP in the SDK

\> Extend Claude Code with custom tools using Model Context Protocol servers

\#\# Overview

Model Context Protocol (MCP) servers extend Claude Code with custom tools and capabilities. MCPs can run as external processes, connect via HTTP/SSE, or execute directly within your SDK application.

\#\# Configuration

\#\#\# Basic Configuration

Configure MCP servers in \`.mcp.json\` at your project root:

\<CodeGroup\>  
  \`\`\`json TypeScript  
  {  
    "mcpServers": {  
      "filesystem": {  
        "command": "npx",  
        "args": \["@modelcontextprotocol/server-filesystem"\],  
        "env": {  
          "ALLOWED\_PATHS": "/Users/me/projects"  
        }  
      }  
    }  
  }  
  \`\`\`

  \`\`\`json Python  
  {  
    "mcpServers": {  
      "filesystem": {  
        "command": "python",  
        "args": \["-m", "mcp\_server\_filesystem"\],  
        "env": {  
          "ALLOWED\_PATHS": "/Users/me/projects"  
        }  
      }  
    }  
  }  
  \`\`\`  
\</CodeGroup\>

\#\#\# Using MCP Servers in SDK

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  import { query } from "@anthropic-ai/claude-code";

  for await (const message of query({  
    prompt: "List files in my project",  
    options: {  
      mcpConfig: ".mcp.json",  
      allowedTools: \["mcp\_\_filesystem\_\_list\_files"\]  
    }  
  })) {  
    if (message.type \=== "result" && message.subtype \=== "success") {  
      console.log(message.result);  
    }  
  }  
  \`\`\`

  \`\`\`python Python  
  from anthropic\_claude\_code import query

  async for message in query(  
      prompt="List files in my project",  
      options={  
          "mcpConfig": ".mcp.json",  
          "allowedTools": \["mcp\_\_filesystem\_\_list\_files"\]  
      }  
  ):  
      if message\["type"\] \== "result" and message\["subtype"\] \== "success":  
          print(message\["result"\])  
  \`\`\`  
\</CodeGroup\>

\#\# Transport Types

\#\#\# stdio Servers

External processes communicating via stdin/stdout:

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  // .mcp.json configuration  
  {  
    "mcpServers": {  
      "my-tool": {  
        "command": "node",  
        "args": \["./my-mcp-server.js"\],  
        "env": {  
          "DEBUG": "${DEBUG:-false}"  
        }  
      }  
    }  
  }  
  \`\`\`

  \`\`\`python Python  
  \# .mcp.json configuration  
  {  
    "mcpServers": {  
      "my-tool": {  
        "command": "python",  
        "args": \["./my\_mcp\_server.py"\],  
        "env": {  
          "DEBUG": "${DEBUG:-false}"  
        }  
      }  
    }  
  }  
  \`\`\`  
\</CodeGroup\>

\#\#\# HTTP/SSE Servers

Remote servers with network communication:

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  // SSE server configuration  
  {  
    "mcpServers": {  
      "remote-api": {  
        "type": "sse",  
        "url": "https://api.example.com/mcp/sse",  
        "headers": {  
          "Authorization": "Bearer ${API\_TOKEN}"  
        }  
      }  
    }  
  }

  // HTTP server configuration  
  {  
    "mcpServers": {  
      "http-service": {  
        "type": "http",  
        "url": "https://api.example.com/mcp",  
        "headers": {  
          "X-API-Key": "${API\_KEY}"  
        }  
      }  
    }  
  }  
  \`\`\`

  \`\`\`python Python  
  \# SSE server configuration  
  {  
    "mcpServers": {  
      "remote-api": {  
        "type": "sse",  
        "url": "https://api.example.com/mcp/sse",  
        "headers": {  
          "Authorization": "Bearer ${API\_TOKEN}"  
        }  
      }  
    }  
  }

  \# HTTP server configuration  
  {  
    "mcpServers": {  
      "http-service": {  
        "type": "http",  
        "url": "https://api.example.com/mcp",  
        "headers": {  
          "X-API-Key": "${API\_KEY}"  
        }  
      }  
    }  
  }  
  \`\`\`  
\</CodeGroup\>

\#\#\# SDK MCP Servers

In-process servers running within your application. For detailed information on creating custom tools, see the \[Custom Tools guide\](/en/docs/claude-code/sdk/custom-tools):

\#\# Resource Management

MCP servers can expose resources that Claude can list and read:

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  import { query } from "@anthropic-ai/claude-code";

  // List available resources  
  for await (const message of query({  
    prompt: "What resources are available from the database server?",  
    options: {  
      mcpConfig: ".mcp.json",  
      allowedTools: \["mcp\_\_list\_resources", "mcp\_\_read\_resource"\]  
    }  
  })) {  
    if (message.type \=== "result") console.log(message.result);  
  }  
  \`\`\`

  \`\`\`python Python  
  from anthropic\_claude\_code import query

  \# List available resources  
  async for message in query(  
      prompt="What resources are available from the database server?",  
      options={  
          "mcpConfig": ".mcp.json",  
          "allowedTools": \["mcp\_\_list\_resources", "mcp\_\_read\_resource"\]  
      }  
  ):  
      if message\["type"\] \== "result":  
          print(message\["result"\])  
  \`\`\`  
\</CodeGroup\>

\#\# Authentication

\#\#\# Environment Variables

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  // .mcp.json with environment variables  
  {  
    "mcpServers": {  
      "secure-api": {  
        "type": "sse",  
        "url": "https://api.example.com/mcp",  
        "headers": {  
          "Authorization": "Bearer ${API\_TOKEN}",  
          "X-API-Key": "${API\_KEY:-default-key}"  
        }  
      }  
    }  
  }

  // Set environment variables  
  process.env.API\_TOKEN \= "your-token";  
  process.env.API\_KEY \= "your-key";  
  \`\`\`

  \`\`\`python Python  
  \# .mcp.json with environment variables  
  {  
    "mcpServers": {  
      "secure-api": {  
        "type": "sse",  
        "url": "https://api.example.com/mcp",  
        "headers": {  
          "Authorization": "Bearer ${API\_TOKEN}",  
          "X-API-Key": "${API\_KEY:-default-key}"  
        }  
      }  
    }  
  }

  \# Set environment variables  
  import os  
  os.environ\["API\_TOKEN"\] \= "your-token"  
  os.environ\["API\_KEY"\] \= "your-key"  
  \`\`\`  
\</CodeGroup\>

\#\#\# OAuth2 Authentication

OAuth2 MCP authentication in-client is not currently supported.

\#\# Error Handling

Handle MCP connection failures gracefully:

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  import { query } from "@anthropic-ai/claude-code";

  for await (const message of query({  
    prompt: "Process data",  
    options: {  
      mcpServers: {  
        "data-processor": dataServer  
      }  
    }  
  })) {  
    if (message.type \=== "system" && message.subtype \=== "init") {  
      // Check MCP server status  
      const failedServers \= message.mcp\_servers.filter(  
        s \=\> s.status \!== "connected"  
      );  
        
      if (failedServers.length \> 0\) {  
        console.warn("Failed to connect:", failedServers);  
      }  
    }  
      
    if (message.type \=== "result" && message.subtype \=== "error\_during\_execution") {  
      console.error("Execution failed");  
    }  
  }  
  \`\`\`

  \`\`\`python Python  
  from anthropic\_claude\_code import query

  async for message in query(  
      prompt="Process data",  
      options={  
          "mcpServers": {  
              "data-processor": data\_server  
          }  
      }  
  ):  
      if message\["type"\] \== "system" and message\["subtype"\] \== "init":  
          \# Check MCP server status  
          failed\_servers \= \[  
              s for s in message\["mcp\_servers"\]  
              if s\["status"\] \!= "connected"  
          \]  
            
          if failed\_servers:  
              print(f"Failed to connect: {failed\_servers}")  
        
      if message\["type"\] \== "result" and message\["subtype"\] \== "error\_during\_execution":  
          print("Execution failed")  
  \`\`\`  
\</CodeGroup\>

\#\# Related Resources

\* \[Custom Tools Guide\](/en/docs/claude-code/sdk/custom-tools) \- Detailed guide on creating SDK MCP servers  
\* \[TypeScript SDK Reference\](/en/docs/claude-code/sdk/sdk-typescript)  
\* \[Python SDK Reference\](/en/docs/claude-code/sdk/sdk-python)  
\* \[SDK Permissions\](/en/docs/claude-code/sdk/sdk-permissions)  
\* \[Common Workflows\](/en/docs/claude-code/common-workflows)

