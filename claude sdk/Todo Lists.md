\# Todo Lists

\> Track and display todos using the Claude Code SDK for organized task management

Todo tracking provides a structured way to manage tasks and display progress to users. The Claude Code SDK includes built-in todo functionality that helps organize complex workflows and keep users informed about task progression.

\#\#\# Todo Lifecycle

Todos follow a predictable lifecycle:

1\. \*\*Created\*\* as \`pending\` when tasks are identified  
2\. \*\*Activated\*\* to \`in\_progress\` when work begins  
3\. \*\*Completed\*\* when the task finishes successfully  
4\. \*\*Removed\*\* when all tasks in a group are completed

\#\#\# When Todos Are Used

The SDK automatically creates todos for:

\* \*\*Complex multi-step tasks\*\* requiring 3 or more distinct actions  
\* \*\*User-provided task lists\*\* when multiple items are mentioned  
\* \*\*Non-trivial operations\*\* that benefit from progress tracking  
\* \*\*Explicit requests\*\* when users ask for todo organization

\#\# Examples

\#\#\# Monitoring Todo Changes

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  import { query } from "@anthropic-ai/claude-code";

  for await (const message of query({  
    prompt: "Optimize my React app performance and track progress with todos",  
    options: { maxTurns: 15 }  
  })) {  
    // Todo updates are reflected in the message stream  
    if (message.type \=== "tool\_use" && message.name \=== "TodoWrite") {  
      const todos \= message.input.todos;  
        
      console.log("Todo Status Update:");  
      todos.forEach((todo, index) \=\> {  
        const status \= todo.status \=== "completed" ? "✅" :   
                      todo.status \=== "in\_progress" ? "🔧" : "❌";  
        console.log(\`${index \+ 1}. ${status} ${todo.content}\`);  
      });  
    }  
  }  
  \`\`\`

  \`\`\`python Python  
  from anthropic\_claude\_code import query

  async for message in query(  
      prompt="Optimize my React app performance and track progress with todos",  
      options={"max\_turns": 15}  
  ):  
      \# Todo updates are reflected in the message stream  
      if message.get("type") \== "tool\_use" and message.get("name") \== "TodoWrite":  
          todos \= message\["input"\]\["todos"\]  
            
          print("Todo Status Update:")  
          for i, todo in enumerate(todos):  
              status \= "✅" if todo\["status"\] \== "completed" else \\  
                      "🔧" if todo\["status"\] \== "in\_progress" else "❌"  
              print(f"{i \+ 1}. {status} {todo\['content'\]}")  
  \`\`\`  
\</CodeGroup\>

\#\#\# Real-time Progress Display

\<CodeGroup\>  
  \`\`\`typescript TypeScript  
  import { query } from "@anthropic-ai/claude-code";

  class TodoTracker {  
    private todos: any\[\] \= \[\];  
      
    displayProgress() {  
      if (this.todos.length \=== 0\) return;  
        
      const completed \= this.todos.filter(t \=\> t.status \=== "completed").length;  
      const inProgress \= this.todos.filter(t \=\> t.status \=== "in\_progress").length;  
      const total \= this.todos.length;  
        
      console.log(\`\\nProgress: ${completed}/${total} completed\`);  
      console.log(\`Currently working on: ${inProgress} task(s)\\n\`);  
        
      this.todos.forEach((todo, index) \=\> {  
        const icon \= todo.status \=== "completed" ? "✅" :   
                    todo.status \=== "in\_progress" ? "🔧" : "❌";  
        const text \= todo.status \=== "in\_progress" ? todo.activeForm : todo.content;  
        console.log(\`${index \+ 1}. ${icon} ${text}\`);  
      });  
    }  
      
    async trackQuery(prompt: string) {  
      for await (const message of query({  
        prompt,  
        options: { maxTurns: 20 }  
      })) {  
        if (message.type \=== "tool\_use" && message.name \=== "TodoWrite") {  
          this.todos \= message.input.todos;  
          this.displayProgress();  
        }  
      }  
    }  
  }

  // Usage  
  const tracker \= new TodoTracker();  
  await tracker.trackQuery("Build a complete authentication system with todos");  
  \`\`\`

  \`\`\`python Python  
  from anthropic\_claude\_code import query  
  from typing import List, Dict

  class TodoTracker:  
      def \_\_init\_\_(self):  
          self.todos: List\[Dict\] \= \[\]  
        
      def display\_progress(self):  
          if not self.todos:  
              return  
            
          completed \= len(\[t for t in self.todos if t\["status"\] \== "completed"\])  
          in\_progress \= len(\[t for t in self.todos if t\["status"\] \== "in\_progress"\])  
          total \= len(self.todos)  
            
          print(f"\\nProgress: {completed}/{total} completed")  
          print(f"Currently working on: {in\_progress} task(s)\\n")  
            
          for i, todo in enumerate(self.todos):  
              icon \= "✅" if todo\["status"\] \== "completed" else \\  
                    "🔧" if todo\["status"\] \== "in\_progress" else "❌"  
              text \= todo\["activeForm"\] if todo\["status"\] \== "in\_progress" else todo\["content"\]  
              print(f"{i \+ 1}. {icon} {text}")  
        
      async def track\_query(self, prompt: str):  
          async for message in query(  
              prompt=prompt,  
              options={"max\_turns": 20}  
          ):  
              if message.get("type") \== "tool\_use" and message.get("name") \== "TodoWrite":  
                  self.todos \= message\["input"\]\["todos"\]  
                  self.display\_progress()

  \# Usage  
  tracker \= TodoTracker()  
  await tracker.track\_query("Build a complete authentication system with todos")  
  \`\`\`  
\</CodeGroup\>

\#\# Related Documentation

\* \[TypeScript SDK Reference\](/en/docs/claude-code/sdk/sdk-typescript)  
\* \[Python SDK Reference\](/en/docs/claude-code/sdk/sdk-python)  
\* \[Streaming vs Single Mode\](/en/docs/claude-code/sdk/streaming-vs-single-mode)  
\* \[Custom Tools\](/en/docs/claude-code/sdk/custom-tools)

