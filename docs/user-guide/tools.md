# Tools

Agents can use tools to extend their capabilities — execute code, search the web, run shell commands, and more.

## Available Tools

| Tool | Class | Description |
|------|-------|-------------|
| **ShellTool** | `ShellTool` | Execute shell commands with timeout and whitelist |
| **WebSearchTool** | `WebSearchTool` | Search the web, fetch pages, browser automation |
| **CodeInterpreterTool** | `CodeInterpreterTool` | Execute Python code in a sandbox |
| **FileSearchTool** | `FileSearchTool` | Search files by glob pattern or content |
| **FunctionTool** | `FunctionTool` | Register custom Python functions as tools |

## Assigning Tools to Agents

```python
from gmas.core import AgentProfile

agent = AgentProfile(
    agent_id="analyst",
    display_name="Analyst",
    tools=["shell", "web_search", "code_interpreter"],
)
```

Tools can be specified as strings (resolved by name from the registry) or as `BaseTool` instances.

## ToolRegistry

The global registry manages all available tools:

```python
from gmas.tools import ToolRegistry, ShellTool, WebSearchTool

registry = ToolRegistry()

# Register tool instances
registry.register(ShellTool(timeout=10))
registry.register(WebSearchTool(max_results=5))

# List registered tools
print(registry.list_tools())  # ['shell', 'web_search']

# Execute a tool by name
from gmas.tools import ToolCall
result = registry.execute(ToolCall(name="shell", arguments={"command": "ls"}))
print(result.output)
```

## ShellTool

Execute shell commands with safety controls:

```python
from gmas.tools import ShellTool

shell = ShellTool(
    timeout=30,                          # command timeout in seconds
    max_output_size=8192,                # max output bytes
    working_dir="/tmp",                  # working directory
    allowed_commands=["ls", "cat", "grep"],  # whitelist (None = all)
)

result = shell.execute(command="ls -la")
print(result.output)
print(result.success)
```

## CodeInterpreterTool

Execute Python code in an isolated subprocess:

```python
from gmas.tools import CodeInterpreterTool

interpreter = CodeInterpreterTool(
    timeout=30,
    max_output_size=8192,
    safe_mode=True,  # restricts builtins and imports (default)
)

result = interpreter.execute(code="import math; print(math.sqrt(144))")
print(result.output)  # "12.0"
```

In safe mode, only these modules are importable: `math`, `statistics`, `json`, `re`, `datetime`, `collections`, `itertools`, `functools`, `random`.

## WebSearchTool

Full web search with multiple providers and browser automation:

```python
from gmas.tools import WebSearchTool

search = WebSearchTool(
    max_results=5,
    fetch_content=True,        # fetch page content
    max_content_length=4000,   # truncate content
)

# Search
result = search.execute(query="Python async patterns")

# Fetch a specific URL
result = search.execute(url="https://example.com", action="fetch")

# Advanced: browser automation with Playwright
from gmas.tools.web_search import PlaywrightFetcher

search = WebSearchTool(
    browser_fetcher=PlaywrightFetcher(),
    deep_search="playwright",
)
result = search.execute(
    query="detailed research topic",
    action="crawl",
    max_depth=2,
    max_pages=10,
)
```

### Search Providers

| Provider | Class | API Key Required |
|----------|-------|------------------|
| DuckDuckGo | `DuckDuckGoProvider` | No |
| Serper | `SerperProvider` | Yes |
| Tavily | `TavilyProvider` | Yes |

### Browser Actions

When a browser fetcher is configured, additional actions are available:

| Action | Description |
|--------|-------------|
| `fetch` | Fetch a URL |
| `click` | Click a selector |
| `fill` | Fill a form field |
| `extract_links` | Extract all links |
| `execute_js` | Run JavaScript |
| `crawl` | Crawl linked pages |
| `get_content` | Get page content |
| `screenshot` | Take a screenshot |
| `download` | Download a file |

## FileSearchTool

Search files by name pattern or content:

```python
from gmas.tools import FileSearchTool

search = FileSearchTool(
    base_directory=".",
    max_results=50,
    allowed_extensions=[".py", ".md"],
)

# Glob search
result = search.execute(pattern="**/*.py")

# Content search
result = search.execute(query="RoleGraph", regex=False)

# Read a specific file
result = search.execute(read_file="src/core/graph.py")
```

## Custom Tools

### FunctionTool Decorator

Register plain Python functions as tools:

```python
from gmas.tools import FunctionTool

func_tool = FunctionTool()

@func_tool.register
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@func_tool.register(name="format_json", description="Format JSON string")
def format_json(data: str) -> str:
    import json
    return json.dumps(json.loads(data), indent=2)

# Use in registry
registry.register(func_tool)
```

### BaseTool Subclass

For more control, subclass `BaseTool`:

```python
from gmas.tools import BaseTool, ToolResult

class DatabaseTool(BaseTool):
    name = "database"
    description = "Query the database"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query"},
            },
            "required": ["query"],
        }

    def execute(self, *, query: str = "", **kwargs) -> ToolResult:
        try:
            result = run_query(query)
            return ToolResult(tool_name=self.name, output=result)
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
            )
```

## Tool Execution in Agents

During execution, the runner handles tool calling automatically:

1. The agent's prompt includes tool descriptions from its `tools` list
2. If the LLM response contains a tool call, the runner executes it
3. The tool result is fed back to the LLM
4. This loops up to `max_tool_iterations` (default: 3)

```python
from gmas.execution import RunnerConfig

config = RunnerConfig(max_tool_iterations=5)
```

## OpenAI Function Calling

For OpenAI-compatible models, tools are automatically converted to function schemas:

```python
from gmas.tools import ToolRegistry

registry = ToolRegistry()
registry.register(ShellTool())

# Get OpenAI-compatible schemas
schemas = registry.to_openai_schemas(["shell"])
# [{"type": "function", "function": {"name": "shell", ...}}]
```
