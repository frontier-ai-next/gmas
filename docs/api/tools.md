# Tools API

## Tools Module

```python
from gmas.tools import (
    # Base
    BaseTool,
    ToolCall,
    ToolResult,
    ToolRegistry,
    FunctionTool,
    FunctionWrapper,

    # Decorators
    tool,
    register_tool,
    get_registry,

    # Specific tools
    ShellTool,
    WebSearchTool,
    CodeInterpreterTool,
    FileSearchTool,

    # Web search providers
    DuckDuckGoProvider,
    SerperProvider,
    TavilyProvider,
)
```

## BaseTool

Abstract base class for all tools.

```python
from gmas.tools import BaseTool, ToolResult

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "What this tool does"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "Description"},
            },
            "required": ["param"],
        }

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(tool_name=self.name, output="result")
```

### Methods

| Method | Description |
|--------|-------------|
| `execute(**kwargs) -> ToolResult` | Run the tool (abstract) |
| `to_openai_schema() -> dict` | Generate OpenAI function schema |

## ToolCall

Represents a parsed tool invocation from an LLM response.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Tool name |
| `arguments` | `dict[str, Any]` | Tool arguments |

```python
calls = ToolCall.parse_from_response(llm_response)
for call in calls:
    print(call.name, call.arguments)
```

## ToolResult

Result of a tool execution.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_name` | `str` | required | Name of the tool |
| `success` | `bool` | `True` | Whether execution succeeded |
| `output` | `str` | `""` | Text output |
| `structured_output` | `dict \| None` | `None` | Structured result |
| `error` | `str \| None` | `None` | Error message if failed |

```python
result = ToolResult(tool_name="shell", output="file.txt", success=True)
print(result.to_message())  # formatted message for LLM
```

## ToolRegistry

Central registry for managing tools.

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `(tool: BaseTool)` | Register a tool |
| `function` | `(func=None, *, name=None, description=None)` | Decorator to register function |
| `get` | `(name: str) -> BaseTool \| None` | Get tool by name |
| `has` | `(name: str) -> bool` | Check if tool exists |
| `execute` | `(call: ToolCall) -> ToolResult` | Execute a tool call |
| `execute_all` | `(calls: list[ToolCall]) -> list[ToolResult]` | Execute multiple calls |
| `list_tools` | `() -> list[str]` | List registered tool names |
| `get_tools` | `(names: list[str]) -> list[BaseTool]` | Get tools by names |
| `to_openai_schemas` | `(names: list[str]) -> list[dict]` | OpenAI function schemas |
| `get_tools_for_agent` | `(names: list[str]) -> list[BaseTool]` | Tools for an agent |
| `format_tools_prompt` | `(names: list[str]) -> str` | Format tools as text prompt |

## ShellTool

Execute shell commands with safety controls.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `int` | `30` | Command timeout (seconds) |
| `max_output_size` | `int` | `8192` | Max output bytes |
| `working_dir` | `str \| None` | `None` | Working directory |
| `allowed_commands` | `list[str] \| None` | `None` | Command whitelist |

## CodeInterpreterTool

Execute Python code in an isolated subprocess.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout` | `int` | `30` | Execution timeout (seconds) |
| `max_output_size` | `int` | `8192` | Max output bytes |
| `safe_mode` | `bool` | `True` | Restrict imports and builtins |

Safe mode allows: `math`, `statistics`, `json`, `re`, `datetime`, `collections`, `itertools`, `functools`, `random`.

## WebSearchTool

Full web search with providers and browser automation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `SearchProvider \| None` | `None` | Search provider |
| `max_results` | `int` | `5` | Max search results |
| `max_content_length` | `int` | `4000` | Content truncation |
| `fetch_content` | `bool` | `False` | Auto-fetch result pages |
| `timeout` | `int` | `15` | Request timeout (seconds) |
| `deep_search` | `str \| None` | `None` | Deep search backend |
| `browser_config` | `dict \| None` | `None` | Browser configuration |
| `browser_fetcher` | `BrowserFetcher \| None` | `None` | Custom browser fetcher |
| `cache` | `SearchCache \| bool \| None` | `None` | Cache configuration |
| `deduplicate` | `bool` | `True` | Deduplicate results |
| `policy` | `WebSearchPolicy \| None` | `None` | Scoring policy |

### Search Providers

| Class | API Key | Environment Variable |
|-------|---------|---------------------|
| `DuckDuckGoProvider` | Not required | - |
| `SerperProvider` | Required | `SERPER_API_KEY` |
| `TavilyProvider` | Required | `TAVILY_API_KEY` |

## FileSearchTool

Search files by pattern or content.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_directory` | `str \| Path` | `"."` | Search root |
| `max_results` | `int` | `50` | Max results |
| `max_depth` | `int` | `10` | Max directory depth |
| `max_file_size` | `int` | `100000` | Max file size (bytes) |
| `max_read_size` | `int` | `10000` | Max read size |
| `allowed_extensions` | `list[str] \| None` | `None` | File extension filter |

## FunctionTool

Register Python functions as tools.

```python
from gmas.tools import FunctionTool

tool = FunctionTool()

# Decorator style
@tool.register
def search(query: str) -> str:
    """Search for information."""
    return "results"

# Or with options
@tool.register(name="web_lookup", description="Look up info on the web")
def lookup(query: str, max_results: int = 5) -> str:
    return "results"

# Programmatic
tool.add_function(my_function, name="custom", description="Custom function")

# List registered functions
print(tool.list_functions())
```

## Global Registry

```python
from gmas.tools import tool, get_registry, register_tool

# Decorator for global registry
@tool
def my_function(param: str) -> str:
    """Description."""
    return "result"

# Or register an instance
register_tool(MyTool())

# Access global registry
registry = get_registry()
```
