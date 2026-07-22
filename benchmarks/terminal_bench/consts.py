from pathlib import Path

OPENAI_MAX_RETRIES = 3
LLM_TEMPERATURE = 0.1
LLM_MAX_COMPLETION_TOKENS = 4096
CONDITION_MAX_COMPLETION_TOKENS = 64
PARSE_MAX_RETRIES = 3

AGENTS_CONFIG_PATH = Path(__file__).parent / "agents_config.json"

DEFAULT_OUTPUT_START_CHAR = 0
DEFAULT_OUTPUT_END_CHAR = 5000

FILESYSTEM_TOOL_NAMES = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "directory_tree",
    "search_files",
    "create_directory",
]
GIT_TOOL_NAMES = [
    "git_status",
    "git_diff",
    "git_log",
    "git_add",
    "git_commit",
    "git_checkout",
    "git_show",
]
SHELL_TOOL_NAMES = ["execute_env", "exec"]
CODE_TOOL_NAMES = ["run_code", "run_python"]
PYTEST_TOOL_NAMES = ["run_pytest"]
MEMORY_TOOL_NAMES = ["memory_write", "memory_read", "memory_list", "memory_delete"]
THINKING_TOOL_NAMES = ["think"]
FETCH_TOOL_NAMES = ["fetch_url"]
TOOL_NAMES = (
    SHELL_TOOL_NAMES
    + CODE_TOOL_NAMES
    + PYTEST_TOOL_NAMES
    + FILESYSTEM_TOOL_NAMES
    + GIT_TOOL_NAMES
    + MEMORY_TOOL_NAMES
    + THINKING_TOOL_NAMES
    + FETCH_TOOL_NAMES
)

SERVER_TO_TOOLS: dict[str, list[str]] = {
    "filesystem": FILESYSTEM_TOOL_NAMES,
    "rust-mcp-filesystem": FILESYSTEM_TOOL_NAMES,
    "git": GIT_TOOL_NAMES,
    "shell-sandbox": SHELL_TOOL_NAMES,
    "code-interpreter": CODE_TOOL_NAMES,
    "run-code": CODE_TOOL_NAMES,
    "pytest": PYTEST_TOOL_NAMES,
    "test-runner": PYTEST_TOOL_NAMES,
    "memory": MEMORY_TOOL_NAMES,
    "sequential-thinking": THINKING_TOOL_NAMES,
    "fetch": FETCH_TOOL_NAMES,
}

SINGLE_AGENT_ID = "single_agent"
RUNNER_MAX_LOOP_ITERATIONS = 1_000_000
RUNNER_MAX_TOOL_ITERATIONS = 10
RUNNER_TIMEOUT = 300

MAS_START_NODE = "coordinator"
MAS_END_NODE = "test_verifier"
