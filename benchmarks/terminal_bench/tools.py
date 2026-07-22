"""
ToolBox for the GMAS terminal agent.

Tool names and descriptions are sourced from the official MCP reference server
implementations (modelcontextprotocol/servers):
  - Filesystem tools:        src/filesystem/index.ts
  - Git tools:               src/git/mcp_server_git/server.py
  - Memory tools:            src/memory (knowledge-graph server, simplified to KV store)
  - Sequential-thinking:     src/sequentialthinking (think scratchpad)
  - Fetch tools:             src/fetch (URL fetcher via curl inside container)

NOT IMPLEMENTED (require external API / subscription):
  - context7: authenticated library-documentation service; no local equivalent.

All container tools run inside the Harbor Docker container via BaseEnvironment.exec().
Memory and think tools are local-only (no container call).
Content is passed to the shell via base64 to avoid quoting/escaping issues.
"""

import asyncio
import base64
import concurrent.futures
import shlex
from typing import TYPE_CHECKING, Any

from benchmarks.terminal_bench.consts import (
    CODE_TOOL_NAMES as CONST_CODE_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    DEFAULT_OUTPUT_END_CHAR,
    DEFAULT_OUTPUT_START_CHAR,
)
from benchmarks.terminal_bench.consts import (
    FETCH_TOOL_NAMES as CONST_FETCH_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    FILESYSTEM_TOOL_NAMES as CONST_FILESYSTEM_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    GIT_TOOL_NAMES as CONST_GIT_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    MEMORY_TOOL_NAMES as CONST_MEMORY_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    PYTEST_TOOL_NAMES as CONST_PYTEST_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    SHELL_TOOL_NAMES as CONST_SHELL_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    THINKING_TOOL_NAMES as CONST_THINKING_TOOL_NAMES,
)
from benchmarks.terminal_bench.consts import (
    TOOL_NAMES as CONST_TOOL_NAMES,
)
from gmas.tools import FunctionWrapper, ToolRegistry

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

# Hard deadline for each individual container command.
_EXEC_TIMEOUT_SEC: int = 120


class ToolBox:
    """Environment-bound tool implementations sourced from MCP reference servers."""

    # ── Tool name lists by category ──────────────────────────────────────────
    FILESYSTEM_TOOL_NAMES = CONST_FILESYSTEM_TOOL_NAMES
    GIT_TOOL_NAMES = CONST_GIT_TOOL_NAMES
    SHELL_TOOL_NAMES = CONST_SHELL_TOOL_NAMES

    # Python snippets executed inside the task container.
    CODE_TOOL_NAMES = CONST_CODE_TOOL_NAMES

    # In-process key-value store (no container call required).
    MEMORY_TOOL_NAMES = CONST_MEMORY_TOOL_NAMES

    # Pytest execution inside the task container.
    PYTEST_TOOL_NAMES = CONST_PYTEST_TOOL_NAMES

    # Structured reasoning scratchpad (local, no container call).
    THINKING_TOOL_NAMES = CONST_THINKING_TOOL_NAMES

    # URL fetching via curl inside the container.
    # context7 is NOT included — it requires an external authenticated API.
    FETCH_TOOL_NAMES = CONST_FETCH_TOOL_NAMES

    TOOL_NAMES = CONST_TOOL_NAMES

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self, environment: "BaseEnvironment") -> None:
        self.environment = environment
        # Every tool call (query + raw output) is appended here for metadata.
        self.tool_outputs: list[dict[str, Any]] = []
        # In-process session memory (key → value), shared across all agents in a run.
        self._memory: dict[str, str] = {}
        if "run_pytest" in self.TOOL_NAMES:
            self._ensure_pytest_available()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _exec(self, cmd: str) -> str:
        """
        Run *cmd* in the container synchronously and return the string result.

        Enforces the deadline in Harbor's environment layer. The extra Future
        timeout is only a last-resort guard in case the environment call itself
        wedges before Harbor can return.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(
            asyncio.run,
            self.environment.exec(cmd, timeout_sec=_EXEC_TIMEOUT_SEC),
        )
        try:
            return str(future.result(timeout=_EXEC_TIMEOUT_SEC + 5))
        except concurrent.futures.TimeoutError:
            return f"Error: command timed out after {_EXEC_TIMEOUT_SEC}s"
        except Exception as exc:
            return f"Error executing command: {exc}"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _slice_output(output: str, start_char: int, end_char: int) -> str:
        """Return the requested character range with a compact truncation note."""
        start = max(0, int(start_char))
        end = max(start, int(end_char))
        if start == 0 and end >= len(output):
            return output
        chunk = output[start:end]
        return (
            f"[output chars {start}:{end} of {len(output)}; set start_char/end_char to inspect another range]\n{chunk}"
        )

    def _run(
        self,
        tool_name: str,
        cmd: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """Execute *cmd* in the container, record the call, and return the output."""
        output = self._exec(cmd)
        model_output = self._slice_output(output, start_char, end_char)
        self.tool_outputs.append(
            {
                "tool": tool_name,
                "query": cmd,
                "output": output,
                "model_output": model_output,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        return model_output

    def _local(
        self,
        tool_name: str,
        output: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """Record a local (no container exec) tool call and return its output."""
        model_output = self._slice_output(output, start_char, end_char)
        self.tool_outputs.append(
            {
                "tool": tool_name,
                "output": output,
                "model_output": model_output,
                "start_char": start_char,
                "end_char": end_char,
            }
        )
        return model_output

    @staticmethod
    def _b64(text: str) -> str:
        """Base64-encode *text* so it can be safely embedded in a shell command."""
        return base64.b64encode(text.encode()).decode()

    @staticmethod
    def _shell_arguments(arguments: str) -> str:
        """Parse a user-provided argument list and safely quote every token."""
        return " ".join(shlex.quote(argument) for argument in shlex.split(arguments))

    def _ensure_pytest_available(self) -> None:
        """Best-effort pytest install for containers that do not ship with it."""
        check_cmd = "python -c " + shlex.quote(
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
        )
        check_output = self._exec(check_cmd)
        if "return_code=0" in check_output:
            return

        install_cmd = (
            "(python -m pip install --quiet pytest || "
            "(python -m ensurepip --upgrade >/tmp/ensurepip.log 2>&1 && "
            "python -m pip install --quiet pytest))"
        )
        install_output = self._exec(install_cmd)
        self.tool_outputs.append(
            {
                "tool": "setup_pytest",
                "query": install_cmd,
                "output": install_output,
                "model_output": self._slice_output(
                    install_output,
                    DEFAULT_OUTPUT_START_CHAR,
                    DEFAULT_OUTPUT_END_CHAR,
                ),
                "start_char": DEFAULT_OUTPUT_START_CHAR,
                "end_char": DEFAULT_OUTPUT_END_CHAR,
            }
        )

    def _run_python_code(
        self,
        tool_name: str,
        code: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """Execute Python code in the task container for Python snippet tools."""
        if not code.strip():
            return self._local(tool_name, "Error: no Python code provided.")
        b64 = self._b64(code)
        runner = (
            "import base64; "
            f"code = base64.b64decode({b64!r}).decode(); "
            f"exec(compile(code, '<{tool_name}>', 'exec'), {{}})"
        )
        return self._run(tool_name, f"python3 -c {shlex.quote(runner)}", start_char, end_char)

    # ── Shell tool ───────────────────────────────────────────────────────────

    def execute_env(
        self,
        query: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Execute a shell command in the task environment and return its output.
        Use this for any operation not covered by the specialised tools.
        Set start_char/end_char to choose the returned output character range.
        """
        return self._run("execute_env", query, start_char, end_char)

    def exec_tool(
        self,
        cmd: list[str],
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Alias for execute_env that accepts an argv-style command list.
        Use this when you need to execute a shell command and the caller emits
        exec({"cmd": ["bash", "-lc", "..."]}). Set start_char/end_char to
        choose the returned output character range.
        """
        query = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(arg)) for arg in cmd)
        return self._run("exec", query, start_char, end_char)

    def run_code(
        self,
        code: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Execute Python code in the task environment and return its output.
        Use this for quick calculations, parsing, or short scripts that should
        run inside the benchmark container. Set start_char/end_char to choose
        the returned output character range.
        """
        return self._run_python_code("run_code", code, start_char, end_char)

    def run_python(
        self,
        code: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Execute Python code in the task environment and return its output.
        Alias for run_code, provided because models often ask for a Python
        execution tool by this name. Set start_char/end_char to choose the
        returned output character range.
        """
        return self._run_python_code("run_python", code, start_char, end_char)

    def run_pytest(
        self,
        target: str = "",
        extra_args: str = "",
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Run Python tests with pytest; this is the only correct way to execute pytest-style Python tests.
        Optional target is a test file, directory, or
        node id. Optional extra_args is passed to pytest as shell-like args.
        Set start_char/end_char to choose the returned output character range.
        """
        args = ["python", "-m", "pytest", "--root-user-action=ignore"]
        if target:
            args.append(target)
        if extra_args:
            args.extend(shlex.split(extra_args))
        cmd = " ".join(shlex.quote(arg) for arg in args)
        return self._run("run_pytest", cmd, start_char, end_char)

    # ── Filesystem tools (descriptions from MCP filesystem reference server) ─

    def read_file(
        self,
        path: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Read the complete contents of a file from the file system as text.
        Handles various text encodings and provides detailed error messages if
        the file cannot be read. Use this tool when you need to examine the
        contents of a single file. Set start_char/end_char to choose the
        returned output character range.
        """
        return self._run("read_file", f"cat {shlex.quote(path)}", start_char, end_char)

    def write_file(self, path: str, content: str) -> str:
        """
        Create a new file or completely overwrite an existing file with new
        content. Use with caution as it will overwrite existing files without
        warning. Handles text content with proper encoding.
        """
        runner = (
            "import base64, pathlib; "
            f"path=base64.b64decode({self._b64(path)!r}).decode(); "
            f"content=base64.b64decode({self._b64(content)!r}).decode(); "
            "target=pathlib.Path(path); "
            "target.parent.mkdir(parents=True, exist_ok=True); "
            "target.write_text(content, encoding='utf-8'); "
            "print(f'Successfully wrote to {path}')"
        )
        cmd = f"python3 -c {shlex.quote(runner)}"
        return self._run("write_file", cmd)

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """
        Make a targeted text replacement in a file. Replaces the first
        occurrence of *old_text* with *new_text*. Returns a confirmation or
        an error if the text was not found. Equivalent to the MCP filesystem
        server edit_file with a single edit operation.
        """
        runner = (
            "import base64; "
            f"path=base64.b64decode({self._b64(path)!r}).decode(); "
            f"old=base64.b64decode({self._b64(old_text)!r}).decode(); "
            f"new=base64.b64decode({self._b64(new_text)!r}).decode(); "
            "content=open(path, encoding='utf-8').read(); "
            "assert old in content, f'old_text not found in {path}'; "
            "open(path, 'w', encoding='utf-8').write(content.replace(old, new, 1)); "
            "print(f'Replaced 1 occurrence in {path}')"
        )
        cmd = f"python3 -c {shlex.quote(runner)}"
        return self._run("edit_file", cmd)

    def list_directory(
        self,
        path: str = ".",
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Get a detailed listing of all files and directories in a specified
        path. Results clearly distinguish between files and directories with
        [FILE] and [DIR] prefixes. Essential for understanding directory
        structure and finding specific files within a directory. Set
        start_char/end_char to choose the returned output character range.
        """
        runner = (
            "import base64, os; "
            f"path=base64.b64decode({self._b64(path)!r}).decode(); "
            "entries=sorted(os.scandir(path), key=lambda entry: entry.name); "
            "print('\\n'.join(('[DIR]  '+entry.name if entry.is_dir() else '[FILE] '+entry.name) "
            "for entry in entries))"
        )
        cmd = f"python3 -c {shlex.quote(runner)}"
        return self._run("list_directory", cmd, start_char, end_char)

    def directory_tree(
        self,
        path: str = ".",
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Get a recursive tree view of files and directories. Each entry shows
        its name and type. Useful for understanding the full structure of a
        project at a glance. Set start_char/end_char to choose the returned
        output character range.
        """
        return self._run(
            "directory_tree",
            f"find {shlex.quote(path)} | sort | head -200",
            start_char,
            end_char,
        )

    def search_files(
        self,
        path: str,
        pattern: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Recursively search for files whose names match a shell glob pattern
        (e.g. '*.py', 'test_*'). Returns full paths to all matching items.
        Great for finding files when you don't know their exact location. Set
        start_char/end_char to choose the returned output character range.
        """
        return self._run(
            "search_files",
            f"find {shlex.quote(path)} -name {shlex.quote(pattern)} | sort | head -100",
            start_char,
            end_char,
        )

    def create_directory(self, path: str) -> str:
        """
        Create a new directory or ensure a directory exists. Can create
        multiple nested directories in one operation. If the directory already
        exists, this operation succeeds silently.
        """
        return self._run(
            "create_directory",
            f"mkdir -p {shlex.quote(path)} && printf '%s\\n' {shlex.quote(f'Created: {path}')}",
        )

    # ── Memory tools (descriptions from MCP memory reference server) ──────────
    # Implemented as a simple in-process key-value store instead of a full
    # knowledge graph. The API mirrors the essential read/write/list operations.

    def memory_write(self, key: str, value: str) -> str:
        """
        Store information in the agent's session memory under *key*.
        Overwrites any previous value at the same key. Use this to save
        findings, intermediate results, or notes for later retrieval.
        """
        self._memory[key] = value
        return self._local("memory_write", f"Stored {len(value)} chars at key '{key}'.")

    def memory_read(
        self,
        key: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Retrieve a value previously stored in session memory by its *key*.
        Returns the stored text, or an informative error if the key is absent.
        Set start_char/end_char to choose the returned output character range.
        """
        if key in self._memory:
            return self._local("memory_read", self._memory[key], start_char, end_char)
        return self._local(
            "memory_read",
            f"Key '{key}' not found. Use memory_list() to see available keys.",
        )

    def memory_list(
        self,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        List all keys currently stored in the agent's session memory.
        Returns a newline-separated list of keys, or a notice if memory is empty.
        Set start_char/end_char to choose the returned output character range.
        """
        result = "Stored keys:\n" + "\n".join(f"  - {k}" for k in self._memory) if self._memory else "Memory is empty."
        return self._local("memory_list", result, start_char, end_char)

    def memory_delete(self, key: str) -> str:
        """
        Delete a key and its value from the agent's session memory.
        Silent no-op if the key does not exist.
        """
        removed = self._memory.pop(key, None)
        msg = f"Deleted key '{key}'." if removed is not None else f"Key '{key}' was not in memory."
        return self._local("memory_delete", msg)

    # ── Think tool (descriptions from MCP sequential-thinking server) ─────────

    def think(self, thought: str) -> str:
        """
        Record a structured reasoning step without executing any command.
        Use this to think through complex problems, plan next steps, or reason
        about ambiguous situations before taking action. Returns the thought
        unchanged so it appears in the conversation context.
        """
        return self._local("think", thought)

    # ── Fetch tool (descriptions from MCP fetch reference server) ─────────────

    def fetch_url(
        self,
        url: str,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Fetch content from a URL and return a bounded character range.
        Useful for retrieving documentation pages, API references, or any
        publicly accessible web resource. Runs curl inside the container so
        the request originates from the task network. Set start_char/end_char
        to choose the returned output character range.
        """
        cmd = f"curl -s -L --max-time 30 -H 'User-Agent: Mozilla/5.0' {shlex.quote(url)}"
        return self._run("fetch_url", cmd, start_char, end_char)

    # ── Git tools (descriptions from MCP git reference server) ───────────────

    def git_status(
        self,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Show the working tree status — which files are modified, staged,
        or untracked. Always run this first to understand the current repo state.
        """
        return self._run("git_status", "git status", start_char, end_char)

    def git_diff(
        self,
        target: str = "",
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Show changes between commits, commit and working tree, etc.
        *target* can be empty (unstaged changes), 'HEAD' (all uncommitted
        changes), a branch name, a commit hash, or 'branch1..branch2'. Set
        start_char/end_char to choose the returned output character range.
        """
        arg = shlex.quote(target) if target else ""
        return self._run("git_diff", f"git diff {arg}".strip(), start_char, end_char)

    def git_log(
        self,
        max_count: int = 20,
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Show the commit log with one-line summaries decorated with branch and
        tag labels. *max_count* limits the number of commits shown. Set
        start_char/end_char to choose the returned output character range.
        """
        return self._run(
            "git_log",
            f"git log --oneline --decorate --graph --all -n {int(max_count)}",
            start_char,
            end_char,
        )

    def git_add(self, files: str) -> str:
        """
        Add file contents to the index (stage for commit). *files* is a
        space-separated list of paths, or '.' to stage all changes.
        """
        return self._run("git_add", f"git add {self._shell_arguments(files)}")

    def git_commit(self, message: str) -> str:
        """
        Record the staged changes to the repository with *message* as the
        commit message.
        """
        return self._run("git_commit", f"git commit -m {shlex.quote(message)}")

    def git_checkout(self, target: str) -> str:
        """
        Switch branches, restore working-tree files, or recover commits.
        *target* can be a branch name, commit hash, or 'BRANCH -- path/to/file'
        to restore a specific file from another branch.
        """
        return self._run("git_checkout", f"git checkout {self._shell_arguments(target)}")

    def git_show(
        self,
        revision: str = "HEAD",
        start_char: int = DEFAULT_OUTPUT_START_CHAR,
        end_char: int = DEFAULT_OUTPUT_END_CHAR,
    ) -> str:
        """
        Show the contents and metadata of a commit, tag, or tree object.
        *revision* defaults to HEAD. Useful for inspecting what a specific
        commit changed. Set start_char/end_char to choose the returned output
        character range.
        """
        return self._run("git_show", f"git show {shlex.quote(revision)}", start_char, end_char)

    # ── Registry ─────────────────────────────────────────────────────────────

    def make_registry(self) -> ToolRegistry:
        """Return a ToolRegistry with every tool in TOOL_NAMES bound to this environment."""
        registry = ToolRegistry()
        for tool_name in self.TOOL_NAMES:
            method_name = "exec_tool" if tool_name == "exec" else tool_name
            method = getattr(self, method_name)
            registry.register(
                FunctionWrapper(
                    method,
                    tool_name=tool_name,
                    tool_description=method.__doc__.strip().split("\n")[0],
                )
            )
        return registry
