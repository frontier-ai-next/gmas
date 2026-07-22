#!/usr/bin/env python3
r"""
Smoke test: GAIA benchmark search stacks return non-empty data.

Loads ``benchmarks.gaia.run`` (the same code paths as the real benchmark).

**Run from the repository root**

1. Install the project with benchmark dependencies::

       uv sync --extra benchmarks

2. Install Playwright browsers (needed for gMAS deep + LangGraph deep)::

       uv run playwright install chromium

3. (Optional, Windows) Bypass corporate proxy for local HTTP, same as ``examples/playground_web_gmas.ipynb``::

       set NO_PROXY=*
       :: PowerShell: $env:NO_PROXY = '*'

4. Run this script::

       uv run python -m benchmarks.gaia.smoke

Exit code ``0`` if web search paths work; deep paths may print ``SKIP`` without failing the run.

**Then run the GAIA benchmark** (needs ``LLM_API_KEY``, ``LLM_BASE_URL``, ``LLM_MODEL`` in ``.env``)::

       uv run python -m benchmarks.gaia.run --max-samples 1 --levels 1 --parallel 1

Quick search-only probe (no LLM)::

       uv run python -m benchmarks.gaia.run --probe-search
"""

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_benchmark() -> Any:
    root = Path(__file__).resolve().parent
    path = root / "run.py"
    spec = importlib.util.spec_from_file_location("_gaia_bench_smoke", path)
    if spec is None or spec.loader is None:
        msg = f"Cannot load benchmark module from {path}"
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ok(label: str, body: str, *, min_len: int = 40) -> bool:
    bad = body.startswith("Error:") or "No results found" in body[:120]
    if bad or len(body.strip()) < min_len:
        print(f"  FAIL  {label}")
        print(f"         preview: {body[:200]!r}")
        return False
    print(f"  OK    {label} ({len(body)} chars)")
    return True


async def _langgraph_deep_extract_example(b: Any) -> str | None:
    if b.async_playwright is None:
        return None
    async with b.async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            tools = b._lc_playwright_tools(browser)
            nav = next((tool for tool in tools if tool.name == "navigate_browser"), None)
            ext = next((tool for tool in tools if tool.name == "extract_text"), None)
            if nav is None or ext is None:
                return None
            await nav.ainvoke({"url": "https://example.com/"})
            return str(await ext.ainvoke({}))
        finally:
            await browser.close()


def main() -> int:
    print("Loading benchmarks.gaia.run …")
    b = _load_benchmark()

    query = "capital of France"
    failures = 0

    print("\n--- gMAS web (WebSearchTool + DuckDuckGoProvider) ---")
    try:
        web_tool = b._make_web_tool()
        out = b._execute_web_search(web_tool, {"query": query})
        web_tool.close()
        if not _ok("gMAS web", out):
            failures += 1
    except Exception as exc:
        print(f"  FAIL  gMAS web: {exc}")
        failures += 1

    print(f"\n--- LangGraph web (LangChain DuckDuckGoSearchResults, {b.DDGS_BACKEND}) ---")
    try:
        out = b._lc_invoke_ddg_search(query)
        if not _ok("LangChain DDG", out):
            failures += 1
    except Exception as exc:
        print(f"  FAIL  LangChain DDG: {exc}")
        failures += 1

    print("\n--- LangGraph tool dispatch (web_search, deep=False) ---")
    try:
        out = b._lg_dispatch("web_search", {"query": query}, deep=False)
        if not _ok("LangGraph _lg_dispatch web", out):
            failures += 1
    except Exception as exc:
        print(f"  FAIL  LangGraph dispatch: {exc}")
        failures += 1

    print("\n--- gMAS deep (WebSearchTool + Playwright) ---")
    try:
        deep_tool = b._make_deep_tool()
        out = b._execute_web_search(deep_tool, {"query": query})
        deep_tool.close()
        if not _ok("gMAS deep", out, min_len=80):
            failures += 1
    except Exception as exc:
        print(f"  SKIP  gMAS deep (install playwright + browsers?): {exc}")

    print("\n--- LangGraph deep (navigate + extract_text) ---")
    try:
        txt = asyncio.run(_langgraph_deep_extract_example(b))
        if txt is None:
            print("  SKIP  PlayWrightBrowserToolkit (playwright / bs4 / browser install)")
        elif not _ok("LangGraph Playwright extract", txt, min_len=20):
            failures += 1
    except Exception as exc:
        print(f"  SKIP  LangGraph deep: {exc}")

    print("\n--- Summary ---")
    if failures:
        print(f"  Required checks failed: {failures} (web stacks must work for a fair benchmark run).")
        return 1
    print("  All required web checks passed. Deep paths may have been skipped; see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
