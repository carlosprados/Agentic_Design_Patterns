# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pure Python refactoring of Antonio Gulli's "Agentic Design Patterns" book. Converts original Jupyter Notebooks into executable `.py` scripts across 21 chapters (01-21), each demonstrating a different agentic AI pattern. Always respect the original intellectual property.

## Commands

```bash
uv sync                                      # Install/sync all dependencies
uv run 01_Prompt_Chaining/prompt_chaining_basics.py  # Run any example script
uv add <package>                             # Add a new dependency
```

There is no test suite, linter config, or build step. Verification is done by running each script with `uv run` and checking it executes without errors.

## Architecture

### Framework: LangGraph / LangChain (exclusively)

All examples use **LangGraph** (preferred) or **LangChain LCEL** as the sole agentic framework. When writing new examples or modifying existing ones, always prefer LangGraph (`StateGraph`, conditional edges, nodes) over plain LangChain chains. Use LangChain LCEL only for simple sequential pipelines where a full graph is overkill.

- **LangGraph**: `StateGraph`, `TypedDict` state, `add_conditional_edges`, `START`/`END`, `MemorySaver` for persistence. Preferred for any workflow with branching, loops, or multi-agent coordination.
- **LangChain LCEL**: `ChatPromptTemplate`, `StrOutputParser`, `RunnableParallel`, `RunnablePassthrough`. Used for simple sequential chains.
- **LLM provider**: Centralized in `shared/llm.py` via `get_llm(temperature=X)`. Supports Gemini (default) and Ollama. Never instantiate `ChatGoogleGenerativeAI` directly in scripts — always use `from shared.llm import get_llm`. Switch provider via `LLM_PROVIDER=ollama` in `.env`.

### Chapter layout

Each numbered directory (`01_Prompt_Chaining/`, `02_Routing/`, etc.) contains 1-6 standalone Python scripts. Folder naming keeps the prefix numbering. File names follow the pattern `langgraph_*.py` (for StateGraph examples) or `langchain_*.py` (for LCEL chains) in snake_case. Scripts go directly inside their chapter folder.

### Script structure convention

Scripts follow a consistent pattern: `sys.path.insert` for project root, `from shared.llm import get_llm`, `load_dotenv()`, helper/tool definitions, agent/chain setup function, `main()` (often async with `asyncio.run`), and `if __name__ == "__main__"` guard.

## Coding Rules

- **No notebooks**: Do not create or commit `.ipynb` files. Everything must be pure Python.
- **Python >= 3.12** required (`.python-version` file present).
- **API keys**: Via `.env` and `os.environ` (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`). Never hardcode keys. Use `python-dotenv` or direct `os.environ` access.
- **Type hints**: On all function signatures and class members.
- **Docstrings**: On all classes and public functions. Briefly explain the agentic pattern being implemented.
- **Logging**: Use the standard `logging` module instead of `print()` for production-level output.
- **Modularity**: Encapsulate logic in classes or functions. Move reusable logic to a `shared/` or `utils/` folder if it applies to multiple chapters.

## Workflow for New/Refactored Scripts

1. Read the original notebook logic (archived in `notebooks/`).
2. Design the clean Python equivalent.
3. Implement in the appropriate chapter folder.
4. Verify execution with `uv run`.
5. Update chapter README if specific instructions are needed.

## Status

All 21 chapters have been converted to LangGraph/LangChain. Verify each script with `uv run` before marking as stable.
