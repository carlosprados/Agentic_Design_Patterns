# Agentic Design Patterns - Pure Python Implementation

Plain Python conversion of the code from Antonio Gulli's **Agentic Design Patterns** great book. This project refactors the original book notebooks into executable `.py` scripts, enabling easier integration, debugging, and production usage.


## 📚 Source Material & Credits

This repository is a code adaptation project. The intellectual property, core algorithms, and agentic concepts belong to **Antonio Gulli**.

* **Book:** *Agentic Design Patterns: A Hands-On Guide to Building Intelligent Systems* by **Antonio Gulli**.
* **Original Repository:** [sarwarbeing-ai/Agentic_Design_Patterns](https://github.com/sarwarbeing-ai/Agentic_Design_Patterns/tree/main)

This project serves strictly as a **refactoring layer** to enable developers to run these patterns as standard Python modules rather than Jupyter Notebooks.

## 🚀 Motivation

While Jupyter Notebooks are excellent for exploration, moving to pure Python scripts offers several advantages for engineering robust Agentic AI systems:

* **Production Readiness:** Easier to deploy in Docker containers or cloud functions.
* **Maintainability:** Compatible with standard linters, formatters, and IDE features.
* **Modularity:** Easier to import classes and functions into larger applications.
* **Version Control:** Cleaner diffs and history tracking in Git.
* **Cost Efficiency:** Default LLM is **Gemini 2.5 Flash** (Google's free tier). Also supports **Ollama** for fully local execution.
* **Provider Flexibility:** A single `shared/llm.py` module abstracts the LLM provider. Switch between Gemini and Ollama with one environment variable — zero code changes.

## 📂 Repository Contents

The primary goal of this repository is to ensure that **every example is fully functional** in a pure Python environment. There is a continuous effort to test and verify each script to guarantee reliability and performance.

This repository covers the patterns and frameworks discussed in the book, organized by the original structure:

### Part 1: Foundational Patterns

* **01 Prompt Chaining** ✅: Sequential workflows and pipelines.
* **02 Routing** ✅: Dynamic decision-making and intent classification.
* **03 Parallelization** ✅: Concurrent execution of independent sub-tasks.
* **04 Reflection** ✅: Self-correction and critique loops (Generator-Critic).
* **05 Tool Use** ✅: Function calling and external API integration.
* **06 Planning** ✅: Breaking complex goals into actionable steps.
* **07 Multi-Agent** ✅: Orchestrating teams of specialized agents.

### Part 2: Advanced Capabilities

* **08 Memory Management** ✅: Short-term context vs. Long-term persistence.
* **09 Learning and Adaptation** ✅: Evolving agent behavior over time.
* **10 Model Context Protocol (MCP)** ✅: Standardizing connection to external resources.
* **11 Goal Setting and Monitoring** ✅: Tracking objectives and milestones.

### Part 3: Robustness & Reliability

* **12 Exception Handling and Recovery** ✅: Recovery strategies and fallback mechanisms.
* **13 Human-in-the-Loop** ✅: Integration of human oversight and approval.
* **14 Knowledge Retrieval (RAG)** ✅: Grounding agents in external data.

### Part 4: Optimization & Safety

* **15 Inter-Agent Communication (A2A)** ✅: Protocols for agent-to-agent talk.
* **16 Resource-Aware Optimization** ✅: Managing compute and cost.
* **17 Reasoning Techniques** ✅: CoT, ReAct, and Tree of Thoughts.
* **18 Guardrails/Safety Patterns** ✅: Input/Output validation and safety boundaries.
* **19 Evaluation and Monitoring** ✅: Assessing agent performance and reliability.
* **20 Prioritization** ✅: Managing task urgency and resource allocation.
* **21 Exploration and Discovery** ✅: Autonomous learning and information seeking.

## 🛠️ Tech Stack

All examples use a single framework stack for consistency and simplicity:

* **LangGraph:** For stateful, graph-based agent workflows (preferred for all patterns with branching, loops, or multi-agent coordination).
* **LangChain LCEL:** For simple sequential chains and prompt pipelines.
* **LLM Provider:** Centralized in [`shared/llm.py`](shared/llm.py). Default: **Google Gemini 2.5 Flash** (Free Tier). Also supports **Ollama** for local models.

### Switching LLM Providers

All scripts use `get_llm()` from `shared/llm.py` instead of instantiating a model directly. To switch providers, set environment variables in your `.env` file:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Provider to use: `gemini` or `ollama` |
| `GOOGLE_API_KEY` | — | Required for Gemini |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name (requires `ollama` running locally) |

**Example — run with Ollama:**

```bash
# .env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b

# Then run any example as usual
uv run 02_Routing/langgraph_coordinator_routing.py
```

> **Note:** Tool-calling examples (Ch05, Ch07 agent-as-tool, Ch20) require models with good function-calling support. With Ollama, `llama3.1` and `qwen2.5` work well; smaller models may struggle.

## ⚙️ Installation

1. **Clone this repository:**

    ```bash
    git clone https://github.com/carlosprados/Agentic_Design_Patterns.git 
    cd Agentic_Design_Patterns
    ```

2. **Initialize environment and install dependencies:**
    Using `uv` is recommended for faster and more reliable dependency management.

    ```bash
    # Create a virtual environment and sync dependencies
    uv sync
    ```

3. **Set up Environment Variables:**
    Create a `.env` file in the root directory:

    ```env
    # Option A: Gemini (default)
    GOOGLE_API_KEY=AIza...

    # Option B: Ollama (local, no API key needed)
    # LLM_PROVIDER=ollama
    # OLLAMA_MODEL=llama3.1:8b
    ```

## 💻 Usage

Run any example with `uv run` from the project root:

```bash
uv run 06_Planning/langchain_planning_writer.py
uv run 02_Routing/langgraph_coordinator_routing.py
uv run 07_Multi_Agent/langgraph_loop_agent.py
```

## 🤝 Contributing

Contributions are welcome! If you find an issue with the conversion logic (e.g., a missing import that was implicit in the notebook), please open an issue or Pull Request.

For theoretical questions regarding the patterns themselves, please refer to Antonio Gulli's original book.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

This project also respects the license of the original repository. Please refer to the source repository for licensing information regarding the code logic.
