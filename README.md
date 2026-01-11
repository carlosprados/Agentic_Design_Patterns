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
* **Cost Efficiency:** Standardized on **Gemini 2.5 Flash** to ensure full functionality within Google's free tier.

## 📂 Repository Contents

The primary goal of this repository is to ensure that **every example is fully functional** in a pure Python environment. There is a continuous effort to test and verify each script to guarantee reliability and performance.

This repository covers the patterns and frameworks discussed in the book, organized by the original structure:

### Part 1: Foundational Patterns

* **01 Prompt Chaining** ✅: Sequential workflows and pipelines.
* **02 Routing** ✅: Dynamic decision-making and intent classification.
* **03 Parallelization** ✅: Concurrent execution of independent sub-tasks.
* **04 Reflection** ⏳: Self-correction and critique loops (Generator-Critic).
* **05 Tool Use** ⏳: Function calling and external API integration.
* **06 Planning** ⏳: Breaking complex goals into actionable steps.
* **07 Multi-Agent** ⏳: Orchestrating teams of specialized agents.

### Part 2: Advanced Capabilities

* **08 Memory Management** ⏳: Short-term context vs. Long-term persistence.
* **09 Learning and Adaptation** ⏳: Evolving agent behavior over time.
* **10 Model Context Protocol (MCP)** ⏳: Standardizing connection to external resources.
* **11 Goal Setting and Monitoring** ⏳: Tracking objectives and milestones.

### Part 3: Robustness & Reliability

* **12 Exception Handling and Recovery** ⏳: Recovery strategies and fallback mechanisms.
* **13 Human-in-the-Loop** ⏳: Integration of human oversight and approval.
* **14 Knowledge Retrieval (RAG)** ⏳: Grounding agents in external data.

### Part 4: Optimization & Safety

* **15 Inter-Agent Communication (A2A)** ⏳: Protocols for agent-to-agent talk.
* **16 Resource-Aware Optimization** ⏳: Managing compute and cost.
* **17 Reasoning Techniques** ⏳: CoT, ReAct, and Tree of Thoughts.
* **18 Guardrails/Safety Patterns** ⏳: Input/Output validation and safety boundaries.
* **19 Evaluation and Monitoring** ⏳: Assessing agent performance and reliability.
* **20 Prioritization** ⏳: Managing task urgency and resource allocation.
* **21 Exploration and Discovery** ⏳: Autonomous learning and information seeking.

## 🛠️ Tech Stack

Following the book's examples, this repository utilizes the following frameworks:

* **LangChain / LangGraph:** For graph-based agent flows.
* **Google Agent Developer Kit (ADK):** For enterprise-grade agent construction.
* **CrewAI:** For role-based multi-agent orchestration.
* **LLM Providers:** Optimized for **Google Gemini 2.5 Flash** (Free Tier compatible).

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
    Create a `.env` file in the root directory and add your API keys:

    ```env
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AIza...
    ANTHROPIC_API_KEY=sk-...
    ```

## 💻 Usage

Navigate to the specific pattern folder and run the script using `uv run`. For example, to run the **Planning Pattern** example:

```bash
uv run 06_Planning/planning_agent.py
```

## 🤝 Contributing

Contributions are welcome! If you find an issue with the conversion logic (e.g., a missing import that was implicit in the notebook), please open an issue or Pull Request.

For theoretical questions regarding the patterns themselves, please refer to Antonio Gulli's original book.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

This project also respects the license of the original repository. Please refer to the source repository for licensing information regarding the code logic.
