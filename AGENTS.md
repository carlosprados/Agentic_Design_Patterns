# AI Agent Guidelines (AGENTS.md)

Welcome, AI Agent. This repository is dedicated to refactoring the examples from Antonio Gulli's book **"Agentic Design Patterns"** into pure, executable Python scripts. To maintain consistency and quality across the project, follow these rules.

## 🎯 Core Objective
- **Convert Notebooks to Scripts:** Transform Jupyter Notebooks into standard Python modules (`.py`).
- **Production Readiness:** Ensure code is modular, well-documented, and ready for deployment.

## 🛠️ Tech Stack & Environment
- **Python Version:** Always target `Python >= 3.12`.
- **Dependency Management:** Use `uv`. 
    - Sync environment: `uv sync`
    - Add dependencies: `uv add <package>`
    - Run scripts: `uv run <script_path>`
- **Core Frameworks:** LangChain, LangGraph, Google ADK, CrewAI.

## 📂 Project Structure
- **Folder Naming:** Maintain the prefix numbering (e.g., `01_Prompt_Chaining/`, `02_Routing/`).
- **File Naming:** Use snake_case for filenames (e.g., `intent_router.py`).
- **Scripts Position:** Place main executable scripts directly inside their respective pattern folders.

## 📜 Coding Rules
1. **No Notebooks:** Do not create or commit `.ipynb` files.
2. **Environment Variables:** Use a `.env` file for all API keys. Never hardcode keys. Use `python-dotenv` or equivalent if necessary, but prioritize standard environment access via `os.environ`.
3. **Modularity:** Encapsulate logic in classes or functions. Move reusable logic to a `shared/` or `utils/` folder if it applies to multiple chapters.
4. **Typing:** Use Python type hints for all function signatures and class members.
5. **Documentation:** Include docstrings for all classes and public functions. Briefly explain the agentic pattern being implemented.
6. **Logging:** Use the standard `logging` module instead of `print()` for production-level output.

## 🔑 Authentication
Agents should assume the following environment variables are available:
- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`

## 🚀 Workflow
1. Read the original notebook logic.
2. Design the clean Python equivalent.
3. Implement in the appropriate folder.
4. Verify execution with `uv run`.
5. Update `README.md` in the folder if specific instructions are needed.

---
> [!IMPORTANT]
> Always respect the original intellectual property of Antonio Gulli while refactoring.
