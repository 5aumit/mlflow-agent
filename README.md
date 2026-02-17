# MLflow Experiment Agentic Chatbot

A CLI-based assistant for ML experimentation, inspired by Claude Code, that helps researchers query, analyze, and gain insights from MLflow experiment logs.

## Features
- Natural language Q&A about ML experiments (regression, classification, neural nets)
- Reads and analyzes MLflow logs for metrics, parameters, and results
- Compares runs, detects patterns (e.g., overfitting), and summarizes findings
- Modular inference engine supporting Groq, OpenAI, Ollama, and more
- Extensible agentic architecture (LangGraph-ready)

## Project Structure
```
mlflow-analyzer/
├── data/                # MLflow experiment logs and artifacts
├── mlflow_agentic_env/  # Conda environment (ignored by git)
├── src/
│   ├── llm/             # Inference engine for LLM providers
│   ├── notebooks/       # Data generation and validation notebooks
│   └── scripts/         # Utility scripts
├── environment.yml      # Conda environment definition
├── requirements.txt     # Python dependencies
├── .gitignore           # Files/folders to ignore in git
└── README.md            # Project documentation
```
