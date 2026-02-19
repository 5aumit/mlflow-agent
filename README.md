# Floki: MLflow Experiment Agentic Chatbot

> **⚠️ Work in Progress:** This project is actively being developed. Features, structure, and documentation may change frequently.

Floki is named after the legendary Viking engineer Flóki Vilgerðarson, who built innovative boats that enabled Vikings to explore new lands. This project aims to empower ML researchers to explore their experiment logs with the same spirit of discovery.

A CLI-based assistant for ML experimentation, inspired by Claude Code, that helps researchers query, analyze, and gain insights from MLflow experiment logs.
# Floki: MLflow Experiment Agentic Chatbot


Floki is named after the legendary Viking engineer Flóki Vilgerðarson, who built innovative boats that enabled Vikings to explore new lands. This project aims to empower ML researchers to explore their experiment logs with the same spirit of discovery.

A CLI-based assistant for ML experimentation, inspired by Claude Code, that helps researchers query, analyze, and gain insights from MLflow experiment logs.

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
