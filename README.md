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

## Getting Started
1. **Clone the repository**
2. **Set up the environment:**
   ```bash
   conda env create -f environment.yml
   conda activate ./mlflow_agentic_env/
   ```
3. **Generate experiment data:**
   - Run the Jupyter notebooks in `src/notebooks/data/` to populate MLflow logs.
4. **Run the CLI agent:**
   - (Coming soon) Use the CLI to ask questions about your experiments.

## Inference Engine
- Supports Groq (default), with placeholders for OpenAI and Ollama.
- Easily extensible to other LLM providers.

## Agentic System
- Designed for integration with agent frameworks (e.g., LangGraph).
- Agent interprets user queries, plans actions, and synthesizes answers using MLflow data and LLMs.

## Contributing
Pull requests and suggestions are welcome!

## License
MIT License
