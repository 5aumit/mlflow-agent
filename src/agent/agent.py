"""
Agent setup for MLflow Experiment Q&A
-------------------------------------
Integrates MLflow tools and LLM inference engine for agentic Q&A.
"""
from typing import Any
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add src/ to path
from llm.inference_engine import GroqEngine  # Add other engines as needed
from mlflow_tools import data_access

# Simple tool registry for agent
TOOLS = {
    'list_experiments': data_access.list_experiments,
    'list_runs': data_access.list_runs,
    'get_run_metrics': data_access.get_run_metrics,
    'get_run_params': data_access.get_run_params,
    'find_best_run_by_metric': data_access.find_best_run_by_metric,
}

class MLflowAgent:
    def __init__(self, inference_engine: Any):
        self.inference_engine = inference_engine
        self.tools = TOOLS

    def answer_query(self, query: str) -> str:
        # For now, just echo the query and show available tools
        # Later: Use LLM to parse query and call tools as needed
        response = self.inference_engine.generate_response(
            f"User query: {query}\nAvailable tools: {list(self.tools.keys())}"
        )
        return response

# Example CLI loop
if __name__ == "__main__":
    # Read Groq API key from config.json
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
    # config_path = "config.json"  # Assuming config.json is in the current working directory
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        groq_api_key = config.get('groq_api_key', None)
        groq_model = config.get('groq_model', 'meta-llama/llama-guard-4-12b')
        groq_params = config.get('groq_params', {})
        if not groq_api_key:
            raise ValueError("groq_api_key not found in config.json")
    except Exception as e:
        print(f"Error loading Groq config from config.json: {e}")
        exit(1)

    engine = GroqEngine(api_key=groq_api_key, model=groq_model, **groq_params)
    agent = MLflowAgent(engine)
    print("MLflow Experiment Agent CLI. Type 'exit' to quit.")
    while True:
        user_query = input("\n> ")
        if user_query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        answer = agent.answer_query(user_query)
        print(f"\n{answer}")
