"""
LangGraph-based agent for MLflow Experiment Q&A
----------------------------------------------
This agent uses LangGraph to orchestrate LLM reasoning and MLflow tool calls.
"""
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from llm.inference_engine import GroqEngine
from mlflow_tools import data_access

import asyncio
import json
from langchain.agents import create_agent            # correct agent factory
from langchain.tools import tool     

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

groq_api_key = config.get('groq_api_key')
groq_model = config.get('groq_model', 'moonshotai/kimi-k2-instruct')
groq_params = config.get('groq_params', {})

# Set up inference engine
llm = GroqEngine(api_key=groq_api_key, model=groq_model, **groq_params).llm
mlflow_tools = data_access.get_all_tools()

agent = create_agent(
    model=llm,                       # pass model instance, not llm.generate_response
    tools=mlflow_tools,
    system_prompt="You are an MLflow experiment assistant. Use tools as needed.",
)


def run_query(user_query: str):
    messages = [{"role": "user", "content": user_query}]
    result = agent.invoke({"messages": messages})
    return result


def main():
    print("LangGraph MLflow Agent CLI. Type 'exit' to quit.")
    while True:
        user_query = input("\n> ")
        if user_query.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = run_query(user_query)
        # Print the last message content if available, else print the whole result
        try:
            print(f"\n{result['messages'][-1].content}")
        except Exception:
            print(f"\n{result}")

if __name__ == "__main__":
    main()
