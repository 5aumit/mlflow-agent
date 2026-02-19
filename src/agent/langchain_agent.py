"""
LangGraph-based agent for MLflow Experiment Q&A
----------------------------------------------
This agent uses LangGraph to orchestrate LLM reasoning and MLflow tool calls.
"""

import os
import json
import sys
import time
import threading
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from llm.inference_engine import GroqEngine, get_llm_from_config
from mlflow_tools import data_access

import logging
# logging.getLogger("mlflow").setLevel(logging.ERROR)

os.environ["MLFLOW_LOGGING_LEVEL"] = "WARNING"

import asyncio
import json
from langchain.agents import create_agent            # correct agent factory
from langchain.tools import tool     


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
print(f"Loading config from: {CONFIG_PATH}")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# LLM selection logic
llm_config = config.get('llm', {})
print(f"Groq API Key: {llm_config.get('groq_api_key', 'Not Set')}")
print(f"Loading Model from config: {llm_config.get('groq_model', 'Not Set')}")
llm = get_llm_from_config(llm_config)

mlflow_tools = data_access.get_all_tools()

agent = create_agent(
    model=llm,
    tools=mlflow_tools,
    system_prompt="You are an MLflow experiment assistant. Use tools as needed.",
)


def run_query(user_query: str):
    messages = [{"role": "user", "content": user_query}]
    result = agent.invoke({"messages": messages})
    return result



def loading_animation(message, duration=3):
    spinner = ['|', '/', '-', '\\']
    print(message, end='', flush=True)
    print('\n', end='', flush=True)
    for i in range(duration * 4):
        print(f' {spinner[i % 4]}', end='\r', flush=True)
        time.sleep(0.25)
    print(' ' * (len(message) + 2), end='\r')

def main():
    print("\n==============================")
    print("  Welcome to MLflow Agent CLI  ")
    print("==============================")
    print("Initializing agent and loading tools...")
    loading_animation("Starting up, please wait...", duration=3)
    print("Agent is ready! Type 'exit' to quit.")
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
