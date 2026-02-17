from abc import ABC, abstractmethod

from langchain_groq import ChatGroq
from langchain_core.tools import Tool as LangChainTool
from typing import List

class InferenceEngine(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate a response from the inference provider given a prompt."""
        pass


class GroqEngine(InferenceEngine):
    def __init__(self, api_key: str, model: str = "moonshotai/kimi-k2-instruct", **kwargs):
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = kwargs
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model,
            **self.generation_kwargs
        )

    def generate_response(self, prompt: str, model: str = None, **kwargs) -> str:
        # Simple synchronous call for compatibility
        return self.llm.invoke(prompt)

    def bind_tools(self, tools: List[LangChainTool]):
        # Return a new ChatGroq instance with tools bound, as per LangChain API
        return self.llm.bind_tools(tools)

class OpenAIEngine(InferenceEngine):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Placeholder for OpenAI client initialization
        pass

    def generate_response(self, prompt: str) -> str:
        # TODO: Implement OpenAI API call
        raise NotImplementedError("OpenAI integration not implemented yet.")

class OllamaEngine(InferenceEngine):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        # Placeholder for Ollama client initialization
        pass

    def generate_response(self, prompt: str) -> str:
        # TODO: Implement Ollama API call
        raise NotImplementedError("Ollama integration not implemented yet.")
