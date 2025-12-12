"""
LLM client wrappers for different providers.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Make API call to the LLM"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client wrapper"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def call(self, prompt: str, temperature: float = 0.1, **kwargs) -> str:
        """Make API call to OpenAI"""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            **kwargs
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic Claude API client wrapper"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Anthropic client"""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client
    
    def call(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        """Make API call to Anthropic"""
        client = self._get_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

