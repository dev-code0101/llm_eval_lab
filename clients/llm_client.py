"""
LLM client wrappers for different providers.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional


# Provider configuration mapping
_PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HF_TOKEN"
}


def get_api_key_for_provider(provider: str) -> Optional[str]:
    """
    Get the appropriate API key for the given provider from environment variables.
    
    Args:
        provider: Provider name (openai, anthropic, huggingface)
    
    Returns:
        API key string or None if not found
    """
    env_var = _PROVIDER_ENV_VARS.get(provider.lower())
    if env_var:
        return os.getenv(env_var)
    return None


def get_api_key_env_var_name(provider: str) -> str:
    """
    Get the environment variable name for the given provider.
    
    Args:
        provider: Provider name (openai, anthropic, huggingface)
    
    Returns:
        Environment variable name, or "API_KEY" as default if provider not recognized
    """
    return _PROVIDER_ENV_VARS.get(provider.lower(), "API_KEY")


def resolve_api_key(provider: str, config_api_key: Optional[str] = None) -> Optional[str]:
    """
    Resolve API key for a provider from environment variables or config.
    
    Priority:
    1. Environment variable (provider-specific)
    2. Config API key (if provided)
    
    Args:
        provider: Provider name (openai, anthropic, huggingface)
        config_api_key: Optional API key from config file
    
    Returns:
        API key string or None if not found
    """
    # Try environment variable first
    env_api_key = get_api_key_for_provider(provider)
    if env_api_key:
        return env_api_key
    
    # Fall back to config API key
    return config_api_key


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
        self.api_key = api_key or get_api_key_for_provider("openai")
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
        self.api_key = api_key or get_api_key_for_provider("anthropic")
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


class HuggingFaceClient(LLMClient):
    """Hugging Face API client wrapper (OpenAI-compatible)"""
    
    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or get_api_key_for_provider("huggingface")
        self.base_url = base_url or "https://router.huggingface.co/v1"
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Hugging Face client (using OpenAI-compatible API)"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")
            except Exception as e:
                print(f"\n\nUnknown Error during HuggingFaceClient initialization: {e}")
                if not self.api_key:
                    print(f"API key is not set for provider: {self.provider}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        return self._client
    
    def call(self, prompt: str, temperature: float = 0.1, **kwargs) -> str:
        """Make API call to Hugging Face (OpenAI-compatible endpoint)"""
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"},
            # **kwargs
        )
        return response.choices[0].message.content

