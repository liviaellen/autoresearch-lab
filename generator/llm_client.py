"""
Provider-agnostic LLM client powered by LiteLLM.

Supports any model LiteLLM supports:
    - OpenAI:     "gpt-4o", "gpt-4o-mini"
    - Anthropic:  "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"
    - Ollama:     "ollama/llama3.2", "ollama/mistral", "ollama/deepseek-r1"
    - Together:   "together_ai/meta-llama/Llama-3-70b-chat-hf"
    - Groq:       "groq/llama-3.1-70b-versatile"
    - OpenRouter: "openrouter/meta-llama/llama-3-70b-instruct"
    - vLLM:       "hosted_vllm/model-name"
    - Any OpenAI-compatible API via base_url

Environment variables for API keys:
    OPENAI_API_KEY      — OpenAI
    ANTHROPIC_API_KEY   — Anthropic/Claude
    TOGETHER_API_KEY    — Together AI
    GROQ_API_KEY        — Groq
    OPENROUTER_API_KEY  — OpenRouter
    (Ollama needs no key, just a running server)
"""

import os
from dataclasses import dataclass, field


def _get_litellm():
    """Lazy import litellm — only needed when LLM mode is used."""
    try:
        import litellm
        return litellm
    except ImportError:
        raise ImportError(
            "litellm is required for LLM mode. Install it with:\n"
            "  uv pip install litellm\n"
            "  # or\n"
            "  pip install autoresearch-lab[llm]"
        )


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    model: str = "ollama/llama3.2"
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: str | None = None
    api_key: str | None = None
    stop: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMConfig":
        """Create config from environment variables."""
        return cls(
            model=model or os.environ.get("AUTORESEARCH_MODEL", "ollama/llama3.2"),
            base_url=os.environ.get("AUTORESEARCH_BASE_URL"),
            api_key=os.environ.get("AUTORESEARCH_API_KEY"),
        )


# Common model presets for quick selection
PRESETS = {
    "local": "ollama/llama3.2",
    "local-large": "ollama/llama3.1:70b",
    "local-small": "ollama/llama3.2:1b",
    "local-code": "ollama/deepseek-coder-v2",
    "local-mistral": "ollama/mistral",
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "claude-opus": "claude-opus-4-20250514",
    "groq-llama": "groq/llama-3.1-70b-versatile",
    "together-llama": "together_ai/meta-llama/Llama-3-70b-chat-hf",
}


def resolve_model(model_or_preset: str) -> str:
    """Resolve a preset name to a full model string, or pass through as-is."""
    return PRESETS.get(model_or_preset, model_or_preset)


def chat(
    messages: list[dict],
    config: LLMConfig | None = None,
    **kwargs,
) -> str:
    """Send a chat completion request and return the response text.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts.
        config: LLM configuration. Uses defaults if None.
        **kwargs: Additional kwargs passed to litellm.completion.

    Returns:
        The assistant's response text.
    """
    if config is None:
        config = LLMConfig.from_env()

    call_kwargs = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        **kwargs,
    }

    if config.base_url:
        call_kwargs["base_url"] = config.base_url
    if config.api_key:
        call_kwargs["api_key"] = config.api_key
    if config.stop:
        call_kwargs["stop"] = config.stop

    litellm = _get_litellm()
    response = litellm.completion(**call_kwargs)
    return response.choices[0].message.content


def chat_stream(
    messages: list[dict],
    config: LLMConfig | None = None,
    **kwargs,
):
    """Stream a chat completion, yielding chunks of text.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts.
        config: LLM configuration.
        **kwargs: Additional kwargs passed to litellm.completion.

    Yields:
        Text chunks as they arrive.
    """
    if config is None:
        config = LLMConfig.from_env()

    call_kwargs = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": True,
        **kwargs,
    }

    if config.base_url:
        call_kwargs["base_url"] = config.base_url
    if config.api_key:
        call_kwargs["api_key"] = config.api_key
    if config.stop:
        call_kwargs["stop"] = config.stop

    litellm = _get_litellm()
    response = litellm.completion(**call_kwargs)
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content
