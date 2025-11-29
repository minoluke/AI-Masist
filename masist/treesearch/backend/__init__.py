"""Backend module for LLM integrations"""

import logging
import os

from . import backend_anthropic, backend_openai
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

logger = logging.getLogger(__name__)

# 環境変数でAPI呼び出しログを制御 (デフォルト: 有効)
LOG_API_CALLS = os.environ.get("MASIST_LOG_API_CALLS", "1") == "1"


def get_ai_client(model: str, **model_kwargs):
    """
    Get the appropriate AI client based on the model string.

    Args:
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        **model_kwargs: Additional keyword arguments for model configuration.
    Returns:
        An instance of the appropriate AI client.
    """
    if "claude-" in model:
        return backend_anthropic.get_ai_client(model=model, **model_kwargs)
    else:
        return backend_openai.get_ai_client(model=model, **model_kwargs)


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message
        user_message (PromptType | None): Uncompiled user message
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at.
        max_tokens (int | None, optional): Maximum number of tokens to generate.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    # API呼び出しログ
    if LOG_API_CALLS:
        if "deepseek" in model.lower():
            logger.info(f"[API] DeepSeek ({model})")
        elif "claude-" in model:
            logger.info(f"[API] Anthropic ({model})")
        else:
            logger.info(f"[API] OpenAI ({model})")

    query_func = backend_anthropic.query if "claude-" in model else backend_openai.query
    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message=compile_prompt_to_md(system_message) if system_message else None,
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output


__all__ = [
    "query",
    "get_ai_client",
    "FunctionSpec",
    "OutputType",
    "PromptType",
    "compile_prompt_to_md",
]
