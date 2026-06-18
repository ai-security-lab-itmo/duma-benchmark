from __future__ import annotations

from urllib.parse import urlparse


DEFAULT_PROVIDER = "openai"


def normalize_model_ref(model: str | None) -> str:
    """Normalize model references to canonical `vendor/model` format.

    Canonical examples:
    - `openai/gpt-4o`
    - `meta-llama/llama-3.3-70b-instruct`
    - `qwen/qwen3-235b-a22b`

    Supported legacy inputs:
    - `gpt-4o` -> `openai/gpt-4o`
    - `openai/gpt-4o` -> `openai/gpt-4o`
    - `openai/meta-llama/llama-3.3-70b-instruct`
      -> `meta-llama/llama-3.3-70b-instruct`
    - `openrouter/openai/gpt-4o` -> `openai/gpt-4o`
    - `openrouter/meta-llama/llama-3.3-70b-instruct`
      -> `meta-llama/llama-3.3-70b-instruct`
    """
    if model is None:
        return ""
    value = str(model).strip()
    if not value:
        return value

    if value.startswith("openrouter/"):
        inner = value.split("/", 1)[1]
        if "/" in inner:
            return inner
        return f"openai/{inner}"

    if value.startswith("huggingface/"):
        inner = value.split("/", 1)[1]
        if "/" in inner:
            return inner
        return f"huggingface/{inner}"

    if value.startswith("openai/"):
        inner = value.split("/", 1)[1]
        if "/" in inner:
            return inner
        return f"openai/{inner}"

    if value.startswith("anthropic/"):
        inner = value.split("/", 1)[1]
        if "/" in inner:
            return inner
        return f"anthropic/{inner}"

    if "/" not in value:
        # Backward compatibility: plain model ids are treated as OpenAI-family ids.
        return f"openai/{value}"

    return value


def is_reasoning_model(model: str | None) -> bool:
    """Return True for OpenAI reasoning models (gpt-5 family, o1/o3/o4 series).

    These models reject ``temperature`` values other than the default (1); the
    benchmark sends ``temperature=0`` to every agent, so it must be stripped for
    them. Matching is done on the canonical (vendor-stripped) model id.
    """
    if not model:
        return False
    canonical = normalize_model_ref(model).lower()
    # Drop a leading ``vendor/`` so ``openai/gpt-5-mini`` -> ``gpt-5-mini``.
    bare = canonical.split("/", 1)[1] if "/" in canonical else canonical
    if bare.startswith("gpt-5"):
        return True
    # o1 / o3 / o4 reasoning series, e.g. "o3-mini", "o1", "o4-mini".
    for prefix in ("o1", "o3", "o4"):
        if bare == prefix or bare.startswith(f"{prefix}-"):
            return True
    return False


def infer_provider(
    model: str | None,
    api_base: str | None = None,
    explicit_provider: str | None = None,
) -> str:
    """Infer LiteLLM provider from explicit setting, model ref, and api_base."""
    if explicit_provider:
        return explicit_provider.strip().lower()

    if isinstance(model, str):
        model_l = model.strip().lower()
        if model_l.startswith("openrouter/"):
            return "openrouter"
        if model_l.startswith("huggingface/"):
            return "huggingface"

    if api_base:
        host = urlparse(api_base).netloc.lower()
        if "openrouter.ai" in host:
            return "openrouter"
        if "router.huggingface.co" in host or "api-inference.huggingface.co" in host:
            return "huggingface"

    return DEFAULT_PROVIDER


def to_litellm_model(model: str | None, provider: str) -> str:
    """Build LiteLLM model string from canonical model ref + provider."""
    canonical = normalize_model_ref(model)
    if not canonical:
        return canonical

    provider_l = provider.strip().lower()
    provider_prefix = f"{provider_l}/"
    if canonical.startswith(provider_prefix):
        return canonical

    return f"{provider_l}/{canonical}"


def normalize_for_reporting(model: str | None) -> str:
    """Normalize model names for metrics/reporting comparisons."""
    return normalize_model_ref(model)
