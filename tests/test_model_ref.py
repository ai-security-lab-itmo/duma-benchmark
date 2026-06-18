import pytest

from duma.utils.model_ref import (
    infer_provider,
    is_reasoning_model,
    normalize_for_reporting,
    normalize_model_ref,
    to_litellm_model,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-5", True),
        ("gpt-5-mini", True),
        ("openai/gpt-5-nano", True),
        ("o1", True),
        ("o3-mini", True),
        ("o4-mini", True),
        ("openrouter/openai/gpt-5", True),
        ("gpt-4o", False),
        ("gpt-4.1-mini", False),
        ("anthropic/claude-opus-4.5", False),
        ("deepseek/deepseek-v3.2", False),
        (None, False),
        ("", False),
    ],
)
def test_is_reasoning_model(model, expected):
    assert is_reasoning_model(model) is expected


def test_normalize_model_ref_plain_openai_model():
    assert normalize_model_ref("gpt-4o-mini") == "openai/gpt-4o-mini"


def test_normalize_model_ref_legacy_openrouter_format():
    assert normalize_model_ref("openrouter/openai/gpt-4o") == "openai/gpt-4o"


def test_normalize_model_ref_provider_prefixed_non_openai_vendor():
    assert (
        normalize_model_ref("openai/meta-llama/llama-3.3-70b-instruct")
        == "meta-llama/llama-3.3-70b-instruct"
    )


def test_normalize_model_ref_vendor_model_is_stable():
    assert normalize_model_ref("qwen/qwen3-235b-a22b") == "qwen/qwen3-235b-a22b"


def test_infer_provider_prefers_explicit_provider():
    provider = infer_provider(
        model="meta-llama/llama-3.3-70b-instruct",
        api_base="https://api.vsellm.ru/v1",
        explicit_provider="openai",
    )
    assert provider == "openai"


def test_infer_provider_from_api_base():
    assert (
        infer_provider(model="openai/gpt-4o", api_base="https://openrouter.ai/api/v1")
        == "openrouter"
    )
    assert (
        infer_provider(
            model="meta-llama/llama-3.1-8b-instruct",
            api_base="https://router.huggingface.co/v1",
        )
        == "huggingface"
    )


def test_to_litellm_model_openai_provider_keeps_vendor_model():
    assert (
        to_litellm_model("meta-llama/llama-3.3-70b-instruct", "openai")
        == "openai/meta-llama/llama-3.3-70b-instruct"
    )


def test_to_litellm_model_openrouter_provider_wraps_canonical_model():
    assert to_litellm_model("openai/gpt-4o", "openrouter") == "openrouter/openai/gpt-4o"


def test_normalize_for_reporting_uses_canonical_format():
    assert (
        normalize_for_reporting("openrouter/openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    )
