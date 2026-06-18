"""Tests for transient-error classification and the backoff retry wrapper."""

import pytest
from litellm import (
    APIConnectionError,
    APIError,
    BadGatewayError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from duma.utils import llm_utils
from duma.utils.llm_utils import _completion_with_retry, _is_transient_llm_error


def _rate_limit():
    return RateLimitError("rate limited", llm_provider="openai", model="m")


def _timeout():
    return Timeout("timed out", model="m", llm_provider="openai")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm_utils.time, "sleep", lambda *_: None)


def test_transient_classification_true():
    assert _is_transient_llm_error(_rate_limit())
    assert _is_transient_llm_error(_timeout())
    assert _is_transient_llm_error(
        APIConnectionError(message="conn", llm_provider="openai", model="m")
    )
    assert _is_transient_llm_error(
        ServiceUnavailableError(message="down", llm_provider="openai", model="m")
    )
    # Real provider 5xx classes litellm actually raises. These descend from
    # openai.APIError (NOT litellm.APIError), so they must be classified via their
    # status_code, not via an isinstance(litellm.APIError) check.
    assert _is_transient_llm_error(
        InternalServerError("boom", model="m", llm_provider="openai")
    )
    assert _is_transient_llm_error(
        BadGatewayError("bad gw", model="m", llm_provider="openai")
    )
    # A generic litellm.APIError carrying a 429 status is still transient.
    assert _is_transient_llm_error(
        APIError(status_code=429, message="e", llm_provider="openai", model="m")
    )


def test_transient_classification_false():
    assert not _is_transient_llm_error(
        BadRequestError("bad", model="m", llm_provider="openai")
    )
    assert not _is_transient_llm_error(
        NotFoundError("missing", model="m", llm_provider="openai")
    )
    assert not _is_transient_llm_error(
        APIError(status_code=400, message="e", llm_provider="openai", model="m")
    )
    assert not _is_transient_llm_error(ValueError("boom"))


def test_retry_succeeds_after_transient_blips():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _timeout()
        return "ok"

    assert _completion_with_retry(call, max_retries=3) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhaustion():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise _rate_limit()

    with pytest.raises(RateLimitError):
        _completion_with_retry(call, max_retries=2)
    assert calls["n"] == 3  # initial + 2 retries


def test_permanent_error_not_retried():
    calls = {"n": 0}

    def call():
        calls["n"] += 1
        raise BadRequestError("bad request", model="m", llm_provider="openai")

    with pytest.raises(BadRequestError):
        _completion_with_retry(call, max_retries=3)
    assert calls["n"] == 1  # no retries for a permanent error
