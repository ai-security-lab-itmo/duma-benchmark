import argparse
from types import SimpleNamespace

import pytest

import duma.cli as cli
from duma import config
from duma.evaluator import evaluator_nl_assertions
from duma.registry import RegistryInfo


def _args(**overrides):
    data = {
        "domain": None,
        "domains": None,
        "task_set_name": None,
        "task_ids": None,
        "num_tasks": None,
        "agent": "llm_agent",
        "agent_llm": "gpt-4o-mini",
        "agent_llm_args": {"temperature": 0.0},
        "agent_base_url": None,
        "agent_provider": None,
        "evaluator_llm": None,
        "user": "user_simulator",
        "user_llm": "gpt-4o-mini",
        "user_llm_args": {"temperature": 0.0},
        "user_base_url": None,
        "user_provider": None,
        "api_key_env": None,
        "local_models": False,
        "num_trials": 1,
        "max_steps": 5,
        "max_errors": 5,
        "save_to": None,
        "max_concurrency": 1,
        "seed": 123,
        "log_level": "INFO",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _options():
    return RegistryInfo(
        domains=["collab", "crm_leak"],
        agents=["llm_agent"],
        users=["user_simulator"],
        task_sets=["collab", "crm_leak"],
    )


def test_run_command_requires_domain_or_domains(monkeypatch):
    monkeypatch.setattr(cli, "get_options", lambda: _options())
    with pytest.raises(ValueError, match="Either --domain or --domains"):
        cli.run_command(_args())


def test_run_command_rejects_domain_and_domains_together(monkeypatch):
    monkeypatch.setattr(cli, "get_options", lambda: _options())
    with pytest.raises(ValueError, match="Cannot use both --domain and --domains"):
        cli.run_command(_args(domain="collab", domains=["crm_leak"]))


def test_run_command_expands_domains_all(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "get_options", lambda: _options())
    monkeypatch.setattr(
        cli,
        "run_domains",
        lambda domains, config: called.update({"domains": domains, "domain": config.domain}) or "ok",
    )
    cli.run_command(_args(domains=["all"]))
    assert called["domains"] == ["collab", "crm_leak"]
    assert called["domain"] == ""


def test_run_command_domain_all_uses_multi_domain(monkeypatch):
    called = {}

    monkeypatch.setattr(cli, "get_options", lambda: _options())
    monkeypatch.setattr(
        cli,
        "run_domains",
        lambda domains, config: called.update({"domains": domains}) or "ok",
    )
    cli.run_command(_args(domain="all"))
    assert called["domains"] == ["collab", "crm_leak"]


@pytest.mark.parametrize(
    "judge, expected_model",
    [(None, config.DEFAULT_LLM_NL_ASSERTIONS), ("openai/gpt-4o", "openai/gpt-4o")],
)
def test_run_command_passes_evaluator_model_to_judge(monkeypatch, judge, expected_model):
    default_model = config.DEFAULT_LLM_NL_ASSERTIONS
    calls = []

    def generate(**kwargs):
        calls.append((kwargs["model"], kwargs["temperature"]))
        return SimpleNamespace(content='{"results": []}')

    def run_domain(run_config):
        assert run_config.llm_user == "openai/gpt-4o-mini"
        evaluator_nl_assertions.NLAssertionsEvaluator.evaluate_nl_assertions(
            [], ["The response follows policy."]
        )

    monkeypatch.setattr(cli, "get_options", _options)
    monkeypatch.setattr(cli, "run_domain", run_domain)
    monkeypatch.setattr(evaluator_nl_assertions, "generate", generate)
    parser = argparse.ArgumentParser()
    cli.add_run_args(parser)
    args = ["--domain", "collab", "--user-llm", "gpt-4o-mini"]
    if judge is not None:
        args.extend(["--evaluator-llm", judge])
    cli.run_command(parser.parse_args(args))

    assert calls == [(expected_model, 0.0)]
    assert config.DEFAULT_LLM_NL_ASSERTIONS == default_model


def test_build_llm_args_local_models_strips_api_credentials():
    args = cli._build_llm_args(
        model="gpt-4o-mini",
        base_args={"temperature": 0.2, "api_key": "k", "api_base": "https://x"},
        base_url=None,
        provider=None,
        api_key_env=None,
        use_local=True,
    )
    assert args == {"temperature": 0.2}


def test_build_llm_args_provider_specific_env_injection(monkeypatch):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    args = cli._build_llm_args(
        model="openrouter/openai/gpt-4o",
        base_args={},
        base_url=None,
        provider=None,
        api_key_env=None,
        use_local=False,
    )
    assert args["api_key"] == "or-key"
    assert args["api_base"] == "https://openrouter.ai/api/v1"
    assert args["custom_llm_provider"] == "openrouter"


def test_build_llm_args_custom_api_key_env(monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "custom-key")
    args = cli._build_llm_args(
        model="gpt-4o-mini",
        base_args={},
        base_url="https://custom.base/v1",
        provider=None,
        api_key_env="MY_API_KEY",
        use_local=False,
    )
    assert args["api_key"] == "custom-key"
    assert args["api_base"] == "https://custom.base/v1"
    assert args["custom_llm_provider"] == "openai"


def test_build_llm_args_explicit_provider_for_vendor_model(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "test-key")
    args = cli._build_llm_args(
        model="meta-llama/llama-3.3-70b-instruct",
        base_args={},
        base_url="https://api.example.com/v1",
        provider="openai",
        api_key_env="CUSTOM_API_KEY",
        use_local=False,
    )
    assert args["api_key"] == "test-key"
    assert args["api_base"] == "https://api.example.com/v1"
    assert args["custom_llm_provider"] == "openai"
