import argparse
import json
import os

from duma.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
)
from duma.data_model.simulation import RunConfig
from duma.run import get_options, run_domain, run_domains
from duma.utils.model_ref import infer_provider, normalize_model_ref


def add_run_args(parser):
    """Add run arguments to a parser."""
    domains = get_options().domains
    domain_choices = domains + ["all"]
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=domain_choices,
        help="The domain to run the simulation on (use --domains for multiple domains, or 'all' for every domain)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        type=str,
        choices=domain_choices,
        help="List of domains to run the simulation on (saves to multi-domain format). Use 'all' to run every registered domain.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=get_options().agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=(
            "The model to use for the agent in vendor/model format "
            f"(e.g. openai/gpt-4o, meta-llama/llama-3.3-70b-instruct). Default is {DEFAULT_LLM_AGENT}."
        ),
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_AGENT},
        help=f"The arguments to pass to the LLM for the agent. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_AGENT}}}'.",
    )
    parser.add_argument(
        "--agent-base-url",
        type=str,
        default=None,
        help="Optional base URL for the agent LLM API (overrides api_base in agent-llm-args).",
    )
    parser.add_argument(
        "--agent-provider",
        type=str,
        default=None,
        help=(
            "Optional LiteLLM provider for agent model routing "
            "(e.g. openai, openrouter, huggingface). "
            "If omitted, inferred from base URL and model."
        ),
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=get_options().users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=(
            "The model to use for the user simulator in vendor/model format "
            f"(e.g. openai/gpt-4o, qwen/qwen3-235b-a22b). Default is {DEFAULT_LLM_USER}."
        ),
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_USER},
        help=f"The arguments to pass to the LLM for the user. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_USER}}}'.",
    )
    parser.add_argument(
        "--user-base-url",
        type=str,
        default=None,
        help="Optional base URL for the user LLM API (overrides api_base in user-llm-args).",
    )
    parser.add_argument(
        "--user-provider",
        type=str,
        default=None,
        help=(
            "Optional LiteLLM provider for user model routing "
            "(e.g. openai, openrouter, huggingface). "
            "If omitted, inferred from base URL and model."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help=(
            "Environment variable name for the LLM API key. "
            "Defaults to provider-specific values (e.g., OPENROUTER_API_KEY for openrouter/*, "
            "OPENAI_API_KEY otherwise)."
        ),
    )
    parser.add_argument(
        "--local-models",
        action="store_true",
        help="Use local models (skip injecting api_key/api_base).",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=get_options().task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs. If not provided, will run all tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="The number of tasks to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a row in the simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>.json. If not provided, will save to <domain>_<agent>_<user>_<llm_agent>_<llm_user>_<timestamp>.json. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )


def _default_api_key_env_for_provider(provider: str | None) -> str:
    if not provider:
        return "OPENAI_API_KEY"
    provider = provider.lower()
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "huggingface":
        return "HUGGINGFACE_API_KEY"
    return "OPENAI_API_KEY"


def _default_api_base_for_provider(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"
    if provider == "huggingface":
        return "https://router.huggingface.co/v1"
    return os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


def _build_llm_args(
    model: str | None,
    base_args: dict,
    base_url: str | None,
    provider: str | None,
    api_key_env: str | None,
    use_local: bool,
) -> dict:
    """Merge base args with optional api_base and api_key injection.

    If use_local is True, api_key/api_base are stripped to support local providers.
    """
    args = dict(base_args or {})
    if use_local:
        args.pop("api_key", None)
        args.pop("api_base", None)
        args.pop("custom_llm_provider", None)
        return args

    inferred_provider = infer_provider(
        model=model,
        api_base=base_url or args.get("api_base"),
        explicit_provider=provider,
    )

    if base_url:
        args["api_base"] = base_url
    else:
        args.setdefault("api_base", _default_api_base_for_provider(inferred_provider))

    env_name = api_key_env or _default_api_key_env_for_provider(inferred_provider)
    api_key = os.getenv(env_name)
    if api_key:
        args.setdefault("api_key", api_key)

    # Keep provider explicit for LiteLLM routing; model remains canonical vendor/model.
    args.setdefault("custom_llm_provider", inferred_provider)

    return args


def main():
    parser = argparse.ArgumentParser(description="Duma command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    add_run_args(run_parser)
    run_parser.set_defaults(func=lambda args: run_command(args))

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to the simulation results file to view",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'collab', 'crm_leak')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Start command
    start_parser = subparsers.add_parser("start", help="Start all servers")
    start_parser.set_defaults(func=lambda args: run_start_servers())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


def run_command(args):
    """Run command handler that supports both single and multi-domain runs."""
    agent_llm_args = _build_llm_args(
        args.agent_llm,
        args.agent_llm_args,
        args.agent_base_url,
        args.agent_provider,
        args.api_key_env,
        args.local_models,
    )
    user_llm_args = _build_llm_args(
        args.user_llm,
        args.user_llm_args,
        args.user_base_url,
        args.user_provider,
        args.api_key_env,
        args.local_models,
    )
    agent_model = normalize_model_ref(args.agent_llm)
    user_model = normalize_model_ref(args.user_llm)

    config = RunConfig(
        domain=args.domain or "",  # Will be ignored if domains is provided
        is_remote=False,
        task_set_name=args.task_set_name,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        agent=args.agent,
        llm_agent=agent_model,
        llm_args_agent=agent_llm_args,
        user=args.user,
        llm_user=user_model,
        llm_args_user=user_llm_args,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        save_to=args.save_to,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        log_level=args.log_level,
    )

    available_domains = get_options().domains

    if args.domains:
        # Multi-domain run
        if args.domain:
            raise ValueError(
                "Cannot use both --domain and --domains. Use --domains for multiple domains."
            )
        domains = available_domains if "all" in args.domains else args.domains
        return run_domains(domains=domains, config=config)
    else:
        # Single domain run
        if not args.domain:
            raise ValueError("Either --domain or --domains must be specified.")
        if args.domain == "all":
            # Treat single-domain 'all' as a multi-domain run
            return run_domains(domains=available_domains, config=config)
        config.domain = args.domain
        return run_domain(config)


def run_view_simulations(args):
    from duma.scripts.view_simulations import main as view_main

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
    )


def run_show_domain(args):
    from duma.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_start_servers():
    from duma.scripts.start_servers import main as start_main

    start_main()


def run_check_data():
    from duma.scripts.check_data import main as check_data_main

    check_data_main()


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from duma.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(args.paths, args.output_dir)


if __name__ == "__main__":
    main()
