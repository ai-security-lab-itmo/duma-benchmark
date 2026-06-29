import os

# SIMULATION
DEFAULT_MAX_STEPS = 100
# Dual-control conversations should not run away (an adversarial user can push a
# correctly-refusing agent indefinitely). Cap dual runs well below the solo cap so
# they terminate as MAX_STEPS / USER_STOP instead of burning tokens to 200 steps.
# See docs/arr/dual_control_remediation_plan.md §1.4.
DEFAULT_MAX_STEPS_DUAL = 100
# The user simulator gives up (natural USER_STOP) after this many consecutive
# unproductive turns (no tool activity), instead of looping to the step cap.
DEFAULT_USER_GIVE_UP_TURNS = 6
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1"
DEFAULT_LLM_USER = "gpt-4.1"
DEFAULT_LLM_TEMPERATURE_AGENT = 0.0
DEFAULT_LLM_TEMPERATURE_USER = 0.0
DEFAULT_LLM_ARGS_AGENT = {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
DEFAULT_LLM_ARGS_USER = {"temperature": DEFAULT_LLM_TEMPERATURE_USER}

DEFAULT_LLM_NL_ASSERTIONS = "openai/gpt-4o-mini"
DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE = 0.0
# The NL-assertions judge is NOT hardcoded to a specific proxy. It only pins an
# endpoint when one is explicitly configured (env EVALUATOR_API_BASE/KEY, or the
# duma run --evaluator-* flags via configure_nl_evaluator); otherwise generate()
# routes it the same way as the agent/user (e.g. OpenRouter), avoiding a surprising
# dependency on one (possibly flaky) proxy.
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
    "temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE,
}
_EVALUATOR_API_BASE = os.environ.get("EVALUATOR_API_BASE")
_EVALUATOR_API_KEY = os.environ.get("EVALUATOR_API_KEY") or os.environ.get(
    "VSE_LLM_API_KEY"
)
if _EVALUATOR_API_BASE:
    DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_base"] = _EVALUATOR_API_BASE
if _EVALUATOR_API_KEY:
    DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_key"] = _EVALUATOR_API_KEY


def configure_nl_evaluator(
    llm: "str | None" = None,
    api_base: "str | None" = None,
    api_key: "str | None" = None,
) -> None:
    """Override the NL-assertions judge endpoint at runtime (e.g. from the CLI).

    Called once before a run starts; the evaluator reads these globals at call time.
    """
    global DEFAULT_LLM_NL_ASSERTIONS
    if llm:
        DEFAULT_LLM_NL_ASSERTIONS = llm
    if api_base:
        DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_base"] = api_base
    if api_key:
        DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_key"] = api_key


DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1"
DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE = 0.0
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"temperature": DEFAULT_LLM_ENV_INTERFACE_TEMPERATURE}

# LITELLM
DEFAULT_MAX_RETRIES = 3
# Orchestrator-level resilience: a single transient API error (429/5xx/timeout) on
# agent OR user generation must not kill an entire run. generate() retries transient
# failures with exponential backoff + jitter before the error becomes fatal.
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0
RETRY_JITTER_SECONDS = 0.5
# Run-level resilience: if a whole simulation still ends as an infrastructure crash
# (run_status == ERRORED) after per-call retries, re-run it up to this many times
# before giving up. Recovers a data point instead of merely excluding it.
DEFAULT_MAX_RUN_RETRIES = 2
# Caching is opt-in via env (LLM_CACHE_ENABLED=true). Default cache type is "local"
# (in-process, safe without external infra); set DEFAULT_LLM_CACHE_TYPE=redis to share
# a prompt cache across the experiment runner's subprocesses.
LLM_CACHE_ENABLED = os.environ.get("LLM_CACHE_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_LLM_CACHE_TYPE = os.environ.get("DEFAULT_LLM_CACHE_TYPE", "local")

# REDIS CACHE
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""
REDIS_PREFIX = "duma"
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# LANGFUSE
USE_LANGFUSE = False  # If True, make sure all the env variables are set for langfuse.

# API
API_PORT = 8000
