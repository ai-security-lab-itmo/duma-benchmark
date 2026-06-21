import os

# SIMULATION
DEFAULT_MAX_STEPS = 200
# Dual-control conversations should not run away (an adversarial user can push a
# correctly-refusing agent indefinitely). Cap dual runs well below the solo cap so
# they terminate as MAX_STEPS / USER_STOP instead of burning tokens to 200 steps.
# See docs/arr/dual_control_remediation_plan.md §1.4.
DEFAULT_MAX_STEPS_DUAL = 50
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
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
    "temperature": DEFAULT_LLM_NL_ASSERTIONS_TEMPERATURE,
    "api_base": os.environ.get("EVALUATOR_API_BASE", "https://api.vsellm.ru/v1"),
    "api_key": os.environ.get(
        "EVALUATOR_API_KEY", os.environ.get("VSE_LLM_API_KEY", "")
    ),
}

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
