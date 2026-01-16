#!/usr/bin/env bash
set -euo pipefail

# Max-load runner for retry_failed_experiments.py.
# Defaults to OpenRouter models (https://openrouter.ai/models).
# Auto-backs off concurrency if rate limits/timeouts detected.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

API_KEY_ENV_NAME="${API_KEY_ENV_NAME:-OPENROUTER_API_KEY}"
: "${!API_KEY_ENV_NAME:?${API_KEY_ENV_NAME} is not set in environment}"

# Export OPENAI_API_KEY too if caller uses OpenRouter, for compatibility with
# any scripts/tools that still look for OPENAI_API_KEY.
if [ "${API_KEY_ENV_NAME}" = "OPENROUTER_API_KEY" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${OPENROUTER_API_KEY}"
fi

MODELS=("openrouter/openai/gpt-4o" "openrouter/openai/gpt-4o-mini")
TEMPS=("0.0" "0.5" "1.0")
NUM_TRIALS="10"

# Aggressive starting point.
# Approx total parallel simulations ~= MAX_CONCURRENCY * TAU2_MAX_CONCURRENCY
MAX_CONCURRENCY="4"        # parallel tau2 run processes
TAU2_MAX_CONCURRENCY="4"   # parallel trials inside one tau2 run
PROCESS_TIMEOUT_SECONDS="14400"  # 4h per tau2 run

MIN_MAX_CONCURRENCY="1"
MIN_TAU2_MAX_CONCURRENCY="1"

mkdir -p logs
RUN_LOG="logs/max_load_retry_$(date +%Y%m%d_%H%M%S).log"

log() {
  echo "$*" | tee -a "${RUN_LOG}"
}

log "Logging to: ${RUN_LOG}"
log "Start: max_concurrency=${MAX_CONCURRENCY}, tau2_max_concurrency=${TAU2_MAX_CONCURRENCY}"

while true; do
  log "== Running with max_concurrency=${MAX_CONCURRENCY} tau2_max_concurrency=${TAU2_MAX_CONCURRENCY} =="

  set +e

  # Run via pseudo-TTY so rich progress bars render.
  # Note: log will include terminal control sequences.
  script -q /dev/null \
    ./.venv/bin/python -u scripts/retry_failed_experiments.py \
      --models "${MODELS[@]}" \
      --temperatures "${TEMPS[@]}" \
      --num-trials "${NUM_TRIALS}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --tau2-max-concurrency "${TAU2_MAX_CONCURRENCY}" \
      --process-timeout-seconds "${PROCESS_TIMEOUT_SECONDS}" \
    2>&1 | tee -a "${RUN_LOG}"

  exit_code=${PIPESTATUS[0]}
  set -e

  # Stop if script itself failed hard (e.g., dependency error)
  if [ "${exit_code}" -ne 0 ]; then
    log "retry_failed_experiments.py exited with code ${exit_code}"
  fi

  # If everything is complete, exit.
  if rg -q "No failed experiments to retry|All experiments have complete result files" "${RUN_LOG}"; then
    log "All done."
    exit 0
  fi

  # Detect rate limits (OpenAI/LiteLLM style) and back off.
  if rg -qi "rate limit|ratelimit|429|too many requests|try again in|tpm|rpm" "${RUN_LOG}"; then
    log "Rate limit detected -> backing off"

    # Prefer reducing internal concurrency first.
    if [ "${TAU2_MAX_CONCURRENCY}" -gt "${MIN_TAU2_MAX_CONCURRENCY}" ]; then
      TAU2_MAX_CONCURRENCY=$((TAU2_MAX_CONCURRENCY - 1))
      continue
    fi

    # Then reduce number of parallel processes.
    if [ "${MAX_CONCURRENCY}" -gt "${MIN_MAX_CONCURRENCY}" ]; then
      MAX_CONCURRENCY=$((MAX_CONCURRENCY - 1))
      continue
    fi

    log "Already at minimum concurrency; cannot back off further."
    exit 1
  fi

  # If timeouts recorded by our retry script, also back off a bit.
  if [ -f retry_errors.log ] && rg -qi "TIMEOUT" retry_errors.log; then
    log "Timeouts detected -> backing off"
    if [ "${TAU2_MAX_CONCURRENCY}" -gt "${MIN_TAU2_MAX_CONCURRENCY}" ]; then
      TAU2_MAX_CONCURRENCY=$((TAU2_MAX_CONCURRENCY - 1))
    elif [ "${MAX_CONCURRENCY}" -gt "${MIN_MAX_CONCURRENCY}" ]; then
      MAX_CONCURRENCY=$((MAX_CONCURRENCY - 1))
    fi
    continue
  fi

  log "No rate limit detected; continuing with same concurrency."
done
