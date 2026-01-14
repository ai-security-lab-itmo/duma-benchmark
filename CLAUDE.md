# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

τ²-Bench is a Python benchmark framework for evaluating AI agents on security-focused conversational tasks. It simulates conversations between an AI agent and a user, then evaluates agent adherence to security policies and robustness against adversarial scenarios.

## Build & Development Commands

```bash
# Installation (Python 3.10+)
pip install -e .

# Testing
make test              # Run pytest tests/
pytest tests/test_domains/collab  # Single domain tests

# Linting
make lint              # Check with ruff
make format            # Format with ruff
make lint-fix          # Auto-fix lint issues
make check-all         # Both lint and format

# CLI usage
tau2 run --domain collab --agent-llm gpt-4o-mini --num-tasks 5
tau2 view              # Interactive results browser
tau2 domain collab     # Start API server + view docs
tau2 check-data        # Verify data directory setup
```

## Architecture

### Core Simulation Loop

```
CLI (cli.py) → run.py → Orchestrator
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
               Agent (LLM)         User (LLM)
                    ↓                   ↓
               Environment ←── Tools (domain-specific)
                    ↓
               Evaluator pipeline → Metrics
```

**Orchestrator** (`src/tau2/orchestrator/orchestrator.py`): Manages the message passing loop between agent, user, and environment until task completion or max steps.

**Registry** (`src/tau2/registry.py`): Maps domain/agent/user names to their implementations. All components are registered here.

### Key Components

| Module | Purpose |
|--------|---------|
| `src/tau2/agent/llm_agent.py` | LLM agent using LiteLLM for multi-provider support |
| `src/tau2/user/user_simulator.py` | LLM-based user that follows task persona |
| `src/tau2/environment/environment.py` | Holds domain policy, tools, and state |
| `src/tau2/evaluator/` | Multi-stage evaluation (action, env, communication, output) |
| `src/tau2/config.py` | Default LLM models, simulation limits, caching settings |

### Domain Structure

Each domain in `src/tau2/domains/{domain}/` follows this pattern:
- `data_model.py`: Pydantic models for domain state
- `tools.py`: ToolKit subclass with `@is_tool()` decorated methods
- `environment.py`: Factory functions `get_environment()` and `get_tasks()`
- `user_tools.py` (optional): User-only tools

Corresponding data in `data/tau2/domains/{domain}/`:
- `policy.md`: Rules the agent must follow
- `tasks.json`: Task definitions with user scenarios and evaluation criteria
- `db.json` or `db.toml`: Initial domain state

### Evaluation

Rewards are 0-1 scores composed from multiple evaluators:
- **EnvironmentEvaluator**: Post-simulation state assertions
- **ActionEvaluator**: Required tool calls verification
- **CommunicateEvaluator**: Communication quality (NLP-based)
- **OutputAssertionsEvaluator**: LLM-as-judge output checks

Task passes when reward ≥ 0.999999. Pass^k metric measures success probability across k trials.

## Data Model

Key Pydantic models in `src/tau2/data_model/`:
- `RunConfig`: Simulation configuration (domains, LLMs, parallelism)
- `SimulationRun`: Single task trial with messages and reward
- `Task`: User scenario + evaluation criteria
- `Message` types: `UserMessage`, `AssistantMessage`, `ToolResponseMessage`

## Environment Variables

Copy `.env.example` to `.env` and configure:
- LiteLLM API keys for your LLM providers
- `LLM_CACHE_ENABLED=True` for Redis caching
- `USE_LANGFUSE=True` for observability

## Adding a New Domain

1. Create `src/tau2/domains/my_domain/` with `data_model.py`, `tools.py`, `environment.py`
2. Add data files to `data/tau2/domains/my_domain/`
3. Register in `src/tau2/registry.py`
4. Add tests to `tests/test_domains/my_domain/`

## Leaderboard Submission

```bash
tau2 submit prepare data/tau2/simulations/*.json --output ./submission
tau2 submit validate ./submission
```
Requires results from all 6 core domains. Submit via PR to `web/leaderboard/public/submissions/`.
