# scripts/

Utilities for experiment settings and results in `data/duma/simulations`.

## Paper protocol

| Experiment | Target models | Evaluation regime | Trials per model and task |
|---|---|---|---|
| Main comparison | 14 (Table 4) | Solo and dual-control | 4 per regime |
| User-temperature sweep | 9 (Appendix B) | Dual-control | 5 per temperature (0.0, 0.5, 1.0) |

Use `--num-trials 4` or `--num-trials 5` explicitly; `duma run` defaults
to one trial. Use `--user-llm openai/gpt-4o-mini` in dual-control and
`--agent llm_agent_solo --user dummy_user` in solo. The communication judge
is `--evaluator-llm openai/gpt-4o` at temperature 0.0 in both regimes.
The simulator temperature does not change the agent or judge temperature.

The experiment helper has the same simulator and judge defaults. Its
default model and domain lists are examples, not the paper's experiment
matrix. Use the paper's model settings, task list, and seeds rather than
assuming the helper defaults reproduce the reported numbers.

## Verify metrics correctness

```bash
./.venv/bin/python scripts/test_all_metrics.py
```

## Example suite configuration

[`benchmark_config_example.json`](benchmark_config_example.json) illustrates
the suite configuration with one trial and a limited model and domain list.
It is not a paper reproduction configuration.
