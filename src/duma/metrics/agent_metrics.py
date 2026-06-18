import math
import re

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from duma.data_model.simulation import Results, RunStatus

# Runs whose reward is a measurement artifact rather than a policy outcome and must
# therefore be dropped from the pass@k denominator (not counted as failures).
EXCLUDED_RUN_STATUSES = {RunStatus.ERRORED.value, RunStatus.INVALID.value}


def is_successful(reward: float | None) -> bool:
    """
    Check if the reward is successful.
    """
    if reward is None:
        return False
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


class AgentMetrics(BaseModel):
    avg_reward: float
    pass_hat_ks: dict[int, float]
    avg_agent_cost: float
    success_rate: float

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
            "success_rate": self.success_rate,
        }
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()
    # Exclude infrastructure crashes and inert (zero-tool) runs: their reward is a
    # measurement artifact and must not inflate the pass@k denominator as failures.
    if "run_status" in df.columns:
        n_before = len(df)
        df = df[~df["run_status"].isin(EXCLUDED_RUN_STATUSES)].copy()
        n_excluded = n_before - len(df)
        if n_excluded:
            logger.warning(
                f"Excluded {n_excluded} infra-crashed/invalid run(s) from metrics "
                f"(run_status in {sorted(EXCLUDED_RUN_STATUSES)})."
            )
    if df.empty:
        logger.warning("No evaluable runs remain after excluding crashed/invalid runs.")
        return df, 0
    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = int(df.info_num_trials.max())

    # After exclusions some tasks may keep fewer than max_k trials. We deliberately do
    # NOT clamp max_k globally (that would truncate pass^k depth for every task because
    # one task lost trials); instead get_tasks_pass_hat_k computes each pass^k per task
    # only where enough trials survive. Surface the shortfall.
    survivors = df.task_id.value_counts()
    short = survivors[survivors < max_k]
    if not short.empty:
        logger.warning(
            f"{len(short)} task(s) have fewer than {max_k} evaluable trials after "
            "exclusions; deeper pass^k averages only over tasks retaining >=k trials. "
            f"Examples: {short.head(5).to_dict()}"
        )
    return df, max_k


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    if df.empty or max_k < 1:
        return pd.DataFrame()
    dfs = []
    for k in range(1, max_k + 1):
        res = df.groupby("task_id")["success"].apply(
            lambda g, k=k: (
                pass_hat_k(len(g), int(g.sum()), k) if len(g) >= k else float("nan")
            )
        )
        res.name = f"pass^{k}"
        dfs.append(res)
    df_pass_hat_k = pd.concat(dfs, axis=1)
    task_columns = [
        "task_num_agent_actions",
        "task_num_user_actions",
        "task_num_actions",
    ]
    df_task_infos = df.groupby("task_id").first()[task_columns]
    df_pass_hat_k = df_task_infos.join(df_pass_hat_k)
    return df_pass_hat_k


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    if df.empty or df_pass_hat_k.empty:
        return df, df_pass_hat_k
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df, df_pass_hat_k


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute metrics for the agent.
    - average reward
    - pass^k
    """
    df, df_pass_hat_k = prepare_dfs(results)
    avg_reward = df.reward.mean() if not df.empty else 0.0
    success_rate = df.success.mean() if not df.empty else 0.0
    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()
    avg_agent_cost = df.agent_cost.mean() if not df.empty else 0.0
    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        avg_agent_cost=avg_agent_cost,
        success_rate=success_rate,
    )


def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward}")
    print(f"✅ Success rate: {metrics.success_rate}")
    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
