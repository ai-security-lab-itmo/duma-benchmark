"""
Functions for collecting and analyzing simulation results.

This module provides utilities to load simulation files, compute metrics,
and generate comprehensive dataframes for analysis.
"""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binom

from tau2.data_model.simulation import MultiDomainResults, Results
from tau2.metrics.agent_metrics import is_successful, pass_hat_k


def load_simulation_file(file_path: str | Path) -> Dict[str, Results]:
    """
    Load a simulation file and return a dictionary mapping domain names to Results.
    Handles both single-domain (Results) and multi-domain (MultiDomainResults) formats.

    Args:
        file_path: Path to the simulation JSON file

    Returns:
        Dictionary mapping domain names to Results objects
    """
    file_path = Path(file_path)

    # Try to load as MultiDomainResults first
    try:
        multi_domain_results = MultiDomainResults.load(file_path)
        return multi_domain_results.domains
    except Exception:
        # Fall back to single-domain Results format
        try:
            results = Results.load(file_path)
            domain_name = results.info.environment_info.domain_name
            return {domain_name: results}
        except Exception as e:
            raise ValueError(f"Failed to load simulation file {file_path}: {e}")


def load_simulations(file_paths: List[str | Path]) -> Dict[str, Results]:
    """
    Load multiple simulation files and combine them into a single dictionary.

    Args:
        file_paths: List of paths to simulation JSON files

    Returns:
        Dictionary mapping domain names to Results objects
        (if multiple files have the same domain, they will be merged)
    """
    all_domains = {}

    for file_path in file_paths:
        domains = load_simulation_file(file_path)
        for domain_name, results in domains.items():
            if domain_name in all_domains:
                # Merge simulations from the same domain
                all_domains[domain_name].simulations.extend(deepcopy(results.simulations))
                # Merge tasks (avoid duplicates)
                existing_task_ids = {task.id for task in all_domains[domain_name].tasks}
                for task in results.tasks:
                    if task.id not in existing_task_ids:
                        all_domains[domain_name].tasks.append(deepcopy(task))
            else:
                all_domains[domain_name] = deepcopy(results)

    return all_domains


def compute_task_metrics(results: Results, task_id: str) -> Dict[str, Any]:
    """
    Compute metrics for a specific task within a Results object.

    Args:
        results: Results object containing simulations
        task_id: ID of the task to compute metrics for

    Returns:
        Dictionary containing computed metrics
    """
    # Filter simulations for this task
    task_simulations = [sim for sim in results.simulations if sim.task_id == task_id]

    if not task_simulations:
        return {}

    # Compute basic metrics
    rewards = [sim.reward_info.reward if sim.reward_info else 0.0 for sim in task_simulations]
    successes = [is_successful(r) for r in rewards]
    agent_costs = [sim.agent_cost if sim.agent_cost else 0.0 for sim in task_simulations]
    user_costs = [sim.user_cost if sim.user_cost else 0.0 for sim in task_simulations]
    durations = [sim.duration for sim in task_simulations]
    num_messages = [len(sim.messages) for sim in task_simulations]

    num_trials = len(task_simulations)
    success_count = sum(successes)

    metrics = {
        "num_trials": num_trials,
        "success_count": success_count,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_agent_cost": float(np.mean(agent_costs)) if agent_costs else None,
        "avg_user_cost": float(np.mean(user_costs)) if user_costs else None,
        "avg_duration": float(np.mean(durations)),
        "avg_num_messages": float(np.mean(num_messages)),
    }

    # Compute pass^k metrics
    if num_trials > 0:
        for k in range(1, min(num_trials + 1, 5)):  # Compute pass^1 to pass^4
            if num_trials >= k:
                metrics[f"pass^{k}"] = float(pass_hat_k(num_trials, success_count, k))
        
        # Compute statistical significance metrics for pass@1
        stats = compute_statistical_significance(success_count, num_trials, method='wilson')
        metrics['pass^1_ci_lower'] = stats['ci_lower']
        metrics['pass^1_ci_upper'] = stats['ci_upper']
        metrics['pass^1_proportion'] = stats['proportion']

    return metrics


def generate_metrics_table(simulation_files: List[str | Path]) -> pd.DataFrame:
    """
    Generate a comprehensive metrics table from simulation files.

    Args:
        simulation_files: List of paths to simulation JSON files

    Returns:
        DataFrame with columns: domain, user_model, user_model_params,
        agent_model, agent_model_params, task, and various metrics
    """
    rows = []

    # Process each file separately to preserve parameter differences
    for file_path in simulation_files:
        # Load domains from this file
        file_domains = load_simulation_file(file_path)

        for domain_name, results in file_domains.items():
            # Extract configuration info from this specific Results object
            user_model = results.info.user_info.llm
            user_model_params = (
                json.dumps(results.info.user_info.llm_args)
                if results.info.user_info.llm_args
                else "{}"
            )
            agent_model = results.info.agent_info.llm
            agent_model_params = (
                json.dumps(results.info.agent_info.llm_args)
                if results.info.agent_info.llm_args
                else "{}"
            )

            # Get unique tasks for this domain
            task_ids = set(sim.task_id for sim in results.simulations)

            for task_id in task_ids:
                # Compute metrics for this task
                task_metrics = compute_task_metrics(results, task_id)

                if not task_metrics:
                    continue

                # Create row
                row = {
                    "domain": domain_name,
                    "user_model": user_model,
                    "user_model_params": user_model_params,
                    "agent_model": agent_model,
                    "agent_model_params": agent_model_params,
                    "task": task_id,
                    **task_metrics,
                }

                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Reorder columns to put metrics at the end
    metric_columns = [
        col
        for col in df.columns
        if col
        not in [
            "domain",
            "user_model",
            "user_model_params",
            "agent_model",
            "agent_model_params",
            "task",
        ]
    ]
    column_order = [
        "domain",
        "user_model",
        "user_model_params",
        "agent_model",
        "agent_model_params",
        "task",
    ] + metric_columns
    df = df[column_order]

    return df


def visualize_metrics(
    simulation_files: List[str | Path],
    show_table: bool = True,
    show_summary: bool = True,
) -> pd.DataFrame:
    """
    Visualize metrics from simulation files.

    Args:
        simulation_files: List of paths to simulation JSON files
        show_table: Whether to display the full table
        show_summary: Whether to display summary statistics

    Returns:
        DataFrame with metrics
    """
    # Generate metrics table
    df = generate_metrics_table(simulation_files)

    if df.empty:
        print("No data found in simulation files.")
        return df

    if show_summary:
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"\nTotal unique configurations: {len(df)}")
        print(f"Domains: {df['domain'].nunique()} ({', '.join(df['domain'].unique())})")
        print(f"Tasks: {df['task'].nunique()}")
        print(f"User models: {df['user_model'].nunique()}")
        print(f"Agent models: {df['agent_model'].nunique()}")

        if "avg_reward" in df.columns:
            print(f"\nOverall average reward: {df['avg_reward'].mean():.4f}")
        if "pass^1" in df.columns:
            print(f"Overall pass^1: {df['pass^1'].mean():.4f}")
        if "avg_agent_cost" in df.columns and df["avg_agent_cost"].notna().any():
            print(f"Overall average agent cost: {df['avg_agent_cost'].mean():.4f}")

    if show_table:
        print("\n" + "=" * 80)
        print("METRICS TABLE")
        print("=" * 80)
        # Display with better formatting
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 50)
        print(df.to_string(index=False))

    return df


def wilson_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson confidence interval for binomial proportion.
    
    Suitable for small sample sizes (n >= 3). The Wilson interval performs
    better than normal approximation for small samples and is more accurate
    than Clopper-Pearson for small n.
    
    Args:
        successes: Number of successful trials
        trials: Total number of trials
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf((1 + confidence) / 2)  # z-score for confidence level
    p_hat = successes / trials
    
    denominator = 1 + (z**2 / trials)
    center = (p_hat + (z**2 / (2 * trials))) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1 - p_hat) / trials) + (z**2 / (4 * trials**2)))
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return (lower, upper)


def fisher_exact_test(group1_successes: int, group1_trials: int, 
                     group2_successes: int, group2_trials: int) -> Tuple[float, str]:
    """
    Perform Fisher's exact test to compare two binomial proportions.
    
    Suitable for small sample sizes. More appropriate than chi-square for n < 20.
    
    Args:
        group1_successes: Number of successes in group 1
        group1_trials: Total trials in group 1
        group2_successes: Number of successes in group 2
        group2_trials: Total trials in group 2
    
    Returns:
        Tuple of (p-value, significance_level) where significance_level is:
        '***' for p < 0.001, '**' for p < 0.01, '*' for p < 0.05, '' otherwise
    """
    group1_failures = group1_trials - group1_successes
    group2_failures = group2_trials - group2_successes
    
    contingency_table = [[group1_successes, group1_failures],
                        [group2_successes, group2_failures]]
    
    try:
        _, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
    except ValueError:
        # Edge case handling
        if group1_successes == group2_successes == 0 or \
           group1_failures == group2_failures == 0:
            p_value = 1.0
        else:
            p_value = 1.0
    
    # Determine significance level
    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = ''
    
    return (p_value, sig_level)


def bootstrap_confidence_interval(
    successes: int, 
    trials: int, 
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for binomial proportion.
    
    Alternative method for small samples using resampling.
    
    Args:
        successes: Number of successful trials
        trials: Total number of trials
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 0.0)
    
    # Create original data
    data = [1] * successes + [0] * (trials - successes)
    
    # Bootstrap resampling
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=trials, replace=True)
        bootstrap_samples.append(np.mean(sample))
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_samples, 100 * alpha / 2)
    upper = np.percentile(bootstrap_samples, 100 * (1 - alpha / 2))
    
    return (max(0.0, lower), min(1.0, upper))


def compute_statistical_significance(
    successes: int,
    trials: int,
    reference_successes: Optional[int] = None,
    reference_trials: Optional[int] = None,
    method: str = 'wilson'
) -> Dict[str, Any]:
    """
    Compute statistical significance metrics for binomial proportion.
    
    Args:
        successes: Number of successful trials
        trials: Total number of trials
        reference_successes: Number of successes in reference group (for comparison)
        reference_trials: Total trials in reference group (for comparison)
        method: Method for CI calculation ('wilson' or 'bootstrap')
    
    Returns:
        Dictionary with:
        - proportion: success rate
        - ci_lower: lower bound of confidence interval
        - ci_upper: upper bound of confidence interval
        - p_value: p-value from Fisher's exact test (if reference provided)
        - significance: significance level ('***', '**', '*', '')
    """
    if trials == 0:
        return {
            'proportion': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'p_value': None,
            'significance': ''
        }
    
    proportion = successes / trials
    
    # Calculate confidence interval
    if method == 'bootstrap':
        ci_lower, ci_upper = bootstrap_confidence_interval(successes, trials)
    else:  # Default to Wilson
        ci_lower, ci_upper = wilson_confidence_interval(successes, trials)
    
    result = {
        'proportion': proportion,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': None,
        'significance': ''
    }
    
    # Compare with reference if provided
    if reference_successes is not None and reference_trials is not None:
        p_value, significance = fisher_exact_test(
            successes, trials,
            reference_successes, reference_trials
        )
        result['p_value'] = p_value
        result['significance'] = significance
    
    return result


def format_pass_k_with_ci(
    successes: int,
    trials: int,
    method: str = 'wilson',
    significance: str = ''
) -> str:
    """
    Format pass@k metric with confidence interval for LaTeX tables.
    
    Args:
        successes: Number of successful trials
        trials: Total number of trials
        method: Method for CI calculation
        significance: Significance level to append
    
    Returns:
        Formatted string like "3/6 (50%) [CI: 18%--82%]***"
    """
    if trials == 0:
        return "0/0 (0%)"
    
    proportion = successes / trials
    ci_lower, ci_upper = wilson_confidence_interval(successes, trials) if method == 'wilson' \
                        else bootstrap_confidence_interval(successes, trials)
    
    # Format as percentage
    pct_str = f"{proportion:.0%}" if proportion < 0.01 or proportion > 0.99 else f"{proportion:.1%}"
    
    # Format CI
    ci_lower_pct = f"{ci_lower:.0%}" if ci_lower < 0.01 or ci_lower > 0.99 else f"{ci_lower:.1%}"
    ci_upper_pct = f"{ci_upper:.0%}" if ci_upper < 0.01 or ci_upper > 0.99 else f"{ci_upper:.1%}"
    
    result = f"{successes}/{trials} ({pct_str}) [CI: {ci_lower_pct}--{ci_upper_pct}]"
    if significance:
        result += significance
    
    return result

