#!/usr/bin/env python3
"""
Script to generate LaTeX tables with statistical significance from simulation results.

This script:
1. Loads simulation results
2. Computes all metrics (pass@1, ASR, avg_reward, costs, duration, messages) with confidence intervals
3. Performs statistical significance tests
4. Generates LaTeX table code with confidence intervals and significance markers
"""

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tau2.data_model.simulation import MultiDomainResults, Results
from tau2.metrics.agent_metrics import is_successful

# Import statistical functions
# Note: result_collection is in the same directory
from result_collection import (
    load_simulation_file,
    load_simulations,
    wilson_confidence_interval,
    fisher_exact_test,
    compute_statistical_significance,
    format_pass_k_with_ci,
    t_confidence_interval,
    t_test_independent,
    compare_conversion_metrics,
    compare_continuous_metrics,
    compute_task_metrics,
)


def _normalize_model_name(model: str | None) -> str:
    """Normalize provider-prefixed model names for reporting/comparison."""
    if not isinstance(model, str):
        return str(model)
    if model.startswith("openrouter/"):
        return model.split("/")[-1]
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def escape_latex(text: str) -> str:
    """
    Escape special LaTeX characters in text.
    
    Args:
        text: Text to escape
    
    Returns:
        Escaped text safe for LaTeX
    """
    # Escape special characters
    text = str(text)
    text = text.replace('\\', '\\textbackslash{}')
    text = text.replace('_', '\\_')
    text = text.replace('&', '\\&')
    text = text.replace('%', '\\%')
    text = text.replace('$', '\\$')
    text = text.replace('#', '\\#')
    text = text.replace('^', '\\textasciicircum{}')
    text = text.replace('{', '\\{')
    text = text.replace('}', '\\}')
    return text


def format_continuous_metric_with_ci(
    values: List[float],
    metric_name: str,
    significance: str = '',
    precision: int = 2
) -> str:
    """
    Format continuous metric with confidence interval for LaTeX tables.
    
    Args:
        values: List of metric values
        metric_name: Name of the metric (for formatting)
        significance: Significance level to append
        precision: Number of decimal places
    
    Returns:
        Formatted string like "0.85 [CI: 0.72--0.98]*"
    """
    if len(values) == 0:
        return "N/A"
    
    mean = float(np.mean(values))
    ci_lower, ci_upper = t_confidence_interval(values)
    
    # Format based on metric type
    if 'cost' in metric_name.lower():
        # Costs in dollars - экранируем $ для LaTeX, используем меньше знаков после запятой
        mean_str = f"\\${mean:.{min(precision, 2)}f}"
        ci_str = f"\\${ci_lower:.{min(precision, 2)}f}--\\${ci_upper:.{min(precision, 2)}f}"
    elif 'duration' in metric_name.lower():
        # Duration in seconds - округляем до целых
        mean_str = f"{mean:.0f}s"
        ci_str = f"{ci_lower:.0f}--{ci_upper:.0f}s"
    else:
        # General metric - используем меньше знаков
        mean_str = f"{mean:.{min(precision, 2)}f}"
        ci_str = f"{ci_lower:.{min(precision, 2)}f}--{ci_upper:.{min(precision, 2)}f}"
    
    # Более компактный формат: убираем "CI:" и используем короткий формат
    result = f"{mean_str} [{ci_str}]"
    if significance:
        result += significance
    
    return result


def format_asr_with_ci(
    successes: int,
    trials: int,
    significance: str = ''
) -> str:
    """
    Format ASR (Attack Success Rate) with confidence interval.
    
    Args:
        successes: Number of failures (trials - successes)
        trials: Total number of trials
        significance: Significance level to append
    
    Returns:
        Formatted string like "50\\% [CI: 18\\%--82\\%]*" (экранированные % для LaTeX)
    """
    if trials == 0:
        return "N/A"
    
    failures = trials - successes
    asr = failures / trials
    # ASR CI is 1 - pass@1 CI (inverted)
    pass1_ci_lower, pass1_ci_upper = wilson_confidence_interval(successes, trials)
    asr_ci_lower = 1 - pass1_ci_upper
    asr_ci_upper = 1 - pass1_ci_lower
    
    asr_pct = f"{asr:.0%}" if asr < 0.01 or asr > 0.99 else f"{asr:.1%}"
    ci_lower_pct = f"{asr_ci_lower:.0%}" if asr_ci_lower < 0.01 or asr_ci_lower > 0.99 else f"{asr_ci_lower:.1%}"
    ci_upper_pct = f"{asr_ci_upper:.0%}" if asr_ci_upper < 0.01 or asr_ci_upper > 0.99 else f"{asr_ci_upper:.1%}"
    
    # Экранировать % для LaTeX
    asr_pct = asr_pct.replace('%', '\\%')
    ci_lower_pct = ci_lower_pct.replace('%', '\\%')
    ci_upper_pct = ci_upper_pct.replace('%', '\\%')
    
    # Более компактный формат: убираем "CI:" для экономии места
    result = f"{asr_pct} [{ci_lower_pct}--{ci_upper_pct}]"
    if significance:
        result += significance
    
    return result


def compute_domain_stats(
    result_files: List[Path],
    domain: str,
    reference_model: str = "gpt-4o-mini",
    reference_temp: float = 0.0
) -> Dict:
    """
    Compute aggregated statistics for a domain.
    
    Args:
        result_files: List of result file paths
        domain: Domain name
        reference_model: Reference model for comparison
        reference_temp: Reference temperature for comparison
    
    Returns:
        Dictionary with aggregated metrics and raw values for statistical tests
    """
    reference_model = _normalize_model_name(reference_model)

    # Load all results for this domain
    domain_results = []
    for file_path in result_files:
        try:
            domains = load_simulation_file(file_path)
            if domain in domains:
                domain_results.append(domains[domain])
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue
    
    if not domain_results:
        return {}
    
    # Aggregate metrics across all tasks and configurations
    all_pass1_values = []
    all_asr_values = []
    all_rewards = []
    all_agent_costs = []
    all_user_costs = []
    all_durations = []
    all_num_messages = []
    
    # Store raw values for statistical tests
    reference_pass1_successes = 0
    reference_pass1_trials = 0
    reference_rewards = []
    reference_agent_costs = []
    reference_user_costs = []
    reference_durations = []
    reference_num_messages = []
    
    # Process each result
    for results in domain_results:
        model = _normalize_model_name(results.info.agent_info.llm)
        temp = (
            results.info.agent_info.llm_args.get("temperature", 0.0)
            if results.info.agent_info.llm_args
            else 0.0
        )
        
        task_ids = set(sim.task_id for sim in results.simulations)
        
        for task_id in task_ids:
            metrics = compute_task_metrics(results, task_id)
            if not metrics:
                continue
            
            # Check if this is the reference configuration
            is_reference = (model == reference_model and temp == reference_temp)
            
            # Collect metrics
            if 'pass^1' in metrics:
                all_pass1_values.append(metrics['pass^1'])
                all_asr_values.append(metrics.get('ASR', 1.0 - metrics['pass^1']))
                
                if is_reference:
                    reference_pass1_successes += metrics.get('success_count', 0)
                    reference_pass1_trials += metrics.get('num_trials', 0)
            
            if 'avg_reward' in metrics:
                all_rewards.append(metrics['avg_reward'])
                if is_reference:
                    # Get raw reward values
                    task_sims = [s for s in results.simulations if s.task_id == task_id]
                    reference_rewards.extend([
                        s.reward_info.reward if s.reward_info else 0.0 
                        for s in task_sims
                    ])
            
            if metrics.get('avg_agent_cost') is not None:
                all_agent_costs.append(metrics['avg_agent_cost'])
                if is_reference:
                    task_sims = [s for s in results.simulations if s.task_id == task_id]
                    reference_agent_costs.extend([
                        s.agent_cost if s.agent_cost else 0.0 
                        for s in task_sims
                    ])
            
            if metrics.get('avg_user_cost') is not None:
                all_user_costs.append(metrics['avg_user_cost'])
                if is_reference:
                    task_sims = [s for s in results.simulations if s.task_id == task_id]
                    reference_user_costs.extend([
                        s.user_cost if s.user_cost else 0.0 
                        for s in task_sims
                    ])
            
            if 'avg_duration' in metrics:
                all_durations.append(metrics['avg_duration'])
                if is_reference:
                    task_sims = [s for s in results.simulations if s.task_id == task_id]
                    reference_durations.extend([
                        s.duration for s in task_sims
                    ])
            
            if 'avg_num_messages' in metrics:
                all_num_messages.append(metrics['avg_num_messages'])
                if is_reference:
                    task_sims = [s for s in results.simulations if s.task_id == task_id]
                    reference_num_messages.extend([
                        len(s.messages) for s in task_sims
                    ])
    
    # Compute aggregated statistics
    stats = {
        'domain': domain,
        'pass1_mean': float(np.mean(all_pass1_values)) if all_pass1_values else 0.0,
        'asr_mean': float(np.mean(all_asr_values)) if all_asr_values else 0.0,
        'avg_reward_mean': float(np.mean(all_rewards)) if all_rewards else 0.0,
        'avg_agent_cost_mean': float(np.mean(all_agent_costs)) if all_agent_costs else None,
        'avg_user_cost_mean': float(np.mean(all_user_costs)) if all_user_costs else None,
        'avg_duration_mean': float(np.mean(all_durations)) if all_durations else 0.0,
        'avg_num_messages_mean': float(np.mean(all_num_messages)) if all_num_messages else 0.0,
        # Raw values for statistical tests
        'reference_pass1_successes': reference_pass1_successes,
        'reference_pass1_trials': reference_pass1_trials,
        'reference_rewards': reference_rewards,
        'reference_agent_costs': reference_agent_costs,
        'reference_user_costs': reference_user_costs,
        'reference_durations': reference_durations,
        'reference_num_messages': reference_num_messages,
    }
    
    return stats


def compare_with_reference(
    current_values: List[float],
    reference_values: List[float],
    metric_type: str = 'continuous'
) -> Tuple[float, str]:
    """
    Compare current values with reference using appropriate statistical test.
    
    Args:
        current_values: Current metric values
        reference_values: Reference metric values
        metric_type: 'continuous' or 'discrete'
    
    Returns:
        Tuple of (p-value, significance_level)
    """
    if len(current_values) == 0 or len(reference_values) == 0:
        return (1.0, '')
    
    if metric_type == 'discrete':
        # For discrete metrics, values should be [successes, trials] pairs
        if len(current_values) == 2 and len(reference_values) == 2:
            return compare_conversion_metrics(
                int(current_values[0]), int(current_values[1]),
                int(reference_values[0]), int(reference_values[1])
            )
    else:
        # Continuous metrics
        return compare_continuous_metrics(current_values, reference_values)
    
    return (1.0, '')


def _short_domain_label(domain: str) -> str:
    # Keep it compact for tables; detailed names stay in text.
    return {
        "mail_rag_phishing": "mail_rag",
        "output_handling": "output",
        "collab": "collab",
    }.get(domain, domain)


def _short_model_label(model: str) -> str:
    if model == "gpt-4o":
        return "4o"
    if model == "gpt-4o-mini":
        return "4o-mini"
    return model


def generate_detailed_metrics_table_latex(
    result_files: List[Path],
    domains: List[str],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate LaTeX table with detailed metrics for each task.
    
    Args:
        result_files: List of result file paths
        domains: List of domain names
        output_path: Optional path to save the table
    
    Returns:
        LaTeX table code as string
    """
    rows = []
    
    # Process each domain
    for domain in domains:
        # Load all results for this domain
        domain_results = []
        for file_path in result_files:
            try:
                domains_dict = load_simulation_file(file_path)
                if domain in domains_dict:
                    domain_results.append(domains_dict[domain])
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        if not domain_results:
            continue
        
        # Process each result configuration
        for results in domain_results:
            model = _normalize_model_name(results.info.agent_info.llm)
            # Temperature is configured on the user model.
            temp = (
                results.info.user_info.llm_args.get("temperature", 0.0)
                if results.info.user_info.llm_args
                else 0.0
            )
            
            task_ids = set(sim.task_id for sim in results.simulations)
            
            for task_id in task_ids:
                # Create temporary Results object for this task only
                task_sims = [deepcopy(s) for s in results.simulations if s.task_id == task_id]
                task_tasks = [deepcopy(t) for t in results.tasks if t.id == task_id]
                
                if not task_sims:
                    continue
                
                # Create temporary Results for metric computation
                temp_results = Results(
                    timestamp=results.timestamp,
                    info=results.info,
                    tasks=task_tasks,
                    simulations=task_sims
                )
                
                metrics = compute_task_metrics(temp_results, task_id)
                if not metrics:
                    continue
                
                # Format metrics
                pass1_str = format_pass_k_with_ci(
                    metrics.get('success_count', 0),
                    metrics.get('num_trials', 0),
                    significance=''
                )
                
                asr_str = format_asr_with_ci(
                    metrics.get('success_count', 0),
                    metrics.get('num_trials', 0),
                    significance=''
                )
                
                # Continuous metrics
                task_sims_raw = [s for s in results.simulations if s.task_id == task_id]
                rewards_raw = [s.reward_info.reward if s.reward_info else 0.0 for s in task_sims_raw]
                durations_raw = [float(s.duration) for s in task_sims_raw]
                num_messages_raw = [float(len(s.messages)) for s in task_sims_raw]
                
                reward_str = format_continuous_metric_with_ci(rewards_raw, 'reward', '') if rewards_raw else "N/A"
                duration_str = format_continuous_metric_with_ci(durations_raw, 'duration', '') if durations_raw else "N/A"
                num_messages_str = format_continuous_metric_with_ci(num_messages_raw, 'num_messages', '') if num_messages_raw else "N/A"
                
                rows.append({
                    'domain': domain,
                    'model': model,
                    'temp': temp,
                    'task': task_id,
                    'pass1': pass1_str,
                    'asr': asr_str,
                    'avg_reward': reward_str,
                    'avg_duration': duration_str,
                    'avg_num_messages': num_messages_str,
                })
    
    # Generate LaTeX table
    # Используем более компактные колонки и landscape для широкой таблицы
    # Уменьшаем шрифт и ширину колонок для лучшего размещения
    latex_lines = [
        "\\begin{landscape}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{longtable}{p{2.2cm}p{1.5cm}p{0.5cm}p{2.3cm}p{2.4cm}p{2.0cm}p{1.7cm}p{1.6cm}p{1.5cm}}",
        "\\caption{Детальные метрики по кейсам} \\label{tab:detailed_metrics} \\\\",
        "\\toprule",
        "\\shortstack{\\textbf{Домен}} & \\shortstack{\\textbf{Модель}} & \\textbf{T} & \\textbf{Кейс} & \\textbf{pass@1} & \\textbf{ASR} & \\shortstack{\\textbf{avg}\\\\\\textbf{reward}} & \\shortstack{\\textbf{duration}\\\\\\textbf{(s)}} & \\shortstack{\\textbf{num}\\\\\\textbf{msgs}} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{9}{c}{\\tablename\\ \\thetable{} -- продолжение} \\\\",
        "\\toprule",
        "\\shortstack{\\textbf{Домен}} & \\shortstack{\\textbf{Модель}} & \\textbf{T} & \\textbf{Кейс} & \\textbf{pass@1} & \\textbf{ASR} & \\shortstack{\\textbf{avg}\\\\\\textbf{reward}} & \\shortstack{\\textbf{duration}\\\\\\textbf{(s)}} & \\shortstack{\\textbf{num}\\\\\\textbf{msgs}} \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{9}{r}{Продолжение на следующей странице} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    
    for row in rows:
        # Экранировать все текстовые поля
        domain_escaped = escape_latex(_short_domain_label(row['domain']))
        model_escaped = escape_latex(_short_model_label(row['model']))
        # Сократить имя кейса: последние 2 сегмента (меньше коллизий чем просто 'trigger').
        task_name = row['task']
        parts = task_name.split('_') if '_' in task_name else [task_name]
        task_short = '_'.join(parts[-2:]) if len(parts) >= 2 else task_name

        if len(task_short) > 18:
            chunks = [
                escape_latex(task_short[i : i + 12])
                for i in range(0, len(task_short), 12)
            ]
            task_escaped = "\\shortstack{" + "\\\\".join(chunks) + "}"
        else:
            task_escaped = escape_latex(task_short)
        
        # Экранировать значения метрик, которые могут содержать спецсимволы
        # Но не экранируем уже отформатированные строки с CI, так как они уже содержат правильное форматирование
        pass1_val = row['pass1']  # Уже отформатировано с escape
        asr_val = row['asr']  # Уже отформатировано
        avg_reward_val = row['avg_reward']  # Уже отформатировано
        avg_duration_val = row['avg_duration']  # Уже отформатировано
        avg_num_messages_val = row['avg_num_messages']  # Уже отформатировано

        # Проверим, что все значения не пустые
        if not all([
            domain_escaped,
            model_escaped,
            task_escaped,
            pass1_val,
            asr_val,
            avg_reward_val,
            avg_duration_val,
            avg_num_messages_val,
        ]):
            print(
                f"Warning: Missing values in row for {row.get('domain', 'unknown')}/{row.get('task', 'unknown')}"
            )
            continue

        latex_lines.append(
            f"{domain_escaped} & {model_escaped} & {row['temp']} & {task_escaped} & "
            f"{pass1_val} & {asr_val} & {avg_reward_val} & "
            f"{avg_duration_val} & {avg_num_messages_val} \\\\" 
        )

    
    latex_lines.append("\\end{longtable}")
    latex_lines.append("\\normalsize",)  # Возвращаем нормальный размер шрифта
    latex_lines.append("\\end{landscape}")
    
    latex_code = "\n".join(latex_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code)
        print(f"Detailed metrics table saved to {output_path}")
    
    return latex_code


def generate_aggregated_table_latex(
    result_files: List[Path],
    domains: List[str],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate LaTeX table with aggregated metrics by domain.
    
    Args:
        result_files: List of result file paths
        domains: List of domain names
        output_path: Optional path to save the table
    
    Returns:
        LaTeX table code as string
    """
    # Compute stats for each domain
    domain_stats_list = []
    for domain in domains:
        stats = compute_domain_stats(result_files, domain)
        if stats:
            domain_stats_list.append(stats)
    
    if not domain_stats_list:
        return ""
    
    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Агрегированные метрики по доменам (смешанные конфигурации)}",
        "\\label{tab:aggregated_domains}",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Домен} & \\textbf{pass@1} & \\textbf{ASR} & \\textbf{avg\\_reward} & \\textbf{avg\\_agent\\_cost} & \\textbf{avg\\_user\\_cost} & \\textbf{avg\\_duration} & \\textbf{avg\\_num\\_messages} \\\\",
        "\\midrule",
    ]
    
    # Функция для экранирования LaTeX специальных символов
    def escape_latex(text: str) -> str:
        """Экранировать специальные символы LaTeX."""
        if not isinstance(text, str):
            text = str(text)
        text = text.replace('_', '\\_')
        text = text.replace('&', '\\&')
        text = text.replace('%', '\\%')
        text = text.replace('$', '\\$')
        text = text.replace('#', '\\#')
        text = text.replace('^', '\\textasciicircum{}')
        text = text.replace('{', '\\{')
        text = text.replace('}', '\\}')
        text = text.replace('~', '\\textasciitilde{}')
        return text

    
    for stats in domain_stats_list:
        domain = escape_latex(stats['domain'])
        pass1 = f"{stats['pass1_mean']:.2f}"
        asr = f"{stats['asr_mean']:.2f}"
        reward = f"{stats['avg_reward_mean']:.2f}"
        agent_cost = (
            f"\\${stats['avg_agent_cost_mean']:.2f}"
            if stats["avg_agent_cost_mean"] is not None
            else "N/A"
        )
        user_cost = (
            f"\\${stats['avg_user_cost_mean']:.2f}"
            if stats["avg_user_cost_mean"] is not None
            else "N/A"
        )
        duration = f"{stats['avg_duration_mean']:.2f}s"
        num_messages = f"{stats['avg_num_messages_mean']:.1f}"
        
        latex_lines.append(
            f"{domain} & {pass1} & {asr} & {reward} & {agent_cost} & {user_cost} & {duration} & {num_messages} \\\\"
        )
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    latex_code = "\n".join(latex_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code)
        print(f"Aggregated table saved to {output_path}")
    
    return latex_code


def _aggregate_successes_trials_by_domain(
    result_files: List[Path],
    *,
    domain: str,
    model: str,
    temperature: float,
) -> tuple[int, int]:
    """Aggregate successes/trials across all tasks in domain for config."""
    total_successes = 0
    total_trials = 0

    norm_model = _normalize_model_name(model)

    for file_path in result_files:
        try:
            domains_dict = load_simulation_file(file_path)
        except Exception:
            continue

        if domain not in domains_dict:
            continue

        results = domains_dict[domain]
        results_model = _normalize_model_name(results.info.agent_info.llm)
        # Temperature is configured on the *user* model in our experiments.
        results_temp = (
            float(results.info.user_info.llm_args.get("temperature", 0.0))
            if results.info.user_info.llm_args
            else 0.0
        )

        if results_model != norm_model:
            continue
        if results_temp != float(temperature):
            continue

        task_ids = {sim.task_id for sim in results.simulations}
        for task_id in task_ids:
            metrics = compute_task_metrics(results, task_id)
            if not metrics:
                continue
            total_successes += int(metrics.get("success_count", 0))
            total_trials += int(metrics.get("num_trials", 0))

    return total_successes, total_trials


def _format_p_value_scientific(p_value: float) -> str:
    # LaTeX-friendly scientific notation
    return f"{p_value:.2e}"


def generate_model_domain_table_latex(
    result_files: List[Path],
    domains: List[str],
    models: List[str],
    temperatures: List[float],
    output_path: Optional[Path] = None,
) -> str:
    """Generate model×domain pass@1 table (aggregated across tasks)."""

    header_domains = [escape_latex(d) for d in domains]

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Сравнение устойчивости моделей по доменам (pass@1)}",
        "\\label{tab:aggregated}",
        "\\begin{tabular}{l" + "c" * len(domains) + "}",
        "\\toprule",
        "\\textbf{Модель} & "
        + " & ".join(f"\\textbf{{{d}}}" for d in header_domains)
        + " \\\\",
        "\\midrule",
    ]

    # Ensure stable ordering
    for model in models:
        m = _normalize_model_name(model)
        for temp in temperatures:
            row_label = escape_latex(f"{m} (T={temp})")
            cells: list[str] = []
            for domain in domains:
                succ, trials = _aggregate_successes_trials_by_domain(
                    result_files, domain=domain, model=m, temperature=float(temp)
                )
                if trials == 0:
                    cells.append("N/A")
                    continue
                rate = succ / trials
                pct = (100.0 * rate)
                cells.append(f"{succ}/{trials} ({pct:.1f}\\%)")

            latex_lines.append(row_label + " & " + " & ".join(cells) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex_code = "\n".join(latex_lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code)
        print(f"Model-domain table saved to {output_path}")

    return latex_code


def generate_significance_table_latex(
    result_files: List[Path],
    domains: List[str],
    temperatures: List[float],
    *,
    model_a: str = "gpt-4o",
    model_b: str = "gpt-4o-mini",
    output_path: Optional[Path] = None,
) -> str:
    """Generate p-value table comparing model_a vs model_b at fixed T."""


def generate_temperature_significance_table_latex(
    result_files: List[Path],
    domains: List[str],
    models: List[str],
    temperatures: List[float],
    *,
    output_path: Optional[Path] = None,
) -> str:
    """Generate p-value table for temperature effects within each model.

    Uses Fisher exact on aggregated successes/trials by domain.
    p-value applies to pass@1 and ASR.
    """

    temps = sorted({float(t) for t in temperatures})
    if len(temps) < 2:
        raise ValueError("Need at least 2 temperatures for comparison")

    # Pairwise comparisons
    pairs: list[tuple[float, float]] = []
    for i in range(len(temps)):
        for j in range(i + 1, len(temps)):
            pairs.append((temps[i], temps[j]))

    header_cols = " & ".join([f"\\shortstack{{\\textbf{{p}}\\\\\\textbf{{{a:g} vs {b:g}}}}}" for a, b in pairs])

    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\caption{Влияние температуры пользователя: p-value попарных сравнений (Fisher exact; p относится к pass@1 и ASR)}",
        "\\label{tab:temp_significance}",
        "\\begin{tabular}{ll" + "c" * len(pairs) + "}",
        "\\toprule",
        "\\textbf{Домен} & \\textbf{Модель} & " + header_cols + " \\\\",
        "\\midrule",
    ]

    for domain in domains:
        domain_tex = escape_latex(domain)
        for model in models:
            model_norm = _normalize_model_name(model)
            model_tex = escape_latex(model_norm)

            # Precompute aggregated successes/trials for each temp
            stats_by_temp: dict[float, tuple[int, int]] = {}
            for t in temps:
                stats_by_temp[t] = _aggregate_successes_trials_by_domain(
                    result_files, domain=domain, model=model_norm, temperature=float(t)
                )

            cells: list[str] = []
            for a, b in pairs:
                succ_a, trials_a = stats_by_temp[a]
                succ_b, trials_b = stats_by_temp[b]
                p_val, _ = fisher_exact_test(succ_a, trials_a, succ_b, trials_b)
                cells.append(_format_p_value_scientific(float(p_val)))

            latex_lines.append(f"{domain_tex} & {model_tex} & " + " & ".join(cells) + " \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex_code = "\n".join(latex_lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code)
        print(f"Temperature significance table saved to {output_path}")

    return latex_code

    model_a = _normalize_model_name(model_a)
    model_b = _normalize_model_name(model_b)

    # Table is wide; keep it compact.
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\caption{Статистическая значимость различий (Fisher exact, двусторонний; p относится к pass@1 и ASR)}",
        "\\label{tab:significance}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "\\textbf{Домен} & \\textbf{T} & "
        "\\shortstack{\\textbf{4o}\\\\\\textbf{pass@1}} & "
        "\\shortstack{\\textbf{4o-mini}\\\\\\textbf{pass@1}} & "
        "\\shortstack{\\textbf{4o}\\\\\\textbf{ASR}} & "
        "\\shortstack{\\textbf{4o-mini}\\\\\\textbf{ASR}} & "
        "\\shortstack{\\textbf{p}} \\\\",
        "\\midrule",
    ]

    for domain in domains:
        domain_tex = escape_latex(domain)
        for temp in temperatures:
            succ_a, trials_a = _aggregate_successes_trials_by_domain(
                result_files, domain=domain, model=model_a, temperature=float(temp)
            )
            succ_b, trials_b = _aggregate_successes_trials_by_domain(
                result_files, domain=domain, model=model_b, temperature=float(temp)
            )

            # p-value for pass@1 (success vs failure)
            p_pass, _ = fisher_exact_test(succ_a, trials_a, succ_b, trials_b)

            # ASR = 1 - pass@1, so p-value is the same; we show it once.

            pass_a = succ_a / trials_a if trials_a else 0.0
            pass_b = succ_b / trials_b if trials_b else 0.0
            asr_a = 1.0 - pass_a
            asr_b = 1.0 - pass_b

            # Keep columns narrow (integer percents).
            pass_a_str = (
                f"{succ_a}/{trials_a} ({100.0 * pass_a:.0f}\\%)" if trials_a else "N/A"
            )
            pass_b_str = (
                f"{succ_b}/{trials_b} ({100.0 * pass_b:.0f}\\%)" if trials_b else "N/A"
            )
            asr_a_str = f"{100.0 * asr_a:.0f}\\%" if trials_a else "N/A"
            asr_b_str = f"{100.0 * asr_b:.0f}\\%" if trials_b else "N/A"

            p_str = _format_p_value_scientific(float(p_pass))

            latex_lines.append(
                f"{domain_tex} & {temp:g} & {pass_a_str} & {pass_b_str} & "
                f"{asr_a_str} & {asr_b_str} & {p_str} \\\\"
            )

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex_code = "\n".join(latex_lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_code)
        print(f"Significance table saved to {output_path}")

    return latex_code


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from simulation results")
    parser.add_argument("result_files", nargs="+", type=Path, help="Paths to simulation result files")
    parser.add_argument("--domains", nargs="+", default=["mail_rag_phishing", "collab", "output_handling"],
                       help="Domain names to process")
    parser.add_argument("--table-type", choices=["detailed", "aggregated", "both"], default="both",
                       help="Type of table to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/paper_template"),
                       help="Output directory for LaTeX tables")
    
    args = parser.parse_args()
    
    # Generate tables
    if args.table_type in ["detailed", "both"]:
        detailed_path = args.output_dir / "detailed_metrics_table.tex"
        generate_detailed_metrics_table_latex(args.result_files, args.domains, detailed_path)
    
    if args.table_type in ["aggregated", "both"]:
        aggregated_path = args.output_dir / "aggregated_table.tex"
        generate_aggregated_table_latex(args.result_files, args.domains, aggregated_path)
    
    print("Table generation complete!")


if __name__ == "__main__":
    main()
