#!/usr/bin/env python3
"""
Script to generate LaTeX tables with statistical significance from simulation results.

This script:
1. Loads simulation results
2. Computes pass@1 metrics with Wilson confidence intervals
3. Performs Fisher's exact test for statistical significance
4. Generates LaTeX table code with confidence intervals and significance markers
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from tau2.data_model.simulation import MultiDomainResults, Results
from tau2.metrics.agent_metrics import is_successful

# Import statistical functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from result_collection import (
    load_simulation_file,
    wilson_confidence_interval,
    fisher_exact_test,
    compute_statistical_significance,
    format_pass_k_with_ci,
)


def compute_domain_stats(results_dict: Dict[str, Results], domain: str) -> Dict[str, Dict]:
    """
    Compute statistics for a specific domain across all models.
    
    Returns:
        Dictionary mapping (model, temperature) to stats dict
    """
    stats = {}
    
    if domain not in results_dict:
        return stats
    
    results = results_dict[domain]
    
    # Group by model and temperature
    groups = {}
    for sim in results.simulations:
        model = results.info.agent_info.llm
        temp = results.info.agent_info.llm_args.get('temperature', 0.0)
        key = (model, temp)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(sim)
    
    # Compute statistics for each group
    for (model, temp), sims in groups.items():
        successes = sum(1 for sim in sims if sim.reward_info and is_successful(sim.reward_info.reward))
        trials = len(sims)
        
        ci_lower, ci_upper = wilson_confidence_interval(successes, trials)
        
        stats[f"{model} (T={temp})"] = {
            'successes': successes,
            'trials': trials,
            'proportion': successes / trials if trials > 0 else 0.0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }
    
    return stats


def compare_with_reference(
    stats: Dict[str, Dict],
    reference_key: str = "GPT-4o-mini (T=0.0)"
) -> Dict[str, str]:
    """
    Compare all models with reference and compute significance levels.
    
    Returns:
        Dictionary mapping model keys to significance markers
    """
    significance = {}
    
    if reference_key not in stats:
        return significance
    
    ref_stats = stats[reference_key]
    
    for model_key, model_stats in stats.items():
        if model_key == reference_key:
            significance[model_key] = ''
            continue
        
        _, sig_level = fisher_exact_test(
            model_stats['successes'], model_stats['trials'],
            ref_stats['successes'], ref_stats['trials']
        )
        significance[model_key] = sig_level
    
    return significance


def generate_latex_table_row(
    model_key: str,
    stats: Dict,
    significance: str = '',
    domain_key: Optional[str] = None
) -> str:
    """
    Generate a LaTeX table row with formatted statistics.
    """
    successes = stats['successes']
    trials = stats['trials']
    proportion = stats['proportion']
    ci_lower = stats['ci_lower']
    ci_upper = stats['ci_upper']
    
    # Format percentage
    if proportion < 0.01 or proportion > 0.99:
        pct_str = f"{proportion:.0%}"
    else:
        pct_str = f"{proportion:.1%}"
    
    # Format CI
    if ci_lower < 0.01 or ci_lower > 0.99:
        ci_lower_str = f"{ci_lower:.0%}"
    else:
        ci_lower_str = f"{ci_lower:.1%}"
    
    if ci_upper < 0.01 or ci_upper > 0.99:
        ci_upper_str = f"{ci_upper:.0%}"
    else:
        ci_upper_str = f"{ci_upper:.1%}"
    
    result = f"{successes}/{trials} ({pct_str}) [CI: {ci_lower_str}--{ci_upper_str}]"
    if significance:
        result += significance
    
    return result


def generate_aggregated_table_latex(
    simulation_files: List[Path],
    domains: List[str],
    output_file: Optional[Path] = None
) -> str:
    """
    Generate LaTeX code for aggregated results table with statistical significance.
    """
    # Load all simulation files
    all_results = {}
    for file_path in simulation_files:
        file_results = load_simulation_file(file_path)
        for domain_name, results in file_results.items():
            if domain_name not in all_results:
                all_results[domain_name] = results
            else:
                # Merge simulations
                all_results[domain_name].simulations.extend(results.simulations)
    
    # Compute statistics for each domain
    domain_stats = {}
    for domain in domains:
        if domain in all_results:
            domain_stats[domain] = compute_domain_stats(all_results, domain)
    
    # Find all unique model configurations
    all_models = set()
    for stats in domain_stats.values():
        all_models.update(stats.keys())
    all_models = sorted(all_models)
    
    # Compare with reference (GPT-4o-mini T=0.0)
    reference = "gpt-4o-mini (T=0.0)"
    significance_map = {}
    for domain in domains:
        if domain in domain_stats:
            domain_sig = compare_with_reference(domain_stats[domain], reference)
            significance_map[domain] = domain_sig
    
    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{threeparttable}",
        "\\caption{Сравнение устойчивости моделей по доменам (pass@1) с 95\\% доверительными интервалами Уилсона}",
        "\\label{tab:aggregated}",
        "\\begin{tabular}{l" + "c" * len(domains) + "}",
        "\\toprule",
        "\\textbf{Модель} & " + " & ".join([f"\\textbf{{{domain}}}" for domain in domains]) + " \\\\",
        "\\midrule",
    ]
    
    # Add rows for each model
    for model_key in all_models:
        row_values = [model_key.replace("_", "\\_")]
        for domain in domains:
            if domain in domain_stats and model_key in domain_stats[domain]:
                stats = domain_stats[domain][model_key]
                sig = significance_map.get(domain, {}).get(model_key, '')
                cell = generate_latex_table_row(model_key, stats, sig)
                row_values.append(cell)
            else:
                row_values.append("---")
        
        latex_lines.append(" & ".join(row_values) + " \\\\")
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item \\textit{Примечание:} Формат: X/Y (Z\\%) [CI: L\\%--U\\%], где X/Y --- число успешных случаев из общего числа, Z\\% --- процент успешности, L\\% и U\\% --- нижняя и верхняя границы 95\\% доверительного интервала Уилсона\\cite{Wilson_1927}. Для сравнения статистической значимости использовался точный критерий Фишера\\cite{Fisher_1935} (см. раздел~\\ref{sec:method}). Значения с статистической значимостью относительно базовой модели (GPT-4o-mini, T=0.0) могут быть отмечены символами: $***$ ($p < 0.001$), $**$ ($p < 0.01$), $*$ ($p < 0.05$).",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ])
    
    latex_code = "\n".join(latex_lines)
    
    if output_file:
        output_file.write_text(latex_code)
        print(f"LaTeX table saved to {output_file}")
    
    return latex_code


def main():
    """Main function to generate statistical tables."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables with statistical significance from simulation results"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="Paths to simulation JSON files"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["mail_rag_phishing", "collab", "output_handling"],
        help="Domains to include in the table"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Generate table
    latex_code = generate_aggregated_table_latex(
        args.input_files,
        args.domains,
        args.output
    )
    
    if not args.output:
        print(latex_code)


if __name__ == "__main__":
    main()
