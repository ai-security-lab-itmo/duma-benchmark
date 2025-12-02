#!/usr/bin/env python3
"""
Script to run tau2 benchmarks with multiple parameter configurations in parallel.

This script allows you to define a list of test cases with different LLM configurations
and runs them in parallel, generating separate simulation files for each case.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


@dataclass
class BenchmarkCase:
    """Configuration for a single benchmark run."""
    agent_llm: str
    user_llm: str
    user_llm_args: Optional[dict] = None
    name: Optional[str] = None  # Optional name for the case (used in save_to)

    def __post_init__(self):
        """Generate name if not provided."""
        if self.name is None:
            agent_name = self.agent_llm.split("/")[-1]
            user_name = self.user_llm.split("/")[-1]
            temp_str = ""
            if self.user_llm_args and "temperature" in self.user_llm_args:
                temp_str = f"_temp{self.user_llm_args['temperature']}"
            self.name = f"{agent_name}_{user_name}{temp_str}"


def run_single_benchmark(
        case: BenchmarkCase,
        domains: List[str],
        base_args: dict,
        output_dir: Path,
) -> tuple[str, bool, str]:
    """
    Run a single benchmark case.
    
    Args:
        case: BenchmarkCase configuration
        domains: List of domain names to run
        base_args: Base arguments for tau2 run command
        output_dir: Directory to save results
        
    Returns:
        Tuple of (case_name, success, output_file_path or error_message)
    """
    console = Console()

    # Build command
    cmd = [
        sys.executable, "-m", "tau2.cli", "run",
        "--domains", *domains,
        "--agent-llm", case.agent_llm,
        "--user-llm", case.user_llm,
    ]

    # Add base arguments
    for key, value in base_args.items():
        if value is not None:
            if key == "user_llm_args":
                # Skip user_llm_args in base_args, use case.user_llm_args instead
                continue
            elif key == "agent_llm_args":
                if value:
                    cmd.extend(["--agent-llm-args", json.dumps(value)])
            else:
                # Convert key from snake_case to kebab-case
                arg_name = key.replace("_", "-")
                cmd.extend([f"--{arg_name}", str(value)])

    # Add user_llm_args from case if provided
    if case.user_llm_args:
        cmd.extend(["--user-llm-args", json.dumps(case.user_llm_args)])

    # Set save_to to include case name
    save_name = f"{case.name}_{'_'.join(domains)}"
    cmd.extend(["--save-to", save_name])

    try:
        # Run command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd(),
        )

        # Find the output file
        output_file = output_dir / f"{save_name}.json"

        if output_file.exists():
            return (case.name, True, str(output_file))
        else:
            return (case.name, False, f"Output file not found: {output_file}")

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {e.stderr[:500] if e.stderr else str(e)}"
        return (case.name, False, error_msg)
    except Exception as e:
        return (case.name, False, f"Unexpected error: {str(e)}")


def run_benchmark_suite(
        cases: List[BenchmarkCase],
        domains: List[str],
        base_args: dict,
        max_workers: int = 2,
        output_dir: Optional[Path] = None,
) -> dict:
    """
    Run multiple benchmark cases in parallel.
    
    Args:
        cases: List of BenchmarkCase configurations
        domains: List of domain names to run
        base_args: Base arguments for tau2 run command
        max_workers: Maximum number of parallel processes
        output_dir: Directory to save results (default: data/simulations)
        
    Returns:
        Dictionary mapping case names to (success, output_file) tuples
    """
    if output_dir is None:
        from tau2.utils.utils import DATA_DIR
        output_dir = DATA_DIR / "simulations"

    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.print(f"\n[bold cyan]Running {len(cases)} benchmark cases with {max_workers} parallel workers[/bold cyan]")
    console.print(f"Domains: {', '.join(domains)}")
    console.print(f"Output directory: {output_dir}\n")

    results = {}

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=len(cases))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    run_single_benchmark,
                    case,
                    domains,
                    base_args,
                    output_dir,
                ): case.name
                for case in cases
            }

            # Process completed tasks
            for future in as_completed(futures):
                case_name = futures[future]
                try:
                    result_name, success, output = future.result()
                    results[result_name] = (success, output)
                    if success:
                        progress.update(task, description=f"✅ {result_name} completed")
                    else:
                        progress.update(task, description=f"❌ {result_name} failed")
                except Exception as e:
                    results[case_name] = (False, f"Exception: {str(e)}")
                    progress.update(task, description=f"❌ {case_name} error")

                progress.advance(task)

    return results


def print_results_summary(results: dict, console: Console):
    """Print a summary table of results."""
    table = Table(title="Benchmark Results Summary")
    table.add_column("Case Name", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Output File / Error", style="dim")

    success_count = 0
    for case_name, (success, output) in results.items():
        status = "✅ Success" if success else "❌ Failed"
        if success:
            success_count += 1
        table.add_row(case_name, status, output)

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]Successfully completed: {success_count}/{len(results)}[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Run tau2 benchmarks with multiple configurations in parallel"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file with cases and settings",
        default="scripts/benchmark_config_example.json"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel processes (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results (default: data/tau2/simulations)",
    )

    args = parser.parse_args()

    console = Console()

    # Load configuration

    with open(args.config, "r") as f:
        config = json.load(f)
    cases = [
        BenchmarkCase(
            agent_llm=case["agent_llm"],
            user_llm=case["user_llm"],
            user_llm_args=case.get("user_llm_args"),
            name=case.get("name"),
        )
        for case in config["cases"]
    ]

    domains = config["domains"]
    base_args = config.get("base_args", {})

    # Run benchmarks
    output_dir = Path(args.output_dir) if args.output_dir else None
    results = run_benchmark_suite(
        cases=cases,
        domains=domains,
        base_args=base_args,
        max_workers=args.max_workers,
        output_dir=output_dir,
    )

    # Print summary
    print_results_summary(results, console)

    # Exit with error code if any failed
    failed = sum(1 for success, _ in results.values() if not success)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
