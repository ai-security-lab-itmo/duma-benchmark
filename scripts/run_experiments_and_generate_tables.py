#!/usr/bin/env python3
"""
Автоматический запуск экспериментов, обработка результатов и генерация таблиц/визуализаций.

Этот скрипт выполняет полный цикл:
1. Запуск всех экспериментов
2. Обработка результатов и вычисление метрик
3. Генерация LaTeX таблиц
4. Создание визуализаций
5. Обновление template.tex
6. Компиляция PDF (опционально)
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add project root and src to path to import duma (must be before importing result_collection)
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from result_collection import (
    load_simulations,
    load_simulation_file,
    compute_task_metrics,
)
from duma.utils.model_ref import normalize_model_ref
from generate_statistical_tables import (
    generate_detailed_metrics_table_latex,
    generate_aggregated_table_latex,
    generate_model_domain_table_latex,
    generate_significance_table_latex,
    generate_temperature_significance_table_latex,
)


def _model_id_for_results(model: str | None) -> str:
    """Normalize model names to canonical vendor/model format."""
    return normalize_model_ref(model)


def _sanitize_model_for_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


def _normalize_model_name(model: str | None) -> str:
    return normalize_model_ref(model)


def run_single_experiment(
    cmd: List[str], output_file: Path, exp_name: str, idx: int, total: int
) -> Tuple[bool, float, Optional[str]]:
    """
    Запустить один эксперимент.

    Returns:
        Tuple of (success, duration, error_message)
    """
    exp_start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=Path.cwd(),
        )

        # Ждем завершения
        stdout, stderr = process.communicate()
        returncode = process.returncode

        exp_duration = time.time() - exp_start_time

        if returncode != 0:
            error_msg = (
                stderr[:500]
                if stderr
                else stdout[:500]
                if stdout
                else "No error message"
            )
            return (False, exp_duration, error_msg)
        else:
            return (True, exp_duration, None)

    except Exception as e:
        exp_duration = time.time() - exp_start_time
        return (False, exp_duration, str(e))


def run_all_experiments(
    models: List[str],
    temperatures: List[float],
    domains: Dict[str, List[str]],
    num_trials: int = 10,
    max_concurrency: int = 3,
    run_dir_name: str = "",  # Subdirectory name under simulations/ (e.g. "run_20260304_143000")
    parallel_experiments: int = 4,  # Количество параллельных экспериментов
    force_rerun: bool = False,  # Принудительно перезапустить все эксперименты
    user_llm: Optional[str] = None,  # Модель для симуляции пользователя
    solo: bool = False,  # Режим solo (агент без пользователя)
    agent_base_url: Optional[str] = None,  # Base URL для провайдера LLM
    api_key_env: Optional[str] = None,  # Имя переменной окружения для API ключа
    duma_max_concurrency: int = 1,  # Внутренняя параллельность duma run
    agent_temperature: float = 0.0,  # Фиксированная температура агента
) -> Path:
    """
    Запустить все эксперименты.

    Args:
        models: Список моделей ['gpt-4o', 'gpt-4o-mini', 'gpt-5.1', 'gpt-5.2']
        temperatures: Список температур [0.0, 0.5, 1.0]
        domains: Словарь {domain: [task_ids]}
        num_trials: Количество прогонов на конфигурацию
        max_concurrency: Максимальное количество параллельных запусков
        run_dir_name: Subdirectory name under simulations/ (e.g. "run_20260304_143000")

    Returns:
        Path to results directory
    """
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be > 0")
    if duma_max_concurrency <= 0:
        raise ValueError("duma_max_concurrency must be > 0")

    # duma сохраняет файлы в data/duma/simulations/ автоматически
    # --save-to принимает имя файла БЕЗ расширения (supports subdirectory paths)
    from duma.utils.utils import DATA_DIR

    actual_output_dir = DATA_DIR / "simulations" / run_dir_name
    actual_output_dir.mkdir(parents=True, exist_ok=True)

    # Функция для загрузки всех задач домена из tasks.json
    def get_all_tasks_for_domain(domain_name: str) -> List[str]:
        """Загрузить все задачи домена из tasks.json."""
        # Проверяем несколько возможных путей (правильный путь первым)
        possible_paths = [
            DATA_DIR
            / "duma"
            / "domains"
            / domain_name
            / "tasks.json",  # Правильный путь
            DATA_DIR / "domains" / domain_name / "tasks.json",
            Path("data/duma/domains") / domain_name / "tasks.json",
            Path(__file__).parent.parent
            / "data"
            / "duma"
            / "domains"
            / domain_name
            / "tasks.json",
        ]

        tasks_file = None
        for path in possible_paths:
            if path.exists():
                tasks_file = path
                break

        if not tasks_file:
            print(f"Warning: tasks.json not found for domain {domain_name}")
            print(f"  Checked paths: {[str(p) for p in possible_paths]}")
            return []

        try:
            with open(tasks_file, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)
                task_ids = [task["id"] for task in tasks_data if "id" in task]
                print(f"  Found {len(task_ids)} tasks in {tasks_file}")
                return task_ids
        except Exception as e:
            print(
                f"Warning: Could not load tasks for {domain_name} from {tasks_file}: {e}"
            )
            import traceback

            traceback.print_exc()
            return []

    # Кэш для задач доменов, чтобы не загружать их многократно
    domain_tasks_cache = {}

    # In solo mode there is no user, so the temperature loop is irrelevant;
    # we run each model/domain/task once with agent_temperature.
    # In dual-control mode, temperatures control the user simulator and
    # agent_temperature is used for the agent.
    user_temperatures = [agent_temperature] if solo else temperatures

    commands = []
    for model in models:
        for user_temp in user_temperatures:
            for domain, tasks in domains.items():
                # Если задачи не указаны, загрузить все задачи домена (с кэшированием)
                if not tasks:
                    if domain not in domain_tasks_cache:
                        domain_tasks_cache[domain] = get_all_tasks_for_domain(domain)
                    tasks = domain_tasks_cache[domain]
                    if not tasks:
                        print(f"Warning: No tasks found for domain {domain}, skipping")
                        continue
                    if model == models[0] and user_temp == user_temperatures[0]:
                        # Показываем только при первой загрузке
                        print(f"Loaded {len(tasks)} tasks for domain {domain}")

                for task in tasks:
                    # Формируем имя файла без расширения (duma добавит .json автоматически)
                    model_id = _model_id_for_results(model)

                    if solo:
                        file_name = f"paper_results_solo_{domain}_{_sanitize_model_for_filename(model_id)}_T{agent_temperature}_{task}"
                    else:
                        # Определяем модель пользователя
                        effective_user_llm = user_llm if user_llm else model
                        user_model_id = _model_id_for_results(effective_user_llm)
                        file_name = f"paper_results_{domain}_{_sanitize_model_for_filename(model_id)}_U{_sanitize_model_for_filename(user_model_id)}_UT{user_temp}_{task}"

                    # Фактический путь к файлу результата
                    output_file = actual_output_dir / f"{file_name}.json"

                    # --save-to path relative to simulations/ (includes run subdir)
                    save_to_name = f"{run_dir_name}/{file_name}" if run_dir_name else file_name

                    # Формируем команду
                    use_local_models = not agent_base_url and not api_key_env
                    if solo:
                        cmd = [
                            "duma",
                            "run",
                            "--domain",
                            domain,
                            "--agent",
                            "llm_agent_solo",
                            "--user",
                            "dummy_user",
                            "--agent-llm",
                            model,
                            "--agent-llm-args",
                            json.dumps({"temperature": agent_temperature}),
                            "--num-trials",
                            str(num_trials),
                            "--task-ids",
                            task,
                            "--save-to",
                            save_to_name,
                            "--max-concurrency",
                            str(duma_max_concurrency),
                        ]
                    else:
                        effective_user_llm = user_llm if user_llm else model
                        cmd = [
                            "duma",
                            "run",
                            "--domain",
                            domain,
                            "--agent-llm",
                            model,
                            "--agent-llm-args",
                            json.dumps({"temperature": agent_temperature}),
                            "--user-llm",
                            effective_user_llm,
                            "--user-llm-args",
                            json.dumps({"temperature": user_temp}),
                            "--num-trials",
                            str(num_trials),
                            "--task-ids",
                            task,
                            "--save-to",
                            save_to_name,
                            "--max-concurrency",
                            str(duma_max_concurrency),
                        ]
                    # Добавляем base URL и API key env если указаны
                    if agent_base_url:
                        cmd += ["--agent-base-url", agent_base_url]
                        if not solo:
                            cmd += ["--user-base-url", agent_base_url]
                    if api_key_env:
                        cmd += ["--api-key-env", api_key_env]
                    if use_local_models:
                        cmd += ["--local-models"]
                    # Проверка: убедимся, что все аргументы - строки
                    cmd = [str(arg) for arg in cmd]
                    commands.append((cmd, output_file))

    print(f"Total configurations to run: {len(commands)}")

    # Запустить команды параллельно с прогрессом
    completed = 0
    skipped = 0
    errors = 0
    start_time = time.time()
    times = []  # Для расчета среднего времени

    # Использовать tqdm если доступен, иначе простой вывод
    # Для thread-safe работы с tqdm используем lock
    from threading import Lock

    pbar_lock = Lock()

    if HAS_TQDM:
        # Используем правильные параметры для обновления на месте
        # mininterval и miniters для уменьшения частоты обновлений
        pbar = tqdm(
            total=len(commands),
            desc="Running experiments",
            unit="exp",
            file=sys.stderr,
            ncols=120,
            leave=True,
            mininterval=1.0,
            miniters=1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
    else:
        pbar = None

    # Функция для выполнения одного эксперимента
    def run_single_experiment(idx_cmd_file):
        idx, cmd, output_file = idx_cmd_file
        exp_start_time = time.time()
        thread_id = threading.current_thread().ident
        cmd_str = " ".join(cmd)  # Для логирования

        # Если файл существует и не требуется принудительный перезапуск, пропускаем
        if output_file.exists() and not force_rerun:
            with pbar_lock:
                if HAS_TQDM:
                    pbar.update(1)
            return (True, 0, "skipped", output_file.name)

        # Если требуется принудительный перезапуск, удаляем существующий файл
        if output_file.exists() and force_rerun:
            output_file.unlink()

        # Показать текущий эксперимент
        model = cmd[cmd.index("--agent-llm") + 1] if "--agent-llm" in cmd else "unknown"
        domain = cmd[cmd.index("--domain") + 1] if "--domain" in cmd else "unknown"
        task = cmd[cmd.index("--task-ids") + 1] if "--task-ids" in cmd else "unknown"
        short_name = f"{domain[:15]}/{model[:10]}/{task.split('_')[-1][:20]}"

        # Запустить команду
        try:
            # Логируем начало выполнения для отладки параллельности
            start_msg = f"[Thread-{thread_id % 10000}] [{idx}] 🚀 START: {short_name}"
            print(start_msg, flush=True)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=Path.cwd(),
            )

            # Ждем завершения (это блокирует, но в разных потоках процессы выполняются параллельно)
            stdout, stderr = process.communicate()
            returncode = process.returncode

            end_msg = f"[Thread-{thread_id % 10000}] [{idx}] ✅ DONE: {short_name} ({time.time() - exp_start_time:.1f}s)"
            print(end_msg, flush=True)

            exp_duration = time.time() - exp_start_time

            if returncode != 0:
                # Собираем полное сообщение об ошибке
                error_parts = []
                if stderr:
                    error_parts.append(f"STDERR: {stderr}")
                if stdout:
                    error_parts.append(f"STDOUT: {stdout}")
                if not error_parts:
                    error_parts.append("No error message")
                error_msg = "\n".join(error_parts)
                # Для возврата используем короткую версию
                error_msg_short = (
                    stderr[:500]
                    if stderr
                    else stdout[:500]
                    if stdout
                    else "No error message"
                )
                with pbar_lock:
                    if HAS_TQDM:
                        pbar.update(1)
                return (False, exp_duration, error_msg, short_name)
            else:
                with pbar_lock:
                    if HAS_TQDM:
                        pbar.update(1)
                return (True, exp_duration, None, short_name)

        except Exception as e:
            exp_duration = time.time() - exp_start_time
            with pbar_lock:
                if HAS_TQDM:
                    pbar.update(1)
            return (False, exp_duration, str(e), short_name)

    # Запустить команды параллельно
    # Используем max_concurrency для ограничения параллельности
    # Проверим, что duma доступен (только при первом запуске)
    if commands:
        duma_path = shutil.which("duma")
        if not duma_path:
            raise FileNotFoundError(
                "duma command not found in PATH. "
                "Make sure duma is installed: pip install -e ."
            )

    # Подготовить данные для параллельного выполнения
    indexed_commands = [
        (idx, cmd, output_file) for idx, (cmd, output_file) in enumerate(commands, 1)
    ]

    print(
        "🚀 Starting parallel execution with "
        f"max_workers={max_concurrency}, duma_max_concurrency={duma_max_concurrency}"
    )
    print(f"   Total experiments: {len(indexed_commands)}")

    # Очистить лог ошибок при старте (используем абсолютный путь)
    project_root = Path(__file__).parent.parent
    error_log_file = project_root / "experiment_errors.log"
    if error_log_file.exists():
        error_log_file.unlink()

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Отправить все задачи сразу - они будут выполняться параллельно
        future_to_item = {
            executor.submit(run_single_experiment, item): item
            for item in indexed_commands
        }

        print(f"✅ All {len(future_to_item)} tasks submitted to thread pool")
        print(f"   Executing up to {max_concurrency} experiments in parallel...\n")

        # Обработать результаты по мере завершения
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            idx = item[0]
            try:
                success, duration, error_msg, name = future.result()
                if duration > 0:  # Не считаем пропущенные
                    times.append(duration)

                if success:
                    if error_msg == "skipped":
                        skipped += 1
                    else:
                        completed += 1
                else:
                    errors += 1
                    # Всегда логируем ошибки для диагностики
                    print(f"\n   ❌ ERROR [{idx}] after {duration:.1f}s: {name}")
                    if error_msg:
                        error_preview = (
                            error_msg[:500] if len(error_msg) > 500 else error_msg
                        )
                        print(f"   Error: {error_preview}")
                        # Сохранить полную ошибку в файл для анализа (используем абсолютный путь)
                        project_root = Path(__file__).parent.parent
                        error_log_file = project_root / "experiment_errors.log"
                        try:
                            # Получить команду из item
                            item_cmd = item[1] if len(item) > 1 else None
                            cmd_str = (
                                " ".join(item_cmd) if item_cmd else "unknown command"
                            )
                            with open(error_log_file, "a", encoding="utf-8") as f:
                                f.write(f"\n{'=' * 80}\n")
                                f.write(f"Experiment [{idx}]: {name}\n")
                                f.write(f"Command: {cmd_str}\n")
                                f.write(f"Duration: {duration:.1f}s\n")
                                f.write(f"Full error message:\n{error_msg}\n")
                                f.write(f"{'=' * 80}\n")
                        except Exception as log_err:
                            print(f"   ⚠️  Failed to write to error log: {log_err}")
            except Exception as e:
                errors += 1
                project_root = Path(__file__).parent.parent
                error_log_file = project_root / "experiment_errors.log"
                print(f"\n   ❌ Exception in experiment {idx}: {e}")
                try:
                    with open(error_log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n{'=' * 80}\n")
                        f.write(
                            f"Exception in experiment [{idx}]: {item[3] if len(item) > 3 else 'unknown'}\n"
                        )
                        f.write(f"Exception type: {type(e).__name__}\n")
                        f.write(f"Exception message: {str(e)}\n")
                        import traceback

                        f.write(f"Traceback:\n{traceback.format_exc()}\n")
                        f.write(f"{'=' * 80}\n")
                except Exception as log_err:
                    print(f"   ⚠️  Failed to write exception to error log: {log_err}")

    if HAS_TQDM:
        pbar.close()

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print(f"\n{'=' * 80}")
    print(f"📊 Summary:")
    print(f"   ✅ Completed: {completed}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   ❌ Errors: {errors}")
    print(f"   ⏱️  Total time: {total_time_str}")
    if completed > 0 and times:
        avg_time = np.mean(times)
        print(f"   📈 Average time per experiment: {avg_time:.1f}s")
    print(f"{'=' * 80}\n")

    # Показать информацию о логе ошибок
    project_root = Path(__file__).parent.parent
    error_log_file = project_root / "experiment_errors.log"
    if error_log_file.exists() and errors > 0:
        print(
            f"⚠️  {errors} errors occurred. Full error details saved to: {error_log_file}"
        )
        print(f"   To view errors: cat {error_log_file}\n")

    return actual_output_dir


def process_all_results(
    results_dir: Path, result_files: Optional[List[Path]] = None
) -> Dict:
    """
    Обработать все результаты и вычислить метрики.

    Args:
        results_dir: Директория с результатами

    Returns:
        Словарь с метриками для всех конфигураций
    """
    if result_files is None:
        result_files = list(results_dir.glob("paper_results_*.json"))

    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    print(f"Loading {len(result_files)} result files...")

    # Обрабатывать каждый файл отдельно, так как они могут содержать разные конфигурации
    all_metrics = {}

    for file_path in result_files:
        try:
            # Загрузить домены из этого файла
            file_domains = load_simulation_file(file_path)

            for domain, results in file_domains.items():
                # Извлечь информацию о конфигурации
                model = _normalize_model_name(results.info.agent_info.llm)
                user_model = _normalize_model_name(results.info.user_info.llm)
                # Температура берется из user_info.llm_args (пользовательская модель)
                temp = (
                    results.info.user_info.llm_args.get("temperature", 0.0)
                    if results.info.user_info.llm_args
                    else 0.0
                )

                # Получить уникальные задачи для этого домена
                task_ids = set(sim.task_id for sim in results.simulations)

                print(
                    f"   Processing {domain}: {len(task_ids)} tasks, {len(results.simulations)} simulations"
                )

                for task_id in task_ids:
                    # Фильтруем симуляции для этой задачи
                    task_sims = [
                        sim for sim in results.simulations if sim.task_id == task_id
                    ]
                    if not task_sims:
                        print(f"      Warning: No simulations found for task {task_id}")
                        continue

                    metrics = compute_task_metrics(results, task_id)
                    if metrics:
                        key = f"{domain}_{model}_T{temp}_{task_id}"
                        all_metrics[key] = {
                            "domain": domain,
                            "model": model,
                            "user_model": user_model,
                            "temperature": temp,
                            "task": task_id,
                            **metrics,
                        }
                        print(
                            f"      ✅ Computed metrics for {task_id}: pass@1={metrics.get('pass^1', 'N/A'):.2f}, ASR={metrics.get('ASR', 'N/A'):.2f}"
                        )
                    else:
                        print(f"      ⚠️  No metrics computed for {task_id}")
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue

    print(f"Computed metrics for {len(all_metrics)} configurations")

    if len(all_metrics) == 0:
        print("⚠️  WARNING: No metrics computed!")
        print("   This might mean:")
        print("   - Files are empty or corrupted")
        print("   - Simulations don't have the expected structure")
        print("   - Task IDs don't match")
        # Попробуем показать структуру одного файла для отладки
        if result_files:
            try:
                from duma.data_model.simulation import Results

                sample = Results.load(result_files[0])
                print(f"\n   Sample file structure:")
                print(f"   - Domain: {sample.info.environment_info.domain_name}")
                print(f"   - Agent LLM: {sample.info.agent_info.llm}")
                print(f"   - User LLM: {sample.info.user_info.llm}")
                print(f"   - Number of simulations: {len(sample.simulations)}")
                if sample.simulations:
                    print(
                        f"   - First simulation task_id: {sample.simulations[0].task_id}"
                    )
                    print(
                        f"   - All task_ids: {set(s.task_id for s in sample.simulations)}"
                    )
            except Exception as debug_e:
                print(f"   Could not inspect file structure: {debug_e}")

    return all_metrics


def generate_visualizations(metrics: Dict, output_dir: Path):
    """
    Создать визуализации и экспортировать для LaTeX.

    Args:
        metrics: Словарь с метриками
        output_dir: Директория для сохранения графиков
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of all figures that should be generated
    all_figure_names = [
        "pass1_by_domain",
        "pass2_by_domain",
        "pass3_by_domain",
        "pass4_by_domain",
        "asr_by_domain",
        "temperature_effect",
        "metrics_heatmap",
    ]

    if not metrics:
        print("⚠️  No metrics to visualize - creating placeholder files")
        # Создать пустые заглушки для LaTeX
        for fig_name in all_figure_names:
            placeholder_path = output_dir / f"{fig_name}.pdf"
            # Создать минимальный PDF заглушку
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"Placeholder: {fig_name}\n\nNo data available\n\nRun experiments to generate visualizations",
                    ha="center",
                    va="center",
                    fontsize=12,
                    wrap=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                plt.savefig(placeholder_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"   Created placeholder: {placeholder_path}")
            except Exception as e:
                print(f"   Could not create placeholder {fig_name}: {e}")
        return

    # Преобразовать в DataFrame для удобства
    df = pd.DataFrame(list(metrics.values()))

    if df.empty or len(metrics) == 0:
        print("⚠️  No metrics to visualize - creating placeholder files")
        # Создать пустые заглушки для LaTeX, чтобы компиляция не падала
        for fig_name in [
            "pass1_by_domain",
            "asr_by_domain",
            "temperature_effect",
            "metrics_heatmap",
        ]:
            placeholder_path = output_dir / f"{fig_name}.pdf"
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(
                    0.5,
                    0.5,
                    f"Placeholder: {fig_name}\n\nNo data available yet.\nRun experiments to generate visualizations.",
                    ha="center",
                    va="center",
                    fontsize=12,
                    wrap=True,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                plt.savefig(placeholder_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"   Created placeholder: {placeholder_path.name}")
            except Exception as e:
                print(f"   ⚠️  Could not create placeholder {fig_name}: {e}")
        return

    # Научный стиль для журналов
    # Используем стиль с минимальными декорациями и профессиональными шрифтами
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "text.usetex": False,  # Отключаем LaTeX для совместимости
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.5,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Цветовая палитра для научных публикаций (colorblind-friendly)
    # Используем палитру, которая хорошо работает в черно-белом и цветном виде
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    sns.set_palette(colors)

    # 1. График pass@1 по доменам и моделям
    fig, ax = plt.subplots(figsize=(6, 4))
    if "domain" in df.columns and "model" in df.columns:
        if "success_count" in df.columns and "num_trials" in df.columns:
            pass1_agg = (
                df.groupby(["model", "domain"], as_index=False)
                .agg({"success_count": "sum", "num_trials": "sum"})
                .copy()
            )
            pass1_agg["pass^1"] = pass1_agg["success_count"] / pass1_agg["num_trials"]
        elif "pass^1" in df.columns:
            pass1_agg = (
                df.groupby(["model", "domain"], as_index=False)
                .agg({"pass^1": "mean"})
                .copy()
            )
        else:
            pass1_agg = None

        if pass1_agg is not None:
            pivot_pass1 = pass1_agg.pivot_table(
                values="pass^1", index="model", columns="domain", aggfunc="mean"
            )
            pivot_pass1.plot(
                kind="bar", ax=ax, width=0.7, edgecolor="black", linewidth=0.5
            )
            ax.set_ylabel("pass@1", fontsize=11)
            ax.set_xlabel("Модель", fontsize=11)
            ax.legend(
                title="Домен",
                fontsize=9,
                frameon=True,
                fancybox=False,
                edgecolor="black",
            )
            ax.grid(True, alpha=0.2, axis="y", linestyle="--", linewidth=0.5)
            ax.set_ylim([0, 1.05])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                output_dir / "pass1_by_domain.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()
            print(f"Saved: {output_dir / 'pass1_by_domain.pdf'}")

    # 1b. Графики pass@k (k=2,3,4) по доменам и моделям
    for k in [2, 3, 4]:
        pass_k_col = f"pass^{k}"
        if pass_k_col not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        if "domain" in df.columns and "model" in df.columns:
            passk_agg = (
                df.groupby(["model", "domain"], as_index=False)
                .agg({pass_k_col: "mean"})
                .copy()
            )

            if passk_agg is not None and not passk_agg.empty:
                pivot_passk = passk_agg.pivot_table(
                    values=pass_k_col, index="model", columns="domain", aggfunc="mean"
                )
                pivot_passk.plot(
                    kind="bar", ax=ax, width=0.7, edgecolor="black", linewidth=0.5
                )
                ax.set_ylabel(f"pass@{k}", fontsize=11)
                ax.set_xlabel("Model", fontsize=11)
                ax.legend(
                    title="Domain",
                    fontsize=9,
                    frameon=True,
                    fancybox=False,
                    edgecolor="black",
                )
                ax.grid(True, alpha=0.2, axis="y", linestyle="--", linewidth=0.5)
                ax.set_ylim([0, 1.05])
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(
                    output_dir / f"pass{k}_by_domain.pdf",
                    dpi=300,
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                plt.close()
                print(f"Saved: {output_dir / f'pass{k}_by_domain.pdf'}")

    # 1c. Графики pass@k по доменам, разделенные по температуре
    if "temperature" in df.columns:
        temperatures = sorted(df["temperature"].unique())
        for temp in temperatures:
            temp_subset = df[df["temperature"] == temp]
            if temp_subset.empty:
                continue

            for k in [1, 2, 3, 4]:
                pass_k_col = f"pass^{k}"
                if pass_k_col not in temp_subset.columns:
                    continue

                fig, ax = plt.subplots(figsize=(6, 4))
                if "domain" in temp_subset.columns and "model" in temp_subset.columns:
                    passk_agg = (
                        temp_subset.groupby(["model", "domain"], as_index=False)
                        .agg({pass_k_col: "mean"})
                        .copy()
                    )

                    if passk_agg is not None and not passk_agg.empty:
                        pivot_passk = passk_agg.pivot_table(
                            values=pass_k_col,
                            index="model",
                            columns="domain",
                            aggfunc="mean",
                        )
                        pivot_passk.plot(
                            kind="bar",
                            ax=ax,
                            width=0.7,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        ax.set_ylabel(f"pass@{k}", fontsize=11)
                        ax.set_xlabel("Model", fontsize=11)
                        ax.set_title(f"Temperature = {temp}", fontsize=12)
                        ax.legend(
                            title="Domain",
                            fontsize=9,
                            frameon=True,
                            fancybox=False,
                            edgecolor="black",
                        )
                        ax.grid(
                            True, alpha=0.2, axis="y", linestyle="--", linewidth=0.5
                        )
                        ax.set_ylim([0, 1.05])
                        plt.xticks(rotation=45, ha="right")
                        plt.tight_layout()
                        # Filename: pass1_by_domain_T0.0.pdf, etc.
                        temp_str = f"{temp}".replace(".", "_")
                        plt.savefig(
                            output_dir / f"pass{k}_by_domain_T{temp_str}.pdf",
                            dpi=300,
                            bbox_inches="tight",
                            pad_inches=0.1,
                        )
                        plt.close()
                        print(
                            f"Saved: {output_dir / f'pass{k}_by_domain_T{temp_str}.pdf'}"
                        )

    # 1d. Comprehensive pass@k heatmap by model, domain, and temperature
    if "temperature" in df.columns and "domain" in df.columns and "model" in df.columns:
        from matplotlib import colors as mcolors

        for k in [1, 2, 3, 4]:
            pass_k_col = f"pass^{k}"
            if pass_k_col not in df.columns:
                continue

            # Aggregate by model, temperature, domain
            agg_passk = (
                df.groupby(["model", "temperature", "domain"], as_index=False)
                .agg({pass_k_col: "mean"})
                .copy()
            )

            if agg_passk.empty:
                continue

            # Make readable row labels
            agg_passk["model_T"] = agg_passk.apply(
                lambda r: f"{r['model']} (T={float(r['temperature']):g})", axis=1
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            heatmap_data = agg_passk.pivot_table(
                values=pass_k_col, index="model_T", columns="domain", aggfunc="mean"
            )

            # Stable ordering: sort by model then temperature
            heatmap_data = heatmap_data.reindex(
                sorted(
                    heatmap_data.index,
                    key=lambda s: (
                        s.split(" (T=")[0],
                        float(s.split("T=")[1].rstrip(")")),
                    ),
                )
            )

            cmap = "cividis"
            norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                norm=norm,
                ax=ax,
                cbar_kws={"label": f"pass@{k}", "shrink": 0.85},
                linewidths=0.5,
                linecolor="white",
                square=False,
                annot_kws={"fontsize": 8},
            )
            ax.set_xlabel("Domain", fontsize=11)
            ax.set_ylabel("Model (T)", fontsize=11)
            ax.set_title(f"pass@{k} by Model, Domain, and Temperature", fontsize=12)
            ax.tick_params(labelsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            plt.tight_layout()
            plt.savefig(
                output_dir / f"pass{k}_heatmap.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()
            print(f"Saved: {output_dir / f'pass{k}_heatmap.pdf'}")

    # 2. График ASR по доменам и моделям
    fig, ax = plt.subplots(figsize=(6, 4))
    if "domain" in df.columns and "model" in df.columns:
        if "success_count" in df.columns and "num_trials" in df.columns:
            asr_agg = (
                df.groupby(["model", "domain"], as_index=False)
                .agg({"success_count": "sum", "num_trials": "sum"})
                .copy()
            )
            asr_agg["ASR"] = 1.0 - (asr_agg["success_count"] / asr_agg["num_trials"])
        elif "ASR" in df.columns:
            asr_agg = (
                df.groupby(["model", "domain"], as_index=False)
                .agg({"ASR": "mean"})
                .copy()
            )
        else:
            asr_agg = None

        if asr_agg is not None:
            pivot_asr = asr_agg.pivot_table(
                values="ASR", index="model", columns="domain", aggfunc="mean"
            )
            pivot_asr.plot(
                kind="bar", ax=ax, width=0.7, edgecolor="black", linewidth=0.5
            )
            ax.set_ylabel("ASR", fontsize=11)
            ax.set_xlabel("Модель", fontsize=11)
            ax.legend(
                title="Домен",
                fontsize=9,
                frameon=True,
                fancybox=False,
                edgecolor="black",
            )
            ax.grid(True, alpha=0.2, axis="y", linestyle="--", linewidth=0.5)
            ax.set_ylim([0, 1.05])
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                output_dir / "asr_by_domain.pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()
            print(f"Saved: {output_dir / 'asr_by_domain.pdf'}")

    # 3. График влияния температуры
    # ВАЖНО: для сопоставимости с таблицами/heatmap агрегируем по (model, temperature, domain).
    if "temperature" in df.columns and "domain" in df.columns and "model" in df.columns:
        group_cols = ["model", "temperature", "domain"]

        if "success_count" in df.columns and "num_trials" in df.columns:
            discrete_agg = (
                df.groupby(group_cols, as_index=False)
                .agg({"success_count": "sum", "num_trials": "sum"})
                .copy()
            )
            discrete_agg["pass^1"] = (
                discrete_agg["success_count"] / discrete_agg["num_trials"]
            )
            discrete_agg["ASR"] = 1.0 - discrete_agg["pass^1"]
        else:
            base_metrics = [m for m in ["pass^1", "ASR"] if m in df.columns]
            if not base_metrics:
                discrete_agg = df[group_cols].drop_duplicates().copy()
            else:
                discrete_agg = (
                    df.groupby(group_cols, as_index=False)
                    .agg({m: "mean" for m in base_metrics})
                    .copy()
                )

        continuous_metrics = [
            m
            for m in [
                "avg_reward",
                "avg_agent_cost",
                "avg_duration",
                "avg_num_messages",
            ]
            if m in df.columns
        ]
        if continuous_metrics:
            cont_agg = (
                df.groupby(group_cols, as_index=False)
                .agg({m: "mean" for m in continuous_metrics})
                .copy()
            )
            temp_df = discrete_agg.merge(cont_agg, on=group_cols, how="left")
        else:
            temp_df = discrete_agg

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        metric_names = [
            "pass^1",
            "ASR",
            "avg_reward",
            "avg_agent_cost",
            "avg_duration",
            "avg_num_messages",
        ]
        metric_labels = {
            "pass^1": "pass@1",
            "ASR": "ASR",
            "avg_reward": "Средняя награда",
            "avg_agent_cost": "Стоимость агента ($)",
            "avg_duration": "Длительность (с)",
            "avg_num_messages": "Число сообщений",
        }

        models = sorted(temp_df["model"].unique())
        linestyles = ["-", "--", ":", "-."]
        markers = ["o", "s", "^", "D"]
        model_style = {
            model: {
                "linestyle": linestyles[idx % len(linestyles)],
                "marker": markers[idx % len(markers)],
            }
            for idx, model in enumerate(models)
        }

        domains = sorted(temp_df["domain"].unique())

        for idx, metric in enumerate(metric_names):
            if metric not in temp_df.columns:
                continue

            ax = axes[idx // 3, idx % 3]
            for domain_idx, domain in enumerate(domains):
                domain_color = colors[domain_idx % len(colors)]
                for model in models:
                    subset = temp_df[
                        (temp_df["domain"] == domain) & (temp_df["model"] == model)
                    ]
                    if subset.empty:
                        continue

                    subset_sorted = subset.sort_values("temperature")
                    ax.plot(
                        subset_sorted["temperature"],
                        subset_sorted[metric],
                        label=f"{domain} ({model})",
                        color=domain_color,
                        linestyle=model_style[model]["linestyle"],
                        marker=model_style[model]["marker"],
                        linewidth=1.5,
                        markersize=5,
                        markerfacecolor=domain_color,
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                    )

            ax.set_xlabel("Температура", fontsize=10)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
            ax.legend(
                fontsize=7, frameon=True, fancybox=False, edgecolor="black", ncol=2
            )
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(
            output_dir / "temperature_effect.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()
        print(f"Saved: {output_dir / 'temperature_effect.pdf'}")

    # 4. Heatmap метрики pass@1 по доменам и (модель, температура)
    # ВАЖНО: делаем тот же тип агрегации, что и в таблицах: суммируем successes/trials
    # по всем кейсам домена для каждой пары (model, temperature).
    if (
        "success_count" in df.columns
        and "num_trials" in df.columns
        and "temperature" in df.columns
        and "domain" in df.columns
        and "model" in df.columns
    ):
        from matplotlib import colors as mcolors

        agg = (
            df.groupby(["model", "temperature", "domain"], as_index=False)
            .agg({"success_count": "sum", "num_trials": "sum"})
            .copy()
        )
        agg["pass@1"] = agg["success_count"] / agg["num_trials"]

        # Make readable row labels
        agg["model_T"] = agg.apply(
            lambda r: f"{r['model']} (T={float(r['temperature']):g})", axis=1
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap_data = agg.pivot_table(
            values="pass@1", index="model_T", columns="domain", aggfunc="mean"
        )

        # Stable ordering: sort by model then temperature.
        heatmap_data = heatmap_data.reindex(
            sorted(
                heatmap_data.index,
                key=lambda s: (s.split(" (T=")[0], float(s.split("T=")[1].rstrip(")"))),
            )
        )

        cmap = "cividis"
        norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            norm=norm,
            ax=ax,
            cbar_kws={"label": "pass@1", "shrink": 0.85},
            linewidths=0.5,
            linecolor="white",
            square=False,
            annot_kws={"fontsize": 8},
        )
        ax.set_xlabel("Домен", fontsize=11)
        ax.set_ylabel("Модель (T)", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(
            output_dir / "metrics_heatmap.pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()
        print(f"Saved: {output_dir / 'metrics_heatmap.pdf'}")

    print(f"\nAll visualizations saved to {output_dir}")


def update_template_tex(
    template_path: Path,
    detailed_table_path: Path,
    model_domain_table_path: Path,
    significance_table_path: Path,
    temp_significance_table_path: Path,
    visualizations_dir: Path,
):
    """
    Обновить template.tex с новыми таблицами и визуализациями.
    """
    template_content = template_path.read_text()

    # 1. Добавить описание расчета метрик в раздел 4.1.3 (после списка метрик)
    metrics_description = """
\\subsubsection{Методы расчета метрик}

\\textbf{Дискретные метрики:}
\\begin{itemize}
    \\item \\textbf{pass@1:} доля кейсов, в которых агент успешно прошёл хотя бы один ассерт. Вычисляется как $\\frac{\\text{число успешных кейсов}}{\\text{общее число кейсов}}$.
    \\item \\textbf{ASR:} Attack Success Rate, вычисляется как $1 - \\text{pass@1}$.
\\end{itemize}

\\textbf{Непрерывные метрики:}
\\begin{itemize}
    \\item \\textbf{avg\\_reward:} среднее арифметическое наград за выполнение задачи по всем прогонам: $\\bar{r} = \\frac{1}{n}\\sum_{i=1}^{n} r_i$, где $r_i$ --- награда $i$-го прогона, $n$ --- число прогонов.
    \\item \\textbf{avg\\_duration:} среднее арифметическое времени выполнения симуляции в секундах: $\\bar{t} = \\frac{1}{n}\\sum_{i=1}^{n} t_i$.
    \\item \\textbf{avg\\_num\\_messages:} среднее арифметическое числа сообщений в диалоге: $\\bar{m} = \\frac{1}{n}\\sum_{i=1}^{n} m_i$.
\\end{itemize}

Для всех метрик вычисляются 95\\% доверительные интервалы: t-интервалы для непрерывных метрик, интервалы Уилсона для дискретных метрик.
"""

    # Найти место для вставки (после списка метрик)
    metrics_list_marker = (
        "\\item \\textbf{avg\\_num\\_messages} --- среднее число сообщений в диалоге."
    )
    if (
        metrics_list_marker in template_content
        and "Методы расчета метрик" not in template_content
    ):
        # Вставить после списка метрик, перед следующим подразделом
        insert_pos = template_content.find(metrics_list_marker) + len(
            metrics_list_marker
        )
        # Найти конец списка
        end_list = template_content.find("\\end{itemize}", insert_pos)
        if end_list != -1:
            template_content = (
                template_content[: end_list + len("\\end{itemize}")]
                + metrics_description
                + template_content[end_list + len("\\end{itemize}") :]
            )

    import re

    # 2) Заменить пилотную таблицу tab:results на автосгенерированную
    if detailed_table_path.exists():
        detailed_rel = detailed_table_path.relative_to(template_path.parent)
        detailed_section = (
            "\n% Автосгенерированная таблица (детальные метрики)\n"
            f"\\input{{{detailed_rel}}}\n"
        )

        template_content = template_content.replace(
            "В таблице~\\ref{tab:results} представлены результаты экспериментов по всем доменам.",
            "В таблице~\\ref{tab:detailed_metrics} представлены детальные результаты экспериментов по всем доменам.",
        )

        tab_results_pattern = (
            r"\\begin\{longtable\}.*?\\label\{tab:results\}.*?\\end\{longtable\}"
        )
        template_content, n_subs = re.subn(
            tab_results_pattern,
            lambda _m: detailed_section,
            template_content,
            flags=re.DOTALL,
        )

        if n_subs == 0:
            print(
                "Warning: could not find tab:results longtable; "
                "skipping detailed table insertion"
            )

    # 3) Обновить агрегированную таблицу в Results & Discussion
    if model_domain_table_path.exists():
        model_domain_rel = model_domain_table_path.relative_to(template_path.parent)
        model_domain_section = (
            "\n% Автосгенерированная таблица (model×domain pass@1)\n"
            f"\\input{{{model_domain_rel}}}\n"
        )

        # Replace only inside the Results & Discussion aggregated subsection
        agg_subsection_start = "\\subsection{Агрегированные результаты}"
        viz_subsection_start = "\\subsection{Визуализация результатов}"
        if (
            agg_subsection_start in template_content
            and viz_subsection_start in template_content
        ):
            start_idx = template_content.find(agg_subsection_start)
            end_idx = template_content.find(viz_subsection_start, start_idx)
            subsection = template_content[start_idx:end_idx]

            table_pattern = r"\\begin\{table\}\[htbp\].*?\\end\{table\}"
            subsection, n_subs = re.subn(
                table_pattern,
                lambda _m: model_domain_section,
                subsection,
                flags=re.DOTALL,
            )

            if n_subs > 0:
                template_content = (
                    template_content[:start_idx]
                    + subsection
                    + template_content[end_idx:]
                )

    # 4) Вставить таблицы статистической значимости (модели и температуры)
    viz_marker = "\\subsection{Визуализация результатов}"

    if (
        significance_table_path.exists()
        and "significance_table.tex" not in template_content
    ):
        sig_rel = significance_table_path.relative_to(template_path.parent)
        sig_section = (
            "\n\\subsection{Статистическая значимость (gpt-4o vs gpt-4o-mini)}\n"
            f"\\input{{{sig_rel}}}\n"
        )
        if viz_marker in template_content:
            template_content = template_content.replace(
                viz_marker, sig_section + "\n" + viz_marker
            )

    if (
        temp_significance_table_path.exists()
        and "temperature_significance_table.tex" not in template_content
    ):
        temp_rel = temp_significance_table_path.relative_to(template_path.parent)
        temp_section = (
            "\n\\subsection{Статистическая значимость (влияние температуры)}\n"
            f"\\input{{{temp_rel}}}\n"
        )
        if viz_marker in template_content:
            template_content = template_content.replace(
                viz_marker, temp_section + "\n" + viz_marker
            )

    # 4. Обновить раздел с визуализациями (не удалять attack-плоты)
    visualizations_section = f"""
\\subsection{{Визуализация результатов}}

На рисунках~\\ref{{fig:attack_flow}}--\\ref{{fig:attack_timeline_output}} представлены визуализации потоков атак по доменам, а на рисунках~\\ref{{fig:pass1_by_domain}}--\\ref{{fig:metrics_heatmap}} --- визуализации метрик по доменам и моделям.

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figs/attack_flow.pdf}}
\\caption{{Визуализация потока атак по доменам безопасности}}
\\label{{fig:attack_flow}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figs/attack_sankey.pdf}}
\\caption{{Поток атаки: от вектора атаки к результату безопасности}}
\\label{{fig:attack_sankey}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figs/attack_timeline_mail_rag_phishing.pdf}}
\\caption{{Временная диаграмма потока сообщений: отравление RAG}}
\\label{{fig:attack_timeline_rag}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figs/attack_timeline_collab.pdf}}
\\caption{{Временная диаграмма потока сообщений: межагентное отравление}}
\\label{{fig:attack_timeline_collab}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.95\\textwidth]{{figs/attack_timeline_output_handling.pdf}}
\\caption{{Временная диаграмма потока сообщений: инъекция в вывод}}
\\label{{fig:attack_timeline_output}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{figs/pass1_by_domain.pdf}}
\\caption{{Метрика pass@1 по доменам для различных моделей}}
\\label{{fig:pass1_by_domain}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{figs/asr_by_domain.pdf}}
\\caption{{Attack Success Rate (ASR) по доменам для различных моделей}}
\\label{{fig:asr_by_domain}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{figs/temperature_effect.pdf}}
\\caption{{Влияние температуры пользовательской модели на метрики}}
\\label{{fig:temperature_effect}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{figs/metrics_heatmap.pdf}}
\\caption{{Heatmap всех метрик по доменам и моделям}}
\\label{{fig:metrics_heatmap}}
\\end{{figure}}
"""

    analysis_marker = "\\subsection{Анализ результатов}"
    viz_marker = "\\subsection{Визуализация результатов}"

    if viz_marker in template_content and analysis_marker in template_content:
        start_idx = template_content.find(viz_marker)
        end_idx = template_content.find(analysis_marker, start_idx)
        if end_idx != -1:
            template_content = (
                template_content[:start_idx]
                + visualizations_section
                + "\n"
                + template_content[end_idx:]
            )
    elif analysis_marker in template_content:
        template_content = template_content.replace(
            analysis_marker, visualizations_section + "\n" + analysis_marker
        )

    # Сохранить обновленный файл
    template_path.write_text(template_content)
    print(f"Updated {template_path}")


def compile_pdf(template_path: Path) -> bool:
    """Скомпилировать PDF из LaTeX."""
    template_dir = template_path.parent
    original_dir = os.getcwd()

    try:
        os.chdir(template_dir)

        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            template_path.name,
        ]

        # Первая компиляция - используем errors='replace' для обработки не-UTF8 символов
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if result.returncode != 0:
            print(f"Error compiling LaTeX (first pass):")
            if result.stderr:
                print(result.stderr[:1000])  # Первые 1000 символов ошибки
            if result.stdout:
                # Показать последние строки вывода для диагностики
                lines = result.stdout.split("\n")
                error_lines = [l for l in lines if "error" in l.lower() or "!" in l]
                if error_lines:
                    print("\nLaTeX errors found:")
                    for line in error_lines[-10:]:  # Последние 10 ошибок
                        print(f"  {line}")
            return False

        # Вторая компиляция для обновления ссылок
        result2 = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if result2.returncode != 0:
            print(
                f"Warning: Second compilation had errors (this is often OK for references)"
            )

        # После os.chdir(template_dir) нужно использовать относительный путь
        pdf_path = Path("template.pdf")
        if pdf_path.exists():
            # Полный путь для вывода
            full_pdf_path = template_dir / "template.pdf"
            print(f"✅ PDF compiled: {full_pdf_path}")
            return True
        else:
            print("❌ PDF file not found after compilation")
            # Попробуем показать содержимое директории для диагностики
            print(f"Files in {template_dir}:")
            for f in sorted(template_dir.glob("*.pdf"))[:5]:
                print(f"  - {f.name}")
            return False

    finally:
        os.chdir(original_dir)


def main():
    """Главная функция для автоматизации всего процесса."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Автоматический запуск экспериментов, обработка результатов и генерация таблиц/визуализаций"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-5.1",
            "openai/gpt-5.2",
        ],
        help="Модели агента для тестирования",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=None,
        help="Модель для симуляции пользователя (по умолчанию: такая же как agent)",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0],
        help="Температуры пользовательской модели (по умолчанию: 0.0, 0.5, 1.0). В solo-режиме игнорируется.",
    )
    parser.add_argument(
        "--agent-temperature",
        type=float,
        default=0.0,
        help="Фиксированная температура агента (по умолчанию: 0.0)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["mail_rag_phishing", "collab", "output_handling"],
        help="Домены для тестирования",
    )
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Количество прогонов на конфигурацию"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Максимальное количество параллельных запусков экспериментов (процессов duma run)",
    )
    parser.add_argument(
        "--duma-max-concurrency",
        type=int,
        default=1,
        help="Внутренняя параллельность в каждом duma run (параллельные trials внутри одного эксперимента)",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        help="Пропустить запуск экспериментов (только обработка)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Разрешить генерацию таблиц/графиков при неполных данных (по умолчанию скрипт остановится)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Принудительно перезапустить все эксперименты (удалить существующие результаты)",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Пропустить генерацию визуализаций",
    )
    parser.add_argument(
        "--compile-pdf",
        action="store_true",
        default=True,
        help="Скомпилировать PDF после обновления (по умолчанию: True)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Директория для поиска результатов (по умолчанию data/duma/simulations/)",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=None,
        help="Путь к template.tex (по умолчанию: docs/paper_template/template.tex или docs/paper_template_solo/template.tex в solo режиме)",
    )
    parser.add_argument(
        "--solo",
        action="store_true",
        help="Run in solo mode (agent without user, --agent llm_agent_solo --user dummy_user)",
    )
    parser.add_argument(
        "--agent-base-url",
        type=str,
        default=None,
        help="Base URL for agent LLM provider (e.g. https://api.vsellm.ru/v1)",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable name for API key (e.g. VSE_LLM_API_KEY)",
    )

    args = parser.parse_args()

    # Resolve default template-path based on solo mode
    if args.template_path is None:
        if args.solo:
            args.template_path = Path("docs/paper_template_solo/template.tex")
        else:
            args.template_path = Path("docs/paper_template/template.tex")

    # Определить задачи для каждого домена
    # Если список задач пустой, скрипт автоматически загрузит все задачи из tasks.json
    tasks_by_domain = {
        "mail_rag_phishing": [],  # Пустой список = загрузить все задачи
        "collab": [],  # Пустой список = загрузить все задачи
        "output_handling": [],  # Пустой список = загрузить все задачи
    }

    # Если домен не указан в tasks_by_domain, используем пустой список (загрузит все задачи)
    domains_dict = {d: tasks_by_domain.get(d, []) for d in args.domains}

    from duma.utils.utils import DATA_DIR

    # Create timestamped run directory name (or use --results-dir for processing)
    if args.skip_experiments and args.results_dir is not None:
        # Processing an existing run directory
        run_dir_name = ""
        results_dir = args.results_dir
    else:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"run_{run_timestamp}"
        results_dir = DATA_DIR / "simulations" / run_dir_name

    print(f"Run directory: {results_dir}")

    # Шаг 1: Запустить эксперименты
    if not args.skip_experiments:
        print("=" * 80)
        print("ШАГ 1: Запуск экспериментов")
        print("=" * 80)

        # Save run config snapshot
        results_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "models": args.models,
            "agent_temperature": args.agent_temperature,
            "user_temperatures": args.temperatures,
            "domains": args.domains,
            "num_trials": args.num_trials,
            "user_llm": args.user_llm,
            "solo": args.solo,
            "max_concurrency": args.max_concurrency,
            "duma_max_concurrency": args.duma_max_concurrency,
            "agent_base_url": args.agent_base_url,
            "api_key_env": args.api_key_env,
            "force_rerun": args.force_rerun,
        }
        config_path = results_dir / "run_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        print(f"Run config saved to {config_path}")

        run_all_experiments(
            args.models,
            args.temperatures,
            domains_dict,
            args.num_trials,
            args.max_concurrency,
            run_dir_name=run_dir_name,
            force_rerun=args.force_rerun,
            user_llm=args.user_llm,
            solo=args.solo,
            agent_base_url=args.agent_base_url,
            api_key_env=args.api_key_env,
            duma_max_concurrency=args.duma_max_concurrency,
            agent_temperature=args.agent_temperature,
        )
    else:
        print("Пропуск запуска экспериментов (--skip-experiments)")

    # Шаг 2: Обработать результаты
    print("\n" + "=" * 80)
    print("ШАГ 2: Обработка результатов")
    print("=" * 80)

    # Сформировать ожидаемый список файлов результатов для выбранных
    # доменов/моделей/температур, чтобы:
    # - проверить полноту данных
    # - не смешивать результаты с другими запусками
    def get_all_tasks_for_domain(domain_name: str) -> List[str]:
        possible_paths = [
            results_dir.parent / "duma" / "domains" / domain_name / "tasks.json",
            results_dir.parent / "domains" / domain_name / "tasks.json",
            Path("data/duma/domains") / domain_name / "tasks.json",
        ]
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tasks_data = json.load(f)
                    if isinstance(tasks_data, list):
                        return [t["id"] for t in tasks_data if "id" in t]
                    if isinstance(tasks_data, dict):
                        tasks_list = tasks_data.get("tasks", [])
                        return [t["id"] for t in tasks_list if "id" in t]
                except Exception:
                    continue
        return []

    def is_complete_result_file(
        path: Path, task_id: str
    ) -> tuple[bool, int, str | None]:
        if not path.exists():
            return (False, 0, "missing")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sims = data.get("simulations", [])
            task_ids = {s.get("task_id") for s in sims if isinstance(s, dict)}
            if sims and task_id not in task_ids:
                return (False, len(sims), f"wrong task_id {task_ids}")
            if len(sims) < args.num_trials:
                return (False, len(sims), "incomplete")
            return (True, len(sims), None)
        except Exception as e:
            return (False, 0, f"unreadable: {e}")

    expected_files: list[tuple[Path, str]] = []
    for domain in args.domains:
        domain_tasks = domains_dict.get(domain) or get_all_tasks_for_domain(domain)
        if not domain_tasks:
            print(f"⚠️  No tasks found for domain {domain}")
            continue

        for model in args.models:
            model_id = _model_id_for_results(model)
            file_model = _sanitize_model_for_filename(model_id)
            if args.solo:
                for task_id in domain_tasks:
                    stem = f"paper_results_solo_{domain}_{file_model}_T{args.agent_temperature}_{task_id}"
                    expected_files.append((results_dir / f"{stem}.json", task_id))
            else:
                # Определяем модель пользователя
                effective_user_llm = args.user_llm if args.user_llm else model
                user_model_id = _model_id_for_results(effective_user_llm)
                file_user_model = _sanitize_model_for_filename(user_model_id)
                for temp in args.temperatures:
                    for task_id in domain_tasks:
                        stem = f"paper_results_{domain}_{file_model}_U{file_user_model}_UT{temp}_{task_id}"
                        expected_files.append((results_dir / f"{stem}.json", task_id))

    missing_or_incomplete: list[str] = []
    selected_result_files: list[Path] = []
    seen_paths: set[str] = set()

    for path, task_id in expected_files:
        ok, sim_count, reason = is_complete_result_file(path, task_id)
        if ok:
            key = str(path)
            if key not in seen_paths:
                selected_result_files.append(path)
                seen_paths.add(key)
        else:
            missing_or_incomplete.append(
                f"{path.name} ({reason}, {sim_count}/{args.num_trials})"
            )

    print(
        f"Expected files: {len(expected_files)} | complete: {len(selected_result_files)} | missing/incomplete: {len(missing_or_incomplete)}"
    )
    if missing_or_incomplete and not args.allow_incomplete:
        print("\n❌ Dataset is incomplete; not updating tables/figs/template.")
        print("   To force generation anyway, pass --allow-incomplete")
        print("\nFirst missing/incomplete files:")
        for line in missing_or_incomplete[:20]:
            print(f"  - {line}")
        return

    # Шаг 2: Обработать результаты (только выбранные result files)
    metrics = process_all_results(results_dir, result_files=selected_result_files)

    # Сохранить метрики в JSON для дальнейшего использования
    metrics_file = results_dir / "summary_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_file}")

    # Шаг 3: Сгенерировать таблицы LaTeX
    print("\n" + "=" * 80)
    print("ШАГ 3: Генерация LaTeX таблиц")
    print("=" * 80)
    result_files = selected_result_files

    template_dir = args.template_path.parent
    detailed_table_path = template_dir / "detailed_metrics_table.tex"
    aggregated_table_path = template_dir / "aggregated_table.tex"
    model_domain_table_path = template_dir / "model_domain_table.tex"
    significance_table_path = template_dir / "significance_table.tex"
    temp_significance_table_path = template_dir / "temperature_significance_table.tex"

    generate_detailed_metrics_table_latex(
        result_files, args.domains, detailed_table_path
    )

    # Note: this table aggregates across mixed configs; kept for reference.
    generate_aggregated_table_latex(result_files, args.domains, aggregated_table_path)

    generate_model_domain_table_latex(
        result_files,
        args.domains,
        [_normalize_model_name(m) for m in args.models],
        [float(t) for t in args.temperatures],
        model_domain_table_path,
    )

    generate_significance_table_latex(
        result_files,
        args.domains,
        [float(t) for t in args.temperatures],
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        output_path=significance_table_path,
    )

    if len(args.temperatures) >= 2:
        generate_temperature_significance_table_latex(
            result_files,
            args.domains,
            [_normalize_model_name(m) for m in args.models],
            [float(t) for t in args.temperatures],
            output_path=temp_significance_table_path,
        )
    else:
        print("Skipping temperature significance table (need at least 2 temperatures)")

    print(
        "Tables generated: "
        f"{detailed_table_path}, {aggregated_table_path}, "
        f"{model_domain_table_path}, {significance_table_path}, {temp_significance_table_path}"
    )

    # Шаг 4: Сгенерировать визуализации
    if not args.skip_visualizations:
        print("\n" + "=" * 80)
        print("ШАГ 4: Генерация визуализаций")
        print("=" * 80)
        figs_dir = template_dir / "figs"
        generate_visualizations(metrics, figs_dir)
    else:
        print("Пропуск генерации визуализаций (--skip-visualizations)")
        figs_dir = template_dir / "figs"

    # Шаг 5: Обновить template.tex
    print("\n" + "=" * 80)
    print("ШАГ 5: Обновление template.tex")
    print("=" * 80)
    update_template_tex(
        args.template_path,
        detailed_table_path,
        model_domain_table_path,
        significance_table_path,
        temp_significance_table_path,
        figs_dir,
    )

    # Шаг 6: Скомпилировать PDF
    if args.compile_pdf:
        print("\n" + "=" * 80)
        print("ШАГ 6: Компиляция PDF")
        print("=" * 80)
        compile_pdf(args.template_path)
    else:
        print("\nДля компиляции PDF запустите:")
        print(
            f"  cd docs/paper_template && pdflatex -interaction=nonstopmode -halt-on-error template.tex"
        )

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print(f"Results directory: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
