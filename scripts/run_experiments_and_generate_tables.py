#!/usr/bin/env python3
"""
Автоматический запуск экспериментов.

Этот скрипт запускает все эксперименты и сохраняет файлы симуляций.
"""

import json
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

import numpy as np

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

from duma.utils.model_ref import normalize_model_ref


def _model_id_for_results(model: str | None) -> str:
    """Normalize model names to canonical vendor/model format."""
    return normalize_model_ref(model)


def _sanitize_model_for_filename(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")


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
            start_msg = f"[Thread-{thread_id % 10000}] [{idx}] START: {short_name}"
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

            end_msg = f"[Thread-{thread_id % 10000}] [{idx}] DONE: {short_name} ({time.time() - exp_start_time:.1f}s)"
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
        "Starting parallel execution with "
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

        print(f"All {len(future_to_item)} tasks submitted to thread pool")
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
                    print(f"\n   ERROR [{idx}] after {duration:.1f}s: {name}")
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
                            print(f"   Failed to write to error log: {log_err}")
            except Exception as e:
                errors += 1
                project_root = Path(__file__).parent.parent
                error_log_file = project_root / "experiment_errors.log"
                print(f"\n   Exception in experiment {idx}: {e}")
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
                    print(f"   Failed to write exception to error log: {log_err}")

    if HAS_TQDM:
        pbar.close()

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print(f"\n{'=' * 80}")
    print(f"Summary:")
    print(f"   Completed: {completed}")
    print(f"   Skipped: {skipped}")
    print(f"   Errors: {errors}")
    print(f"   Total time: {total_time_str}")
    if completed > 0 and times:
        avg_time = np.mean(times)
        print(f"   Average time per experiment: {avg_time:.1f}s")
    print(f"{'=' * 80}\n")

    # Показать информацию о логе ошибок
    project_root = Path(__file__).parent.parent
    error_log_file = project_root / "experiment_errors.log"
    if error_log_file.exists() and errors > 0:
        print(
            f"{errors} errors occurred. Full error details saved to: {error_log_file}"
        )
        print(f"   To view errors: cat {error_log_file}\n")

    return actual_output_dir


def main():
    """Главная функция: запуск экспериментов."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Запуск экспериментов DUMA-Bench"
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
        "--force-rerun",
        action="store_true",
        help="Принудительно перезапустить все эксперименты (удалить существующие результаты)",
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

    # Если домен не указан в tasks_by_domain, используем пустой список (загрузит все задачи)
    domains_dict = {d: [] for d in args.domains}

    from duma.utils.utils import DATA_DIR

    # Create timestamped run directory name
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{run_timestamp}"
    results_dir = DATA_DIR / "simulations" / run_dir_name

    print(f"Run directory: {results_dir}")

    # Запустить эксперименты
    print("=" * 80)
    print("Запуск экспериментов")
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

    print("\n" + "=" * 80)
    print("ГОТОВО!")
    print(f"Results directory: {results_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
