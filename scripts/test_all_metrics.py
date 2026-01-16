#!/usr/bin/env python3
"""
Скрипт для проверки всех метрик на существующих результатах.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from result_collection import (
    load_simulations,
    compute_task_metrics,
    t_confidence_interval,
    wilson_confidence_interval,
    compare_continuous_metrics,
    compare_conversion_metrics,
)
from tau2.utils.utils import DATA_DIR


def test_all_metrics():
    """Проверить вычисление всех метрик."""
    print("=" * 80)
    print("ПРОВЕРКА ВСЕХ МЕТРИК")
    print("=" * 80)
    
    # Найти все файлы результатов
    results_dir = DATA_DIR / "simulations"
    result_files = list(results_dir.glob("paper_results_*.json"))
    
    if not result_files:
        print(f"❌ Не найдено файлов результатов в {results_dir}")
        print("   Запустите сначала эксперименты или укажите другую директорию")
        return False
    
    print(f"📁 Найдено {len(result_files)} файлов результатов")
    
    # Загрузить результаты
    print("\n📊 Загрузка результатов...")
    all_domains = load_simulations(result_files)
    
    if not all_domains:
        print("❌ Не удалось загрузить результаты")
        return False
    
    print(f"✅ Загружено {len(all_domains)} доменов")
    
    # Проверить метрики для каждого домена и задачи
    all_metrics_found = set()
    metrics_errors = []
    
    required_metrics = {
        'pass^1', 'ASR', 'avg_reward', 'avg_agent_cost', 'avg_user_cost',
        'avg_duration', 'avg_num_messages',
        'pass^1_ci_lower', 'pass^1_ci_upper',
        'avg_reward_ci_lower', 'avg_reward_ci_upper',
        'avg_duration_ci_lower', 'avg_duration_ci_upper',
        'avg_num_messages_ci_lower', 'avg_num_messages_ci_upper'
    }
    
    print("\n🔍 Проверка метрик...")
    for domain, results in all_domains.items():
        task_ids = set(sim.task_id for sim in results.simulations)
        
        for task_id in task_ids:
            metrics = compute_task_metrics(results, task_id)
            
            if not metrics:
                metrics_errors.append(f"{domain}/{task_id}: нет метрик")
                continue
            
            # Проверить наличие всех требуемых метрик
            missing = required_metrics - set(metrics.keys())
            if missing:
                metrics_errors.append(f"{domain}/{task_id}: отсутствуют метрики {missing}")
            
            # Проверить корректность значений
            if 'pass^1' in metrics and 'ASR' in metrics:
                expected_asr = 1.0 - metrics['pass^1']
                actual_asr = metrics['ASR']
                if abs(expected_asr - actual_asr) > 0.0001:
                    metrics_errors.append(
                        f"{domain}/{task_id}: ASR не соответствует (ожидается {expected_asr:.4f}, получено {actual_asr:.4f})"
                    )
            
            # Проверить доверительные интервалы
            if 'pass^1_ci_lower' in metrics and 'pass^1_ci_upper' in metrics:
                ci_lower = metrics['pass^1_ci_lower']
                ci_upper = metrics['pass^1_ci_upper']
                if ci_lower > ci_upper:
                    metrics_errors.append(f"{domain}/{task_id}: неправильный CI для pass@1")
                if ci_lower < 0 or ci_upper > 1:
                    metrics_errors.append(f"{domain}/{task_id}: CI для pass@1 вне диапазона [0,1]")
            
            # Проверить непрерывные метрики
            for metric_name in ['avg_reward', 'avg_duration', 'avg_num_messages']:
                if metric_name in metrics:
                    ci_lower_key = f"{metric_name}_ci_lower"
                    ci_upper_key = f"{metric_name}_ci_upper"
                    if ci_lower_key in metrics and ci_upper_key in metrics:
                        ci_lower = metrics[ci_lower_key]
                        ci_upper = metrics[ci_upper_key]
                        if ci_lower > ci_upper:
                            metrics_errors.append(f"{domain}/{task_id}: неправильный CI для {metric_name}")
            
            all_metrics_found.update(metrics.keys())
    
    # Вывести результаты
    print(f"\n✅ Найдено {len(all_metrics_found)} различных метрик")
    print(f"   Метрики: {', '.join(sorted(all_metrics_found))}")
    
    if metrics_errors:
        print(f"\n❌ Найдено {len(metrics_errors)} ошибок:")
        for error in metrics_errors[:10]:  # Показать первые 10
            print(f"   - {error}")
        if len(metrics_errors) > 10:
            print(f"   ... и еще {len(metrics_errors) - 10} ошибок")
        return False
    else:
        print("\n✅ Все метрики вычислены корректно!")
        return True


def test_statistical_functions():
    """Проверить статистические функции."""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА СТАТИСТИЧЕСКИХ ФУНКЦИЙ")
    print("=" * 80)
    
    errors = []
    
    # Тест Wilson CI
    try:
        ci_lower, ci_upper = wilson_confidence_interval(5, 10)
        if not (0 <= ci_lower <= ci_upper <= 1):
            errors.append("Wilson CI: неправильный диапазон")
        print("✅ Wilson CI работает корректно")
    except Exception as e:
        errors.append(f"Wilson CI: {e}")
    
    # Тест t-интервал
    try:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci_lower, ci_upper = t_confidence_interval(values)
        if ci_lower > ci_upper:
            errors.append("t-интервал: неправильный диапазон")
        print("✅ t-интервал работает корректно")
    except Exception as e:
        errors.append(f"t-интервал: {e}")
    
    # Тест сравнения непрерывных метрик
    try:
        group1 = [1.0, 2.0, 3.0]
        group2 = [4.0, 5.0, 6.0]
        p_value, sig = compare_continuous_metrics(group1, group2)
        if not (0 <= p_value <= 1):
            errors.append("compare_continuous_metrics: неправильный p-value")
        print("✅ compare_continuous_metrics работает корректно")
    except Exception as e:
        errors.append(f"compare_continuous_metrics: {e}")
    
    # Тест сравнения дискретных метрик
    try:
        p_value, sig = compare_conversion_metrics(5, 10, 3, 10)
        if not (0 <= p_value <= 1):
            errors.append("compare_conversion_metrics: неправильный p-value")
        print("✅ compare_conversion_metrics работает корректно")
    except Exception as e:
        errors.append(f"compare_conversion_metrics: {e}")
    
    if errors:
        print(f"\n❌ Найдено {len(errors)} ошибок в статистических функциях:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("\n✅ Все статистические функции работают корректно!")
        return True


if __name__ == "__main__":
    print("🧪 Запуск полной проверки всех метрик\n")
    
    # Проверить статистические функции
    stats_ok = test_statistical_functions()
    
    # Проверить метрики на данных
    metrics_ok = test_all_metrics()
    
    # Итоговый результат
    print("\n" + "=" * 80)
    if stats_ok and metrics_ok:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        sys.exit(0)
    else:
        print("❌ НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
        sys.exit(1)
