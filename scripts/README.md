# scripts/

Минимальный набор полезных команд для работы с результатами в `data/simulations`.

## Быстро пересобрать paper (таблицы + графики + PDF)

```bash
./.venv/bin/python scripts/run_experiments_and_generate_tables.py \
  --skip-experiments \
  --models gpt-4o gpt-4o-mini \
  --temperatures 0.0 0.5 1.0 \
  --domains mail_rag_phishing collab output_handling \
  --num-trials 10 \
  --results-dir data/simulations
```

Выходные артефакты:
- `docs/paper_template/template.tex`, `docs/paper_template/template.pdf`
- таблицы: `docs/paper_template/model_domain_table.tex`, `docs/paper_template/significance_table.tex`, `docs/paper_template/temperature_significance_table.tex`, `docs/paper_template/detailed_metrics_table.tex`
- графики: `docs/paper_template/figs/*.pdf`

## Запустить/дозапустить эксперименты (основной пайплайн)

```bash
./scripts/run_experiments.sh \
  --models gpt-4o gpt-4o-mini \
  --temperatures 0.0 0.5 1.0 \
  --domains mail_rag_phishing collab output_handling \
  --num-trials 10
```

## Дозапуск только недостающих результатов

Если часть файлов в `data/simulations` неполная/отсутствует:

```bash
./.venv/bin/python scripts/retry_failed_experiments.py \
  --models gpt-4o gpt-4o-mini \
  --temperatures 0.0 0.5 1.0 \
  --num-trials 10 \
  --max-concurrency 1 \
  --tau2-max-concurrency 3
```

## Максимальная загрузка (агрессивно)

Авто-откат при rate limit / timeout.

```bash
export OPENAI_API_KEY=...  # или OPENROUTER_API_KEY (если настроен OpenRouter routing)
./scripts/max_load_retry.sh
```

## Проверка корректности метрик

```bash
./.venv/bin/python scripts/test_all_metrics.py
```

## Benchmark suite (если нужен общий запуск)

```bash
./.venv/bin/python scripts/run_benchmark_suite.py --config scripts/benchmark_config_example.json
```

## Архив

Нерелевантные/вспомогательные скрипты и ноутбуки перенесены в `scripts/_archive/`.
