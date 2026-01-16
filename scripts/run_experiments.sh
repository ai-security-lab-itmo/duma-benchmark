#!/bin/bash
# Скрипт для запуска экспериментов с активацией venv

cd "$(dirname "$0")/.."

# Активировать venv если существует
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Запустить скрипт с переданными аргументами
python3 scripts/run_experiments_and_generate_tables.py "$@"
