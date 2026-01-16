#!/bin/bash
# Скрипт для генерации графиков и компиляции PDF

set -e  # Остановить при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Генерация графиков и компиляция PDF"
echo "=========================================="
echo ""

# Шаг 1: Обработка результатов и генерация таблиц
echo "Шаг 1: Обработка результатов и генерация таблиц..."
python scripts/run_experiments_and_generate_tables.py \
    --skip-experiments \
    --compile-pdf

echo ""
echo "=========================================="
echo "Готово! PDF скомпилирован:"
echo "  docs/paper_template/template.pdf"
echo "=========================================="
