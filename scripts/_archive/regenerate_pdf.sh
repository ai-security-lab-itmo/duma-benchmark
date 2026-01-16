#!/bin/bash
# Простой скрипт для регенерации PDF

cd "$(dirname "$0")/.."

echo "Генерация таблиц и графиков..."
python3 scripts/run_experiments_and_generate_tables.py --skip-experiments --compile-pdf

echo ""
echo "Проверка результата..."
if [ -f "docs/paper_template/template.pdf" ]; then
    echo "✅ PDF создан: docs/paper_template/template.pdf"
    ls -lh docs/paper_template/template.pdf
else
    echo "❌ PDF не найден"
    echo "Попытка компиляции вручную..."
    cd docs/paper_template
    pdflatex -interaction=nonstopmode template.tex
    if [ -f "template.pdf" ]; then
        echo "✅ PDF создан после ручной компиляции"
    else
        echo "❌ Ошибка компиляции. Проверьте логи выше."
    fi
fi
