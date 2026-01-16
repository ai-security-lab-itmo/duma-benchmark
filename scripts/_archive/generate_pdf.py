#!/usr/bin/env python3
"""
Простой скрипт для генерации графиков и компиляции PDF.

Использование:
    python scripts/generate_pdf.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Генерация графиков и компиляция PDF."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print("=" * 80)
    print("Генерация графиков и компиляция PDF")
    print("=" * 80)
    print()
    
    # Запустить основной скрипт
    cmd = [
        sys.executable,
        str(script_dir / "run_experiments_and_generate_tables.py"),
        "--skip-experiments",
        "--compile-pdf"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=project_root, check=True)
        print()
        print("=" * 80)
        print("✅ Готово! PDF скомпилирован:")
        print(f"   {project_root / 'docs' / 'paper_template' / 'template.pdf'}")
        print("=" * 80)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при выполнении: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Прервано пользователем")
        return 1

if __name__ == "__main__":
    sys.exit(main())
