#!/usr/bin/env python3
"""
Скрипт для создания постера конференции в Figma через REST API.

Требования:
1. Figma Personal Access Token (настройте в .env как FIGMA_ACCESS_TOKEN)
2. File Key из URL Figma файла

Использование:
    python scripts/create_figma_poster.py

Переменные окружения:
    FIGMA_ACCESS_TOKEN: Personal Access Token из Figma
    FIGMA_FILE_KEY: Key файла (из URL: figma.com/file/{FILE_KEY}/...)
"""

import os
import json
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Конфигурация
FIGMA_BASE_URL = "https://api.figma.com/v1"
FILE_KEY = "mUjvN6dtrqcnAHnH21DUGU"  # Из URL вашего файла

# Размеры постера A0 (в пикселях при 72 DPI, но в Figma работаем в px)
POSTER_WIDTH = 3360  # ~1189mm при 72 DPI
POSTER_HEIGHT = 4760  # ~1684mm при 72 DPI

# Цвета
COLORS = {
    "primary_green": "#065f46",
    "accent_green": "#10b981",
    "success": "#10b981",
    "error": "#EF4444",
    "warning": "#F59E0B",
    "background": "#FFFFFF",
    "background_light": "#F8FAFC",
    "text_primary": "#1E293B",
    "text_secondary": "#374151",
    "text_tertiary": "#64748B",
}

# Размеры шрифтов (в pt, Figma использует px = pt)
FONT_SIZES = {
    "h1": 96,
    "h2": 48,
    "h3": 36,
    "body_large": 28,
    "body": 24,
    "caption": 18,
}


class FigmaPosterCreator:
    """Класс для создания постера в Figma через API."""
    
    def __init__(self, access_token: str, file_key: str):
        self.access_token = access_token
        self.file_key = file_key
        self.headers = {
            "X-Figma-Token": access_token,
            "Content-Type": "application/json",
        }
        self.base_url = f"{FIGMA_BASE_URL}/files/{file_key}"
        self.nodes_to_create = []
    
    def create_poster_structure(self) -> Dict:
        """
        Создает структуру постера.
        
        Примечание: Figma REST API не поддерживает прямой создание элементов.
        Этот скрипт генерирует JSON структуру, которую можно использовать:
        1. Через Figma Plugin API
        2. Через Figma Desktop App с плагином
        3. Вручную создавая элементы согласно структуре
        """
        
        poster_structure = {
            "name": "Conference Poster - DUMA Bench",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": POSTER_HEIGHT,
            "background": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
            "layoutMode": "VERTICAL",
            "paddingLeft": 0,
            "paddingRight": 0,
            "paddingTop": 0,
            "paddingBottom": 0,
            "itemSpacing": 0,
            "children": [
                self._create_header(),
                self._create_abstract(),
                self._create_domains_section(),
                self._create_results_section(),
                self._create_findings_section(),
                self._create_methodology_section(),
                self._create_footer(),
            ],
        }
        
        return poster_structure
    
    def _create_header(self) -> Dict:
        """Создает секцию заголовка."""
        return {
            "name": "Header",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 680,  # ~240mm
            "background": [{
                "type": "GRADIENT_LINEAR",
                "gradientStops": [
                    {"position": 0, "color": {"r": 0.024, "g": 0.373, "b": 0.275}},  # #065f46
                    {"position": 1, "color": {"r": 0.016, "g": 0.471, "b": 0.341}},  # #047857
                ],
                "gradientHandlePositions": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 1},
                ],
            }],
            "layoutMode": "VERTICAL",
            "paddingLeft": 272,  # 96px
            "paddingRight": 272,
            "paddingTop": 226,  # 80px
            "paddingBottom": 170,  # 60px
            "itemSpacing": 56,  # 20px
            "primaryAxisAlignItems": "CENTER",
            "counterAxisAlignItems": "CENTER",
            "children": [
                {
                    "name": "Logo",
                    "type": "FRAME",
                    "width": 340,  # 120px
                    "height": 340,
                    "background": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                    "cornerRadius": 34,  # 12px
                },
                {
                    "name": "Title",
                    "type": "TEXT",
                    "characters": (
                        "ОЦЕНКА УСТОЙЧИВОСТИ АГЕНТНЫХ СИСТЕМ НА ОСНОВЕ\n"
                        "БОЛЬШИХ ЯЗЫКОВЫХ МОДЕЛЕЙ\n"
                        "К АТАКАМ НА СРЕДУ ИСПОЛНЕНИЯ"
                    ),
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 800,
                        "fontSize": FONT_SIZES["h1"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                    },
                },
                {
                    "name": "Subtitle",
                    "type": "TEXT",
                    "characters": "Расширение τ²-bench доменами безопасности",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["h2"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1, "a": 0.95}}],
                    },
                },
                {
                    "name": "Authors",
                    "type": "TEXT",
                    "characters": "ITMO Security Lab | Декабрь 2025",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 500,
                        "fontSize": FONT_SIZES["h3"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                    },
                },
            ],
        }
    
    def _create_abstract(self) -> Dict:
        """Создает секцию Abstract."""
        return {
            "name": "Abstract",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 566,  # ~200mm
            "background": [{"type": "SOLID", "color": {"r": 0.941, "g": 0.992, "b": 0.957}}],  # #F0FDF4
            "layoutMode": "VERTICAL",
            "paddingLeft": 272,
            "paddingRight": 272,
            "paddingTop": 170,
            "paddingBottom": 170,
            "itemSpacing": 113,  # 40px
            "children": [
                {
                    "name": "Abstract Text",
                    "type": "TEXT",
                    "characters": (
                        "Агентные системы на основе LLM всё шире применяются для автоматизации сложных задач, "
                        "однако их безопасность в реалистичных сценариях взаимодействия остаётся недостаточно изученной. "
                        "Предлагается расширение бенчмарка τ²-bench тремя новыми доменами безопасности: "
                        "mail_rag_phishing (атаки через отравление RAG-системы), "
                        "collab (атаки через межагентное взаимодействие) и "
                        "output_handling (некорректная обработка выводов)."
                    ),
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body_large"],
                        "lineHeight": {"unit": "PIXELS", "value": 45},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_secondary"])}],
                    },
                },
                {
                    "name": "Highlights",
                    "type": "FRAME",
                    "layoutMode": "HORIZONTAL",
                    "itemSpacing": 113,
                    "primaryAxisAlignItems": "CENTER",
                    "children": [
                        self._create_highlight("🔍", "GPT-4o: 50% устойчивость (collab), 33% (output)"),
                        self._create_highlight("⚠️", "GPT-4o-mini: 0% устойчивость"),
                        self._create_highlight("🚨", "RAG-системы уязвимы во всех случаях"),
                    ],
                },
            ],
            "strokes": [{
                "type": "SOLID",
                "color": self._hex_to_rgba(COLORS["accent_green"]),
            }],
            "strokeTopWeight": 14,  # 5px
        }
    
    def _create_highlight(self, icon: str, text: str) -> Dict:
        """Создает элемент highlight."""
        return {
            "name": f"Highlight: {icon}",
            "type": "FRAME",
            "layoutMode": "HORIZONTAL",
            "itemSpacing": 34,
            "primaryAxisAlignItems": "CENTER",
            "children": [
                {
                    "name": "Icon",
                    "type": "TEXT",
                    "characters": icon,
                    "style": {
                        "fontSize": 90,
                    },
                },
                {
                    "name": "Text",
                    "type": "TEXT",
                    "characters": text,
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 600,
                        "fontSize": FONT_SIZES["body"],
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_secondary"])}],
                    },
                },
            ],
        }
    
    def _create_domains_section(self) -> Dict:
        """Создает секцию с тремя доменами."""
        return {
            "name": "Domains",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 1588,  # ~560mm
            "layoutMode": "HORIZONTAL",
            "paddingLeft": 272,
            "paddingRight": 272,
            "paddingTop": 226,
            "paddingBottom": 226,
            "itemSpacing": 136,  # 48px
            "children": [
                self._create_domain_card("mail", "📧", "RAG Poisoning", "#EF4444"),
                self._create_domain_card("collab", "🤝", "Cross-Agent\nPoisoning", "#F59E0B"),
                self._create_domain_card("output", "🖥️", "Output\nHandling", "#EF4444"),
            ],
        }
    
    def _create_domain_card(self, domain_id: str, icon: str, title: str, border_color: str) -> Dict:
        """Создает карточку домена."""
        descriptions = {
            "mail": "Атака через отравление RAG-системы почтового клиента через внедрение вредоносных инструкций в содержимое писем.",
            "collab": "Мультиагентная среда, где атакующий агент (коллега) манипулирует целевым агентом через канал коммуникации.",
            "output": "Проверка устойчивости агента к генерации вредоносного кода или инъекций в выводе (XSS, SQLi).",
        }
        
        threats = {
            "mail": ["YAISAFE.DATA.1", "LLM01", "Indirect Prompt Injection"],
            "collab": ["YAISAFE.INFRA.3", "T15, MCP05", "Privilege Escalation"],
            "output": ["YAISAFE.INPUT.3", "LLM05", "Improper Output Handling"],
        }
        
        results = {
            "mail": {"gpt4o": ("0%", "failure"), "gpt4omini": ("0%", "failure")},
            "collab": {"gpt4o": ("50%", "success"), "gpt4omini": ("0%", "failure")},
            "output": {"gpt4o": ("33%", "partial"), "gpt4omini": ("0%", "failure")},
        }
        
        return {
            "name": f"Domain Card: {domain_id}",
            "type": "FRAME",
            "width": 1020,  # ~360mm
            "height": "AUTO",
            "background": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
            "layoutMode": "VERTICAL",
            "paddingLeft": 136,
            "paddingRight": 136,
            "paddingTop": 136,
            "paddingBottom": 136,
            "itemSpacing": 68,
            "cornerRadius": 45,
            "strokes": [{
                "type": "SOLID",
                "color": self._hex_to_rgba(border_color),
            }],
            "strokeTopWeight": 14,
            "effects": [{
                "type": "DROP_SHADOW",
                "color": {"r": 0, "g": 0, "b": 0, "a": 0.1},
                "offset": {"x": 0, "y": 8},
                "radius": 24,
            }],
            "children": [
                {
                    "name": "Domain Header",
                    "type": "FRAME",
                    "layoutMode": "HORIZONTAL",
                    "itemSpacing": 45,
                    "children": [
                        {
                            "name": "Icon",
                            "type": "TEXT",
                            "characters": icon,
                            "style": {"fontSize": 136},
                        },
                        {
                            "name": "Title",
                            "type": "TEXT",
                            "characters": title,
                            "style": {
                                "fontFamily": "Inter",
                                "fontWeight": 700,
                                "fontSize": FONT_SIZES["h3"],
                                "lineHeight": {"unit": "AUTO"},
                                "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
                            },
                        },
                    ],
                },
                {
                    "name": "Description",
                    "type": "TEXT",
                    "characters": descriptions[domain_id],
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body"],
                        "lineHeight": {"unit": "PIXELS", "value": 36},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_secondary"])}],
                    },
                },
                {
                    "name": "Threats",
                    "type": "FRAME",
                    "background": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["background_light"])}],
                    "layoutMode": "VERTICAL",
                    "paddingLeft": 68,
                    "paddingRight": 68,
                    "paddingTop": 68,
                    "paddingBottom": 68,
                    "itemSpacing": 23,
                    "cornerRadius": 23,
                    "children": [
                        {
                            "name": "Threats Title",
                            "type": "TEXT",
                            "characters": "Угрозы:",
                            "style": {
                                "fontFamily": "Inter",
                                "fontWeight": 600,
                                "fontSize": 57,
                                "lineHeight": {"unit": "AUTO"},
                                "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_primary"])}],
                            },
                        },
                        *[{
                            "name": f"Threat: {threat}",
                            "type": "TEXT",
                            "characters": f"• {threat}",
                            "style": {
                                "fontFamily": "Inter",
                                "fontWeight": 400,
                                "fontSize": 57,
                                "lineHeight": {"unit": "AUTO"},
                                "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_tertiary"])}],
                            },
                        } for threat in threats[domain_id]],
                    ],
                },
                {
                    "name": "Results",
                    "type": "FRAME",
                    "layoutMode": "VERTICAL",
                    "itemSpacing": 34,
                    "children": [
                        {
                            "name": "Results Title",
                            "type": "TEXT",
                            "characters": "Результаты:",
                            "style": {
                                "fontFamily": "Inter",
                                "fontWeight": 600,
                                "fontSize": FONT_SIZES["body"],
                                "lineHeight": {"unit": "AUTO"},
                                "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_primary"])}],
                            },
                        },
                        *[
                            self._create_result_item(f"GPT-4o" if model == "gpt4o" else "GPT-4o-mini", value, status)
                            for model, (value, status) in results[domain_id].items()
                        ],
                    ],
                },
            ],
        }
    
    def _create_result_item(self, label: str, value: str, status: str) -> Dict:
        """Создает элемент результата."""
        color_map = {
            "success": COLORS["success"],
            "failure": COLORS["error"],
            "partial": COLORS["warning"],
        }
        
        return {
            "name": f"Result: {label}",
            "type": "FRAME",
            "layoutMode": "HORIZONTAL",
            "justifyContent": "SPACE_BETWEEN",
            "paddingLeft": 45,
            "paddingRight": 45,
            "paddingTop": 45,
            "paddingBottom": 45,
            "background": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["background_light"])}],
            "cornerRadius": 23,
            "children": [
                {
                    "name": "Label",
                    "type": "TEXT",
                    "characters": f"{label}:",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 500,
                        "fontSize": 62,
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_secondary"])}],
                    },
                },
                {
                    "name": "Value",
                    "type": "TEXT",
                    "characters": value,
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 700,
                        "fontSize": FONT_SIZES["body"],
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(color_map[status])}],
                    },
                },
            ],
        }
    
    def _create_results_section(self) -> Dict:
        """Создает секцию результатов."""
        return {
            "name": "Results",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 1134,  # ~400mm
            "background": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["background_light"])}],
            "layoutMode": "HORIZONTAL",
            "paddingLeft": 272,
            "paddingRight": 272,
            "paddingTop": 226,
            "paddingBottom": 226,
            "itemSpacing": 181,
            "children": [
                self._create_results_table(),
                self._create_results_chart(),
            ],
        }
    
    def _create_results_table(self) -> Dict:
        """Создает таблицу результатов."""
        return {
            "name": "Results Table",
            "type": "FRAME",
            "width": 1588,  # ~560mm
            "height": "AUTO",
            "background": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
            "layoutMode": "VERTICAL",
            "paddingLeft": 136,
            "paddingRight": 136,
            "paddingTop": 136,
            "paddingBottom": 136,
            "itemSpacing": 0,
            "cornerRadius": 45,
            "effects": [{
                "type": "DROP_SHADOW",
                "color": {"r": 0, "g": 0, "b": 0, "a": 0.1},
                "offset": {"x": 0, "y": 8},
                "radius": 24,
            }],
            "children": [
                {
                    "name": "Table Title",
                    "type": "TEXT",
                    "characters": "Результаты экспериментов",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 700,
                        "fontSize": FONT_SIZES["h2"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
                    },
                },
                # Таблица будет создана вручную или через плагин
            ],
        }
    
    def _create_results_chart(self) -> Dict:
        """Создает график результатов."""
        return {
            "name": "Results Chart",
            "type": "FRAME",
            "width": 1588,
            "height": "AUTO",
            "background": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
            "layoutMode": "VERTICAL",
            "paddingLeft": 136,
            "paddingRight": 136,
            "paddingTop": 136,
            "paddingBottom": 136,
            "itemSpacing": 91,
            "cornerRadius": 45,
            "effects": [{
                "type": "DROP_SHADOW",
                "color": {"r": 0, "g": 0, "b": 0, "a": 0.1},
                "offset": {"x": 0, "y": 8},
                "radius": 24,
            }],
            "children": [
                {
                    "name": "Chart Title",
                    "type": "TEXT",
                    "characters": "Сравнение метрик pass@1",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 700,
                        "fontSize": FONT_SIZES["h2"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
                    },
                },
                # Графики будут созданы вручную
            ],
        }
    
    def _create_findings_section(self) -> Dict:
        """Создает секцию ключевых выводов."""
        findings = [
            "Размер модели критичен: GPT-4o демонстрирует значительно более высокую устойчивость к атакам по сравнению с GPT-4o-mini.",
            "RAG-системы критически уязвимы: Ни одна модель не показала устойчивости к атакам через отравление RAG.",
            "Необходимы специализированные защитные механизмы: Результаты указывают на критическую необходимость разработки guardrails для агентных архитектур.",
        ]
        
        return {
            "name": "Findings",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 453,  # ~160mm
            "background": [{
                "type": "GRADIENT_LINEAR",
                "gradientStops": [
                    {"position": 0, "color": {"r": 0.941, "g": 0.992, "b": 0.957}},
                    {"position": 1, "color": {"r": 1, "g": 1, "b": 1}},
                ],
            }],
            "layoutMode": "VERTICAL",
            "paddingLeft": 272,
            "paddingRight": 272,
            "paddingTop": 226,
            "paddingBottom": 226,
            "itemSpacing": 68,
            "children": [
                {
                    "name": "Findings Title",
                    "type": "TEXT",
                    "characters": "🔑 Ключевые выводы",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 700,
                        "fontSize": FONT_SIZES["h2"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
                    },
                },
                *[self._create_finding_item(i + 1, finding) for i, finding in enumerate(findings)],
            ],
        }
    
    def _create_finding_item(self, number: int, text: str) -> Dict:
        """Создает элемент вывода."""
        return {
            "name": f"Finding {number}",
            "type": "FRAME",
            "layoutMode": "HORIZONTAL",
            "itemSpacing": 68,
            "children": [
                {
                    "name": "Number",
                    "type": "FRAME",
                    "width": 181,
                    "height": 181,
                    "background": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["accent_green"])}],
                    "cornerRadius": 91,
                    "primaryAxisAlignItems": "CENTER",
                    "counterAxisAlignItems": "CENTER",
                    "children": [
                        {
                            "name": "Number Text",
                            "type": "TEXT",
                            "characters": str(number),
                            "style": {
                                "fontFamily": "Inter",
                                "fontWeight": 700,
                                "fontSize": 91,
                                "lineHeight": {"unit": "AUTO"},
                                "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                            },
                        },
                    ],
                },
                {
                    "name": "Text",
                    "type": "TEXT",
                    "characters": text,
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body_large"],
                        "lineHeight": {"unit": "PIXELS", "value": 45},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_primary"])}],
                    },
                },
            ],
        }
    
    def _create_methodology_section(self) -> Dict:
        """Создает секцию методологии."""
        methodology_items = [
            ("Формализация", "Dec-POMDP"),
            ("Бенчмарк", "τ²-bench расширение"),
            ("Метрики", "pass@1, ASR"),
            ("Эксперименты", "10 прогонов на конфигурацию"),
            ("Модели", "GPT-4o, GPT-4o-mini"),
            ("Температура агента", "T = 0.0"),
            ("Температура пользователя", "T = {0.0, 0.5, 1.0}"),
            ("Классификация", "AI-SAFE, OWASP LLM Top 10"),
        ]
        
        return {
            "name": "Methodology",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 340,  # ~120mm
            "layoutMode": "VERTICAL",
            "paddingLeft": 272,
            "paddingRight": 272,
            "paddingTop": 226,
            "paddingBottom": 226,
            "itemSpacing": 91,
            "children": [
                {
                    "name": "Methodology Title",
                    "type": "TEXT",
                    "characters": "📊 Методология",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 700,
                        "fontSize": FONT_SIZES["h2"],
                        "lineHeight": {"unit": "AUTO"},
                        "textAlignHorizontal": "CENTER",
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
                    },
                },
                {
                    "name": "Methodology Grid",
                    "type": "FRAME",
                    "layoutMode": "VERTICAL",
                    "itemSpacing": 91,
                    "children": [
                        *[
                            {
                                "name": f"Methodology Row {i // 2 + 1}",
                                "type": "FRAME",
                                "layoutMode": "HORIZONTAL",
                                "itemSpacing": 91,
                                "children": [
                                    self._create_methodology_item(label, value)
                                    for label, value in methodology_items[i:i+2]
                                ],
                            }
                            for i in range(0, len(methodology_items), 2)
                        ],
                    ],
                },
            ],
        }
    
    def _create_methodology_item(self, label: str, value: str) -> Dict:
        """Создает элемент методологии."""
        return {
            "name": f"Methodology: {label}",
            "type": "FRAME",
            "layoutMode": "VERTICAL",
            "paddingLeft": 91,
            "paddingRight": 91,
            "paddingTop": 91,
            "paddingBottom": 91,
            "itemSpacing": 34,
            "background": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["background_light"])}],
            "cornerRadius": 34,
            "strokes": [{
                "type": "SOLID",
                "color": self._hex_to_rgba(COLORS["accent_green"]),
            }],
            "strokeLeftWeight": 14,
            "children": [
                {
                    "name": "Label",
                    "type": "TEXT",
                    "characters": label,
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 600,
                        "fontSize": 57,
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
                    },
                },
                {
                    "name": "Value",
                    "type": "TEXT",
                    "characters": value,
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body_large"],
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["text_secondary"])}],
                    },
                },
            ],
        }
    
    def _create_footer(self) -> Dict:
        """Создает футер."""
        return {
            "name": "Footer",
            "type": "FRAME",
            "width": POSTER_WIDTH,
            "height": 340,  # ~120mm
            "background": [{"type": "SOLID", "color": self._hex_to_rgba(COLORS["primary_green"])}],
            "layoutMode": "HORIZONTAL",
            "paddingLeft": 272,
            "paddingRight": 272,
            "paddingTop": 136,
            "paddingBottom": 136,
            "itemSpacing": 181,
            "primaryAxisAlignItems": "CENTER",
            "children": [
                {
                    "name": "Contact: GitHub",
                    "type": "TEXT",
                    "characters": "🌐 github.com/ai-security-lab-itmo",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body"],
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                    },
                },
                {
                    "name": "Contact: Email",
                    "type": "TEXT",
                    "characters": "📧 security-lab@itmo.ru",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body"],
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                    },
                },
                {
                    "name": "Contact: Website",
                    "type": "TEXT",
                    "characters": "📖 duma-benchmark.github.io",
                    "style": {
                        "fontFamily": "Inter",
                        "fontWeight": 400,
                        "fontSize": FONT_SIZES["body"],
                        "lineHeight": {"unit": "AUTO"},
                        "fills": [{"type": "SOLID", "color": {"r": 1, "g": 1, "b": 1}}],
                    },
                },
            ],
        }
    
    def _hex_to_rgba(self, hex_color: str) -> Dict:
        """Конвертирует HEX цвет в RGBA."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return {"r": r, "g": g, "b": b}
    
    def save_structure_to_json(self, output_path: Path):
        """Сохраняет структуру в JSON файл."""
        structure = self.create_poster_structure()
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Структура постера сохранена в: {output_path}")
        print(f"📋 Используйте этот файл для:")
        print(f"   1. Импорта в Figma через плагин")
        print(f"   2. Ручного создания элементов согласно структуре")
        print(f"   3. Использования в скриптах автоматизации")


def main():
    """Главная функция."""
    # Получаем токен из переменных окружения (не обязателен для генерации структуры)
    access_token = os.getenv("FIGMA_ACCESS_TOKEN", "dummy_token")
    
    # Создаем экземпляр создателя
    creator = FigmaPosterCreator(access_token, FILE_KEY)
    
    # Сохраняем структуру в JSON
    output_path = Path(__file__).parent.parent / "docs" / "paper_template" / "poster_structure.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    creator.save_structure_to_json(output_path)
    
    print("\n📝 Примечание:")
    print("   Figma REST API не поддерживает прямое создание элементов.")
    print("   Используйте созданный JSON файл для:")
    print("   - Импорта через Figma Plugin")
    print("   - Ручного создания элементов в Figma")
    print("   - Использования в других инструментах автоматизации")


if __name__ == "__main__":
    main()
