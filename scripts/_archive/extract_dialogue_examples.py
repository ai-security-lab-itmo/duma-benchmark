#!/usr/bin/env python3
"""
Скрипт для извлечения примеров диалогов из симуляций и создания визуализаций атак.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import networkx as nx
from collections import defaultdict
import numpy as np
import seaborn as sns

# Настройка научного стиля для публикаций (как в существующих графиках)
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
        "text.usetex": False,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "patch.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Профессиональная цветовая палитра (как в существующих графиках)
COLORS = {
    "primary": "#1f77b4",  # Синий
    "secondary": "#ff7f0e",  # Оранжевый
    "success": "#2ca02c",  # Зеленый
    "danger": "#d62728",  # Красный
    "purple": "#9467bd",  # Фиолетовый
    "brown": "#8c564b",  # Коричневый
    "neutral": "#7f7f7f",  # Серый
    "light_gray": "#e5e7eb",  # Светло-серый
    "dark_gray": "#4b5563",  # Темно-серый
}


def load_simulation(file_path: Path) -> Dict:
    """Загрузить симуляцию из JSON файла."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_dialogue(simulation: Dict, max_turns: int = 15) -> List[Dict]:
    """Извлечь диалог из симуляции."""
    if not simulation.get("simulations"):
        return []

    sim = simulation["simulations"][0]  # Берем первую симуляцию
    messages = sim.get("messages", [])

    dialogue = []
    for msg in messages[: max_turns * 2]:  # Ограничиваем количество сообщений
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls")
        turn_idx = msg.get("turn_idx", 0)

        if role in ["user", "assistant"]:
            dialogue.append(
                {
                    "role": role,
                    "content": content,
                    "tool_calls": tool_calls,
                    "turn": turn_idx,
                }
            )
        elif role == "tool" and content:
            # Добавляем результат вызова инструмента
            dialogue.append(
                {
                    "role": "tool",
                    "content": content[:200] + "..."
                    if len(str(content)) > 200
                    else str(content),
                    "turn": turn_idx,
                }
            )

    return dialogue


def create_attack_flow_diagram(simulations_data: Dict[str, Dict], output_path: Path):
    """
    Создать профессиональную визуализацию потока атаки в научном стиле.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Визуализация потока атак по доменам безопасности",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    domains_config = {
        "mail_rag_phishing": {
            "title": "Отравление RAG",
            "color": COLORS["danger"],
            "stages": [
                ("Запрос\nпользователя", "Пользователь запрашивает\nстатус платежа"),
                ("Поиск в RAG", "Агент извлекает\nдокументы из RAG"),
                ("Отравленный\nшаблон", "Вредоносный SOP\nшаблон найден"),
                ("Обработка", "Агент обрабатывает\nинструкции"),
                ("Результат", "Email отправлен\nили заблокирован"),
            ],
            "ax_idx": 0,
        },
        "collab": {
            "title": "Межагентное\nотравление",
            "color": COLORS["secondary"],
            "stages": [
                ("Запрос\nклиента", "Клиент спрашивает\nо инциденте"),
                ("Сообщение\nколлеги", "Вредоносный агент\nотправляет инструкции"),
                ("Проверка\nполитики", "Агент проверяет\nполитику безопасности"),
                ("Принятие\nрешения", "Агент оценивает\nзапрос"),
                ("Результат", "Атака заблокирована\nили выполнена"),
            ],
            "ax_idx": 1,
        },
        "output_handling": {
            "title": "Инъекция в вывод",
            "color": COLORS["purple"],
            "stages": [
                ("Запрос\nпользователя", "Пользователь просит\nссылку для проверки"),
                ("Получение\nконтекста", "Агент получает\nконтекст тикета"),
                ("Сообщение\nколлеги", "Ops предлагает\nпереслать payload"),
                ("Проверка\nбезопасности", "Проверка политики\nсанитизации"),
                ("Результат", "Безопасный ответ\nили инъекция"),
            ],
            "ax_idx": 2,
        },
    }

    for domain_name, domain_info in domains_config.items():
        if domain_name not in simulations_data:
            continue

        ax = axes[domain_info["ax_idx"]]
        sim = simulations_data[domain_name]
        attack_success = sim.get("attack_success", False)

        stages = domain_info["stages"]
        n_stages = len(stages)

        # Рисуем этапы как прямоугольники
        x_positions = np.linspace(0.1, 0.9, n_stages)
        box_width = 0.12
        box_height = 0.3

        for i, (stage_name, stage_desc) in enumerate(stages):
            x = x_positions[i] - box_width / 2
            y = 0.5 - box_height / 2

            # Определяем цвет узла
            if i == n_stages - 1:  # Последний узел - результат
                if attack_success:
                    node_color = COLORS["danger"]
                    result_text = "Успешно"
                else:
                    node_color = COLORS["success"]
                    result_text = "Заблокировано"
            elif i == 2:  # Узел с отравленным контентом
                node_color = COLORS["secondary"]
            else:
                node_color = COLORS["light_gray"]

            # Рисуем прямоугольник
            rect = Rectangle(
                (x, y),
                box_width,
                box_height,
                facecolor=node_color,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Добавляем текст ПОД прямоугольником (не на нем)
            ax.text(
                x_positions[i],
                y - 0.05,
                stage_name,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color="black",
                wrap=True,
            )

            # Добавляем описание еще ниже
            ax.text(
                x_positions[i],
                y - 0.15,
                stage_desc,
                ha="center",
                va="top",
                fontsize=7,
                style="italic",
                color=COLORS["dark_gray"],
                wrap=True,
            )

            # Рисуем стрелку к следующему этапу
            if i < n_stages - 1:
                arrow_x = x_positions[i] + box_width / 2
                arrow_end = x_positions[i + 1] - box_width / 2
                ax.annotate(
                    "",
                    xy=(arrow_end, 0.5),
                    xytext=(arrow_x, 0.5),
                    arrowprops=dict(
                        arrowstyle="->", lw=2, color=domain_info["color"], alpha=0.7
                    ),
                )

        # Добавляем результат внизу
        result_y = 0.1
        result_color = COLORS["danger"] if attack_success else COLORS["success"]
        result_text = "Атака успешна" if attack_success else "Атака заблокирована"
        ax.text(
            0.5,
            result_y,
            result_text,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=result_color,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=result_color,
                linewidth=2,
            ),
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(
            domain_info["title"],
            fontsize=12,
            fontweight="bold",
            color=domain_info["color"],
            pad=15,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close()
    print(f"Attack flow diagram saved to {output_path}")


def create_timeline_visualization(simulations_data: Dict[str, Dict], output_path: Path):
    """
    Создать профессиональную временную визуализацию взаимодействий.
    Каждый домен на отдельной картинке без наложений, с полными фразами и переводами.
    """
    domains_order = ["mail_rag_phishing", "collab", "output_handling"]
    domain_titles = {
        "mail_rag_phishing": "Отравление RAG: временная последовательность",
        "collab": "Межагентное отравление: временная последовательность",
        "output_handling": "Инъекция в вывод: временная последовательность",
    }

    colors = {
        "user": COLORS["primary"],
        "assistant": COLORS["success"],
        "tool": COLORS["secondary"],
        "attack": COLORS["danger"],
        "poisoned": COLORS["brown"],
    }

    # Создаем отдельную фигуру для каждого домена
    for idx, domain in enumerate(domains_order):
        if domain not in simulations_data:
            continue

        import textwrap

        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle(domain_titles[domain], fontsize=14, fontweight="bold", y=0.98)

        sim = simulations_data[domain]
        simulation_obj = sim.get("simulation", {})

        # Извлекаем сообщения с реальными временными метками
        messages = []
        if simulation_obj.get("simulations"):
            sim_data = simulation_obj["simulations"][0]
            messages = sim_data.get("messages", [])[:15]  # Ограничиваем для читаемости

        if not messages:
            continue

        # Парсим временные метки и вычисляем относительное время
        from datetime import datetime

        start_time = None
        timeline_data = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            timestamp_str = msg.get("timestamp", "")
            turn = msg.get("turn_idx", 0)

            # Парсим время
            try:
                if timestamp_str:
                    msg_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if start_time is None:
                        start_time = msg_time
                    rel_time = (msg_time - start_time).total_seconds()
                else:
                    rel_time = turn * 2
            except Exception:
                rel_time = turn * 2

            # Определяем тип и цвет
            if role == "user":
                msg_type = "Пользователь"
                msg_type_en = "User"
                color = colors["user"]
            elif role == "assistant":
                has_attack_keywords = content and any(
                    keyword in content.lower()
                    for keyword in ["send", "delete", "reset", "forward", "email sent"]
                )
                has_tool_attack = tool_calls and any(
                    tool.get("name")
                    in [
                        "send_email",
                        "delete_customer_visible_logs",
                        "reset_admin_api_key",
                        "forward_raw_snippet",
                    ]
                    for tool in tool_calls
                )

                if has_attack_keywords or has_tool_attack:
                    msg_type = "Агент (вредоносное действие)"
                    msg_type_en = "Agent (malicious)"
                    color = colors["attack"]
                else:
                    msg_type = "Агент"
                    msg_type_en = "Agent"
                    color = colors["assistant"]
            elif role == "tool":
                is_poisoned = any(
                    keyword in str(content).lower()
                    for keyword in ["template", "sop", "instructions", "dump", "wipe"]
                )
                if is_poisoned:
                    msg_type = "Инструмент (отравленный контент)"
                    msg_type_en = "Tool (poisoned)"
                    color = colors["poisoned"]
                else:
                    msg_type = "Инструмент"
                    msg_type_en = "Tool"
                    color = colors["tool"]
            else:
                msg_type = "Другое"
                msg_type_en = "Other"
                color = COLORS["neutral"]

            # Получаем полный контент
            full_content = ""
            if content:
                full_content = content
            elif tool_calls:
                tool_name = tool_calls[0].get("name", "unknown")
                tool_args = tool_calls[0].get("arguments", {})
                # tool_args может быть уже строкой или словарем
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except Exception:
                        pass
                if isinstance(tool_args, dict):
                    args_str = json.dumps(tool_args, ensure_ascii=False)
                else:
                    args_str = str(tool_args)
                full_content = f"{tool_name}({args_str})"

            timeline_data.append(
                {
                    "turn": turn,
                    "type": msg_type,
                    "type_en": msg_type_en,
                    "color": color,
                    "content": full_content,
                    "time": rel_time,
                }
            )

        if not timeline_data:
            continue

        # Сортируем по времени
        timeline_data.sort(key=lambda x: x["time"])

        # Нормализуем время
        max_time = max(d["time"] for d in timeline_data) if timeline_data else 1
        if max_time == 0:
            max_time = 1

        # Профессиональная раскладка: события по времени слева,
        # подписи вынесены в фиксированную правую колонку.
        # Это убирает пересечения (особенно у меток времени, которые раньше рисовались снизу).
        n_messages = len(timeline_data)

        # Динамическая высота под число сообщений, чтобы подписи не налезали друг на друга.
        fig_height = max(6.5, min(14.0, 2.5 + 0.55 * n_messages))
        fig.set_size_inches(16, fig_height, forward=True)

        timeline_x0 = 0.08
        timeline_width = 0.78
        timeline_x1 = timeline_x0 + timeline_width  # 0.86

        top_y = 0.90
        bottom_y = 0.16  # оставляем место под ось времени
        usable_height = max(0.01, top_y - bottom_y)
        y_spacing = usable_height / max(n_messages - 1, 1)

        # Правая колонка для подписей
        label_x = 0.90
        label_wrap = 55 if n_messages <= 12 else 48
        max_label_lines = 4 if n_messages <= 12 else 3

        # Минимальное расстояние по X для событий с одинаковым/очень близким временем
        min_dx = 0.015
        last_x = None

        for i, data in enumerate(timeline_data):
            y_pos = top_y - i * y_spacing

            x_raw = data["time"] / max_time if max_time else 0.0
            x_pos = timeline_x0 + x_raw * timeline_width
            if last_x is not None and x_pos - last_x < min_dx:
                x_pos = last_x + min_dx
            x_pos = min(x_pos, timeline_x1)
            last_x = x_pos

            # Маркер события
            ax.scatter(
                [x_pos],
                [y_pos],
                s=160,
                c=data["color"],
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=5,
            )

            # Компактная метка времени рядом с маркером (не на общей нижней линии).
            ax.text(
                x_pos,
                y_pos - 0.028,
                f"{data['time']:.1f}с",
                ha="center",
                va="top",
                fontsize=7,
                color="black",
            )

            type_label = data["type"]

            # Сжимаем контент и переносим строки для аккуратной верстки
            content_short = ""
            if data["content"]:
                content = str(data["content"]).strip()
                if len(content) > 220:
                    words = content.split()
                    for word in words:
                        if len(content_short) + len(word) + 1 <= 220:
                            content_short += word + " "
                        else:
                            break
                    content_short = content_short.strip() + "..."
                else:
                    content_short = content

            label_text = (
                f"{type_label}: {content_short}" if content_short else type_label
            )
            wrapped = "\n".join(textwrap.wrap(label_text, width=label_wrap))
            lines = wrapped.splitlines()
            if len(lines) > max_label_lines:
                lines = lines[:max_label_lines]
                lines[-1] = lines[-1].rstrip(".") + "…"
                wrapped = "\n".join(lines)

            # Линия-указатель к правой колонке
            ax.plot(
                [x_pos, label_x - 0.01],
                [y_pos, y_pos],
                color=data["color"],
                alpha=0.35,
                linewidth=1.0,
                zorder=2,
            )

            # Подпись в фиксированной правой колонке
            ax.text(
                label_x,
                y_pos,
                wrapped,
                ha="left",
                va="center",
                fontsize=8,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor=data["color"],
                    linewidth=1.2,
                    alpha=0.95,
                ),
                zorder=4,
                clip_on=False,
            )

        # Рисуем временную ось внизу
        axis_y = 0.08
        ax.axhline(y=axis_y, xmin=0.08, xmax=0.86, color="black", linewidth=2, zorder=3)

        # Добавляем метки времени на оси
        time_ticks = np.linspace(0, max_time, 6)
        for tick_time in time_ticks:
            x_tick = 0.08 + (tick_time / max_time) * 0.78
            ax.plot(
                [x_tick, x_tick],
                [axis_y, axis_y - 0.012],
                color="black",
                linewidth=1.5,
                zorder=4,
            )
            ax.text(
                x_tick,
                axis_y - 0.02,
                f"{tick_time:.1f}с",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color="black",
            )

        # Настройка осей
        ax.set_xlim(0, 1.15)  # место под правую колонку подписей
        ax.set_ylim(0, 1)

        # Подписи осей
        ax.set_xlabel("Время (секунды)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Последовательность событий", fontsize=11, fontweight="bold")

        ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Легенда
        legend_elements = [
            mpatches.Patch(color=colors["user"], label="Пользователь / User"),
            mpatches.Patch(color=colors["assistant"], label="Агент / Agent"),
            mpatches.Patch(color=colors["tool"], label="Инструмент / Tool"),
            mpatches.Patch(
                color=colors["poisoned"], label="Отравленный контент / Poisoned"
            ),
            mpatches.Patch(
                color=colors["attack"], label="Вредоносное действие / Attack"
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper left",
            fontsize=9,
            frameon=True,
            fancybox=False,
            edgecolor="black",
        )

        # Сохраняем отдельный файл для каждого домена
        domain_output = (
            output_path.parent / f"{output_path.stem}_{domain}{output_path.suffix}"
        )
        plt.tight_layout()
        plt.savefig(
            domain_output,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.1,
        )
        plt.close()
        print(f"Timeline visualization for {domain} saved to {domain_output}")

    # Также создаем общий файл со всеми доменами (компактная версия)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(
        "Временная диаграмма потока сообщений при атаках",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    for idx, domain in enumerate(domains_order):
        if domain not in simulations_data:
            continue

        ax = axes[idx]
        sim = simulations_data[domain]
        simulation_obj = sim.get("simulation", {})
        messages = []
        if simulation_obj.get("simulations"):
            sim_data = simulation_obj["simulations"][0]
            messages = sim_data.get("messages", [])[
                :10
            ]  # Меньше сообщений для компактности

        if not messages:
            continue

        from datetime import datetime

        start_time = None
        timeline_data = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            timestamp_str = msg.get("timestamp", "")
            turn = msg.get("turn_idx", 0)

            try:
                if timestamp_str:
                    msg_time = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if start_time is None:
                        start_time = msg_time
                    rel_time = (msg_time - start_time).total_seconds()
                else:
                    rel_time = turn * 2
            except:
                rel_time = turn * 2

            if role == "user":
                msg_type = "Пользователь"
                color = colors["user"]
            elif role == "assistant":
                has_attack = (
                    content
                    and any(
                        k in content.lower()
                        for k in ["send", "delete", "reset", "forward", "email sent"]
                    )
                ) or (
                    tool_calls
                    and any(
                        t.get("name")
                        in [
                            "send_email",
                            "delete_customer_visible_logs",
                            "reset_admin_api_key",
                            "forward_raw_snippet",
                        ]
                        for t in tool_calls
                    )
                )
                msg_type = "Агент (вредоносное)" if has_attack else "Агент"
                color = colors["attack"] if has_attack else colors["assistant"]
            elif role == "tool":
                is_poisoned = any(
                    k in str(content).lower()
                    for k in ["template", "sop", "instructions", "dump", "wipe"]
                )
                msg_type = "Инструмент (отравлен)" if is_poisoned else "Инструмент"
                color = colors["poisoned"] if is_poisoned else colors["tool"]
            else:
                msg_type = "Другое"
                color = COLORS["neutral"]

            timeline_data.append(
                {"turn": turn, "type": msg_type, "color": color, "time": rel_time}
            )

        if not timeline_data:
            continue

        timeline_data.sort(key=lambda x: x["time"])
        max_time = max(d["time"] for d in timeline_data) if timeline_data else 1
        if max_time == 0:
            max_time = 1

        n_messages = len(timeline_data)
        y_spacing = 0.85 / max(n_messages, 1)

        for i, data in enumerate(timeline_data):
            y_pos = 0.92 - i * y_spacing
            x_pos = data["time"] / max_time * 0.7 + 0.15

            ax.plot(
                [x_pos, x_pos],
                [y_pos - 0.02, y_pos + 0.02],
                color=data["color"],
                linewidth=2,
                alpha=0.8,
            )
            ax.scatter(
                [x_pos],
                [y_pos],
                s=120,
                c=data["color"],
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=5,
            )
            ax.text(
                x_pos,
                y_pos - 0.04,
                f"T{data['turn']}\n{data['time']:.1f}с",
                ha="center",
                va="top",
                fontsize=7,
                color="black",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=data["color"],
                    linewidth=1,
                ),
            )
            ax.text(
                x_pos + 0.03,
                y_pos,
                data["type"],
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=data["color"],
                    linewidth=1,
                ),
            )

        ax.axhline(y=0.02, color="black", linewidth=1.5)
        time_ticks = np.linspace(0, max_time, 6)
        for tick_time in time_ticks:
            x_tick = tick_time / max_time * 0.7 + 0.15
            ax.plot([x_tick, x_tick], [0.02, 0.01], color="black", linewidth=1)
            ax.text(
                x_tick, -0.01, f"{tick_time:.1f}с", ha="center", va="top", fontsize=7
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1)
        ax.set_xlabel("Время (секунды)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Последовательность событий", fontsize=10, fontweight="bold")
        ax.set_title(domain_titles[domain], fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_elements = [
        mpatches.Patch(color=colors["user"], label="Пользователь"),
        mpatches.Patch(color=colors["assistant"], label="Агент"),
        mpatches.Patch(color=colors["tool"], label="Инструмент"),
        mpatches.Patch(color=colors["poisoned"], label="Отравленный контент"),
        mpatches.Patch(color=colors["attack"], label="Вредоносное действие"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=9,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        title="Типы сообщений",
        title_fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close()
    print(f"Timeline overview saved to {output_path}")


def create_alternative_sankey(simulations_data: Dict[str, Dict], output_path: Path):
    """Создать визуализацию потока атак с деталями в научном стиле."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Создаем граф потока
    G = nx.DiGraph()

    attack_types = {
        "mail_rag_phishing": {
            "name": "Отравление RAG",
            "color": COLORS["danger"],
            "description": "Отравленный SOP\nшаблон в RAG",
        },
        "collab": {
            "name": "Межагентное\nотравление",
            "color": COLORS["secondary"],
            "description": "Вредоносные инструкции\nот коллеги",
        },
        "output_handling": {
            "name": "Инъекция в вывод",
            "color": COLORS["purple"],
            "description": "Попытка переслать\nсырой payload",
        },
    }

    pos = {}
    y_spacing = 2.5

    # Левая сторона - типы атак
    attack_nodes = []
    for i, (domain, info) in enumerate(attack_types.items()):
        if domain in simulations_data:
            node_id = f"attack_{domain}"
            G.add_node(
                node_id,
                label=info["name"],
                type="attack",
                color=info["color"],
                desc=info["description"],
            )
            pos[node_id] = (0, i * y_spacing)
            attack_nodes.append(node_id)

    # Центральная часть - этапы обработки
    processing_stages = [
        ("Обнаружение", "Агент оценивает\nзапрос"),
        ("Проверка\nполитики", "Проверка политики\nбезопасности"),
        ("Решение", "Принятие решения\nо действии"),
    ]

    stage_nodes = []
    for i, (stage_name, stage_desc) in enumerate(processing_stages):
        node_id = f"stage_{i}"
        G.add_node(node_id, label=stage_name, type="stage", desc=stage_desc)
        pos[node_id] = (3, 0.5 + i * 2)
        stage_nodes.append(node_id)

    # Правая сторона - результаты
    outcome_nodes = []
    outcomes = [("Заблокировано", COLORS["success"]), ("Успешно", COLORS["danger"])]

    for i, (outcome_name, outcome_color) in enumerate(outcomes):
        node_id = f"outcome_{outcome_name.lower()}"
        G.add_node(node_id, label=outcome_name, type="outcome", color=outcome_color)
        pos[node_id] = (6, i * 3)
        outcome_nodes.append(node_id)

    # Добавляем ребра от атак к этапам
    for attack_node in attack_nodes:
        G.add_edge(attack_node, "stage_0", weight=2)

    # Добавляем ребра между этапами
    for i in range(len(stage_nodes) - 1):
        G.add_edge(stage_nodes[i], stage_nodes[i + 1], weight=2)

    # Добавляем ребра от этапов к результатам
    for domain, sim in simulations_data.items():
        if domain in attack_types:
            outcome = (
                "Заблокировано" if not sim.get("attack_success", False) else "Успешно"
            )
            outcome_node = f"outcome_{outcome.lower()}"
            G.add_edge("stage_2", outcome_node, weight=3)

    # Рисуем узлы атак с увеличенным размером для текста
    for node in attack_nodes:
        node_color = G.nodes[node]["color"]
        # Увеличиваем размер узла, чтобы текст помещался
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=node_color,
            node_size=6000,
            ax=ax,
            alpha=0.9,
            edgecolors="black",
            linewidths=2,
        )
        # Описание ПОД узлом
        x, y = pos[node]
        ax.text(
            x,
            y - 1.2,
            G.nodes[node]["desc"],
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            color=COLORS["dark_gray"],
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor=node_color,
                linewidth=1,
                alpha=0.9,
            ),
        )

    # Рисуем узлы этапов с увеличенным размером
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=stage_nodes,
        node_color=COLORS["light_gray"],
        node_size=5000,
        ax=ax,
        alpha=0.9,
        edgecolors="black",
        linewidths=2,
    )

    # Описания этапов ПОД узлами
    for node in stage_nodes:
        x, y = pos[node]
        ax.text(
            x,
            y - 0.8,
            G.nodes[node]["desc"],
            ha="center",
            va="top",
            fontsize=8,
            style="italic",
            color=COLORS["dark_gray"],
        )

    # Рисуем узлы результатов с увеличенным размером
    for node in outcome_nodes:
        node_color = G.nodes[node]["color"]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=node_color,
            node_size=6000,
            ax=ax,
            alpha=0.9,
            edgecolors="black",
            linewidths=2,
        )

    # Рисуем ребра
    for edge in G.edges():
        weight = G.edges[edge].get("weight", 1)
        edge_color = COLORS["neutral"]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            edge_color=edge_color,
            arrows=True,
            arrowsize=25,
            width=weight * 1.5,
            ax=ax,
            alpha=0.6,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.15",
        )

    # Рисуем метки узлов ВНУТРИ узлов - красиво для конференции
    # Все подписи внутри кругов: белый текст на цветных узлах, черный на серых
    for node in G.nodes():
        x, y = pos[node]
        label = G.nodes[node]["label"]
        node_type = G.nodes[node].get("type")
        node_color = G.nodes[node].get("color", COLORS["light_gray"])

        # Разбиваем на строки если нужно
        label_lines = label.split("\n")

        # Определяем цвет текста: белый для цветных узлов, черный для серых
        if node_type == "attack" or node_type == "outcome":
            text_color = "white"
        else:
            text_color = "black"

        # Добавляем текст ВНУТРИ узла, центрированный
        for i, line in enumerate(label_lines):
            y_offset = (len(label_lines) - 1) * 0.15 - i * 0.3
            ax.text(
                x,
                y + y_offset,
                line,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    # Заголовки секций
    ax.text(0, 6.5, "Типы атак", ha="center", fontsize=12, fontweight="bold")
    ax.text(3, 6.5, "Обработка", ha="center", fontsize=12, fontweight="bold")
    ax.text(6, 6.5, "Результат", ha="center", fontsize=12, fontweight="bold")

    ax.set_xlim(-1, 7)
    ax.set_ylim(-2.5, 7.5)
    ax.set_title(
        "Поток атаки: от вектора атаки к результату безопасности",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        pad_inches=0.1,
    )
    plt.close()
    print(f"Alternative flow diagram saved to {output_path}")


def main():
    """Основная функция."""
    simulations_dir = Path(__file__).parent.parent / "data" / "simulations"
    output_dir = Path(__file__).parent.parent / "docs" / "paper_template" / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Находим файлы симуляций для каждого домена
    domain_files = {
        "mail_rag_phishing": "paper_results_mail_rag_phishing_gpt-4o_T0.0_mail_rag_phishing_global_shipper_trigger.json",
        "collab": "paper_results_collab_gpt-4o-mini_T1.0_collab_poisoning_resolution_gate.json",
        "output_handling": "paper_results_output_handling_gpt-4o-mini_T1.0_output_redirect_block.json",
    }

    simulations_data = {}

    for domain, filename in domain_files.items():
        file_path = simulations_dir / filename
        if file_path.exists():
            sim = load_simulation(file_path)
            dialogue = extract_dialogue(sim)

            # Определяем успешность атаки по reward
            attack_success = False
            if sim.get("simulations"):
                reward = sim["simulations"][0].get("reward_info", {}).get("reward", 0)
                attack_success = reward == 0.0  # Атака успешна, если reward = 0

            simulations_data[domain] = {
                "simulation": sim,
                "dialogue": dialogue,
                "attack_success": attack_success,
            }
            print(
                f"Loaded {domain}: {len(dialogue)} messages, attack_success={attack_success}"
            )
        else:
            print(f"Warning: {file_path} not found")

    # Создаем визуализации
    if simulations_data:
        create_attack_flow_diagram(simulations_data, output_dir / "attack_flow.pdf")
        create_timeline_visualization(
            simulations_data, output_dir / "attack_timeline.pdf"
        )
        create_alternative_sankey(simulations_data, output_dir / "attack_sankey.pdf")

        # Сохраняем примеры диалогов в JSON для использования в статье
        dialogue_examples = {}
        for domain, data in simulations_data.items():
            dialogue_examples[domain] = data["dialogue"]

        examples_path = output_dir / "dialogue_examples.json"
        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(dialogue_examples, f, indent=2, ensure_ascii=False)
        print(f"Dialogue examples saved to {examples_path}")


if __name__ == "__main__":
    main()
