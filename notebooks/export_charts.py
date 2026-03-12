#!/usr/bin/env python3
"""Export all notebook charts as individual images for LaTeX inclusion.

Each notebook cell's charts go into a separate subfolder under OUTPUT_DIR.
All charts use a consistent white-background style suitable for academic papers.
"""

import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "notebooks" / "exported_charts"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── data directories ───────────────────────────────────────────────────────
SOLO_DIRS = [
    ROOT / "data/simulations/run_20260304_210952",
    ROOT / "data/simulations/run_20260308_175733",
    ROOT / "data/simulations/run_20260308_212210",
]
DUAL_DIRS = [
    ROOT / "data/simulations/run_20260304_190834",
    ROOT / "data/simulations/run_20260308_184752",
    ROOT / "data/simulations/run_20260308_212225",
]
DUAL_TEMP_DIRS = [
    ROOT / "data/simulations/run_20260309_181507",
    ROOT / "data/simulations/run_20260311_000916",
    ROOT / "data/simulations/run_20260311_000927",
]

# ── global style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.facecolor": "white",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Consistent color palette (tab10)
COLORS = plt.cm.tab10.colors

# ── data loading ───────────────────────────────────────────────────────────

def load_results(directories: list[Path], run_mode: str) -> list[dict]:
    rows = []
    for directory in directories:
        for f in sorted(directory.glob("paper_results_*.json")):
            with open(f) as fp:
                data = json.load(fp)
            agent_llm = data["info"]["agent_info"]["llm"]
            domain = data["info"]["environment_info"]["domain_name"]
            for sim in data["simulations"]:
                reward = sim["reward_info"]["reward"] if sim.get("reward_info") else 0.0
                rows.append({
                    "run_mode": run_mode,
                    "model": agent_llm,
                    "domain": domain,
                    "task_id": sim["task_id"],
                    "reward": reward,
                    "success": int(abs(reward - 1.0) < 1e-6),
                    "trial": sim.get("trial", 0),
                })
    return rows


def load_results_with_temp(directories: list[Path]) -> list[dict]:
    rows = []
    for directory in directories:
        for f in sorted(directory.glob("paper_results_*.json")):
            with open(f) as fp:
                data = json.load(fp)
            agent_llm = data["info"]["agent_info"]["llm"]
            domain = data["info"]["environment_info"]["domain_name"]
            user_args = data["info"]["user_info"].get("llm_args") or {}
            user_temp = user_args.get("temperature", 0.0)
            for sim in data["simulations"]:
                reward = sim["reward_info"]["reward"] if sim.get("reward_info") else 0.0
                rows.append({
                    "model": agent_llm,
                    "domain": domain,
                    "task_id": sim["task_id"],
                    "user_temp": user_temp,
                    "reward": reward,
                    "success": int(abs(reward - 1.0) < 1e-6),
                    "trial": sim.get("trial", 0),
                })
    return rows


print("Loading data...")
rows_solo = load_results(SOLO_DIRS, "solo")
rows_dual = load_results(DUAL_DIRS, "dual")
df = pd.DataFrame(rows_solo + rows_dual)
# Filter same models as notebook
df = df[~df["model"].isin(["openai/gpt-4.1-nano", "moonshotai/kimi-k2.5"])]

df_t = pd.DataFrame(load_results_with_temp(DUAL_TEMP_DIRS))

# Strip provider prefix from model names (e.g. "openai/gpt-4o" -> "gpt-4o")
def short_model(name: str) -> str:
    return name.split("/", 1)[-1] if "/" in name else name

df["model"] = df["model"].map(short_model)
df_t["model"] = df_t["model"].map(short_model)

print(f"  Solo+Dual: {len(df)} rows  |  Temperature: {len(df_t)} rows")

# ── precompute ─────────────────────────────────────────────────────────────
models = sorted(df.model.unique())
domains = sorted(df.domain.unique())

task_rates = (
    df.groupby(["run_mode", "model", "domain", "task_id"])["success"]
    .mean().reset_index(name="p")
)

temps_t = sorted(df_t.user_temp.unique())
models_t = sorted(df_t.model.unique())
domains_t = sorted(df_t.domain.unique())

task_rates_t = (
    df_t.groupby(["user_temp", "model", "domain", "task_id"])["success"]
    .mean().reset_index(name="p")
)


def passk_table(k):
    tmp = task_rates.copy()
    tmp["passk"] = tmp["p"] ** k
    return tmp.groupby(["run_mode", "model", "domain"])["passk"].mean().reset_index()


def passk_table_t(k):
    tmp = task_rates_t.copy()
    tmp["passk"] = tmp["p"] ** k
    return tmp.groupby(["user_temp", "model", "domain"])["passk"].mean().reset_index()


def savefig(fig, folder: str, name: str):
    d = OUTPUT_DIR / folder
    d.mkdir(exist_ok=True)
    path = d / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A: Solo vs Dual
# ═══════════════════════════════════════════════════════════════════════════

# ── A1: pass^1 by domain (solo vs dual) ──────────────────────────────────
print("\n[A1] pass^1 by domain")
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df.groupby(["domain", "run_mode"])["success"].mean().unstack("run_mode")
pivot[["solo", "dual"]].plot(kind="bar", ax=ax, width=0.7, color=[COLORS[0], COLORS[1]])
ax.set_title("Pass@1 by Domain")
ax.set_ylabel("Pass@1")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
ax.legend(title="Mode")
ax.tick_params(axis="x", rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), ha="right")
plt.tight_layout()
savefig(fig, "A1_pass1_solo_dual", "pass1_by_domain")

# ── A2: pass^1 by model (solo vs dual) ──────────────────────────────────
print("[A2] pass^1 by model")
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df.groupby(["model", "run_mode"])["success"].mean().unstack("run_mode")
pivot[["solo", "dual"]].plot(kind="bar", ax=ax, width=0.7, color=[COLORS[0], COLORS[1]])
ax.set_title("Pass@1 by Model")
ax.set_ylabel("Pass@1")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
ax.legend(title="Mode")
ax.tick_params(axis="x", rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), ha="right")
plt.tight_layout()
savefig(fig, "A1_pass1_solo_dual", "pass1_by_model")

# ── A3: pass^k grid (solo vs dual) ──────────────────────────────────────
print("[A3] pass^k grid (solo vs dual)")
configs = [
    ("solo", 1, "Solo  pass@1"),
    ("solo", 4, "Solo  pass@4"),
    ("dual", 1, "Dual  pass@1"),
    ("dual", 4, "Dual  pass@4"),
]
for mode, k, title in configs:
    fig, ax = plt.subplots(figsize=(10, 5))
    pk = passk_table(k)
    pk = pk[pk.run_mode == mode]
    piv = pk.pivot(index="model", columns="domain", values="passk") \
            .reindex(index=models, columns=domains).fillna(0)

    x = np.arange(len(models))
    n_d = len(domains)
    bw = 0.8 / n_d
    for i, dom in enumerate(domains):
        offset = (i - n_d / 2 + 0.5) * bw
        ax.bar(x + offset, piv[dom], bw, label=dom, color=COLORS[i % len(COLORS)])

    ax.set_ylabel(f"pass@{k}")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=90, ha="center")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Domain", fontsize=8, ncol=2)
    plt.tight_layout()
    fname = f"passk_{mode}_k{k}"
    savefig(fig, "A2_passk_solo_dual", fname)

# ── A4: pass^k summary table (solo vs dual) ─────────────────────────────
print("[A4] pass^k table")
ks = [1, 2, 3, 4]
rows_table = {}
for mode in ["solo", "dual"]:
    p = task_rates.loc[task_rates.run_mode == mode, "p"].values
    rows_table[mode] = [float(np.mean(p ** k)) for k in ks]

fig, ax = plt.subplots(figsize=(4.5, 2.2))
ax.axis("off")
cell_text = [[f"{rows_table['solo'][i]:.3f}", f"{rows_table['dual'][i]:.3f}"] for i in range(len(ks))]
row_labels = [f"pass@{k}" for k in ks]
col_labels = ["Solo", "Dual"]
tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.0, 1.6)
for key, cell in tbl.get_celld().items():
    cell.set_text_props(color="black")
    cell.set_facecolor("white")
    cell.set_edgecolor("#999999")
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#D9E2F3")
    tbl[0, j].set_text_props(fontweight="bold")
for i in range(len(ks)):
    tbl[i + 1, -1].set_facecolor("#F2F2F2")
    tbl[i + 1, -1].set_text_props(fontweight="bold")
    s, d = rows_table["solo"][i], rows_table["dual"][i]
    if s > d:
        tbl[i + 1, 0].set_facecolor("#FFF2CC")
    elif d > s:
        tbl[i + 1, 1].set_facecolor("#FFF2CC")
ax.set_title("Pass@k: Solo vs Dual-Control", fontweight="bold", pad=12)
plt.tight_layout()
savefig(fig, "A3_passk_table", "passk_solo_vs_dual")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B: Temperature Analysis (dual-control)
# ═══════════════════════════════════════════════════════════════════════════

temp_colors = {t: COLORS[i] for i, t in enumerate(temps_t)}

# ── B1: pass^1 by domain grouped by temperature ─────────────────────────
print("\n[B1] pass^1 by domain (temperature)")
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df_t.groupby(["domain", "user_temp"])["success"].mean().unstack("user_temp")
pivot.plot(kind="bar", ax=ax, width=0.75, color=[temp_colors[t] for t in pivot.columns])
ax.set_title("Pass@1 by Domain (per user temperature)")
ax.set_ylabel("Pass@1")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
ax.legend(title="User T", labels=[f"T={c}" for c in pivot.columns])
ax.tick_params(axis="x", rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), ha="right")
plt.tight_layout()
savefig(fig, "B1_pass1_temperature", "pass1_by_domain_temp")

# ── B2: pass^1 by model grouped by temperature ──────────────────────────
print("[B2] pass^1 by model (temperature)")
fig, ax = plt.subplots(figsize=(8, 5))
pivot = df_t.groupby(["model", "user_temp"])["success"].mean().unstack("user_temp")
pivot.plot(kind="bar", ax=ax, width=0.75, color=[temp_colors[t] for t in pivot.columns])
ax.set_title("Pass@1 by Model (per user temperature)")
ax.set_ylabel("Pass@1")
ax.set_xlabel("")
ax.set_ylim(0, 1.05)
ax.legend(title="User T", labels=[f"T={c}" for c in pivot.columns])
ax.tick_params(axis="x", rotation=45)
ax.set_xticklabels(ax.get_xticklabels(), ha="right")
plt.tight_layout()
savefig(fig, "B1_pass1_temperature", "pass1_by_model_temp")

# ── B3: pass^k grid per temperature ─────────────────────────────────────
print("[B3] pass^k grid per temperature")
for t in temps_t:
    for k in [1, 2, 3, 4, 5]:
        fig, ax = plt.subplots(figsize=(10, 5))
        pk = passk_table_t(k)
        pk = pk[pk.user_temp == t]
        piv = pk.pivot(index="model", columns="domain", values="passk") \
                .reindex(index=models_t, columns=domains_t).fillna(0)

        x = np.arange(len(models_t))
        n_d = len(domains_t)
        bw = 0.8 / n_d
        for i, dom in enumerate(domains_t):
            offset = (i - n_d / 2 + 0.5) * bw
            ax.bar(x + offset, piv[dom], bw, label=dom, color=COLORS[i % len(COLORS)])

        ax.set_ylabel(f"pass@{k}")
        ax.set_title(f"User T={t}  pass@{k}", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_t, rotation=90, ha="center")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Domain", fontsize=8, ncol=2)
        plt.tight_layout()
        savefig(fig, "B2_passk_per_temp", f"passk_T{t}_k{k}")

# ── B4: pass^k summary table by temperature ─────────────────────────────
print("[B4] pass^k table (temperature)")
ks = [1, 2, 3, 4]
rows_tbl = {}
for t in temps_t:
    p = task_rates_t.loc[task_rates_t.user_temp == t, "p"].values
    rows_tbl[f"T={t}"] = [float(np.mean(p ** k)) for k in ks]

fig, ax = plt.subplots(figsize=(5.5, 2.5))
ax.axis("off")
col_labels = list(rows_tbl.keys())
cell_text = [[f"{rows_tbl[c][i]:.3f}" for c in col_labels] for i in range(len(ks))]
row_labels = [f"pass@{k}" for k in ks]
tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.0, 1.6)
for key, cell in tbl.get_celld().items():
    cell.set_text_props(color="black")
    cell.set_facecolor("white")
    cell.set_edgecolor("#999999")
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#D9E2F3")
    tbl[0, j].set_text_props(fontweight="bold")
for i in range(len(ks)):
    tbl[i + 1, -1].set_facecolor("#F2F2F2")
    tbl[i + 1, -1].set_text_props(fontweight="bold")
    vals = [rows_tbl[c][i] for c in col_labels]
    max_idx = vals.index(max(vals))
    tbl[i + 1, max_idx].set_facecolor("#FFF2CC")
ax.set_title("Pass@k by User Temperature (dual-control)", fontweight="bold", pad=12)
plt.tight_layout()
savefig(fig, "B3_passk_table_temp", "passk_by_temperature")

# ── B5: pass^k vs temperature — per model ───────────────────────────────
print("[B5] pass^k vs temperature — per model")
for k in [1, 2, 3, 4]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, model in enumerate(models_t):
        vals = []
        for t in temps_t:
            p = task_rates_t.loc[
                (task_rates_t.user_temp == t) & (task_rates_t.model == model), "p"
            ].values
            vals.append(float(np.mean(p ** k)) if len(p) > 0 else 0.0)
        ax.plot(temps_t, vals, marker="o", linewidth=2, markersize=6,
                label=model, color=COLORS[mi % len(COLORS)])
    ax.set_xlabel("User Temperature")
    ax.set_ylabel(f"pass@{k}")
    ax.set_title(f"pass@{k} vs User Temperature (per model)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(temps_t)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Model", fontsize=8, loc="best")
    plt.tight_layout()
    savefig(fig, "B4_passk_vs_temp_model", f"passk_vs_temp_model_k{k}")

# ── B6: pass^k vs temperature — per domain ──────────────────────────────
print("[B6] pass^k vs temperature — per domain")
for k in [1, 2, 3, 4]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for di, domain in enumerate(domains_t):
        vals = []
        for t in temps_t:
            p = task_rates_t.loc[
                (task_rates_t.user_temp == t) & (task_rates_t.domain == domain), "p"
            ].values
            vals.append(float(np.mean(p ** k)) if len(p) > 0 else 0.0)
        ax.plot(temps_t, vals, marker="s", linewidth=2, markersize=6,
                label=domain, color=COLORS[di % len(COLORS)])
    ax.set_xlabel("User Temperature")
    ax.set_ylabel(f"pass@{k}")
    ax.set_title(f"pass@{k} vs User Temperature (per domain)", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(temps_t)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Domain", fontsize=8, loc="best")
    plt.tight_layout()
    savefig(fig, "B5_passk_vs_temp_domain", f"passk_vs_temp_domain_k{k}")

# ── B7: pass^k vs k — aggregate ─────────────────────────────────────────
print("[B7] pass^k vs k — aggregate")
ks_range = list(range(1, 11))
fig, ax = plt.subplots(figsize=(8, 5))
for ti, t in enumerate(temps_t):
    p = task_rates_t.loc[task_rates_t.user_temp == t, "p"].values
    vals = [float(np.mean(p ** k)) for k in ks_range]
    ax.plot(ks_range, vals, marker="o", linewidth=2, markersize=6,
            label=f"T={t}", color=temp_colors[t])
ax.set_xlabel("k")
ax.set_ylabel("pass@k")
ax.set_title("Pass@k vs k — aggregate (all models, all domains)", fontweight="bold")
ax.set_ylim(0, 1.05)
ax.set_xticks(ks_range)
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(title="User Temp")
plt.tight_layout()
savefig(fig, "B6_passk_vs_k", "passk_vs_k_aggregate")

# ── B8: pass^k vs k — per model ─────────────────────────────────────────
print("[B8] pass^k vs k — per model")
for mi, model in enumerate(models_t):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ti, t in enumerate(temps_t):
        p = task_rates_t.loc[
            (task_rates_t.user_temp == t) & (task_rates_t.model == model), "p"
        ].values
        vals = [float(np.mean(p ** k)) if len(p) > 0 else 0.0 for k in ks_range]
        ax.plot(ks_range, vals, marker="o", linewidth=2, markersize=5,
                label=f"T={t}", color=temp_colors[t])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(model, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks_range)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="User Temp")
    plt.tight_layout()
    safe_name = model.replace("/", "_")
    savefig(fig, "B7_passk_vs_k_model", f"passk_vs_k_{safe_name}")

# ── B9: pass^k vs k — per domain ────────────────────────────────────────
print("[B9] pass^k vs k — per domain")
for di, domain in enumerate(domains_t):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ti, t in enumerate(temps_t):
        p = task_rates_t.loc[
            (task_rates_t.user_temp == t) & (task_rates_t.domain == domain), "p"
        ].values
        vals = [float(np.mean(p ** k)) if len(p) > 0 else 0.0 for k in ks_range]
        ax.plot(ks_range, vals, marker="s", linewidth=2, markersize=5,
                label=f"T={t}", color=temp_colors[t])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(domain, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks_range)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="User Temp")
    plt.tight_layout()
    savefig(fig, "B8_passk_vs_k_domain", f"passk_vs_k_{domain}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION C: Pass^k Tables by Domain (dual-control, temperature experiments)
# ═══════════════════════════════════════════════════════════════════════════

ks_full = [1, 2, 3, 4, 5]
col_labels = [f"pass@{k}" for k in ks_full]

# ── C1: pass^k by domain — one table per temperature ─────────────────────
for t in temps_t:
    print(f"[C1] pass^k by domain table (T={t})")
    cell_text_m = []
    for dom in domains_t:
        p = task_rates_t.loc[
            (task_rates_t.user_temp == t) & (task_rates_t.domain == dom), "p"
        ].values
        cell_text_m.append(
            [f"{float(np.mean(p ** k)):.3f}" if len(p) > 0 else "—" for k in ks_full]
        )
    # Aggregate row
    p_all = task_rates_t.loc[task_rates_t.user_temp == t, "p"].values
    cell_text_m.append(
        [f"{float(np.mean(p_all ** k)):.3f}" if len(p_all) > 0 else "—" for k in ks_full]
    )
    row_labels_m = list(domains_t) + ["ALL"]

    n_r = len(row_labels_m)
    fig_h = 1.2 + 0.45 * n_r
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text_m, rowLabels=row_labels_m, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.6)

    for key, cell in tbl.get_celld().items():
        cell.set_text_props(color="black")
        cell.set_facecolor("white")
        cell.set_edgecolor("#999999")
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#D9E2F3")
        tbl[0, j].set_text_props(fontweight="bold")
    for i in range(n_r):
        tbl[i + 1, -1].set_facecolor("#F2F2F2")
        tbl[i + 1, -1].set_text_props(fontweight="bold")
    # ALL row background
    for j in range(len(col_labels)):
        tbl[n_r, j].set_facecolor("#F2F2F2")
    # Highlight best domain per column (excluding ALL)
    for ki in range(len(ks_full)):
        vals = []
        for di in range(len(domains_t)):
            try:
                vals.append(float(cell_text_m[di][ki]))
            except ValueError:
                vals.append(-1)
        if vals:
            best_di = int(np.argmax(vals))
            tbl[best_di + 1, ki].set_facecolor("#FFF2CC")

    ax.set_title(f"Pass@k by Domain (User T={t})", fontweight="bold", pad=14)
    plt.tight_layout()
    savefig(fig, "C1_passk_by_domain", f"passk_by_domain_T{t}")

# ── C2: aggregated table — rows = domain, columns = pass@k (averaged over all temps) ──
print("[C2] pass^k by domain table (aggregated across temperatures)")

cell_text_agg = []
for dom in domains_t:
    p = task_rates_t.loc[task_rates_t.domain == dom, "p"].values
    cell_text_agg.append(
        [f"{float(np.mean(p ** k)):.3f}" if len(p) > 0 else "—" for k in ks_full]
    )
# Aggregate row
p_all = task_rates_t["p"].values
cell_text_agg.append(
    [f"{float(np.mean(p_all ** k)):.3f}" if len(p_all) > 0 else "—" for k in ks_full]
)
row_labels_agg = list(domains_t) + ["ALL"]

n_r = len(row_labels_agg)
fig_h = 1.2 + 0.45 * n_r
fig, ax = plt.subplots(figsize=(7, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText=cell_text_agg, rowLabels=row_labels_agg, colLabels=col_labels,
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 1.6)

for key, cell in tbl.get_celld().items():
    cell.set_text_props(color="black")
    cell.set_facecolor("white")
    cell.set_edgecolor("#999999")
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#D9E2F3")
    tbl[0, j].set_text_props(fontweight="bold")
for i in range(n_r):
    tbl[i + 1, -1].set_facecolor("#F2F2F2")
    tbl[i + 1, -1].set_text_props(fontweight="bold")
# ALL row
for j in range(len(col_labels)):
    tbl[n_r, j].set_facecolor("#F2F2F2")

# Highlight best domain per column (excluding ALL)
for ki in range(len(ks_full)):
    vals = []
    for di in range(len(domains_t)):
        try:
            vals.append(float(cell_text_agg[di][ki]))
        except ValueError:
            vals.append(-1)
    if vals:
        best_di = int(np.argmax(vals))
        tbl[best_di + 1, ki].set_facecolor("#FFF2CC")

ax.set_title("Pass@k by Domain (dual-control, all temperatures)", fontweight="bold", pad=14)
plt.tight_layout()
savefig(fig, "C1_passk_by_domain", "passk_by_domain_all_temps")


print("\nDone!")
