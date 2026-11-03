"""Draw docs/daily_flow.png: how Airflow runs the pipeline each day and feeds the dashboard.

uv run set-quant-alpha/docs/daily_flow.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_suffix(".png")
INK, MUTED = "#1f2933", "#52606d"
DAILY, DAILY_BG = "#1d4ed8", "#eff4ff"
WEEKLY, WEEKLY_BG = "#7c3aed", "#f5f0ff"
FILE_BG, SRC_BG, SERVE_BG = "#f8fafc", "#ecfdf3", "#fff7ed"

fig, ax = plt.subplots(figsize=(20.6, 11), dpi=150)
ax.set(xlim=(0, 20.6), ylim=(0, 11))
ax.axis("off")


def box(x, y, w, h, title, body="", fc="white", ec=INK, lw=1.4, ls="-", size=9.5, title_color=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw, ls=ls
        )
    )
    if body:
        ax.text(
            x + w / 2,
            y + h - 0.2,
            title,
            ha="center",
            va="top",
            fontsize=size + 1.5,
            weight="bold",
            color=title_color or ec,
            family="DejaVu Sans",
        )
        ax.text(
            x + w / 2,
            y + h - 0.55,
            body,
            ha="center",
            va="top",
            fontsize=size,
            color=INK,
            linespacing=1.35,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            linespacing=1.35,
        )


def arrow(p, q, color=INK, ls="-", lw=1.6, label="", label_xy=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            lw=lw,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )
    if label:
        lx, ly = label_xy or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
            style="italic",
            bbox=dict(fc="white", ec="none", pad=1.5),
        )


ax.text(
    10.3,
    10.6,
    "set-quant-alpha: daily run from Airflow to the dashboard",
    ha="center",
    fontsize=18,
    weight="bold",
    color=INK,
)
ax.text(
    10.3,
    10.2,
    "Every task is a BashOperator calling  python scripts/run_pipeline.py --mode <file|live> --steps <step>"
    "  ·  steps share files on disk, so a failed task retries on its own",
    ha="center",
    fontsize=10,
    color=MUTED,
)

# lanes
TX, TW, GAP = 3.9, 2.2, 0.45  # task column layout inside a lane
cols = [TX + i * (TW + GAP) for i in range(5)]
box(3.6, 4.55, 13.35, 5.15, "", fc=DAILY_BG, ec=DAILY, lw=1.2)
ax.text(
    3.8,
    9.45,
    "DAG  set_quant_alpha_daily   ·   Mon-Fri 18:30 Asia/Bangkok (after the SET close)   ·   "
    "retries 2 × 10 min",
    fontsize=10.5,
    weight="bold",
    color=DAILY,
    va="center",
)
box(3.6, 0.75, 13.35, 3.25, "", fc=WEEKLY_BG, ec=WEEKLY, lw=1.2)
ax.text(
    3.8,
    3.9,
    "DAG  set_quant_alpha_weekly\nSaturday 06:00 Asia/Bangkok",
    fontsize=10.5,
    weight="bold",
    color=WEEKLY,
    va="top",
    linespacing=1.3,
)

# daily tasks (bottom row of the lane) and the files they write (top row)
TY, TH, FY, FH = 5.0, 1.75, 7.45, 1.45
daily = [
    ("1  ingest_live", "--mode live\nonly new days\n(+10-day overlap)\nupsert into parquet"),
    (
        "2  transform",
        "factor model, idio vol\n65 features, 55 dummies\nGaussian HMM regime\n(forward-filtered)",
    ),
    ("3  signals", "5-day + 20-day models\n50/50 rank blend\nS5: top-10, 30-rank\nbuffer, no stop"),
    ("4  paper", "6 frozen shadow\nportfolios (S1-S6):\nfills, NAV + IC\ndecay monitor"),
    ("5  report", "weekly backtest\n+ signals + paper\n→ one self-contained\nHTML page"),
]
files = [
    "data/raw/*.parquet\nprices · market · FRED\nCboe · sectors · FF factors",
    "data/processed/\ndataset.parquet",
    "reports/latest_signals.csv\nticker · weight · shares\nscores · HMM regime",
    "data/paper/ (append-only)\nnav · trades · scores\nstate.json",
    "reports/dashboard.html\nreports/index.html",
]
for x, (title, body), f in zip(cols, daily, files, strict=True):
    box(x, TY, TW, TH, title, body, ec=DAILY, size=9)
    box(x, FY, TW, FH, f, fc=FILE_BG, ec=MUTED, lw=1.0, ls="--", size=9)
    arrow((x + TW / 2, TY + TH), (x + TW / 2, FY), color=MUTED)
for a, b in zip(cols[:-1], cols[1:], strict=True):
    arrow((a + TW, TY + TH / 2), (b, TY + TH / 2), color=DAILY, lw=2.2)
for a, b in zip(cols[:2], cols[1:3], strict=True):  # each step reads the previous step's file
    arrow((a + TW, FY + 0.35), (b + 0.15, TY + TH), color=MUTED, lw=1.0, rad=-0.15)

# weekly tasks
WY, WH = 1.15, 2.0
weekly = [
    ("train", "refresh data first,\nthen walk-forward\n+ production refit"),
    ("simulate", "6 exact strategies\nfactsheet, Carhart,\nvalidation pick"),
    ("report", "publish S5 record\n+ trial returns,\nrebuild dashboard"),
]
wcols = cols[2:]
for x, (title, body) in zip(wcols, weekly, strict=True):
    box(x, WY, TW, WH, title, body, ec=WEEKLY)
for a, b in zip(wcols[:-1], wcols[1:], strict=True):
    arrow((a + TW, WY + WH / 2), (b, WY + WH / 2), color=WEEKLY, lw=2.2)
arrow(
    (wcols[0] + TW * 0.5, WY + WH),
    (cols[2] + TW * 0.5, TY),
    color=WEEKLY,
    ls="--",
    label="models/*.joblib\nmeta.json",
    label_xy=(cols[2] + TW * 0.5, 4.25),
)
arrow(
    (wcols[1] + TW * 0.7, WY + WH),
    (cols[4] + TW * 0.3, TY),
    color=WEEKLY,
    ls="--",
    label="sim_results.joblib",
    label_xy=(cols[4] - 0.1, 4.25),
)
ax.text(
    cols[0] + TW * 1.1,
    WY + 0.6,
    "trains the models\nthe daily DAG\nonly reads",
    ha="center",
    va="center",
    fontsize=9,
    color=WEEKLY,
    style="italic",
)

# sources and the scheduler (left)
box(
    0.3,
    4.75,
    2.95,
    2.25,
    "Data sources",
    "Yahoo: screener, prices,\nsectors, TDEX, THB\nFRED (Baa-10y, rates)\n"
    "Cboe VIX term structure\nKen French EM factors",
    fc=SRC_BG,
    ec="#15803d",
    size=9,
)
arrow((3.25, 5.9), (cols[0], 5.9), color="#15803d", lw=2.0)
box(
    0.3,
    1.15,
    2.95,
    3.0,
    "airflow container",
    "scheduler starts runs\nby cron (Asia/Bangkok)\n\nUI  localhost:8080\n"
    "task status, logs,\nretries, manual trigger",
    fc="white",
    ec=INK,
    size=9,
)
arrow((3.25, 3.6), (3.6, 4.75), color=DAILY, lw=1.8)
arrow((3.25, 2.2), (3.6, 2.2), color=WEEKLY, lw=1.8)

# serving (right)
SX, SW = 17.35, 3.0
box(
    SX,
    FY - 0.1,
    SW,
    FH + 0.2,
    "dashboard container",
    "nginx serves reports/\nlocalhost:8000",
    fc=SERVE_BG,
    ec="#c2410c",
    size=9.5,
)
arrow((cols[4] + TW, FY + FH / 2), (SX, FY + FH / 2), color="#c2410c", lw=2.0)
box(
    SX,
    TY - 0.2,
    SW,
    TH + 0.4,
    "Trader, next morning",
    "opens the dashboard:\nequity curve, factsheet,\n"
    "regime, today's orders\n→ buy at the next open",
    fc=SERVE_BG,
    ec="#c2410c",
    size=9.5,
)
arrow((SX + SW / 2, FY - 0.1), (SX + SW / 2, TY + TH + 0.2), color="#c2410c", lw=2.0)

# legend / footer
ax.text(SX, 3.5, "Legend", fontsize=10, weight="bold", color=INK)
for i, (c, ls, t) in enumerate(
    [
        (DAILY, "-", "daily task order"),
        (WEEKLY, "-", "weekly task order"),
        (WEEKLY, "--", "weekly output read daily"),
        (MUTED, "-", "writes / reads file"),
    ]
):
    y = 3.05 - i * 0.42
    arrow((SX, y), (SX + 0.7, y), color=c, ls=ls, lw=1.8)
    ax.text(SX + 0.85, y, t, va="center", fontsize=9, color=INK)
ax.text(
    10.3,
    0.3,
    "docker compose: airflow · dashboard · pipeline (manual one-off run)   ·   host folders data/ models/ "
    "reports/ are mounted into every container, so runs stay incremental across restarts",
    ha="center",
    fontsize=9.5,
    color=MUTED,
)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
