"""Draw docs/strategy_flow.png: how strategy S5 turns data into a weekly 10-stock portfolio.

uv run set-quant-alpha/docs/strategy_flow.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_suffix(".png")
INK, MUTED = "#1f2933", "#52606d"
SRC, SRC_BG = "#047857", "#ecfdf3"
PREP, PREP_BG = "#0f766e", "#f0fdfa"
MODEL, MODEL_BG = "#1d4ed8", "#eff4ff"
BOOK, BOOK_BG = "#7c3aed", "#f5f0ff"
LIVE, LIVE_BG = "#c2410c", "#fff7ed"

fig, ax = plt.subplots(figsize=(18, 11.2), dpi=150)
ax.set(xlim=(0, 18), ylim=(0, 11.2))
ax.axis("off")


def box(x, y, w, h, title, body, ec, fc, size=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14", fc=fc, ec=ec, lw=1.6
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=size + 2,
        weight="bold",
        color=ec,
    )
    ax.text(
        x + w / 2,
        y + h - 0.72,
        body,
        ha="center",
        va="top",
        fontsize=size,
        color=INK,
        linespacing=1.45,
    )


def arrow(p, q, color=INK, ls="-", label="", label_xy=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=18,
            color=color,
            lw=1.8,
            ls=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
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
            fontsize=9.5,
            color=color,
            style="italic",
            bbox=dict(fc="white", ec="none", pad=1.5),
        )


def stage(y, text, color):
    ax.text(
        0.25, y, text, ha="left", va="center", fontsize=11, weight="bold", color=color, rotation=90
    )


ax.text(
    9,
    10.85,
    "Strategy S5: from market data to a weekly 10-stock portfolio",
    ha="center",
    va="center",
    fontsize=18,
    weight="bold",
    color=INK,
)
ax.text(
    9,
    10.4,
    "Every step uses only information available at that day's close; trades happen at the next open.",
    ha="center",
    va="center",
    fontsize=11,
    color=MUTED,
)

# ---- row 1: data
stage(9.0, "DATA", SRC)
box(
    0.8,
    8.2,
    5.2,
    1.85,
    "Thai stock prices",
    "Yahoo Finance: 853 SET stocks\ndaily open / high / low / close / volume",
    SRC,
    SRC_BG,
)
box(
    6.4,
    8.2,
    5.2,
    1.85,
    "Global market data",
    "FRED rates, Cboe VIX, USD/THB, EEM,\nS&P 500, oil, gold (lagged one day)",
    SRC,
    SRC_BG,
)
box(
    12.0,
    8.2,
    5.2,
    1.85,
    "Stock information",
    "sector, size, Thai market breadth\nand market-regime estimates",
    SRC,
    SRC_BG,
)

# ---- row 2: preparation
stage(6.75, "PREPARE", PREP)
box(
    0.8,
    5.95,
    7.6,
    1.75,
    "1  Choose the universe (daily)",
    "liquid: median traded value over 60 days >= THB 20M\ntradable: the 100 most-traded of those (a point-in-time SET100)",
    PREP,
    PREP_BG,
)
box(
    9.6,
    5.95,
    7.6,
    1.75,
    "2  Build 120 inputs per stock",
    "momentum and short-term reversal, volatility, liquidity,\nmarket / baht / EM betas, VIX and market regime, sector",
    PREP,
    PREP_BG,
)

# ---- row 3: models and score
stage(4.3, "SCORE", MODEL)
box(
    0.8,
    3.4,
    5.0,
    1.95,
    "3a  Fast model (5 days)",
    'logistic regression + XGBoost\n"will this stock beat the market\nover the next 5 days?"',
    MODEL,
    MODEL_BG,
)
box(
    12.2,
    3.4,
    5.0,
    1.95,
    "3b  Slow model (20 days)",
    "XGBoost\nranks stocks by expected\nmarket-beating return over 20 days",
    MODEL,
    MODEL_BG,
)
box(
    6.5,
    3.4,
    5.0,
    1.95,
    "4  One combined score",
    "50% rank of the fast model\n+ 50% rank of the slow model\n= a score from 0 to 1 per stock",
    MODEL,
    MODEL_BG,
)

# ---- row 4: portfolio and execution
stage(1.5, "TRADE", BOOK)
box(
    0.8,
    0.45,
    6.2,
    2.15,
    "5  Portfolio rules (every 5 trading days)",
    "buy the 10 highest-scoring stocks\nkeep a stock already held while it stays in the top 30\n"
    "10% in each stock, no stop-loss",
    BOOK,
    BOOK_BG,
)
box(
    7.6,
    0.45,
    4.6,
    2.15,
    "6  Trade at the next open",
    "sells first, then buys\n100-share board lots\ncost 0.168% per side",
    BOOK,
    BOOK_BG,
)
box(
    12.8,
    0.45,
    4.4,
    2.15,
    "7  Track and review",
    "paper portfolio vs TDEX\ndashboard and signal-decay alerts\nreview about May 2027",
    LIVE,
    LIVE_BG,
)

# ---- arrows
arrow((3.4, 8.2), (3.4, 7.7))  # prices -> universe
arrow((5.0, 8.2), (11.0, 7.7))  # prices -> inputs
arrow((9.0, 8.2), (12.4, 7.7))  # global data -> inputs
arrow((14.6, 8.2), (14.6, 7.7))  # stock information -> inputs
arrow((8.4, 6.82), (9.6, 6.82))
arrow((13.4, 5.95), (3.3, 5.35), rad=0.0)
arrow((13.4, 5.95), (14.7, 5.35))
arrow((5.8, 4.35), (6.5, 4.35))
arrow((12.2, 4.35), (11.5, 4.35))
arrow(
    (9.0, 3.4), (3.9, 2.6), color=BOOK, label="weekly decision after the close", label_xy=(6.9, 3.0)
)
arrow((7.0, 1.5), (7.6, 1.5))
arrow((12.2, 1.5), (12.8, 1.5))
arrow(
    (15.0, 2.6),
    (14.7, 3.4),
    color=LIVE,
    ls="--",
    label="both models retrained weekly on new data",
    label_xy=(13.3, 3.0),
)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
