"""Figures for the LLM-vs-LLM negotiation experiments (results/llm_negotiation -> figures/llm)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", os.environ.get("LLM_RES", "llm_negotiation"))
FIG = os.path.join(ROOT, "figures", os.environ.get("LLM_FIG", "llm"))
os.makedirs(FIG, exist_ok=True)
COL = {"Surplus": "#d62728", "Static-BATNA": "#7f7f7f", "Dynamic-BATNA": "#1f77b4"}


def load(exp):
    p = os.path.join(RES, f"episodes_{exp}.jsonl")
    if not os.path.exists(p):
        return None
    df = pd.DataFrame(map(json.loads, open(p)))
    df["cond"] = np.where(df.shock, "shock", "no-shock")
    return df


def ci(x, B=4000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (B, len(x))).mean(1)
    return x.mean(), x.mean() - np.quantile(m, .025), np.quantile(m, .975) - x.mean()


def bars(ax, df, groups, metric, title, colors, xlabels):
    conds = ["no-shock", "shock"]
    w = 0.8 / len(groups)
    for j, g in enumerate(groups):
        vals = [ci(df[(df[groups.name] == g) & (df.cond == c)][metric].astype(float)) for c in conds] \
            if hasattr(groups, "name") else None
        m = [v[0] for v in vals]
        err = np.array([[v[1] for v in vals], [v[2] for v in vals]])
        ax.bar(np.arange(len(conds)) + (j - (len(groups) - 1) / 2) * w, m, w, yerr=err, capsize=3,
               color=colors[g], label=g)
    ax.set_xticks(range(len(conds)), xlabels)
    ax.set_title(title, fontsize=10)
    ax.axhline(0, color="k", lw=0.6)


def fig_A(df):
    groups = pd.Series(["Surplus", "Static-BATNA", "Dynamic-BATNA"], name="buyer")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (m, t) in zip(axes, [("buyer_profit", "Buyer profit (÷ current price)"),
                                 ("seller_profit", "Seller profit (÷ current price)"),
                                 ("joint_profit", "Joint profit (÷ current price)"),
                                 ("below_batna", "Deals above buyer's true BATNA")]):
        bars(ax, df, groups, m, t, COL, ["no shock", "PortWatch shock"])
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Experiment A: LLM buyer vs LLM seller on real Amazon products, real PortWatch shocks "
                 f"(n={df.i.nunique()} products per cell, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_profits.png"), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, c in zip(axes, ["no-shock", "shock"]):
        sub = df[df.cond == c]
        tab = sub.groupby("buyer").outcome.value_counts(normalize=True).unstack(fill_value=0)
        tab = tab.reindex(["Surplus", "Static-BATNA", "Dynamic-BATNA"])
        bottom = np.zeros(len(tab))
        for o, colr in [("deal", "#2ca02c"), ("walk_buyer", "#ff7f0e"), ("walk_seller", "#9467bd"),
                        ("impasse", "#8c564b")]:
            if o in tab:
                ax.bar(tab.index, tab[o], bottom=bottom, color=colr, label=o)
                bottom += tab[o].values
        ax.set_title(f"Outcomes ({c})", fontsize=10)
        ax.set_ylim(0, 1)
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_outcomes.png"), dpi=160)
    plt.close(fig)

    sub = df[(df.cond == "shock") & (df.outcome == "deal")]
    fig, ax = plt.subplots(figsize=(6, 4))
    for b in ["Surplus", "Static-BATNA", "Dynamic-BATNA"]:
        s = sub[sub.buyer == b]
        ax.scatter(s.market_end_ratio, s.deal_price_ratio, color=COL[b], label=b, alpha=.8)
    lim = [0.9, max(1.6, float(sub[["market_end_ratio", "deal_price_ratio"]].max().max()) + .05)] if len(sub) else [.9, 1.6]
    ax.plot(lim, lim, "k--", lw=.8, label="price = market (BATNA)")
    ax.set_xlabel("Market price on deal day ÷ current price")
    ax.set_ylabel("Deal price ÷ current price")
    ax.set_title("Shock deals: points above the line overpay vs the BATNA", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figA_price_vs_batna.png"), dpi=160)
    plt.close(fig)


def fig_B(df):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    cols = {"neutral": "#1f77b4", "emotional": "#d62728"}
    for ax, (m, t) in zip(axes, [("buyer_profit", "Buyer profit"), ("seller_profit", "Seller profit"),
                                 ("below_batna", "Deals above buyer's BATNA")]):
        for j, s in enumerate(["neutral", "emotional"]):
            vals = [ci(df[(df.buyer == b) & (df.seller == s)][m].astype(float)) for b in ["Surplus", "Dynamic-BATNA"]]
            ax.bar(np.arange(2) + (j - .5) * .38, [v[0] for v in vals], .38,
                   yerr=np.array([[v[1] for v in vals], [v[2] for v in vals]]), capsize=3, color=cols[s],
                   label=f"{s} seller")
        ax.set_xticks([0, 1], ["Surplus buyer", "Dynamic-BATNA buyer"])
        ax.set_title(t, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Experiment B: emotional 'evil-twin' seller, PortWatch shock (n={df.i.nunique()} products)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figB_evil_twin.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    a = load("A")
    if a is not None:
        fig_A(a)
    b = load("B")
    if b is not None:
        fig_B(b)
    print("figures ->", FIG)
