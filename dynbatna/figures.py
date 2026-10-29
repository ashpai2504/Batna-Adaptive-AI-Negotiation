"""Figures + markdown tables for the Dynamic-BATNA evaluation. Usage: python -m dynbatna.figures [--run NAME] [--lora-runs base lora]"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DYN_DATA, RESULTS, ROOT

FIG = os.path.join(ROOT, "figures", "dynbatna")
RUNS = os.path.join(RESULTS, "runs")
COND_ORDER = ["none", "static", "dynamic"]
COND_LABEL = {"none": "no notices (rlvr-batna setup)", "static": "notices + static-BATNA reward",
              "dynamic": "notices + dynamic-BATNA reward"}
COLORS = {"none": "#9e9e9e", "static": "#e07b39", "dynamic": "#2b7bba"}
DS_ORDER = ["amazon", "craigslist", "casino", "dnd", "ji", "ALL"]


def _ci(s):
    lo, hi = s.strip("[]").split(",")
    return float(lo), float(hi)


def fig_dataset(stats_path, out):
    st = json.load(open(stats_path))
    ds = [d for d in DS_ORDER if d in st]
    types = sorted({t for d in ds for t in st[d]["by_type"]})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bottom = np.zeros(len(ds))
    for t in types:
        v = np.array([st[d]["by_type"].get(t, 0) for d in ds])
        ax[0].bar(ds, v, bottom=bottom, label=t)
        bottom += v
    ax[0].set_title("Scheduled events per dataset (test panel)")
    ax[0].set_ylabel("events")
    ax[0].legend(fontsize=8)
    up = [st[d]["up_events"] for d in ds]
    dn = [st[d]["down_events"] for d in ds]
    x = np.arange(len(ds))
    ax[1].bar(x - 0.2, up, 0.4, label="alternative improves (up)", color="#4caf50")
    ax[1].bar(x + 0.2, dn, 0.4, label="alternative worsens (down)", color="#c62828")
    ax[1].set_xticks(x, ds)
    ax[1].set_title("Direction of reservation moves")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _grouped(summ, metric, title, out, ylabel=None, datasets=DS_ORDER):
    d = summ[summ.dataset.isin(datasets)].copy()
    d["dataset"] = pd.Categorical(d.dataset, [x for x in DS_ORDER if x in set(d.dataset)])
    d = d.sort_values("dataset")
    ds = list(d.dataset.cat.categories)
    conds = [c for c in COND_ORDER if c in set(d.condition)]
    w = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, c in enumerate(conds):
        sub = d[d.condition == c].set_index("dataset").reindex(ds)
        vals = sub[metric].astype(float).values
        cis = np.array([_ci(s) if isinstance(s, str) else (np.nan, np.nan) for s in sub[metric + "_ci"]])
        err = np.abs(cis.T - vals)
        x = np.arange(len(ds)) + (i - (len(conds) - 1) / 2) * w
        ax.bar(x, vals, w, yerr=err, capsize=2, color=COLORS[c], label=COND_LABEL[c])
    ax.set_xticks(np.arange(len(ds)), ds)
    ax.set_title(title)
    ax.set_ylabel(ylabel or metric)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_forest(tests, out, comparison="dynamic - static", dataset="ALL", title=None):
    t = tests[(tests.comparison == comparison) & (tests.dataset == dataset)].dropna(subset=["diff"])
    t = t[t.metric.isin(["deal_rate", "bargained_ratio_dyn", "reward_dynamic_tau", "reward_static_tau",
                         "deal_below_batna", "decision_accuracy", "regret"])]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(t))
    ax.errorbar(t["diff"], y, xerr=[t["diff"] - t.ci_lo, t.ci_hi - t["diff"]], fmt="o", color="#2b7bba", capsize=3)
    for yi, (m, p, n) in enumerate(zip(t.metric, t.p, t.n_pairs)):
        ax.text(ax.get_xlim()[1], yi, f" p={p:.3f} (n={n})", va="center", fontsize=8)
    ax.set_yticks(y, t.metric)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"paired difference ({comparison}), 95% bootstrap CI")
    ax.set_title(title or f"{comparison}: {dataset}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def md_table(summ, metrics, datasets=DS_ORDER, conds=COND_ORDER):
    rows = ["| dataset | condition | n | " + " | ".join(metrics) + " |", "|" + "---|" * (3 + len(metrics))]
    for ds in datasets:
        for c in conds:
            r = summ[(summ.dataset == ds) & (summ.condition == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = []
            for m in metrics:
                v = r[m]
                ci = r.get(m + "_ci")
                cells.append("–" if pd.isna(v) else (f"{v:.3f} {ci}" if isinstance(ci, str) else f"{v:.3f}"))
            rows.append(f"| {ds} | {c} | {int(r.n)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def md_tests(tests, comparison, dataset="ALL"):
    t = tests[(tests.comparison == comparison) & (tests.dataset == dataset)].dropna(subset=["diff"])
    rows = [f"| metric ({comparison}, {dataset}) | n pairs | diff | 95% CI | p (sign-flip) |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        rows.append(f"| {r.metric} | {int(r.n_pairs)} | {r['diff']:+.4f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p:.4f} |")
    return "\n".join(rows)


def fig_lora(run_dirs, labels, out):
    rows = []
    for rd, lab in zip(run_dirs, labels):
        s = pd.read_csv(os.path.join(rd, "summary.csv"))
        s = s[(s.dataset == "ALL") & (s.condition == "dynamic")].iloc[0]
        rows.append((lab, s))
    metrics = ["deal_rate", "bargained_ratio_dyn", "reward_dynamic_tau", "deal_below_batna", "decision_accuracy", "regret",
               "learner_format_rate"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.8 / len(rows)
    for i, (lab, s) in enumerate(rows):
        vals = [float(s[m]) for m in metrics]
        cis = np.array([_ci(s[m + "_ci"]) for m in metrics])
        x = np.arange(len(metrics)) + (i - (len(rows) - 1) / 2) * w
        ax.bar(x, vals, w, yerr=np.abs(cis.T - vals), capsize=2, label=lab)
    ax.set_xticks(np.arange(len(metrics)), metrics, rotation=15)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Small open model, dynamic condition, baseline panel: base vs LoRA (rejection-sampled SFT)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="gpt5mini_main")
    ap.add_argument("--lora-runs", nargs="*", default=[])
    ap.add_argument("--lora-labels", nargs="*", default=[])
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    rd = os.path.join(RUNS, a.run)
    summ = pd.read_csv(os.path.join(rd, "summary.csv"))
    tests = pd.read_csv(os.path.join(rd, "paired_tests.csv"))
    ev = pd.read_csv(os.path.join(rd, "summary_event_episodes.csv"))
    ev_tests = pd.read_csv(os.path.join(rd, "paired_tests_event_episodes.csv"))
    fig_dataset(os.path.join(DYN_DATA, "stats.test.json"), os.path.join(FIG, "fig1_dataset.png"))
    _grouped(summ, "bargained_ratio", "Bargained ratio (rlvr-batna metric, static BATNA)", os.path.join(FIG, "fig2_bargained_ratio.png"))
    _grouped(summ, "reward_dynamic_tau", "Dynamic-threshold reward (all episodes)", os.path.join(FIG, "fig3_reward_dynamic.png"))
    _grouped(ev, "deal_below_batna", "Deals closed below the current BATNA (event episodes)", os.path.join(FIG, "fig4_below_batna_events.png"), "fraction of deals")
    _grouped(ev, "decision_accuracy", "Normative decision accuracy (event episodes)", os.path.join(FIG, "fig5_decision_accuracy_events.png"))
    _grouped(ev, "regret", "Regret vs best securable outcome (event episodes)", os.path.join(FIG, "fig6_regret_events.png"))
    fig_forest(ev_tests, os.path.join(FIG, "fig7_forest_dyn_vs_static_events.png"), title="dynamic − static reward framing, event episodes (pooled)")
    fig_forest(tests, os.path.join(FIG, "fig8_forest_dyn_vs_static_all.png"), title="dynamic − static reward framing, all 517 episodes")
    fig_forest(ev_tests, os.path.join(FIG, "fig9_forest_static_vs_none_events.png"), comparison="static - none",
               title="static framing − no notices, event episodes (pooled)")
    if a.lora_runs:
        fig_lora([os.path.join(RUNS, r) for r in a.lora_runs], a.lora_labels or a.lora_runs, os.path.join(FIG, "fig10_lora.png"))
    with open(os.path.join(RESULTS, "tables.md"), "w") as f:
        f.write("## Table 1. All episodes (main panel, 517 episodes x 3 conditions)\n\n")
        f.write(md_table(summ, ["deal_rate", "bargained_ratio", "bargained_ratio_dyn", "reward_static_tau", "reward_dynamic_tau",
                                "deal_below_batna", "decision_accuracy", "regret"]) + "\n\n")
        f.write("## Table 2. Event episodes only\n\n")
        f.write(md_table(ev, ["deal_rate", "bargained_ratio_dyn", "reward_dynamic_tau", "deal_below_batna", "decision_accuracy",
                              "regret", "adaptation_latency"]) + "\n\n")
        for comp in ["dynamic - static", "dynamic - none", "static - none"]:
            f.write(f"## Paired tests, event episodes: {comp}\n\n" + md_tests(ev_tests, comp) + "\n\n")
            f.write(f"## Paired tests, all episodes: {comp}\n\n" + md_tests(tests, comp) + "\n\n")
        if a.lora_runs:
            f.write("## LoRA: small open model (dynamic condition, baseline panel)\n\n")
            for r, lab in zip(a.lora_runs, a.lora_labels or a.lora_runs):
                s = pd.read_csv(os.path.join(RUNS, r, "summary.csv"))
                f.write(f"### {lab}\n\n" + md_table(s, ["deal_rate", "bargained_ratio_dyn", "reward_dynamic_tau", "deal_below_batna",
                                                      "decision_accuracy", "regret", "learner_format_rate"], conds=["dynamic"]) + "\n\n")
    print("figures ->", FIG, "\ntables ->", os.path.join(RESULTS, "tables.md"))


if __name__ == "__main__":
    main()
