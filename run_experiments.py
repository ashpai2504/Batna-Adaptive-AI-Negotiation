"""Preliminary benchmark: Surplus vs Static-tau vs Dynamic-BATNA in a multi-echelon network.

Outputs
  data/mock_kaggle_geo_disruptions.csv   synthetic geospatial disruption table
  results/network.db                     SQLite ledger of all 50-episode evaluation runs
  results/episodes_main.csv              per-episode log (4 agents x 2 conditions x 50 episodes)
  results/summary_main.csv, summary.json aggregate metrics with bootstrap CIs
  results/training_history.csv           CEM learning curves
  figures/*.png                          proposal figures
  review_package/                        reviewer bundle (see review_package/DATA_DICTIONARY.md)
"""
from __future__ import annotations

import json
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from negotiation_sim.agents import AGENTS, DynamicBATNAAgent, StaticThresholdAgent, SurplusAgent
from negotiation_sim.db import NetworkDB
from negotiation_sim.env import Config, MultiEchelonEnv, ShockProcess, sample_scenario
from negotiation_sim.geodata import DISTRIBUTOR, PORTS, ShockSampler, load_disruptions
from negotiation_sim.train import cem

ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("data", "results", "figures"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

N_EPISODES = 50
N_REPLICATIONS = 10
EVAL_SEED = 900_000
COLORS = {"Surplus": "#c0504d", "Static-τ": "#e6a23c", "Dyn-BATNA": "#2f6db5", "Dyn-τ (no hedge)": "#7aa6d6"}
METRICS = ["rho_dn", "rho_dn_deliverable", "dn_deal", "chain_impasse", "avoidable_impasse", "below_true_batna", "below_static_tau",
           "stockout", "shortfall", "pareto_efficient", "rho_up", "up_deal", "first_bid_ratio", "net_value", "reward"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(pd.Series(x).dropna(), float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired_boot(a, b, n=5000, seed=0):
    """Paired bootstrap on common-random-number episodes: mean(a-b), 95% CI, two-sided p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if np.all(d == 0):
        return 0.0, 0.0, 0.0, 1.0
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, (n, len(d))).mean(1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(p, 1 / n)


ASCII_NAMES = {"Surplus": "Surplus", "Static-τ": "Static-tau", "Dyn-BATNA": "Dyn-BATNA",
               "Dyn-τ (no hedge)": "Dyn-tau-nohedge"}
LEAD_COLS = ["agent", "condition", "scenario", "episode", "cem_seed", "replication", "kappa_true", "lambda_so",
             "baseline", "metric"]


def write_csv(df, name):
    """CSV export: ASCII agent ids, identifier columns first, floats rounded to 4 dp."""
    out = df.copy()
    for c in ("agent", "baseline"):
        if c in out:
            out[c] = out[c].map(lambda x: ASCII_NAMES.get(x, x))
    lead = [c for c in LEAD_COLS if c in out.columns]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out.round(4).to_csv(os.path.join(ROOT, "results", name), index=False)


def evaluate(policies, cfg, sampler, n, seed0, db_path=None, conditions=(False, True)):
    db = NetworkDB(db_path or ":memory:", log_actions=db_path is not None)
    env = MultiEchelonEnv(cfg, db)
    rows, ep = [], 0
    for shock in conditions:
        for i in range(n):
            sc = sample_scenario(seed0 + i, sampler, shock)  # common random numbers across agents
            for cls, theta in policies:
                r = env.run(sc, cls(theta, cfg), ep)
                r.update(agent=cls.name, condition="shock" if shock else "no-shock", scenario=i)
                if db_path is not None:
                    db.register_episode(ep, ASCII_NAMES[cls.name], r["condition"], i, r)
                rows.append(r)
                ep += 1
    db.commit()
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    cfg = Config()
    geo = load_disruptions()
    geo.to_csv(os.path.join(ROOT, "data", "mock_kaggle_geo_disruptions.csv"), index=False)
    sampler = ShockSampler(geo)

    # ------------------------------------------------------------ training
    policies, hist_rows = [], []
    for cls in AGENTS:
        best = None
        for s in range(3):
            theta, hist = cem(cls, cfg, sampler, iters=15, pop=32, n_eps=64, seed=s)
            for h in hist:
                h.update(agent=cls.name, cem_seed=s)
            hist_rows += hist
            if best is None or hist[-1]["val_reward"] > best[1]:
                best = (theta, hist[-1]["val_reward"])
        policies.append((cls, best[0]))
        print(f"[train] {cls.name:18s} theta={np.round(best[0], 3)} val_reward={best[1]:.3f}")
    hist_df = pd.DataFrame(hist_rows)
    write_csv(hist_df, "training_history.csv")

    # ------------------------------------------------------------ main 50-episode benchmark
    dbp = os.path.join(ROOT, "results", "network.db")
    if os.path.exists(dbp):
        os.remove(dbp)
    main_df = evaluate(policies, cfg, sampler, N_EPISODES, EVAL_SEED, db_path=dbp)
    write_csv(main_df, "episodes_main.csv")

    summ = []
    for (a, c), g in main_df.groupby(["agent", "condition"], sort=False):
        row = dict(agent=a, condition=c, n=len(g))
        for m in METRICS:
            mu, lo, hi = boot_ci(g[m].astype(float))
            row[m], row[m + "_lo"], row[m + "_hi"] = mu, lo, hi
        summ.append(row)
    summ = pd.DataFrame(summ)
    write_csv(summ, "summary_main.csv")

    # paired tests Dyn-BATNA vs each baseline
    tests = []
    for c in ("no-shock", "shock"):
        sub = main_df[main_df.condition == c]
        piv = {m: sub.pivot(index="scenario", columns="agent", values=m).astype(float) for m in
               ["net_value", "chain_impasse", "avoidable_impasse", "below_true_batna", "stockout", "rho_dn"]}
        for base in ("Surplus", "Static-τ"):
            for m, p in piv.items():
                d, lo, hi, pv = paired_boot(p["Dyn-BATNA"], p[base])
                n_pairs = int((p["Dyn-BATNA"].notna() & p[base].notna()).sum())
                tests.append(dict(condition=c, baseline=base, metric=m, n_pairs=n_pairs,
                                  diff_dyn_minus_baseline=d, ci95_lo=lo, ci95_hi=hi, p_two_sided=pv))
    tests = pd.DataFrame(tests)
    write_csv(tests, "paired_tests.csv")

    # ------------------------------------------------------------ replications (robustness)
    reps = []
    for k in range(N_REPLICATIONS):
        d = evaluate(policies, cfg, sampler, N_EPISODES, EVAL_SEED + 10_000 * (k + 1))
        agg = d.groupby(["agent", "condition"])[METRICS].mean().reset_index()
        agg["replication"] = k
        reps.append(agg)
    reps = pd.concat(reps)
    write_csv(reps, "replications.csv")
    rep_summary = reps.groupby(["agent", "condition"])[METRICS].agg(["mean", "std"])

    # ------------------------------------------------------------ sensitivity: true kappa sweep
    sens = []
    for kt in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        cf = Config(kappa=kt, kappa_belief=cfg.kappa)  # agents keep kappa=0.6 belief (misspecified)
        d = evaluate(policies, cf, sampler, 200, 700_000, conditions=(True,))
        g = d.groupby("agent")[["net_value", "avoidable_impasse", "below_true_batna", "stockout"]].mean()
        g["kappa_true"] = kt
        sens.append(g.reset_index())
    sens = pd.concat(sens)
    write_csv(sens, "sensitivity_kappa.csv")

    # sensitivity: stockout penalty weight lambda_so on evaluation net value
    lam_rows = []
    for lam in (0.0, 0.5, 1.0, 2.0):
        cf = Config(lambda_so=lam)
        d = evaluate(policies, cf, sampler, 200, 710_000, conditions=(True,))
        g = d.groupby("agent")["net_value"].mean().rename("net_value").reset_index()
        g["lambda_so"] = lam
        lam_rows.append(g)
    lam_df = pd.concat(lam_rows)
    write_csv(lam_df, "sensitivity_lambda.csv")

    # ------------------------------------------------------------ figures
    fig_geodata(geo, sampler)
    fig_main(summ)
    fig_dynamics(main_df, policies, cfg, sampler, sens)
    fig_training(hist_df)

    # ------------------------------------------------------------ json for the proposal
    con = __import__("sqlite3").connect(dbp)
    db_stats = dict(
        action_rows=con.execute("SELECT COUNT(*) FROM action_log").fetchone()[0],
        capacity_queries=con.execute("SELECT COUNT(*) FROM action_log WHERE action='[QUERY_CAPACITY]'").fetchone()[0],
        invalid_actions=con.execute("SELECT COUNT(*) FROM action_log WHERE valid=0").fetchone()[0],
        upstream_contracts=con.execute("SELECT COUNT(*) FROM upstream_contracts").fetchone()[0],
        downstream_contracts=con.execute("SELECT COUNT(*) FROM downstream_contracts").fetchone()[0],
    )
    out = dict(
        config=cfg.__dict__,
        policies={cls.name: dict(zip(["a_u", "beta_u", "a_d", "beta_d"], np.round(th, 4).tolist()))
                  for cls, th in policies},
        main=summ.round(4).to_dict(orient="records"),
        paired_tests=tests.round(4).to_dict(orient="records"),
        replications={f"{a}|{c}": {m: [round(rep_summary.loc[(a, c), (m, 'mean')], 4),
                                       round(rep_summary.loc[(a, c), (m, 'std')], 4)] for m in METRICS}
                      for a, c in rep_summary.index},
        sensitivity_kappa=sens.round(4).to_dict(orient="records"),
        sensitivity_lambda=lam_df.round(4).to_dict(orient="records"),
        geodata=dict(n_events=len(geo), types=geo.disruption_type.value_counts().to_dict(),
                     shock_severity=main_df[main_df.condition == "shock"].groupby("scenario").severity.first()
                     .describe().round(3).to_dict()),
        db=db_stats,
        runtime_sec=round(time.time() - t0, 1),
    )
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    cols = ["agent", "condition", "rho_dn", "dn_deal", "chain_impasse", "avoidable_impasse", "below_true_batna",
            "stockout", "pareto_efficient", "rho_up", "up_deal", "first_bid_ratio", "rho_dn_deliverable", "net_value"]
    print(summ[cols].round(3).to_string(index=False))
    print(tests[tests.metric == "net_value"].round(4).to_string(index=False))
    build_review_package()
    print(f"done in {time.time() - t0:.1f}s")


REVIEW_FILES = ["summary_main.csv", "paired_tests.csv", "replications.csv", "episodes_main.csv",
                "sensitivity_kappa.csv", "sensitivity_lambda.csv", "training_history.csv", "network.db"]


def build_review_package():
    """Copy reviewer-facing artifacts next to the hand-written DATA_DICTIONARY.md."""
    pkg = os.path.join(ROOT, "review_package")
    os.makedirs(os.path.join(pkg, "data"), exist_ok=True)
    os.makedirs(os.path.join(pkg, "figures"), exist_ok=True)
    for f in REVIEW_FILES:
        shutil.copy2(os.path.join(ROOT, "results", f), os.path.join(pkg, "data", f))
    for f in os.listdir(os.path.join(ROOT, "figures")):
        shutil.copy2(os.path.join(ROOT, "figures", f), os.path.join(pkg, "figures", f))
    shutil.copy2(os.path.join(ROOT, "data", "mock_kaggle_geo_disruptions.csv"),
                 os.path.join(pkg, "data", "mock_geo_disruptions_SYNTHETIC.csv"))
    shutil.copy2(os.path.join(ROOT, "Research_Proposal.md"), os.path.join(pkg, "Research_Proposal.md"))


# ======================================================================== figures
def fig_geodata(geo, sampler):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    types = geo.disruption_type.unique()
    cmap = plt.get_cmap("tab10")
    for k, t in enumerate(sorted(types)):
        g = geo[geo.disruption_type == t]
        ax.scatter(g.longitude % 360, g.latitude, s=4 + 30 * g.severity, alpha=0.35, color=cmap(k),
                   label=t.replace("_", " "))
    p = PORTS.set_index("port")
    for sp, ep in [("Shanghai", "Los Angeles"), ("Ho Chi Minh City", "Oakland"), ("Chennai", "Houston"),
                   ("Manzanillo", "Houston")]:
        xs = [p.loc[sp, "lon"] % 360, p.loc[ep, "lon"] % 360, DISTRIBUTOR[2] % 360]
        ys = [p.loc[sp, "lat"], p.loc[ep, "lat"], DISTRIBUTOR[1]]
        ax.plot(xs, ys, "k--", lw=0.8, alpha=0.7)
    ax.scatter([DISTRIBUTOR[2] % 360], [DISTRIBUTOR[1]], marker="*", s=160, color="k", zorder=5,
               label="Phoenix DC (agent)")
    ax.set_xlabel("longitude (°E, Pacific-centred)"); ax.set_ylabel("latitude")
    ax.set_title(f"Mock geospatial disruption table (n={len(geo)} events, 2019–2025); marker size ∝ severity")
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=0.8, frameon=False)
    fig.savefig(os.path.join(ROOT, "figures", "fig1_geodata.png"))
    plt.close(fig)


def _bars(ax, summ, metric, title, pct=False, agents=None):
    agents = agents or list(COLORS)
    conds = ["no-shock", "shock"]
    w = 0.8 / len(agents)
    for k, a in enumerate(agents):
        vals, err = [], [[], []]
        for c in conds:
            r = summ[(summ.agent == a) & (summ.condition == c)].iloc[0]
            m = r[metric] * (100 if pct else 1)
            vals.append(m)
            err[0].append(m - r[metric + "_lo"] * (100 if pct else 1))
            err[1].append(r[metric + "_hi"] * (100 if pct else 1) - m)
        x = np.arange(len(conds)) + (k - (len(agents) - 1) / 2) * w
        ax.bar(x, vals, w, yerr=err, capsize=2, color=COLORS[a], label=a, error_kw=dict(lw=0.7))
        for xi, v in zip(x, vals):
            ax.text(xi, v + (2 if pct else 0.02), f"{v:.0f}" if pct else f"{v:.2f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(conds)), conds)
    ax.set_title(title)


def fig_main(summ):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
    _bars(axs[0, 0], summ, "rho_dn", "(a) Downstream bargained ratio ρ (closed deals)")
    axs[0, 0].set_ylim(0, 0.9)
    _bars(axs[0, 1], summ, "chain_impasse", "(b) Supply-chain impasse rate (%)", pct=True)
    _bars(axs[1, 0], summ, "below_true_batna", "(c) Deal-closing bias: deals below true BATNA (%)", pct=True)
    _bars(axs[1, 1], summ, "stockout", "(d) Inventory stockout rate (%)", pct=True)
    for ax in axs[:, 1].tolist() + [axs[1, 0]]:
        ax.set_ylim(0, 105)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Preliminary benchmark: {N_EPISODES} paired episodes per condition (95% bootstrap CIs)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(ROOT, "figures", "fig2_main_results.png"))
    plt.close(fig)


def fig_dynamics(main_df, policies, cfg, sampler, sens):
    fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.6))
    ax = axs[0]
    tt = np.linspace(0, cfg.T - 1, 200)
    for s, ts in [(0.25, 2), (0.45, 1), (0.7, 3)]:
        g = np.where(tt >= ts, 1 - np.exp(-cfg.lam * (tt - ts + 1)), 0)
        ax.plot(tt, np.maximum(cfg.tau_min, cfg.tau0 - cfg.kappa * s * g), label=f"$z$={s}, $t_s$={ts}")
    ax.axhline(cfg.tau0, color="k", ls="--", lw=0.8, label="static τ=0.4")
    ax.set_xlabel("turn $t$"); ax.set_ylabel("τ(z_t, t)"); ax.set_ylim(0, 0.45)
    ax.set_title("(a) Dynamic threshold τ(z_t, t)"); ax.legend(fontsize=6, frameon=False)

    ax = axs[1]
    d = main_df[(main_df.condition == "shock") & main_df.dn_deal]
    for a in ["Surplus", "Static-τ", "Dyn-BATNA"]:
        g = d[d.agent == a]
        ax.scatter(g.tau_true_close, g.rho_dn, s=10, alpha=0.7, color=COLORS[a], label=a)
    ax.plot([0, 0.45], [0, 0.45], "k:", lw=0.8)
    ax.set_xlabel("true τ at close"); ax.set_ylabel("ρ of closed deal")
    ax.set_title("(b) Closed deals under shock"); ax.legend(fontsize=6, frameon=False)

    ax = axs[2]
    for a in ["Surplus", "Static-τ", "Dyn-BATNA"]:
        g = sens[sens.agent == a]
        ax.plot(g.kappa_true, 100 * g.below_true_batna, "o-", ms=3, color=COLORS[a], label=a)
    ax.axvline(cfg.kappa, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("true κ (agent assumes κ=0.6)"); ax.set_ylabel("deals below true BATNA (%)")
    ax.set_title("(c) κ misspecification"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig3_dynamics_sensitivity.png"))
    plt.close(fig)


def fig_training(hist):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for a, g in hist.groupby("agent", sort=False):
        m = g.groupby("iteration").val_net_value.agg(["mean", "min", "max"])
        axs[0].plot(m.index, m["mean"], color=COLORS[a], label=a)
        axs[0].fill_between(m.index, m["min"], m["max"], color=COLORS[a], alpha=0.15)
        f = g.groupby("iteration").a_d.mean()
        axs[1].plot(f.index, f.values, color=COLORS[a], label=a)
    axs[0].set_xlabel("CEM iteration"); axs[0].set_ylabel("validation net value")
    axs[0].set_title("(a) Optimisation under each reward (3 seeds)")
    axs[1].set_xlabel("CEM iteration"); axs[1].set_ylabel("opening aspiration $a_d$")
    axs[1].set_title("(b) Learned downstream opening aspiration"); axs[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig4_training.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
