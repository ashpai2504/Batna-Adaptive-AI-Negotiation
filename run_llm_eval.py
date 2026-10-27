"""Zero-shot LLM evaluation on real (PortWatch-calibrated) shocks.

Runs LLM-Surplus, LLM-Static-tau and LLM-Dyn-BATNA on the same scenarios (common random
numbers) as the rule-based agents from `run_experiments.py --data real`, and compares them.

  python run_llm_eval.py --smoke              # 1 shock episode per agent, prints the transcript
  python run_llm_eval.py --episodes 30        # main LLM run (resumable)
  python run_llm_eval.py --fake --episodes 2  # offline plumbing test, no API calls

Outputs (results/llm/): episodes_llm.jsonl (one line per episode incl. transcript),
episodes_llm.csv, summary_llm.csv, paired_llm.csv, usage.json; figures/real/fig6_llm_results.png
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from negotiation_sim.agents import AGENTS
from negotiation_sim.db import NetworkDB
from negotiation_sim.env import Config, MultiEchelonEnv, sample_scenario
from negotiation_sim.geodata import ShockSampler
from negotiation_sim.llm_agent import LLM_AGENTS
from negotiation_sim.realdata import load_real_disruptions

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "llm")
FIG = os.path.join(ROOT, "figures", "real")
EVAL_SEED = 900_000
METRICS = ["net_value", "dn_deal", "chain_impasse", "avoidable_impasse", "below_true_batna", "stockout",
           "rho_dn", "rho_dn_deliverable", "pareto_efficient", "up_deal", "rho_up", "first_bid_ratio",
           "invalid_actions", "format_errors", "capacity_calls", "unqueried_commit_attempts"]
RULE_TWIN = {"LLM-Surplus": "Surplus", "LLM-Static-tau": "Static-τ", "LLM-Dyn-BATNA": "Dyn-BATNA"}


class FakeChat:
    """Scripted stand-in for AzureChat: queries capacity, then accepts if the offer clears 0.4, else counters."""
    usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, messages, tools=None, seed_key=""):
        user = messages[1]["content"]
        if user.startswith("UPSTREAM"):
            offer = float(user.split("current offer: $")[1].split()[0])
            return {"content": f"Thought: counter.\nTalk: How about less?\nAction: [SUBMIT_DEAL] price:{offer * 0.9:.2f}"}
        if not any(m["role"] == "tool" for m in messages):
            return {"content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "query_capacity", "arguments": json.dumps({"kind": "effective"} if "effective" in user else {})}}]}
        score = float(user.split("(your score ")[1].split(")")[0])
        return {"content": "Thought: ok.\nTalk: Deal.\nAction: " + ("[ACCEPT_DEAL]" if score >= 0.4 else "[REJECT_DEAL]")}


def load_setup():
    with open(os.path.join(ROOT, "results", "real", "calibration.json")) as f:
        calib = json.load(f)
    with open(os.path.join(ROOT, "results", "real", "summary.json")) as f:
        summ = json.load(f)
    cfg = Config(lam=calib["lambda_per_turn"])
    sampler = ShockSampler(load_real_disruptions())
    thetas = {name: np.array([p["a_u"], p["beta_u"], p["a_d"], p["beta_d"]]) for name, p in summ["policies"].items()}
    return cfg, sampler, thetas


def run_episode(cls, cfg, sampler, shock, i, client, ep_id):
    sc = sample_scenario(EVAL_SEED + i, sampler, shock)
    db = NetworkDB(":memory:")
    env = MultiEchelonEnv(cfg, db)
    agent = cls(np.full(4, 0.5), cfg)
    agent.setup_llm(client, sc)
    r = env.run(sc, agent, ep_id)
    log = db.conn.execute("SELECT turn, channel, actor, action, payload, valid FROM action_log ORDER BY rowid").fetchall()
    r.update(agent=cls.name, condition="shock" if shock else "no-shock", scenario=i,
             event_name=sc.event.get("event_name"), event_port=sc.event.get("event_port"),
             format_errors=agent.format_errors, capacity_calls=agent.tool_calls["query_capacity"],
             score_calls=agent.tool_calls["score_package"],
             unqueried_commit_attempts=sum(1 for row in log if row[3] in ("[SUBMIT_DEAL]", "[ACCEPT_DEAL]")
                                           and row[1] == "down" and row[2] == "distributor" and row[5] == 0))
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in r.items()}, agent.transcript, log


def rule_based(cfg, sampler, thetas, n, conditions):
    rows = []
    db = NetworkDB(":memory:", log_actions=False)
    env = MultiEchelonEnv(cfg, db)
    for shock in conditions:
        for i in range(n):
            sc = sample_scenario(EVAL_SEED + i, sampler, shock)
            for cls in AGENTS[:3]:
                r = env.run(sc, cls(thetas[cls.name], cfg), len(rows))
                r.update(agent=cls.name, condition="shock" if shock else "no-shock", scenario=i)
                rows.append(r)
    return pd.DataFrame(rows)


def boot(x, n=2000, seed=0):
    x = np.asarray(pd.Series(x, dtype=float).dropna(), float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    bs = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def paired(a, b, n=5000, seed=0):
    d = (np.asarray(a, float) - np.asarray(b, float))
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return 0.0, 0.0, 0.0, 1.0, len(d)
    bs = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(p, 1 / n), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--conditions", default="no-shock,shock")
    ap.add_argument("--agents", default=",".join(c.name for c in LLM_AGENTS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--deployment", default=None)
    args = ap.parse_args()

    out_dir = os.path.join(OUT, "fake") if args.fake else OUT
    os.makedirs(out_dir, exist_ok=True)
    cfg, sampler, thetas = load_setup()
    if args.fake:
        client = FakeChat()
    else:
        from negotiation_sim.llm_client import AzureChat
        client = AzureChat(deployment=args.deployment)

    agents = [c for c in LLM_AGENTS if c.name in args.agents.split(",")]
    conds = [c == "shock" for c in args.conditions.split(",")]
    if args.smoke:
        r, tr, log = run_episode(agents[-1], cfg, sampler, True, 0, client, 0)
        for rec in tr:
            print(f"--- {rec['channel']} t={rec['t']} tools={[(c['name'], c['result']) for c in rec['tool_calls']]}")
            print(rec["reply"][-600:])
        print({k: r[k] for k in ("agent", "up_deal", "dn_deal", "rho_up", "rho_dn", "net_value", "stockout",
                                 "invalid_actions", "format_errors", "capacity_calls")})
        print("usage", client.usage)
        return

    path = os.path.join(out_dir, "episodes_llm.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line)
            done.add((d["result"]["agent"], d["result"]["condition"], d["result"]["scenario"]))
    jobs = [(cls, shock, i) for shock in conds for i in range(args.episodes) for cls in agents
            if (cls.name, "shock" if shock else "no-shock", i) not in done]
    print(f"{len(done)} episodes already done, {len(jobs)} to run with {args.workers} workers")
    lock = threading.Lock()
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex, open(path, "a") as fh:
        futs = {ex.submit(run_episode, cls, cfg, sampler, shock, i, client, k): (cls.name, shock, i)
                for k, (cls, shock, i) in enumerate(jobs)}
        for n_done, f in enumerate(as_completed(futs), 1):
            try:
                r, tr, log = f.result()
            except Exception as e:
                print("episode failed", futs[f], str(e)[:300])
                continue
            with lock:
                fh.write(json.dumps({"result": r, "transcript": tr, "action_log": log}, default=float) + "\n")
                fh.flush()
            if n_done % 10 == 0 or n_done == len(jobs):
                print(f"  {n_done}/{len(jobs)} episodes, {time.time() - t0:.0f}s, usage={client.usage}")

    summarize(out_dir, cfg, sampler, thetas, args.episodes, client)


def summarize(out_dir, cfg, sampler, thetas, n, client):
    rows = [json.loads(line)["result"] for line in open(os.path.join(out_dir, "episodes_llm.jsonl"))]
    llm = pd.DataFrame(rows)
    llm = llm[llm.scenario < n]
    conds = sorted(llm.condition.unique(), key=lambda c: c == "shock")
    rb = rule_based(cfg, sampler, thetas, n, [c == "shock" for c in conds])
    for c in ("format_errors", "capacity_calls", "unqueried_commit_attempts"):
        rb[c] = np.nan
    both = pd.concat([llm, rb], ignore_index=True)
    both.to_csv(os.path.join(out_dir, "episodes_llm.csv"), index=False)

    summ = []
    for (a, c), g in both.groupby(["agent", "condition"], sort=False):
        row = dict(agent=a, condition=c, n=len(g))
        for m in METRICS:
            if m in g:
                row[m], row[m + "_lo"], row[m + "_hi"] = boot(g[m])
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.round(4).to_csv(os.path.join(out_dir, "summary_llm.csv"), index=False)

    tests = []
    for c in conds:
        sub = both[both.condition == c]
        for m in ("net_value", "chain_impasse", "avoidable_impasse", "below_true_batna", "stockout", "rho_dn"):
            p = sub.pivot_table(index="scenario", columns="agent", values=m, aggfunc="first").astype(float)
            pairs = [("LLM-Dyn-BATNA", "LLM-Static-tau"), ("LLM-Dyn-BATNA", "LLM-Surplus"),
                     ("LLM-Static-tau", "LLM-Surplus")] + [(k, v) for k, v in RULE_TWIN.items()]
            for a, b in pairs:
                if a in p and b in p:
                    d, lo, hi, pv, k = paired(p[a], p[b])
                    tests.append(dict(condition=c, a=a, b=b, metric=m, n_pairs=k, diff_a_minus_b=d,
                                      ci95_lo=lo, ci95_hi=hi, p_two_sided=pv))
    tests = pd.DataFrame(tests)
    tests.round(4).to_csv(os.path.join(out_dir, "paired_llm.csv"), index=False)
    with open(os.path.join(out_dir, "usage.json"), "w") as f:
        json.dump(dict(usage=client.usage, deployment=getattr(client, "deployment", "fake"),
                       reasoning_effort=getattr(client, "reasoning", None), episodes_per_cell=n), f, indent=2)
    cols = ["agent", "condition", "n", "net_value", "dn_deal", "chain_impasse", "below_true_batna", "stockout",
            "rho_dn", "up_deal", "first_bid_ratio", "format_errors", "invalid_actions", "capacity_calls"]
    print(summ[[c for c in cols if c in summ]].round(3).to_string(index=False))
    print(tests[tests.metric == "net_value"].round(3).to_string(index=False))
    if "fake" not in out_dir:
        figure(summ)


def figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    order = ["Surplus", "Static-τ", "Dyn-BATNA", "LLM-Surplus", "LLM-Static-tau", "LLM-Dyn-BATNA"]
    colors = ["#e8a5a3", "#f2cd8f", "#9bbbe2", "#c0504d", "#e6a23c", "#2f6db5"]
    panels = [("net_value", "(a) Net network value V", False), ("chain_impasse", "(b) Supply-chain impasse (%)", True),
              ("below_true_batna", "(c) Deals below true walk-away (%)", True), ("stockout", "(d) Stockouts (%)", True)]
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.9))
    for ax, (m, title, pct) in zip(axs, panels):
        conds = [c for c in ("no-shock", "shock") if c in set(summ.condition)]
        w = 0.8 / len(order)
        for k, a in enumerate(order):
            for j, c in enumerate(conds):
                r = summ[(summ.agent == a) & (summ.condition == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                s = 100 if pct else 1
                ax.bar(j + (k - 2.5) * w, r[m] * s, w, color=colors[k],
                       yerr=[[max(0, (r[m] - r[m + "_lo"]) * s)], [max(0, (r[m + "_hi"] - r[m]) * s)]],
                       capsize=1.5, error_kw=dict(lw=0.6), label=a if (j == 0) else None)
        ax.set_xticks(range(len(conds)), conds)
        ax.set_title(title)
    axs[0].legend(fontsize=6, frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Zero-shot LLM agents (saturated colours) vs rule-based twins (light) on real PortWatch shocks", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG, "fig6_llm_results.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
