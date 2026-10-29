"""Run the dynamic-BATNA evaluation on rlvr-batna's frozen manifests and analyse it.

  python -m dynbatna.evaluate run --panel main --conditions none static dynamic --learner gpt-5-mini
  python -m dynbatna.evaluate analyze --run results/dynbatna/runs/<name>

Learner conditions share the identical event schedule per episode (data/dynbatna/<ds>.test.jsonl);
'none' is rlvr-batna's original setup (no events, constant reservation) and reproduces its baseline protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import DATA_PATHS, MANIFESTS, MAX_TURNS, RESULTS
from .episode import CONDITIONS, run_dyn_episode
from .events import load_layer
from .llm import azure_generator, local_generator
from .vendor.envs import get_env
from .vendor.manifest import load_manifest

RUNS = os.path.join(RESULTS, "runs")


def make_gen(spec: str, label=None, **kw):
    """'azure:gpt-5-mini' | 'local:<model>@http://host:port/v1'"""
    kind, _, rest = spec.partition(":")
    if kind == "azure":
        return azure_generator(label=label or rest, model=rest, **kw)
    model, _, url = rest.partition("@")
    return local_generator(model, label=label or model, base_url=url or "http://localhost:8080/v1", **kw)


def run(args):
    name = args.name or f"{args.learner.split(':')[-1].split('@')[0]}_{args.panel}_{time.strftime('%Y%m%d_%H%M')}"
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    learner_gen = make_gen(args.learner, label=args.learner_label, max_tokens=args.max_tokens)
    opponent_gen = make_gen(args.opponent, max_tokens=args.max_tokens)
    meta = {"name": name, "panel": args.panel, "datasets": args.datasets, "conditions": args.conditions,
            "learner": learner_gen.settings(), "opponent": opponent_gen.settings(), "max_turns": MAX_TURNS,
            "economic_regulation": True, "protocol": "rlvr-batna explicit-allocation-v3 + private event notices",
            "limit_per_dataset": args.limit}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    path = os.path.join(out, "episodes.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(d["dataset"], d["episode_id"], d["condition"]) for d in map(json.loads, open(path))}
    jobs = []
    for ds in args.datasets:
        env = get_env(ds)
        scenarios = env.load_scenarios(DATA_PATHS[ds]["test"])
        man = load_manifest(MANIFESTS[args.panel][ds], ds, env, scenarios)
        layer = load_layer(ds, "test")
        entries = man["episodes"][: args.limit] if args.limit else man["episodes"]
        for e in entries:
            for c in args.conditions:
                if (ds, e["episode_id"], c) in done:
                    continue
                jobs.append((ds, env, scenarios, e, layer[(e["scenario_id"], e["learner_agent_id"])], c))
    print(f"{len(jobs)} episodes to run -> {out}", flush=True)
    t0 = time.time()

    def work(job):
        ds, env, scenarios, e, lay, c = job
        sc = deepcopy(scenarios[e["scenario_index"]])
        ep = run_dyn_episode(env, ds, learner_gen, opponent_gen, sc, lay, e, c, learner_gen.label, opponent_gen.label,
                             seed_key=f"{ds}|{e['scenario_id']}|{c}")
        return ep

    n_done = 0
    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                ep = fut.result()
            except Exception as ex:  # noqa: BLE001
                print("episode failed:", str(ex)[:200], flush=True)
                continue
            f.write(json.dumps(asdict(ep)) + "\n")
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"{n_done}/{len(jobs)} done ({time.time() - t0:.0f}s, learner calls {learner_gen.usage['calls']}, "
                      f"opp calls {opponent_gen.usage['calls']})", flush=True)
    json.dump({"learner": learner_gen.usage, "opponent": opponent_gen.usage, "seconds": time.time() - t0},
              open(os.path.join(out, "usage.json"), "w"), indent=1)
    analyze(out)


# ---------------------------------------------------------------------------------------------- analysis
METRICS = ["deal_rate", "bargained_ratio", "bargained_ratio_dyn", "reward_surplus", "reward_static_tau",
           "reward_dynamic_tau", "deal_below_batna", "decision_accuracy", "regret", "first_bid_ratio",
           "pareto_efficient", "joint_surplus_norm", "negotiation_power", "acquisition_ratio",
           "agora_buyer_metric", "adaptation_latency", "learner_format_rate", "turns"]


def _boot(x, B=4000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = np.random.default_rng(seed).choice(x, (B, len(x))).mean(1)
    return (float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975)), int(len(x)))


def _paired(a, b, B=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (np.nan, np.nan, np.nan, np.nan, len(d))
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], (B, len(d)))
    p = float(((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (B + 1))
    bs = rng.choice(d, (B, len(d))).mean(1)
    return (float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), p, len(d))


def frame(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "episodes.jsonl"))]
    df = pd.DataFrame(rows)
    df["deal_rate"] = (df.outcome == "deal").astype(float)
    df["learner_format_rate"] = df.learner_format_ok / df.learner_total_turns.clip(lower=1)
    df["turns"] = df.num_turns
    for c in ("deal_below_batna", "pareto_efficient"):
        df[c] = df[c].map(lambda v: np.nan if v is None else float(v))
    for c in METRICS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    # rlvr-batna clips bargained ratios to [-1, 1] when averaging; regret is capped at one span.
    for c in ("bargained_ratio", "bargained_ratio_dyn"):
        df[c] = df[c].clip(-1, 1)
    df["regret"] = df["regret"].clip(0, 1)
    df["has_events"] = df.n_events_delivered > 0
    return df


def analyze(run_dir):
    df = frame(run_dir)
    summ, tests = [], []
    for (ds, c), g in df.groupby(["dataset", "condition"]):
        row = {"dataset": ds, "condition": c, "n": len(g)}
        for m in METRICS:
            mean, lo, hi, n = _boot(g[m])
            row[m] = round(mean, 4) if n else None
            row[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]" if n else None
        summ.append(row)
    # pooled across datasets (rlvr-batna reports an average BR across datasets)
    for c, g in df.groupby("condition"):
        row = {"dataset": "ALL", "condition": c, "n": len(g)}
        for m in METRICS:
            mean, lo, hi, n = _boot(g[m])
            row[m] = round(mean, 4) if n else None
            row[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]" if n else None
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    conds = [c for c in CONDITIONS if c in set(df.condition)]
    comps = [("dynamic", "static"), ("dynamic", "none"), ("static", "none")]
    for ds in list(df.dataset.unique()) + ["ALL"]:
        sub = df if ds == "ALL" else df[df.dataset == ds]
        for a, b in comps:
            if a not in conds or b not in conds:
                continue
            A = sub[sub.condition == a].set_index(["dataset", "episode_id"])
            Bm = sub[sub.condition == b].set_index(["dataset", "episode_id"])
            ids = A.index.intersection(Bm.index)
            for m in ("deal_rate", "bargained_ratio", "bargained_ratio_dyn", "reward_dynamic_tau", "reward_static_tau",
                      "deal_below_batna", "decision_accuracy", "regret", "first_bid_ratio"):
                d, lo, hi, p, n = _paired(A.loc[ids, m], Bm.loc[ids, m])
                tests.append({"dataset": ds, "comparison": f"{a} - {b}", "metric": m, "n_pairs": n,
                              "diff": round(d, 4) if n else None, "ci_lo": round(lo, 4) if n else None,
                              "ci_hi": round(hi, 4) if n else None, "p": round(p, 4) if n else None})
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(run_dir, "paired_tests.csv"), index=False)
    # Event-conditional view: episodes whose dynamic layer schedules >=1 event (the 'none' condition never
    # delivers notices, so membership is defined by the layer, i.e. by any condition having delivered an event).
    ev_layer = df.groupby(["dataset", "episode_id"]).has_events.max().rename("any_event")
    ev = df.join(ev_layer, on=["dataset", "episode_id"])
    ev = ev[ev.any_event]
    ev_metrics = ["deal_rate", "bargained_ratio", "bargained_ratio_dyn", "reward_static_tau", "reward_dynamic_tau",
                  "deal_below_batna", "decision_accuracy", "regret", "adaptation_latency"]
    ev_rows, ev_tests = [], []
    for ds in list(ev.dataset.unique()) + ["ALL"]:
        sub = ev if ds == "ALL" else ev[ev.dataset == ds]
        for c, g in sub.groupby("condition"):
            row = {"dataset": ds, "condition": c, "n": len(g)}
            for m in ev_metrics:
                mean, lo, hi, n = _boot(g[m])
                row[m] = round(mean, 4) if n else None
                row[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]" if n else None
            ev_rows.append(row)
        for a, b in comps:
            if a not in conds or b not in conds:
                continue
            A = sub[sub.condition == a].set_index(["dataset", "episode_id"])
            Bm = sub[sub.condition == b].set_index(["dataset", "episode_id"])
            ids = A.index.intersection(Bm.index)
            for m in ev_metrics[:-1]:
                d, lo, hi, p, n = _paired(A.loc[ids, m], Bm.loc[ids, m])
                ev_tests.append({"dataset": ds, "comparison": f"{a} - {b}", "metric": m, "n_pairs": n,
                                 "diff": round(d, 4) if n else None, "ci_lo": round(lo, 4) if n else None,
                                 "ci_hi": round(hi, 4) if n else None, "p": round(p, 4) if n else None})
    ev_summ = pd.DataFrame(ev_rows)
    ev_summ.to_csv(os.path.join(run_dir, "summary_event_episodes.csv"), index=False)
    pd.DataFrame(ev_tests).to_csv(os.path.join(run_dir, "paired_tests_event_episodes.csv"), index=False)
    pd.set_option("display.width", 250)
    print("\n== Summary (all episodes) ==")
    print(summ[["dataset", "condition", "n", "deal_rate", "bargained_ratio", "bargained_ratio_dyn", "reward_static_tau",
                "reward_dynamic_tau", "deal_below_batna", "decision_accuracy", "regret", "first_bid_ratio",
                "learner_format_rate"]].to_string(index=False))
    print("\n== Event episodes only ==")
    if ev_summ.empty:
        print("(no event episodes in this run subset)")
    else:
        print(ev_summ[["dataset", "condition", "n"] + ev_metrics].to_string(index=False))
    print("\n== Paired tests, event episodes (ALL) ==")
    if ev_tests:
        print(pd.DataFrame(ev_tests).query("dataset == 'ALL'").to_string(index=False))
    print("\n== Paired tests (ALL) ==")
    print(tests[tests.dataset == "ALL"].to_string(index=False))
    return summ, tests


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--panel", choices=["main", "baseline"], default="main")
    r.add_argument("--datasets", nargs="+", default=list(DATA_PATHS))
    r.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    r.add_argument("--learner", default="azure:gpt-5-mini")
    r.add_argument("--opponent", default="azure:gpt-5-mini")
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--name", default=None)
    r.add_argument("--learner-label", default=None, help="cache/label name for the learner (e.g. qwen1.5b-lora)")
    a_ = sub.add_parser("analyze")
    a_.add_argument("--run", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        analyze(a.run)
