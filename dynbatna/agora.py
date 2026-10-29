"""AgoraBench single-product evaluation with the Dynamic-BATNA layer.

Scope: the five public single-product market regimes (vanilla, negative,
monopoly, installment, deceptive), ten products each. The four "several"
multi-product regimes are intentionally excluded because their acquired-item
similarity term requires a different, item-aware BATNA definition.

The source repository is pinned as third_party/Agorabench. AgoraBench does not
publish a train/test split for these 50 scenario definitions, so this is an
external benchmark panel with repeated model rollouts, not training data.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict

import pandas as pd

from .config import AGORA_DATA, AGORA_MANIFEST, DYN_DATA, MAX_TURNS, RESULTS
from .episode import CONDITIONS, run_dyn_episode
from .evaluate import _boot, _paired, analyze, frame, make_gen
from .events import ShockSource, generate_schedule, llm_notice, template_notice
from .llm import azure_generator
from .vendor.envs import get_env
from .vendor.manifest import build_manifest, load_manifest

RUNS = os.path.join(RESULTS, "runs")
LAYER_PATH = os.path.join(DYN_DATA, "agora.test.jsonl")
SOURCE_COMMIT = "fe5c64097cd341efaaa7105dfffcbc0b45d18833"
MARKETS = ("vanilla", "negative", "monopoly", "installment", "deceptive")


def prepare(repeats=3, seed=42, use_llm=True):
    """Freeze the panel and build one paired event schedule per source scenario."""
    env = get_env("agora")
    scenarios = env.load_scenarios(AGORA_DATA)
    manifest = build_manifest("agora", env, scenarios, seed=seed, num_episodes=len(scenarios), repeats=repeats)
    # AgoraBench's native loop has the seller respond first to a buyer greeting.
    for entry in manifest["episodes"]:
        entry["learner_goes_first"] = False
    manifest["source_repository"] = "https://github.com/ericoh929/Agorabench"
    manifest["source_commit"] = SOURCE_COMMIT
    manifest["scope"] = "five single-product market regimes; multi-product regimes excluded"
    os.makedirs(DYN_DATA, exist_ok=True)
    with open(AGORA_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)

    gen = azure_generator(label="curation-gpt-5-mini", max_tokens=200) if use_llm else None
    rng = random.Random(f"{seed}-agora-test-events")
    shocks = ShockSource(seed=seed)
    rows, seen = [], set()
    for entry in manifest["episodes"]:
        sid = entry["scenario_id"]
        if sid in seen:
            continue
        seen.add(sid)
        scenario = scenarios[entry["scenario_index"]]
        layer = generate_schedule("agora", scenario, "buyer", rng, shocks)
        for event in layer["events"]:
            event["notice_template"] = template_notice("agora", event, layer["kind"])
            if gen:
                text, ok = llm_notice(gen, "agora", event, layer["kind"])
                event["notice"] = text if ok else event["notice_template"]
                event["notice_source"] = "llm" if ok else "template"
            else:
                event["notice"] = event["notice_template"]
                event["notice_source"] = "template"
        rows.append(
            {
                "dataset": "agora",
                "split": "test",
                "episode_id": entry["episode_id"],
                "scenario_id": sid,
                "scenario_index": entry["scenario_index"],
                "learner_agent_id": "buyer",
                "market": scenario["market"],
                "product": scenario["product"],
                **layer,
            }
        )
    with open(LAYER_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    stats = {
        "source_commit": SOURCE_COMMIT,
        "unique_scenarios": len(rows),
        "rollout_repeats": repeats,
        "manifest_episodes": len(manifest["episodes"]),
        "markets": pd.Series([r["market"] for r in rows]).value_counts().to_dict(),
        "with_events": sum(bool(r["events"]) for r in rows),
        "notices": sum(len(r["events"]) for r in rows),
        "llm_notices": sum(e["notice_source"] == "llm" for r in rows for e in r["events"]),
    }
    with open(os.path.join(DYN_DATA, "agora.stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    if gen:
        with open(os.path.join(DYN_DATA, "agora.curation_usage.json"), "w") as f:
            json.dump(gen.usage, f, indent=1)
    print(json.dumps(stats, indent=1))


def _load():
    env = get_env("agora")
    scenarios = env.load_scenarios(AGORA_DATA)
    manifest = load_manifest(AGORA_MANIFEST, "agora", env, scenarios)
    rows = [json.loads(line) for line in open(LAYER_PATH)]
    layer = {(r["scenario_id"], r["learner_agent_id"]): r for r in rows}
    return env, scenarios, manifest, layer


def run(args):
    if not os.path.exists(AGORA_MANIFEST) or not os.path.exists(LAYER_PATH):
        prepare(repeats=args.repeats, seed=args.seed, use_llm=not args.no_llm_notices)
    env, scenarios, manifest, layer = _load()
    name = args.name or f"agora_gpt5mini_{time.strftime('%Y%m%d_%H%M')}"
    out = os.path.join(RUNS, name)
    os.makedirs(out, exist_ok=True)
    learner = make_gen(args.learner, label=args.learner_label, max_tokens=args.max_tokens)
    opponent = make_gen(args.opponent, max_tokens=args.max_tokens)
    entries = manifest["episodes"]
    if args.limit:
        entries = entries[: args.limit]
    meta = {
        "name": name,
        "dataset": "AgoraBench",
        "scope": "single-product regimes",
        "source_repository": "https://github.com/ericoh929/Agorabench",
        "source_commit": SOURCE_COMMIT,
        "source_scenarios": 50,
        "repeats": manifest["repeats"],
        "conditions": args.conditions,
        "learner": learner.settings(),
        "opponent": opponent.settings(),
        "max_turns": MAX_TURNS,
        "protocol": "rlvr-batna explicit-price-v3 + private event notices",
        "note": "AgoraBench economic scenarios/prompts retained; native actions mapped to shared protocol",
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    path = os.path.join(out, "episodes.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(d["episode_id"], d["condition"]) for d in map(json.loads, open(path))}
    jobs = []
    for entry in entries:
        for condition in args.conditions:
            if (entry["episode_id"], condition) not in done:
                jobs.append((entry, condition))
    print(f"{len(jobs)} AgoraBench episodes -> {out}", flush=True)
    started = time.time()

    def work(job):
        entry, condition = job
        scenario = deepcopy(scenarios[entry["scenario_index"]])
        return run_dyn_episode(
            env,
            "agora",
            learner,
            opponent,
            scenario,
            layer[(entry["scenario_id"], "buyer")],
            entry,
            condition,
            learner.label,
            opponent.label,
            seed_key=f"agora|{entry['scenario_id']}|r{entry['repeat']}|{condition}",
        )

    completed = 0
    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(work, j) for j in jobs]):
            try:
                episode = future.result()
            except Exception as exc:  # noqa: BLE001
                print("episode failed:", str(exc)[:200], flush=True)
                continue
            f.write(json.dumps(asdict(episode)) + "\n")
            f.flush()
            completed += 1
            if completed % 25 == 0:
                print(f"{completed}/{len(jobs)} done ({time.time() - started:.0f}s)", flush=True)
    with open(os.path.join(out, "usage.json"), "w") as f:
        json.dump(
            {"learner": learner.usage, "opponent": opponent.usage, "seconds": time.time() - started},
            f,
            indent=1,
        )
    analyze_agora(out)


def analyze_agora(run_dir):
    analyze(run_dir)
    df = frame(run_dir)
    metrics = [
        "deal_rate",
        "bargained_ratio",
        "negotiation_power",
        "agora_buyer_metric",
        "reward_dynamic_tau",
        "deal_below_batna",
        "decision_accuracy",
        "regret",
    ]
    rows = []
    for (market, condition), group in df.groupby(["market", "condition"]):
        row = {"market": market, "condition": condition, "n": len(group)}
        for metric in metrics:
            mean, lo, hi, n = _boot(group[metric])
            row[metric] = round(mean, 4) if n else None
            row[metric + "_ci"] = f"[{lo:.3f}, {hi:.3f}]" if n else None
        rows.append(row)
    market_summary = pd.DataFrame(rows)
    market_summary.to_csv(os.path.join(run_dir, "summary_by_market.csv"), index=False)

    tests = []
    for market in list(MARKETS) + ["ALL"]:
        sub = df if market == "ALL" else df[df.market == market]
        for a, b in (("dynamic", "static"), ("dynamic", "none"), ("static", "none")):
            A = sub[sub.condition == a].set_index("episode_id")
            B = sub[sub.condition == b].set_index("episode_id")
            ids = A.index.intersection(B.index)
            for metric in metrics:
                diff, lo, hi, p, n = _paired(A.loc[ids, metric], B.loc[ids, metric])
                tests.append(
                    {
                        "market": market,
                        "comparison": f"{a} - {b}",
                        "metric": metric,
                        "n_pairs": n,
                        "diff": round(diff, 4) if n else None,
                        "ci_lo": round(lo, 4) if n else None,
                        "ci_hi": round(hi, 4) if n else None,
                        "p": round(p, 4) if n else None,
                    }
                )
    pd.DataFrame(tests).to_csv(os.path.join(run_dir, "paired_tests_by_market.csv"), index=False)
    _make_figures(run_dir, market_summary)
    print("\n== AgoraBench by market ==")
    print(market_summary[["market", "condition", "n"] + metrics].to_string(index=False))
    return market_summary


def _make_figures(run_dir, market_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = os.path.join(os.path.dirname(os.path.dirname(run_dir)), "..", "..", "figures", "dynbatna")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    conditions = ("none", "static", "dynamic")
    colors = {"none": "#9e9e9e", "static": "#e07b39", "dynamic": "#2b7bba"}
    labels = {"none": "no notices", "static": "notices + static BATNA", "dynamic": "notices + dynamic BATNA"}
    metrics = [
        ("deal_rate", "Deal rate"),
        ("agora_buyer_metric", "AgoraBench buyer metric"),
        ("deal_below_batna", "Deals below current BATNA"),
        ("decision_accuracy", "Decision accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(MARKETS))
    width = 0.25
    for ax, (metric, title) in zip(axes.flat, metrics):
        for i, condition in enumerate(conditions):
            sub = market_summary[market_summary.condition == condition].set_index("market").reindex(MARKETS)
            values = sub[metric].astype(float).values
            ci = np.array(
                [[float(v) for v in text.strip("[]").split(",")] for text in sub[metric + "_ci"]]
            )
            error = np.abs(ci.T - values)
            ax.bar(x + (i - 1) * width, values, width, yerr=error, capsize=2,
                   color=colors[condition], label=labels[condition])
        ax.set_xticks(x, MARKETS, rotation=15)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("AgoraBench single-product regimes (30 rollouts per market and condition)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig11_agorabench_by_market.png"), dpi=160)
    plt.close(fig)

    from .figures import fig_forest
    event_tests = pd.read_csv(os.path.join(run_dir, "paired_tests_event_episodes.csv"))
    fig_forest(
        event_tests,
        os.path.join(out, "fig12_agorabench_dynamic_vs_static_events.png"),
        title="AgoraBench: dynamic − static framing, event episodes",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-llm-notices", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    r.add_argument("--learner", default="azure:gpt-5-mini")
    r.add_argument("--learner-label", default=None)
    r.add_argument("--opponent", default="azure:gpt-5-mini")
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--name", default="agora_gpt5mini")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--no-llm-notices", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.repeats, args.seed, not args.no_llm_notices)
    elif args.cmd == "run":
        run(args)
    else:
        analyze_agora(args.run)
