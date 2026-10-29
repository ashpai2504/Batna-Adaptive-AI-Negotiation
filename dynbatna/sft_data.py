"""Rejection-sampling SFT corpus for LoRA (Stage 4 of the design note; the 'dual use' of the dataset).

1. Roll out the Dynamic-BATNA learner (gpt-5-mini) against the fixed opponent on the *training* split layer
   (data/dynbatna/<ds>.train.jsonl; JI is held out, as in rlvr-batna).
2. Keep only trajectories that are verifiably good under the dynamic reward: no normative violation on any
   learner turn, all learner turns well-formed, and either a deal with reward_dynamic_tau >= MIN_REWARD or a
   rational walk-away (standing offers all below r(t)).
3. Emit one SFT example per learner turn: the learner's full context (system prompt with the dynamic block,
   dialogue so far, private notices) -> the learner's Thought/Talk/Action response.
Output: data/dynbatna/sft/{train,valid}.jsonl in mlx-lm chat format {"messages": [...]}.
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

from .config import DATA_PATHS, DYN_DATA, RESULTS
from .episode import run_dyn_episode
from .events import load_layer
from .llm import azure_generator
from .vendor.envs import get_env

MIN_REWARD = 0.4
SFT_DIR = os.path.join(DYN_DATA, "sft")


def rollout(datasets, workers, out_path):
    learner = azure_generator(label="gpt-5-mini-sftgen", max_tokens=512)
    opponent = azure_generator(label="gpt-5-mini", max_tokens=512)
    done = set()
    if os.path.exists(out_path):
        done = {(d["dataset"], d["episode_id"]) for d in map(json.loads, open(out_path))}
    jobs = []
    for ds in datasets:
        env = get_env(ds)
        scenarios = env.load_scenarios(DATA_PATHS[ds]["train"])
        layer = load_layer(ds, "train")
        for (sid, agent), lay in layer.items():
            if (ds, lay["episode_id"]) in done:
                continue
            entry = {"episode_id": lay["episode_id"], "scenario_id": sid, "scenario_index": lay["scenario_index"],
                     "learner_agent_id": agent, "persona": "none",
                     "opponent_agent_id": next(a for a in scenarios[lay["scenario_index"]]["agent_ids"] if a != agent),
                     "learner_goes_first": bool(lay["episode_id"] % 2)}
            jobs.append((ds, env, scenarios, entry, lay))
    print(f"{len(jobs)} training rollouts", flush=True)
    t0 = time.time()

    def work(job):
        ds, env, scenarios, e, lay = job
        return run_dyn_episode(env, ds, learner, opponent, deepcopy(scenarios[e["scenario_index"]]), lay, e, "dynamic",
                               learner.label, opponent.label, seed_key=f"train|{ds}|{e['scenario_id']}")

    n = 0
    with open(out_path, "a") as f, ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(work, j) for j in jobs]):
            try:
                ep = fut.result()
            except Exception as ex:  # noqa: BLE001
                print("rollout failed:", str(ex)[:160], flush=True)
                continue
            f.write(json.dumps(asdict(ep)) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)


def keep(ep) -> bool:
    if ep["learner_format_ok"] != ep["learner_total_turns"] or ep["violations"] > 0:
        return False
    if ep["outcome"] == "deal":
        return ep["reward_dynamic_tau"] >= MIN_REWARD and not ep["deal_below_batna"]
    if ep["outcome"] == "walk_away" and ep["who_terminated"] == "learner":
        return True
    return False


def build_sft(rollout_path, seed=0, valid_frac=0.1):
    eps = [json.loads(l) for l in open(rollout_path)]
    kept = [e for e in eps if keep(e)]
    examples = []
    for e in kept:
        msgs = e["learner_messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "assistant":
                ctx = [{"role": x["role"], "content": x["content"]} for x in msgs[:i]]
                examples.append({"messages": ctx + [{"role": "assistant", "content": m["content"]}],
                                 "dataset": e["dataset"], "episode_id": e["episode_id"]})
    rng = random.Random(seed)
    ep_ids = sorted({(x["dataset"], x["episode_id"]) for x in examples})
    rng.shuffle(ep_ids)
    n_valid = max(1, int(len(ep_ids) * valid_frac))
    valid_ids = set(ep_ids[:n_valid])
    os.makedirs(SFT_DIR, exist_ok=True)
    with open(os.path.join(SFT_DIR, "train.jsonl"), "w") as ft, open(os.path.join(SFT_DIR, "valid.jsonl"), "w") as fv:
        for x in examples:
            f = fv if (x["dataset"], x["episode_id"]) in valid_ids else ft
            f.write(json.dumps({"messages": x["messages"]}) + "\n")
    stats = {"rollouts": len(eps), "kept_episodes": len(kept), "kept_by_dataset": {},
             "kept_outcomes": {}, "sft_examples": len(examples), "valid_episodes": n_valid,
             "selection_rule": {"min_reward_dynamic_tau": MIN_REWARD, "no_violations": True, "all_turns_well_formed": True,
                                "rational_learner_walkaway_kept": True}}
    for e in kept:
        stats["kept_by_dataset"][e["dataset"]] = stats["kept_by_dataset"].get(e["dataset"], 0) + 1
        stats["kept_outcomes"][e["outcome"]] = stats["kept_outcomes"].get(e["outcome"], 0) + 1
    stats["rollout_outcomes"] = {}
    for e in eps:
        stats["rollout_outcomes"][e["outcome"]] = stats["rollout_outcomes"].get(e["outcome"], 0) + 1
    json.dump(stats, open(os.path.join(SFT_DIR, "stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["amazon", "craigslist", "casino", "dnd"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(RESULTS, "sft"), exist_ok=True)
    path = os.path.join(RESULTS, "sft", "rollouts_train.jsonl")
    if not a.build_only:
        rollout(a.datasets, a.workers, path)
    build_sft(path)
