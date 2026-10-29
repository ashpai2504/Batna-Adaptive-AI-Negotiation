"""Dynamic-BATNA dataset layer: event schedules and reservation trajectories b(t) for rlvr-batna scenarios.

Implements Stage 1 of the design note (dynamic-batna-dataset-design.md):
  * each episode carries an event schedule E = {(turn_k, e_k)} delivered privately to the learner;
  * the learner's reservation (outside-option) value r(t), in learner points, follows from the schedule;
  * magnitudes and timing of supply-shock events come from IMF PortWatch measured port impacts
    (data/real/measured_port_impacts.csv, results/real/calibration.json);
  * the natural-language notice for each event is written by an LLM from the structured record and
    mechanically verified (the new reservation value must appear verbatim); templates are the fallback.

Reservation convention (all datasets): r(t) is measured in the *static* scenario's learner points, so that
"accept iff u(offer) >= r(t)" is one rule everywhere.
  price (learner = buyer):  u(P) = B0 - P;  the outside option is buying elsewhere at B(t);  r(t) = B0 - B(t).
  multi-issue:              u(deal) = points;  r(0) = TAU0 * max_points (rlvr-batna's threshold);  r(t) moves.
Direction: "up" = the learner's alternative improved (r rises), "down" = it worsened (r falls).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re

import pandas as pd

from .config import (DATA_PATHS, DYN_DATA, MANIFESTS, MAX_TURNS, MULTI_DATASETS, PORTWATCH_CALIB,
                     PORTWATCH_IMPACTS, PRICE_DATASETS, TAU0)
from .vendor.envs import get_env
from .vendor.manifest import load_manifest
from .vendor.scenarios import scenario_id

LEARNER_TURNS = MAX_TURNS // 2          # learner speaks on 6 of 12 turns
EVENT_TURN_RANGE = (1, 4)               # learner-turn index at which a notice can arrive (0-based)
N_EVENT_PROBS = {0: 0.25, 1: 0.45, 2: 0.30}
SHOCK_ONSET_SPLIT = 0.55                # share of a PortWatch shock delivered at onset; rest two learner turns later
                                        # (g(2d)/g(6d) with g(k)=1-exp(-0.304k): 0.46/0.84 = 0.55)
BOUNDS = {"price": (0.10, 0.80), "multi": (0.10, 0.80)}   # r(t) kept inside these fractions of the span
MIN_MOVE_FRAC = 0.05
EVENT_TYPES = ("PRICE_MOVE", "NEW_OFFER", "OFFER_EXPIRY", "RECOVERY")
TYPE_NAMES = {"TC": "tropical cyclone", "FL": "flood", "EQ": "earthquake", "DR": "drought", "OT": "port disruption"}


# ----------------------------------------------------------------------------- scenario economics
def learner_role(dataset, scenario, manifest_entry=None):
    env = get_env(dataset)
    if manifest_entry:
        return manifest_entry["learner_agent_id"]
    return env.fixed_learner_role() or scenario["agent_ids"][0]


def economics(dataset, scenario, agent_id):
    """Return (kind, r0, span, max_points, extra) for the learner."""
    env = get_env(dataset)
    if dataset in PRICE_DATASETS:
        B0, C = scenario["buyer_budget"], scenario["seller_cost"]
        return "price", 0.0, float(B0 - C), float(B0 - C), {"B0": B0, "C": C}
    mx = env.max_points(scenario, agent_id)
    return "multi", TAU0 * mx, float(mx), float(mx), {}


def display_value(kind, r, extra):
    """Human-facing number stated in notices: alternative price in $ (price) or points (multi)."""
    if kind == "price":
        return int(round(extra["B0"] - r))
    return int(round(r))


# ----------------------------------------------------------------------------- PortWatch shock source
class ShockSource:
    def __init__(self, seed=0):
        imp = pd.read_csv(PORTWATCH_IMPACTS)
        self.imp = imp[imp.severity > 0.05].reset_index(drop=True)
        cal = json.load(open(PORTWATCH_CALIB))
        self.sev_p75 = cal["severity"]["75%"]
        self.onset_rate = cal["onset_rate_per_day"]
        self.recovery_rate = cal["recovery_rate_per_day"]
        self.rng = random.Random(seed)

    def draw(self):
        row = self.imp.iloc[self.rng.randrange(len(self.imp))]
        sev = min(1.0, float(row.severity) / self.sev_p75)
        return {
            "source": "IMF PortWatch / GDACS measured port impact",
            "event_id": str(row.event_id), "event_name": str(row.event_name),
            "event_type": TYPE_NAMES.get(str(row.event_type), "disruption"),
            "port": str(row.portname), "country": str(row.country), "date": str(row.fromdate),
            "severity_raw": round(float(row.severity), 4), "severity_scaled": round(sev, 4),
            "onset_rate_per_day": self.onset_rate,
        }

    def magnitude_frac(self, sev_scaled):
        return 0.15 + 0.45 * sev_scaled                 # 15%..60% of the span, median ~0.41 at p75 severity


# ----------------------------------------------------------------------------- schedule generation
def _clip(r, kind, span, r0):
    lo_f, hi_f = BOUNDS[kind]
    if kind == "price":
        # r = B0 - B(t); B(t) in [C + lo*span, B0 + hi*span]  ->  r in [-hi*span, span - lo*span]
        return max(-hi_f * span, min(span - lo_f * span, r))
    return max(lo_f * span, min(hi_f * span, r))


def generate_schedule(dataset, scenario, agent_id, rng: random.Random, shocks: ShockSource):
    kind, r0, span, max_pts, extra = economics(dataset, scenario, agent_id)
    if kind == "multi":
        r0 = float(round(r0))
    n = rng.choices(list(N_EVENT_PROBS), weights=list(N_EVENT_PROBS.values()))[0]
    turns = sorted(rng.sample(range(EVENT_TURN_RANGE[0], EVENT_TURN_RANGE[1] + 1), n))
    directions = []
    if n == 1:
        directions = [rng.choice(["down", "up"])]
    elif n == 2:
        directions = rng.sample(["down", "up"], 2)
    events, r = [], r0
    had_new_offer = False
    for t, d in zip(turns, directions):
        if d == "down":
            if had_new_offer and rng.random() < 0.5:
                etype, mag, src = "OFFER_EXPIRY", None, {"source": "synthetic (design-note taxonomy)"}
            else:
                src = shocks.draw()
                etype, mag = "PRICE_MOVE", shocks.magnitude_frac(src["severity_scaled"]) * span
        else:
            if any(e["type"] == "PRICE_MOVE" for e in events) and rng.random() < 0.4:
                etype, mag, src = "RECOVERY", None, {"source": "IMF PortWatch recovery rate",
                                                    "recovery_rate_per_day": shocks.recovery_rate}
            else:
                src = shocks.draw()
                etype, mag = "NEW_OFFER", shocks.magnitude_frac(src["severity_scaled"]) * span
                src = {"source": "synthetic (design-note taxonomy); magnitude from PortWatch severity distribution",
                       "severity_scaled": src["severity_scaled"]}
        if etype == "OFFER_EXPIRY":
            target = next(e["r_before"] for e in reversed(events) if e["type"] == "NEW_OFFER")
            steps = [(t, target)]
        elif etype == "RECOVERY":
            shock = next(e for e in reversed(events) if e["type"] == "PRICE_MOVE")
            steps = [(t, r + 0.6 * (shock["r_before"] - r))]
        else:
            sign = -1 if d == "down" else 1
            if etype == "PRICE_MOVE" and t + 2 < LEARNER_TURNS:
                steps = [(t, r + sign * SHOCK_ONSET_SPLIT * mag), (t + 2, r + sign * mag)]
            else:
                steps = [(t, r + sign * mag)]
        for i, (st, target) in enumerate(steps):
            new_r = _clip(target, kind, span, r0)
            # Quantize so the number the learner reads is exactly the number that is scored.
            new_r = float(extra["B0"] - round(extra["B0"] - new_r)) if kind == "price" else float(round(new_r))
            if abs(new_r - r) < MIN_MOVE_FRAC * span or new_r == r:
                continue
            events.append({
                "learner_turn": st, "type": etype, "direction": d, "stage": "update" if i else "onset",
                "r_before": round(r, 4), "r_after": round(new_r, 4),
                "display_before": display_value(kind, r, extra), "display_after": display_value(kind, new_r, extra),
                "move_frac": round((new_r - r) / span, 4), "source": src, "notice": None,
            })
            r = new_r
        had_new_offer = had_new_offer or etype == "NEW_OFFER"
    events.sort(key=lambda e: e["learner_turn"])
    r_path, cur = [], r0
    by_turn = {}
    for e in events:
        by_turn[e["learner_turn"]] = e["r_after"]
    for lt in range(LEARNER_TURNS):
        cur = by_turn.get(lt, cur)
        r_path.append(round(cur, 4))
    return {"kind": kind, "r0": round(r0, 4), "span": span, "max_points": max_pts, "extra": extra,
            "events": events, "r_path": r_path}


# ----------------------------------------------------------------------------- notices (LLM + verification)
CONTEXT = {
    "amazon": "a buyer negotiating with a seller for a product; the buyer's outside option is buying the same item elsewhere",
    "craigslist": "a buyer negotiating with a seller for a used item on Craigslist; the buyer's outside option is buying a similar item from another listing",
    "agora": "a buyer negotiating with a seller in an AgoraBench product market; the buyer's outside option is buying the same product elsewhere",
    "casino": "a camper negotiating with a neighbor over food, water and firewood packages; the camper's outside option is a supply bundle offered by the camp store, valued in the camper's private points",
    "dnd": "a collector negotiating with a partner over books, hats and balls; the collector's outside option is an offer from another collector, valued in private points",
    "ji": "a job applicant negotiating an offer with a recruiter; the applicant's outside option is a competing job offer, valued in the applicant's private points",
}
UNIT = {"price": "dollars", "multi": "points"}


def template_notice(dataset, ev, kind):
    u = "$" if kind == "price" else ""
    pts = "" if kind == "price" else " points"
    after = f"{u}{ev['display_after']}{pts}"
    what = {"amazon": "the same item from another seller", "craigslist": "a comparable item from another listing",
            "agora": "the same product from another seller",
            "casino": "your fallback supply bundle (from the camp store)", "dnd": "your fallback offer (from another collector)",
            "ji": "your competing job offer"}[dataset]
    s = ev["source"]
    if ev["type"] == "PRICE_MOVE":
        if kind == "price":
            lead = (f"{s['event_type'].capitalize()} '{s['event_name']}' has disrupted {s['port']} ({s['country']}); "
                    f"shipping delays pushed the price of {what} up to {after}.")
        else:
            lead = f"Supply disruption: {what} is now worth only {after} to you."
        if ev["stage"] == "update":
            lead = f"Update: the disruption at {s['port']} deepened; {what} now costs {after}." if kind == "price" \
                else f"Update: the disruption worsened; {what} is now worth {after} to you."
        return lead
    if ev["type"] == "NEW_OFFER":
        return (f"A new alternative appeared: {what} is now available at {after}." if kind == "price"
                else f"A new alternative appeared: {what} is now worth {after} to you.")
    if ev["type"] == "OFFER_EXPIRY":
        return (f"Your earlier alternative quote has expired; {what} is back to {after}." if kind == "price"
                else f"Your earlier alternative has expired; {what} is back to {after} to you.")
    return (f"Shipping has partly recovered; {what} is back down to {after}." if kind == "price"
            else f"Conditions have partly recovered; {what} is back up to {after} to you.")


def llm_notice(gen, dataset, ev, kind):
    """Ask an LLM to write a 1-2 sentence private notice; verify the exact new value appears."""
    unit = UNIT[kind]
    val = ev["display_after"]
    must = f"${val}" if kind == "price" else f"{val} points"
    facts = {k: v for k, v in ev["source"].items() if k in ("event_type", "event_name", "port", "country", "date")}
    prompt = (
        f"Write a private environment notice (1-2 sentences, at most 45 words, plain text) for {CONTEXT[dataset]}.\n"
        f"Event type: {ev['type']} ({ev['stage']}). Direction for the negotiator: alternative got "
        f"{'better' if ev['direction'] == 'up' else 'worse'}.\n"
        f"The negotiator's alternative is now exactly: {must}. State this number verbatim as '{must}'.\n"
        + (f"Real-world cause (mention event name and port): {json.dumps(facts)}\n" if facts else "")
        + "Do not give advice, do not mention 'BATNA', do not address the other party. Output only the notice."
    )
    text = gen([{"role": "user", "content": prompt}], seed_key=json.dumps(ev, sort_keys=True))
    text = text.strip().strip('"').split("\n")[0].strip()
    ok = re.search(re.escape(must).replace(r"\ ", r"\s*"), text) is not None and len(text.split()) <= 60
    if facts and ok:
        ok = facts["event_name"].split("-")[0].lower() in text.lower() or facts["port"].lower() in text.lower()
    return text if ok else None, ok


# ----------------------------------------------------------------------------- dataset build
def build(split, datasets, seed=42, n_train=200, use_llm=True, out_dir=DYN_DATA):
    os.makedirs(out_dir, exist_ok=True)
    gen = None
    if use_llm:
        from .llm import azure_generator
        gen = azure_generator(label="curation-gpt-5-mini", max_tokens=200)
    stats = {}
    for ds in datasets:
        env = get_env(ds)
        scenarios = env.load_scenarios(DATA_PATHS[ds][split])
        if split == "test":
            man = load_manifest(MANIFESTS["main"][ds], ds, env, scenarios)
            entries = man["episodes"]
        else:
            rng0 = random.Random(f"{seed}-{ds}-train")
            idx = rng0.sample(range(len(scenarios)), min(n_train, len(scenarios)))
            entries = [{"episode_id": i, "scenario_index": j, "scenario_id": scenario_id(ds, scenarios[j]),
                        "learner_agent_id": learner_role(ds, scenarios[j]) if env.fixed_learner_role()
                        else rng0.choice(scenarios[j]["agent_ids"])} for i, j in enumerate(idx)]
        rng = random.Random(f"{seed}-{ds}-{split}-events")
        shocks = ShockSource(seed=hash((seed, ds, split)) & 0xFFFF)
        rows, n_ev, n_llm_ok, n_llm_try = [], 0, 0, 0
        for e in entries:
            sc = scenarios[e["scenario_index"]]
            sched = generate_schedule(ds, sc, e["learner_agent_id"], rng, shocks)
            for ev in sched["events"]:
                ev["notice_template"] = template_notice(ds, ev, sched["kind"])
                if gen is not None:
                    n_llm_try += 1
                    txt, ok = llm_notice(gen, ds, ev, sched["kind"])
                    n_llm_ok += ok
                    ev["notice"] = txt if ok else ev["notice_template"]
                    ev["notice_source"] = "llm" if ok else "template"
                else:
                    ev["notice"], ev["notice_source"] = ev["notice_template"], "template"
            n_ev += len(sched["events"])
            rows.append({"dataset": ds, "split": split, "episode_id": e["episode_id"],
                         "scenario_id": e["scenario_id"], "scenario_index": e["scenario_index"],
                         "learner_agent_id": e["learner_agent_id"], **sched})
        path = os.path.join(out_dir, f"{ds}.{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_with = sum(1 for r in rows if r["events"])
        up = sum(1 for r in rows for ev in r["events"] if ev["direction"] == "up" and ev["stage"] == "onset")
        dn = sum(1 for r in rows for ev in r["events"] if ev["direction"] == "down" and ev["stage"] == "onset")
        stats[ds] = {"episodes": len(rows), "with_events": n_with, "notices": n_ev, "up_events": up, "down_events": dn,
                     "llm_notices_verified": n_llm_ok, "llm_notices_attempted": n_llm_try,
                     "by_type": pd.Series([ev["type"] for r in rows for ev in r["events"] if ev["stage"] == "onset"])
                     .value_counts().to_dict()}
        print(ds, split, json.dumps(stats[ds]))
    json.dump(stats, open(os.path.join(out_dir, f"stats.{split}.json"), "w"), indent=1)
    if gen is not None:
        json.dump(gen.usage, open(os.path.join(out_dir, f"curation_usage.{split}.json"), "w"), indent=1)
    return stats


def load_layer(dataset, split) -> dict:
    """scenario_id -> dynamic layer row."""
    path = os.path.join(DYN_DATA, f"{dataset}.{split}.jsonl")
    rows = [json.loads(l) for l in open(path)]
    return {(r["scenario_id"], r["learner_agent_id"]): r for r in rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--datasets", nargs="+", default=list(DATA_PATHS))
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build(a.split, a.datasets, seed=a.seed, n_train=a.n_train, use_llm=not a.no_llm)
