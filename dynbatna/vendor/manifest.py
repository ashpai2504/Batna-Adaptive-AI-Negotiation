"""Build a paired evaluation schedule before any concurrent model calls."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .scenarios import digest, scenario_id, serialize_scenario


def build_manifest(
    dataset,
    env,
    scenarios,
    *,
    seed=42,
    num_episodes=200,
    repeats=1,
    persona_weights=None,
):
    """Sample unique scenarios without replacement; repeat only when requested.

    num_episodes is the requested number of unique scenarios, capped at the
    available unique pool. Roles and first mover are balanced over each trial's
    manifest (not a full factorial within each individual scenario).
    """
    for name, value in [("num_episodes", num_episodes), ("repeats", repeats)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not scenarios:
        raise ValueError(f"No usable scenarios for {dataset}")
    pw = persona_weights or {"none": 1.0}
    if any(not math.isfinite(w) or w < 0 for w in pw.values()) or sum(pw.values()) <= 0:
        raise ValueError(
            "persona_weights must be finite, nonnegative and have positive sum"
        )
    from .runner import PERSONAS

    if set(pw) - {"none", *PERSONAS}:
        raise ValueError("Unknown evaluation persona")
    unique = {}
    for i, sc in enumerate(scenarios):
        unique.setdefault(scenario_id(dataset, sc), i)
    # Sort stable IDs so selection does not depend on source row order.
    rng = random.Random(digest([seed, dataset, "paired-evaluation-v1"]))
    selected = rng.sample(sorted(unique), min(num_episodes, len(unique)))
    entries = []
    for trial in range(repeats):
        offset = rng.randrange(2)
        roles = [(i + offset) % 2 for i in range(len(selected))]
        offset = rng.randrange(2)
        first = [bool((i + offset) % 2) for i in range(len(selected))]
        rng.shuffle(roles)
        rng.shuffle(first)
        personas = rng.choices(list(pw), weights=list(pw.values()), k=len(selected))
        for j, sid in enumerate(selected):
            index = unique[sid]
            sc = scenarios[index]
            ids = sc["agent_ids"]
            if len(ids) != 2 or len(set(ids)) != 2:
                raise ValueError("Paired evaluation requires two distinct agent IDs")
            learner = env.fixed_learner_role() or ids[roles[j]]
            if learner not in ids:
                raise ValueError("Fixed learner role is absent from the scenario")
            entries.append(
                {
                    "episode_id": len(entries),
                    "scenario_id": sid,
                    "scenario_index": index,
                    "repeat": trial,
                    "persona": personas[j],
                    "learner_agent_id": learner,
                    "opponent_agent_id": next(a for a in ids if a != learner),
                    "learner_goes_first": first[j],
                }
            )
    return {
        "schema_version": 1,
        "dataset": dataset,
        "seed": seed,
        "pool_sha256": digest([serialize_scenario(dataset, s) for s in scenarios]),
        "requested_unique_scenarios": num_episodes,
        "available_unique_scenarios": len(unique),
        "selected_unique_scenarios": len(selected),
        "repeats": repeats,
        "episodes": entries,
    }


def load_manifest(path, dataset, env, scenarios):
    """Reuse a saved schedule, failing before inference if its pool has changed."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != dataset:
        raise ValueError("Manifest schema or dataset mismatch")
    if manifest.get("pool_sha256") != digest(
        [serialize_scenario(dataset, s) for s in scenarios]
    ):
        raise ValueError("Scenario pool has changed since the manifest was written")
    entries = manifest.get("episodes", [])
    if not entries:
        raise ValueError("Manifest has no episodes")
    from .runner import PERSONAS

    pairs = set()
    for i, entry in enumerate(entries):
        index = entry["scenario_index"]
        if type(index) is not int or not 0 <= index < len(scenarios):
            raise ValueError("Invalid manifest scenario index")
        trial = entry["repeat"]
        if type(trial) is not int or not 0 <= trial < manifest["repeats"]:
            raise ValueError("Invalid manifest repeat index")
        pair = (entry["scenario_id"], trial)
        if pair in pairs:
            raise ValueError("Duplicate scenario within a manifest trial")
        pairs.add(pair)
        sc = scenarios[index]
        if entry["episode_id"] != i or entry["scenario_id"] != scenario_id(dataset, sc):
            raise ValueError("Invalid manifest episode or scenario identity")
        if {entry["learner_agent_id"], entry["opponent_agent_id"]} != set(
            sc["agent_ids"]
        ):
            raise ValueError("Manifest roles do not match scenario")
        if (
            env.fixed_learner_role()
            and entry["learner_agent_id"] != env.fixed_learner_role()
        ):
            raise ValueError("Manifest violates fixed learner role")
        if type(entry["learner_goes_first"]) is not bool:
            raise ValueError("Manifest first mover must be boolean")
        if entry["persona"] not in {"none", *PERSONAS}:
            raise ValueError("Unknown manifest persona")
    selected = {e["scenario_id"] for e in entries}
    if (
        len(selected) != manifest["selected_unique_scenarios"]
        or len(entries) != len(selected) * manifest["repeats"]
    ):
        raise ValueError("Manifest trial counts are inconsistent")
    return manifest
