"""Stable identities and JSON round trips for loaded negotiation scenarios."""

from __future__ import annotations

import hashlib
import json

from .envs import _JIUser


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def serialize_scenario(dataset: str, scenario: dict) -> dict:
    if dataset == "ji":
        return {
            **scenario,
            "users": {k: v.context for k, v in scenario["users"].items()},
        }
    return scenario


def deserialize_scenario(dataset: str, scenario: dict) -> dict:
    if dataset == "ji":
        if not all(isinstance(v, dict) for v in scenario["users"].values()):
            raise ValueError(
                "Legacy JI log contains stringified users; cannot restore scenario"
            )
        return {
            **scenario,
            "users": {k: _JIUser(v) for k, v in scenario["users"].items()},
        }
    return scenario


def scenario_id(dataset: str, scenario: dict) -> str:
    payload = serialize_scenario(dataset, scenario)
    if dataset == "dnd":
        # The same allocation problem viewed from the other seat is one scenario.
        payload = {
            "counts": scenario["counts"],
            "agents": sorted(
                scenario["agents"].values(), key=lambda a: json.dumps(a, sort_keys=True)
            ),
        }
    return digest({"dataset": dataset, "scenario": payload})
