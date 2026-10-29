"""Paths and shared constants for the dynamic-BATNA extension of rlvr-batna."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLVR_DATA = os.path.join(ROOT, "third_party", "rlvr-batna", "data")
DYN_DATA = os.path.join(ROOT, "data", "dynbatna")
RESULTS = os.path.join(ROOT, "results", "dynbatna")
PORTWATCH_IMPACTS = os.path.join(ROOT, "data", "real", "measured_port_impacts.csv")
PORTWATCH_CALIB = os.path.join(ROOT, "results", "real", "calibration.json")

# Dhruv Bansal's (rlvr-batna) dataset paths, test and train splits.
DATA_PATHS = {
    "amazon": {"test": f"{RLVR_DATA}/ahp/test", "train": f"{RLVR_DATA}/ahp/train/ahp.train.json"},
    "craigslist": {"test": f"{RLVR_DATA}/craigslist/revision/test.json",
                   "train": f"{RLVR_DATA}/craigslist/revision/train.json"},
    "casino": {"test": f"{RLVR_DATA}/casino/ca.test.csv", "train": f"{RLVR_DATA}/casino/ca.train.csv"},
    "dnd": {"test": f"{RLVR_DATA}/dnd/dnd.test.csv", "train": f"{RLVR_DATA}/dnd/dnd.train.csv"},
    "ji": {"test": f"{RLVR_DATA}/ji/ji.test.json", "train": f"{RLVR_DATA}/ji/ji.train.json"},
}
MANIFESTS = {
    panel: {ds: f"{RLVR_DATA}/evaluation_manifests/revision/{panel}/{ds}.json" for ds in DATA_PATHS}
    for panel in ("main", "baseline")
}
PRICE_DATASETS = ("amazon", "craigslist")
MULTI_DATASETS = ("casino", "dnd", "ji")

MAX_TURNS = 12            # rlvr-batna evaluation turn limit
TAU0 = 0.4                # rlvr-batna BATNA threshold: initial reservation = TAU0 * max points (multi-issue)
BATNA_PENALTY = -0.5      # rlvr-batna penalty for a deal below threshold
