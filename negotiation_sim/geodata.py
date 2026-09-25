"""Mock Kaggle-style geospatial supply-chain disruption dataset and shock sampler.

The schema mirrors public disruption/event datasets (event id, timestamp, lat/lon,
region, disruption type, severity, duration, lead-time delay, spot-price index).
Values are synthetic; the loader is written so a real CSV with the same columns
can be dropped in via ``load_disruptions(path)``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

PORTS = pd.DataFrame(
    [
        ("Shanghai", 31.23, 121.47, "East Asia"),
        ("Shenzhen", 22.54, 114.06, "East Asia"),
        ("Busan", 35.10, 129.04, "East Asia"),
        ("Kaohsiung", 22.62, 120.30, "East Asia"),
        ("Singapore", 1.29, 103.85, "Southeast Asia"),
        ("Ho Chi Minh City", 10.82, 106.63, "Southeast Asia"),
        ("Chennai", 13.08, 80.27, "South Asia"),
        ("Suez Canal", 30.58, 32.27, "Middle East"),
        ("Rotterdam", 51.92, 4.48, "Europe"),
        ("Panama Canal", 9.08, -79.68, "Central America"),
        ("Manzanillo", 19.05, -104.31, "North America"),
        ("Los Angeles", 33.74, -118.26, "North America"),
        ("Oakland", 37.80, -122.27, "North America"),
        ("Houston", 29.76, -95.36, "North America"),
        ("Savannah", 32.08, -81.09, "North America"),
    ],
    columns=["port", "lat", "lon", "region"],
)

# type -> (Beta a, Beta b) for severity, mean duration (days), relative frequency
DISRUPTION_TYPES = {
    "port_congestion": (2.0, 4.0, 12, 0.30),
    "typhoon_hurricane": (2.5, 3.0, 6, 0.18),
    "labor_strike": (2.0, 3.0, 10, 0.12),
    "earthquake": (3.0, 2.5, 20, 0.05),
    "canal_blockage": (3.5, 2.0, 9, 0.05),
    "cyberattack": (2.0, 3.5, 5, 0.08),
    "rail_derailment": (1.5, 4.0, 4, 0.10),
    "geopolitical_sanction": (2.5, 2.5, 45, 0.12),
}

DISTRIBUTOR = ("Phoenix DC", 33.45, -112.07)
SUPPLIER_PORTS = ["Shanghai", "Shenzhen", "Busan", "Kaohsiung", "Ho Chi Minh City", "Chennai", "Manzanillo"]
ENTRY_PORTS = ["Los Angeles", "Oakland", "Houston"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def generate_mock_disruptions(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = list(DISRUPTION_TYPES)
    probs = np.array([DISRUPTION_TYPES[t][3] for t in types])
    probs /= probs.sum()
    port_idx = rng.integers(0, len(PORTS), n)
    dtype = rng.choice(types, n, p=probs)
    sev = np.array([rng.beta(*DISRUPTION_TYPES[t][:2]) for t in dtype])
    dur = np.array([rng.gamma(2.0, DISRUPTION_TYPES[t][2] / 2.0) for t in dtype])
    base = PORTS.iloc[port_idx].reset_index(drop=True)
    start = pd.Timestamp("2019-01-01")
    days = rng.integers(0, 365 * 7, n)
    df = pd.DataFrame(
        {
            "event_id": [f"EV{i:05d}" for i in range(n)],
            "event_date": start + pd.to_timedelta(days, unit="D"),
            "nearest_port": base["port"],
            "region": base["region"],
            "latitude": base["lat"] + rng.normal(0, 1.5, n),
            "longitude": base["lon"] + rng.normal(0, 1.5, n),
            "disruption_type": dtype,
            "severity": np.round(sev, 3),
            "duration_days": np.round(dur, 1),
            "lead_time_delay_days": np.round(sev * dur * rng.uniform(0.3, 0.9, n), 1),
            "spot_price_index_change": np.round(sev * rng.uniform(0.05, 0.45, n), 3),
        }
    )
    return df.sort_values("event_date").reset_index(drop=True)


def load_disruptions(path: str | None = None, seed: int = 7) -> pd.DataFrame:
    if path is None:
        return generate_mock_disruptions(seed=seed)
    return pd.read_csv(path, parse_dates=["event_date"])


@dataclass
class Lane:
    supplier_port: str
    entry_port: str
    nodes: list  # [(lat, lon), ...]


def make_lane(rng) -> Lane:
    sp = rng.choice(SUPPLIER_PORTS)
    ep = rng.choice(ENTRY_PORTS)
    p = PORTS.set_index("port")
    nodes = [(p.loc[sp, "lat"], p.loc[sp, "lon"]), (p.loc[ep, "lat"], p.loc[ep, "lon"]), DISTRIBUTOR[1:]]
    return Lane(sp, ep, nodes)


def lane_exposure(df: pd.DataFrame, lane: Lane, length_scale_km: float = 800.0) -> np.ndarray:
    """Effective severity of every event on a lane: severity * exp(-d_min / l)."""
    d = np.min(
        np.stack([haversine_km(df["latitude"].values, df["longitude"].values, la, lo) for la, lo in lane.nodes]),
        axis=0,
    )
    return df["severity"].values * np.exp(-d / length_scale_km)


class ShockSampler:
    """Draws lane-specific shocks from the disruption table."""

    def __init__(self, df: pd.DataFrame, min_effective: float = 0.2):
        self.df = df
        self.min_effective = min_effective
        self._cache: dict = {}

    def _exposure(self, lane: Lane):
        key = (lane.supplier_port, lane.entry_port)
        if key not in self._cache:
            s_eff = lane_exposure(self.df, lane)
            years = (self.df["event_date"].max() - self.df["event_date"].min()).days / 365.0
            material = s_eff >= self.min_effective
            # prior per-episode hazard: P(material event within a ~2-day negotiation window)
            rate_per_day = material.sum() / (years * 365.0)
            hazard = float(1 - np.exp(-rate_per_day * 2.0))
            mean_sev = float(s_eff[material].mean()) if material.any() else 0.0
            self._cache[key] = (s_eff, hazard, mean_sev)
        return self._cache[key]

    def prior(self, lane: Lane):
        _, hazard, mean_sev = self._exposure(lane)
        return hazard, mean_sev

    def sample(self, lane: Lane, rng):
        s_eff, _, _ = self._exposure(lane)
        idx = np.flatnonzero(s_eff >= self.min_effective)
        w = s_eff[idx] / s_eff[idx].sum()
        i = int(rng.choice(idx, p=w))
        row = self.df.iloc[i]
        return {
            "event_id": row["event_id"],
            "disruption_type": row["disruption_type"],
            "event_port": row["nearest_port"],
            "severity_eff": float(min(1.0, s_eff[i])),
            "lead_time_delay_days": float(row["lead_time_delay_days"]),
        }
