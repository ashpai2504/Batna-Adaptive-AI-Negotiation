"""Real shocks from IMF PortWatch: measured port-traffic impacts of GDACS disruption events.

For every (event, affected port) pair we compare daily port calls around the event with
the port's own pre-event baseline:

  baseline   mean daily calls over [from - 56d, from - 8d]
  trough     minimum 7-day trailing mean over [from, to + 21d]
  drop       1 - trough / baseline
  recovery   days from trough until the 7-day mean is back to >= 90% of baseline

The same statistic on a placebo window (same port, same dates one year earlier) gives the
drop you see from normal week-to-week noise. Severity is the drop in excess of that noise:

  severity = clip((drop - d0) / (1 - d0), 0, 1),   d0 = median placebo drop

Only ports with a baseline of at least MIN_BASELINE calls/day are kept (too noisy otherwise).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = os.path.join(ROOT, "data", "real")
MIN_BASELINE = 2.0
BASE_WIN = (-56, -8)
POST_PAD = 21
ROLL = 7

EVENT_TYPES = {"TC": "tropical_cyclone", "EQ": "earthquake", "FL": "flood", "DR": "drought",
               "WF": "wildfire", "VO": "volcano", "OT": "other_geopolitical"}


def _series(df, start, end):
    s = df.groupby("date")["portcalls"].first().astype(float)
    idx = pd.date_range(start, end, freq="D")
    return s.reindex(idx, fill_value=0.0)


def measure_pair(daily: pd.DataFrame, fromdate, todate, shift_days: int = 0):
    """Impact statistics for one port around one event window (optionally shifted back for placebo)."""
    f = fromdate - pd.Timedelta(days=shift_days)
    t = todate - pd.Timedelta(days=shift_days)
    s = _series(daily, f + pd.Timedelta(days=BASE_WIN[0]), t + pd.Timedelta(days=POST_PAD + 45))
    base = s[f + pd.Timedelta(days=BASE_WIN[0]): f + pd.Timedelta(days=BASE_WIN[1])]
    if len(base) < 30 or base.mean() < MIN_BASELINE:
        return None
    b = base.mean()
    roll = s.rolling(ROLL, min_periods=ROLL).mean()
    win = roll[f: t + pd.Timedelta(days=POST_PAD)]
    if win.isna().all():
        return None
    trough_day = win.idxmin()
    trough = win.min()
    after = roll[trough_day:]
    back = after[after >= 0.9 * b]
    censored = back.empty
    recovery = (after.index[-1] - trough_day).days if censored else (back.index[0] - trough_day).days
    below = (win < 0.9 * b).sum()
    daily_win = s[f: t + pd.Timedelta(days=POST_PAD)]
    return dict(baseline_calls=b, trough_calls=trough, drop=float(np.clip(1 - trough / b, 0, 1)),
                time_to_trough_days=(trough_day - f).days, recovery_days=int(recovery),
                recovery_censored=bool(censored), days_below_90=int(below),
                deficit_curve=((b - daily_win.values[:30]) / b).tolist())


def measure_impacts(events, links, calls, placebo):
    ev = events.set_index("eventid")
    rows = []
    cg = dict(tuple(calls.groupby(["eventid", "portid"])))
    pg = dict(tuple(placebo.groupby(["eventid", "portid"])))
    for r in links.drop_duplicates(["eventid", "portid"]).itertuples():
        key = (r.eventid, r.portid)
        if key not in cg or r.eventid not in ev.index:
            continue
        e = ev.loc[r.eventid]
        m = measure_pair(cg[key], r.fromdate, r.todate)
        if m is None:
            continue
        p = measure_pair(pg[key], r.fromdate, r.todate, shift_days=364) if key in pg else None
        rows.append(dict(event_id=str(r.eventid), event_name=e["eventname"], event_type=e["eventtype"],
                         alert_level=e["alertlevel"], fromdate=r.fromdate, todate=r.todate,
                         portid=r.portid, portname=r.portname, country=r.country,
                         port_lat=float(r.lat), port_lon=float(r.long),
                         event_lat=float(e["lat"]), event_lon=float(e["long"]),
                         distance_km=float(r.distance_km), placebo_drop=None if p is None else p["drop"], **m))
    return pd.DataFrame(rows)


def placebo_test(imp, n=10000, seed=0):
    """Paired bootstrap: mean(event drop - placebo drop) over pairs with both measured."""
    d = imp.dropna(subset=["placebo_drop"])
    diff = (d["drop"] - d["placebo_drop"]).values
    rng = np.random.default_rng(seed)
    bs = rng.choice(diff, (n, len(diff))).mean(1)
    return dict(n_pairs=len(diff), mean_event_drop=float(d["drop"].mean()),
                mean_placebo_drop=float(d["placebo_drop"].mean()),
                median_placebo_drop=float(d["placebo_drop"].median()),
                diff=float(diff.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                p_one_sided=float(max((bs <= 0).mean(), 1 / n)),
                share_event_gt_placebo=float((diff > 0).mean()))


def mean_deficit_curve(imp, min_sev=0.2, horizon=21):
    """Average daily traffic deficit (share of baseline lost) after onset, material impacts only."""
    m = imp[imp.severity >= min_sev]
    return np.nanmean(np.vstack([np.asarray(c[:horizon], float) for c in m.deficit_curve]), axis=0)


def fit_onset_rate(imp, min_sev=0.2, horizon=21):
    """Onset phase of the mean deficit curve: fit peak * (1 - exp(-a (k + 1))) on days 0..argmax.

    Returns (a per day, peak day, mean curve normalised to its peak)."""
    curve = mean_deficit_curve(imp, min_sev, horizon)
    k_peak = int(np.argmax(curve))
    y = curve[: k_peak + 1] / curve[k_peak]
    x = np.arange(k_peak + 1)
    ks = np.linspace(0.02, 5.0, 2000)
    sse = [np.sum((y - (1 - np.exp(-k * (x + 1)))) ** 2) for k in ks]
    return float(ks[int(np.argmin(sse))]), k_peak, curve / curve[k_peak]


def fit_recovery_rate(imp, min_sev=0.2):
    """Exponential recovery mu (1/day): median of ln(10) / recovery_days over uncensored material impacts."""
    m = imp[(imp.severity >= min_sev) & (~imp.recovery_censored) & (imp.recovery_days > 0)]
    return float(np.median(np.log(10) / m.recovery_days)), len(m)


def build_impacts(save=True):
    events = pd.read_csv(os.path.join(REAL, "portwatch_events.csv"), parse_dates=["fromdate", "todate"])
    links = pd.read_csv(os.path.join(REAL, "portwatch_event_ports.csv"), parse_dates=["fromdate", "todate"])
    calls = pd.read_csv(os.path.join(REAL, "portwatch_daily_calls.csv"), parse_dates=["date"])
    placebo = pd.read_csv(os.path.join(REAL, "placebo_daily_calls.csv"), parse_dates=["date"])
    imp = measure_impacts(events, links, calls, placebo)
    d0 = float(imp["placebo_drop"].median())
    imp["severity"] = np.clip((imp["drop"] - d0) / (1 - d0), 0, 1).round(4)
    imp.attrs["d0"] = d0
    if save:
        imp.drop(columns=["deficit_curve"]).to_csv(os.path.join(REAL, "measured_port_impacts.csv"), index=False)
    return imp


def to_sim_table(imp: pd.DataFrame) -> pd.DataFrame:
    """Map measured impacts onto the simulator's disruption-table schema (one row per event x port)."""
    return pd.DataFrame({
        "event_id": imp["event_id"] + "@" + imp["portid"],
        "event_date": imp["fromdate"],
        "nearest_port": imp["portname"],
        "region": imp["country"],
        "latitude": imp["port_lat"],
        "longitude": imp["port_lon"],
        "disruption_type": imp["event_type"].map(EVENT_TYPES).fillna("other"),
        "event_name": imp["event_name"],
        "severity": imp["severity"],
        "duration_days": imp["days_below_90"],
        "recovery_days": imp["recovery_days"],
        "lead_time_delay_days": imp["days_below_90"],
        "source": "IMF PortWatch / GDACS (measured port-call drop)",
    }).sort_values("event_date").reset_index(drop=True)


def load_real_disruptions() -> pd.DataFrame:
    path = os.path.join(REAL, "real_port_disruptions.csv")
    if not os.path.exists(path):
        to_sim_table(build_impacts()).to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["event_date"])
