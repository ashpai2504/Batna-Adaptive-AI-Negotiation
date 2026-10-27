"""Measure real shock severity and dynamics from IMF PortWatch and write the calibration.

Run scripts/fetch_portwatch.py first. Outputs
  data/real/measured_port_impacts.csv   one row per (event, affected port) with drop, placebo drop, recovery
  data/real/real_port_disruptions.csv   the same impacts in the simulator's disruption-table schema
  results/real/calibration.json         placebo test, onset/recovery rates, lambda per negotiation turn
  figures/real/fig5_real_calibration.png
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from negotiation_sim import realdata as rd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "real")
FIG = os.path.join(ROOT, "figures", "real")
DAYS_PER_TURN = 1.0


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    imp = rd.build_impacts()
    table = rd.to_sim_table(imp)
    table.to_csv(os.path.join(rd.REAL, "real_port_disruptions.csv"), index=False)

    test = rd.placebo_test(imp)
    a_day, k_peak, curve = rd.fit_onset_rate(imp)
    mu_day, n_rec = rd.fit_recovery_rate(imp)
    mat = imp[imp.severity >= 0.2]
    out = dict(
        source="IMF PortWatch disruptions database + Daily_Ports_Data (GDACS events), fetched by scripts/fetch_portwatch.py",
        n_events_total=132, n_event_port_links=490,
        n_measured_pairs=len(imp), n_measured_events=int(imp.event_id.nunique()),
        n_material_pairs=len(mat), min_baseline_calls_per_day=rd.MIN_BASELINE,
        placebo_median_drop_d0=round(imp.attrs["d0"], 4),
        placebo_test=test,
        severity=imp.severity.describe().round(4).to_dict(),
        by_type=imp.groupby("event_type").agg(n=("severity", "size"), mean_severity=("severity", "mean"),
                                              mean_drop=("drop", "mean")).round(4).to_dict(orient="index"),
        onset_rate_per_day=round(a_day, 4), peak_day=k_peak,
        recovery_rate_per_day=round(mu_day, 4), n_recovery=n_rec,
        median_time_to_trough_days=float(mat.time_to_trough_days.median()),
        median_recovery_days=float(mat.recovery_days.median()),
        days_per_turn=DAYS_PER_TURN,
        lambda_per_turn=round(a_day * DAYS_PER_TURN, 4),
        lambda_hand_picked_v01=1.0,
    )
    with open(os.path.join(RES, "calibration.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: out[k] for k in ("n_measured_pairs", "n_measured_events", "placebo_test",
                                          "onset_rate_per_day", "peak_day", "recovery_rate_per_day",
                                          "lambda_per_turn")}, indent=2, default=float))
    figure(imp, test, a_day, k_peak, curve)


def figure(imp, test, a_day, k_peak, curve):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    ax = axs[0]
    bins = np.linspace(0, 1, 21)
    d = imp.dropna(subset=["placebo_drop"])
    ax.hist(d.placebo_drop, bins, alpha=0.6, color="#9e9e9e", label=f"placebo (same port, 1 yr earlier), mean {test['mean_placebo_drop']:.2f}")
    ax.hist(d["drop"], bins, alpha=0.6, color="#c0504d", label=f"during event, mean {test['mean_event_drop']:.2f}")
    ax.set_xlabel("drop in 7-day port calls vs baseline"); ax.set_ylabel("event-port pairs")
    ax.set_title(f"(a) Real events vs placebo (n={test['n_pairs']})\ndiff {test['diff']:+.3f}, 95% CI [{test['ci95'][0]:.3f}, {test['ci95'][1]:.3f}]")
    ax.legend(fontsize=6.5, frameon=False)

    ax = axs[1]
    x = np.arange(len(curve))
    ax.plot(x, curve, "o-", ms=3, color="#2f6db5", label="mean daily deficit (material impacts)")
    xf = np.arange(k_peak + 1)
    ax.plot(xf, 1 - np.exp(-a_day * (xf + 1)), "k--", lw=1, label=f"fit $1-e^{{-a(k+1)}}$, a={a_day:.2f}/day")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("days since event onset"); ax.set_ylabel("deficit / peak deficit")
    ax.set_title("(b) Shock onset is gradual, peaks ~day 6"); ax.legend(fontsize=6.5, frameon=False)

    ax = axs[2]
    for code, g in imp.groupby("event_type"):
        ax.scatter(g.severity, g.recovery_days, s=10, alpha=0.6, label=f"{rd.EVENT_TYPES.get(code, code)} (n={len(g)})")
    ax.set_xlabel("measured severity (placebo-adjusted)"); ax.set_ylabel("days to recover to 90%")
    ax.set_title("(c) Severity and recovery by event type"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_real_calibration.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
