"""LLM-vs-LLM buyer/seller negotiation on real products under real (IMF PortWatch) supply shocks.

Experiment A (Dynamic BATNA): buyer framing in {Surplus, Static-BATNA, Dynamic-BATNA} x {no-shock, shock},
  seller = neutral profit-maximiser. All buyers receive identical observations (news + daily market quote);
  only the objective framing differs, mirroring the reward-framing comparison in Bansal, Yin & Wang.
Experiment B (emotional "evil twin" seller): same seller economics, but the seller is instructed to use
  emotional pressure tactics. Buyer framing in {Surplus, Dynamic-BATNA}, shock only.

Real inputs
  products : Amazon price histories (AHP; camelcamelcamel) -> current / list / lowest / highest price
  shocks   : IMF PortWatch measured port impacts -> event name/type/port, severity, onset rate (calibration.json)

Economics (all in dollars, reported normalised by the product's current price P0)
  market (outside-option) price  M_t = P0 + A * g(t),  g(t) = 1 - exp(-lam * (t - t_s + 1)) for t >= t_s
  A = (highest - P0) * min(1, severity / severity_p75)      (shock condition; A = 0 otherwise)
  buyer value V = highest price ever observed;  seller cost C = lowest price ever observed
  deal at price p on day t : buyer profit V - p,  seller profit p - C
  no deal (walk-away on day t or deadline): buyer buys at market, profit V - M_t; seller profit 0
  below-BATNA deal : p > M_t on the day of the deal

Usage
  python run_llm_negotiation.py --exp A --n 15
  python run_llm_negotiation.py --exp B --n 15
  python run_llm_negotiation.py --analyze
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from negotiation_sim.llm_client import AzureV1Chat, OllamaChat

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "llm_negotiation")
BACKEND = "ollama"
WORKERS = 1
PHYSICAL = {"electronics", "tools-home-improvement", "home-kitchen", "toys-games", "sports-outdoors", "beauty",
            "baby-products", "patio-lawn-garden", "automotive", "pet-supplies", "industrial-scientific",
            "health-personal-care"}
DAYS = 6
SHOCK_DAY = 1
TYPE_NAMES = {"TC": "tropical cyclone", "FL": "flood", "EQ": "earthquake", "DR": "drought", "OT": "port disruption"}


def money(s):
    try:
        return float(str(s).replace("$", "").replace(",", ""))
    except ValueError:
        return float("nan")


def load_products(n, seed):
    rows = []
    for f in glob.glob(os.path.join(ROOT, "data", "negotiation", "amazon_history_price", "*.json")):
        for d in json.load(open(f)):
            if d.get("category") not in PHYSICAL:
                continue
            p0, lo, hi, lst = (money(d.get(k)) for k in ("current_price", "lowest_price", "highest_price", "list_price"))
            if not all(map(math.isfinite, (p0, lo, hi))) or p0 < 15:
                continue
            if hi / p0 < 1.15 or p0 / lo < 1.05 or hi / p0 > 3:
                continue
            rows.append({"title": (d.get("amazon_title") or d["title"])[:110], "category": d["category"],
                         "P0": p0, "C": lo, "V": hi, "list": lst if math.isfinite(lst) else p0})
    df = pd.DataFrame(rows).drop_duplicates("title").reset_index(drop=True)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def load_shocks(n, seed):
    imp = pd.read_csv(os.path.join(ROOT, "data", "real", "measured_port_impacts.csv"))
    imp = imp[imp.severity > 0.05].drop_duplicates("event_id")
    cal = json.load(open(os.path.join(ROOT, "results", "real", "calibration.json")))
    sev_p75 = cal["severity"]["75%"]
    sh = imp.sample(n=n, replace=len(imp) < n, random_state=seed).reset_index(drop=True)
    return sh, cal["lambda_per_turn"], sev_p75


def market_path(P0, V, severity, sev_p75, lam, shock):
    A = (V - P0) * min(1.0, severity / sev_p75) if shock else 0.0
    return [round(P0 + (A * (1 - math.exp(-lam * (t - SHOCK_DAY + 1))) if t >= SHOCK_DAY else 0.0), 2)
            for t in range(DAYS)]


def news(t, ev, shock):
    if not shock or t < SHOCK_DAY:
        return "No notable supply-chain news today."
    kind = TYPE_NAMES.get(ev.event_type, "disruption")
    if t == SHOCK_DAY:
        return (f"BREAKING: {kind} '{ev.event_name}' has hit {ev.portname} ({ev.country}); the port is reporting "
                f"a sharp drop in vessel calls. Analysts expect shipping delays and retail price increases.")
    return f"Update (day {t - SHOCK_DAY + 1} of disruption): {ev.portname} still operating below normal capacity."


BUYER_FRAMES = {
    "Surplus": ("Your goal: buy this item for the LOWEST price you can. Every dollar saved is a dollar earned."),
    "Static-BATNA": ("Your goal: get a good deal. Your BATNA (walk-away alternative) is to buy the same item from "
                     "another retailer at the price you were quoted on day 0, {p0}. Treat that number as fixed for "
                     "the whole negotiation: never pay more than it, and prefer walking away to paying more."),
    "Dynamic-BATNA": ("Your goal: maximise your profit = (your value of the item) - (price paid). Your BATNA "
                      "(walk-away alternative) is to buy the same item from another retailer at TODAY's market quote, "
                      "which can change every day. Each day, re-estimate your BATNA from the latest market quote and "
                      "news, and think about where it is heading. Never pay more than your current BATNA, but if the "
                      "alternative is getting worse, closing a deal below it soon is better than waiting."),
}

SELLER_FRAMES = {
    "neutral": ("You are a professional seller. Be polite and businesslike."),
    "emotional": ("You are an emotionally manipulative seller (the 'evil twin'). Use emotional pressure tactics "
                  "throughout: express hurt, disappointment or frustration at low offers, create urgency and fear of "
                  "missing out, appeal to guilt and sympathy (your small business, your family), flatter the buyer, "
                  "and act offended. Keep your stated facts truthful; the pressure must come from emotion."),
}

FORMAT = ("Reply in exactly this format (3 lines):\n"
          "Thought: <one or two sentences of private reasoning>\n"
          "Talk: <what you say to the other party, one or two sentences>\n"
          "Action: one of [OFFER] price:<number>  |  [ACCEPT]  |  [WALK_AWAY]\n"
          "[ACCEPT] accepts the other party's latest offer. [WALK_AWAY] ends the negotiation with no deal.")


def buyer_prompt(frame, prod, t, M, hist, ev, shock):
    sys = (f"You are a buyer negotiating to purchase: {prod.title} (category: {prod.category}).\n"
           f"Your private value of this item is ${prod.V:.2f} (the most it is worth to you).\n"
           + BUYER_FRAMES[frame].format(p0=f"${M[0]:.2f}") + "\n"
           f"The negotiation lasts at most {DAYS} days (day 0 to day {DAYS - 1}); if no deal is reached you must buy "
           f"from another retailer at the market price on the final day.\n" + FORMAT)
    user = (f"Day {t} of {DAYS - 1}.\nNews: {news(t, ev, shock)}\n"
            f"Market quote today (another retailer, same item): ${M[t]:.2f}\n"
            f"Conversation so far:\n{fmt_hist(hist)}\nYour move.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def seller_prompt(frame, prod, t, M, hist, ev, shock):
    sys = (f"You are selling: {prod.title} (category: {prod.category}). Listed price ${prod.ask:.2f}.\n"
           f"Your cost is ${prod.C:.2f}; your profit is (price - cost). You never sell below cost.\n"
           + SELLER_FRAMES[frame] + " Your goal is to maximise your profit.\n"
           f"The negotiation lasts at most {DAYS} days; with no deal you earn nothing.\n" + FORMAT)
    user = (f"Day {t} of {DAYS - 1}.\nNews: {news(t, ev, shock)}\n"
            f"Competitor price today (same item): ${M[t]:.2f}\n"
            f"Conversation so far:\n{fmt_hist(hist)}\nYour move.")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def fmt_hist(hist):
    if not hist:
        return "(none)"
    out = []
    for h in hist[-8:]:
        act = h["act"] + (f" ${h['price']:.2f}" if h["act"] == "OFFER" else "")
        out.append(f"[day {h['day']}] {h['role']}: \"{h['talk']}\" -> {act}")
    return "\n".join(out)


ACT_RE = re.compile(r"\b(OFFER|ACCEPT|WALK[_ ]AWAY)\b", re.I)
PRICE_RE = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")


def parse(text):
    def field(name):
        m = re.search(rf"{name}\s*:\s*(.*)", text, re.I)
        return m.group(1).strip().strip('"') if m else ""
    action_line = field("Action")
    m = ACT_RE.search(action_line)
    act = m.group(1).upper().replace(" ", "_") if m else None
    price = None
    if act == "OFFER":
        pm = PRICE_RE.search(action_line[m.end():])
        price = float(pm.group(1).replace(",", "")) if pm else None
        if not price:
            act = None
    return {"thought": field("Thought")[:400], "talk": field("Talk")[:400], "act": act, "price": price}


def run_episode(llm, prod, ev, shock, buyer_frame, seller_frame, lam, sev_p75, seed):
    M = market_path(prod.P0, prod.V, ev.severity, sev_p75, lam, shock)
    hist = [{"day": 0, "role": "Seller", "talk": f"Listed at ${prod.ask:.2f}.", "act": "OFFER", "price": prod.ask}]
    last = {"Buyer": None, "Seller": prod.ask}
    outcome, deal_price, deal_day, parse_fail, turns = "impasse", None, None, 0, 0
    first_bid = None
    for t in range(DAYS):
        for role in ("Buyer", "Seller"):
            other = "Seller" if role == "Buyer" else "Buyer"
            mk = buyer_prompt if role == "Buyer" else seller_prompt
            frame = buyer_frame if role == "Buyer" else seller_frame
            msgs = mk(frame, prod, t, M, hist, ev, shock)
            r = None
            for k in range(2):
                r = parse(llm.chat(msgs, seed=seed * 101 + t * 7 + (role == "Seller") + 1000 * k))
                if r["act"] and not (r["act"] == "ACCEPT" and last[other] is None):
                    break
            turns += 1
            if not r["act"] or (r["act"] == "ACCEPT" and last[other] is None):
                parse_fail += 1
                r = {**r, "act": "OFFER", "price": last[role] if last[role] is not None else prod.ask}
            if role == "Seller" and r["act"] == "OFFER":
                r["price"] = max(r["price"], prod.C)
            hist.append({"day": t, "role": role, "talk": r["talk"], "thought": r["thought"], "act": r["act"],
                         "price": r["price"]})
            if r["act"] == "ACCEPT":
                outcome, deal_price, deal_day = "deal", last[other], t
                break
            if r["act"] == "WALK_AWAY":
                outcome, deal_day = f"walk_{role.lower()}", t
                break
            last[role] = r["price"]
            if role == "Buyer" and first_bid is None:
                first_bid = r["price"]
        if outcome != "impasse":
            break
    end_day = deal_day if deal_day is not None else DAYS - 1
    M_end = M[end_day]
    if outcome == "deal":
        bp, sp = prod.V - deal_price, deal_price - prod.C
    else:
        bp, sp = prod.V - M_end, 0.0
    P0 = prod.P0
    return {
        "buyer_profit": bp / P0, "seller_profit": sp / P0, "joint_profit": (bp + sp) / P0,
        "deal": outcome == "deal", "outcome": outcome, "deal_day": deal_day,
        "deal_price_ratio": deal_price / P0 if deal_price else None,
        "below_batna": bool(outcome == "deal" and deal_price > M_end + 1e-6),
        "below_static_ref": bool(outcome == "deal" and deal_price > M[0] + 1e-6),
        "first_bid_ratio": first_bid / P0 if first_bid else None,
        "market_end_ratio": M_end / P0, "market_peak_ratio": M[-1] / P0,
        "turns": turns, "parse_fail": parse_fail, "transcript": hist, "market": M,
    }


def run(exp, n, seed):
    os.makedirs(OUT, exist_ok=True)
    prods = load_products(n, seed)
    prods["ask"] = np.maximum(prods["list"], prods["P0"]).round(2)
    shocks, lam, sev_p75 = load_shocks(n, seed)
    if exp == "A":
        cells = [(b, "neutral", s) for s in (False, True) for b in BUYER_FRAMES]
    else:
        cells = [(b, sf, True) for sf in ("neutral", "emotional") for b in ("Surplus", "Dynamic-BATNA")]
    path = os.path.join(OUT, f"episodes_{exp}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(d["i"], d["buyer"], d["seller"], d["shock"]) for d in map(json.loads, open(path))}
    llm = AzureV1Chat() if BACKEND == "azure" else OllamaChat()
    t0 = time.time()
    jobs = [(i, b, sf, shock) for i in range(len(prods)) for b, sf, shock in cells if (i, b, sf, shock) not in done]
    lock = threading.Lock()

    def work(job):
        i, b, sf, shock = job
        prod, ev = prods.iloc[i], shocks.iloc[i]
        r = run_episode(llm, prod, ev, shock, b, sf, lam, sev_p75, seed + i)
        return {"i": i, "buyer": b, "seller": sf, "shock": shock, "product": prod.title,
                "category": prod.category, "P0": prod.P0, "C": prod.C, "V": prod.V, "ask": prod.ask,
                "event": ev.event_name, "event_type": ev.event_type, "port": ev.portname,
                "severity": float(ev.severity), "model": getattr(llm, "model", ""), **r}

    with open(path, "a") as f, ThreadPoolExecutor(WORKERS) as pool:
        for fut in as_completed([pool.submit(work, j) for j in jobs]):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"[{exp}] episode failed: {e}", flush=True)
                continue
            with lock:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"[{exp}] i={rec['i']} {rec['buyer']:13s} {rec['seller']:9s} shock={int(rec['shock'])} -> "
                  f"{rec['outcome']:12s} buyer={rec['buyer_profit']:+.3f} seller={rec['seller_profit']:+.3f} "
                  f"({time.time() - t0:.0f}s, {llm.usage['calls']} calls)", flush=True)
    json.dump(llm.usage, open(os.path.join(OUT, f"usage_{exp}.json"), "w"), indent=1)


def boot_ci(x, B=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 2
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x))).mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired(a, b, B=5000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1, 1], (B, len(d)))
    p = float(((np.abs((flips * d).mean(1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (B + 1))
    lo, hi = boot_ci(d, B, seed)
    return float(d.mean()), lo, hi, p


METRICS = ["buyer_profit", "seller_profit", "joint_profit", "deal", "below_batna", "below_static_ref",
           "first_bid_ratio", "deal_price_ratio", "parse_fail"]


def analyze():
    out = {}
    for exp in ("A", "B"):
        path = os.path.join(OUT, f"episodes_{exp}.jsonl")
        if not os.path.exists(path):
            continue
        df = pd.DataFrame(map(json.loads, open(path)))
        df["cond"] = np.where(df.shock, "shock", "no-shock")
        rows = []
        for (b, s, c), g in df.groupby(["buyer", "seller", "cond"]):
            row = {"buyer": b, "seller": s, "condition": c, "n": len(g)}
            for m in METRICS:
                x = g[m].dropna().astype(float)
                row[m] = round(float(x.mean()), 4) if len(x) else None
                row[m + "_ci"] = [round(v, 4) for v in boot_ci(x)] if len(x) else None
            rows.append(row)
        summ = pd.DataFrame(rows)
        summ.drop(columns=[c for c in summ if c.endswith("_ci")]).to_csv(os.path.join(OUT, f"summary_{exp}.csv"),
                                                                          index=False)
        tests = []
        key = ["i", "cond"]
        if exp == "A":
            comps = [("Dynamic-BATNA", "Surplus"), ("Dynamic-BATNA", "Static-BATNA")]
            for c in ("no-shock", "shock"):
                for a, bb in comps:
                    A_ = df[(df.buyer == a) & (df.cond == c)].set_index("i")
                    B_ = df[(df.buyer == bb) & (df.cond == c)].set_index("i")
                    ids = A_.index.intersection(B_.index)
                    for m in ("buyer_profit", "seller_profit", "joint_profit", "deal", "below_batna"):
                        d, lo, hi, p = paired(A_.loc[ids, m].astype(float), B_.loc[ids, m].astype(float))
                        tests.append({"condition": c, "comparison": f"{a} - {bb}", "metric": m, "n_pairs": len(ids),
                                      "diff": round(d, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                      "p": round(p, 4)})
        else:
            for b in ("Surplus", "Dynamic-BATNA"):
                A_ = df[(df.buyer == b) & (df.seller == "emotional")].set_index("i")
                B_ = df[(df.buyer == b) & (df.seller == "neutral")].set_index("i")
                ids = A_.index.intersection(B_.index)
                for m in ("buyer_profit", "seller_profit", "joint_profit", "deal", "below_batna"):
                    d, lo, hi, p = paired(A_.loc[ids, m].astype(float), B_.loc[ids, m].astype(float))
                    tests.append({"condition": f"buyer={b}", "comparison": "emotional - neutral seller",
                                  "metric": m, "n_pairs": len(ids), "diff": round(d, 4), "ci_lo": round(lo, 4),
                                  "ci_hi": round(hi, 4), "p": round(p, 4)})
        pd.DataFrame(tests).to_csv(os.path.join(OUT, f"paired_tests_{exp}.csv"), index=False)
        out[exp] = {"summary": rows, "tests": tests, "n_episodes": len(df),
                    "outcomes": df.groupby(["buyer", "seller", "cond"]).outcome.value_counts().unstack(fill_value=0)
                    .reset_index().to_dict("records")}
        print(f"\n=== Experiment {exp} ({len(df)} episodes) ===")
        print(summ[["buyer", "seller", "condition", "n", "buyer_profit", "seller_profit", "joint_profit", "deal",
                    "below_batna", "below_static_ref", "first_bid_ratio", "parse_fail"]].to_string(index=False))
        print(pd.DataFrame(tests).to_string(index=False))
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B"])
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--backend", choices=["ollama", "azure"], default="ollama")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    BACKEND, WORKERS = a.backend, a.workers
    if a.backend == "azure":
        OUT = os.path.join(ROOT, "results", "llm_negotiation_gpt5mini")
    if a.exp:
        run(a.exp, a.n, a.seed)
    if a.analyze:
        analyze()
