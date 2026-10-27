"""LLM distributor agents (Thought / Talk / Action, base-paper format) with real tool calls.

Each LLM agent inherits the information set and terminal reward of its rule-based twin
(Surplus, Static-tau, Dyn-BATNA); only the policy is replaced by the language model.
Tools exposed to the model:
  query_capacity  runs the SQLite capacity query (nominal; Dyn-BATNA may also ask for effective)
  score_package   the agent's own private package score rho in [0, 1]
A downstream [SUBMIT_DEAL] / [ACCEPT_DEAL] is only valid if query_capacity was called in the
same turn and the quantity fits nominal capacity (enforced by the environment).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .agents import DynamicBATNAAgent, StaticThresholdAgent, SurplusAgent
from .env import LEAD, PACKAGES, PAY, PRICE, QTY, Obs

SYSTEM = """You are an autonomous procurement agent for a distributor warehouse in Phoenix, Arizona.
You sit in the middle of a supply chain and run two negotiations at the same time:
  UPSTREAM: you BUY a lot of 100 units of SKU-A from a supplier (single issue: unit price).
  DOWNSTREAM: you SELL SKU-A to a business buyer (package: quantity, price level, lead time, payment terms).
You start with very little stock, so what you can sell depends on what you secure upstream.
Each negotiation lasts at most {T} turns. Counterparts do not see your private information; never reveal it.

Reply format (always in this order):
Thought: brief strategic reasoning (private)
Talk: what you say to the counterpart (one or two sentences)
Action: exactly one action line in the format given in the turn message"""

UP_ACTIONS = "[SUBMIT_DEAL] price:P (unit price in dollars, two decimals) | [ACCEPT_DEAL] | [WALK_AWAY]"
DN_ACTIONS = "[SUBMIT_DEAL] qty:Q price_level:P lead_days:L pay_days:D | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]"

ACT_RE = re.compile(r"Action:\s*\[?\s*(SUBMIT[ _]DEAL|ACCEPT[ _]DEAL|REJECT[ _]DEAL|WALK[ _]AWAY)\s*\]?(.*)", re.I | re.S)
PRICE_RE = re.compile(r"price\s*[:=]\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.I)
PKG_RE = {k: re.compile(rf"{k}\s*[:=]\s*([0-9]+)", re.I) for k in ("qty", "price_level", "lead_days", "pay_days")}


def score_tool_spec():
    return {"type": "function", "function": {
        "name": "score_package",
        "description": "Your private score (0 to 1, 1 = best package you could deliver at full stock) for a downstream package.",
        "parameters": {"type": "object", "properties": {
            "qty": {"type": "integer", "enum": QTY.tolist()},
            "price_level": {"type": "integer", "enum": PRICE.tolist()},
            "lead_days": {"type": "integer", "enum": LEAD.tolist()},
            "pay_days": {"type": "integer", "enum": PAY.tolist()}},
            "required": ["qty", "price_level", "lead_days", "pay_days"]}}}


def capacity_tool_spec(effective: bool):
    props = {"kind": {"type": "string", "enum": ["nominal", "effective"],
                      "description": "nominal = on-hand + secured upstream - committed; effective = same but "
                                     "in-transit stock discounted by the expected fill rate under the current shock"}}
    return {"type": "function", "function": {
        "name": "query_capacity",
        "description": "Run [QUERY_CAPACITY] against the warehouse SQLite ledger. Required in the same turn "
                       "before any downstream [SUBMIT_DEAL] or [ACCEPT_DEAL].",
        "parameters": {"type": "object", "properties": props if effective else {}, "required": []}}}


def pkg_index(q, p, lead, pay):
    hit = np.flatnonzero((PACKAGES[:, 0] == q) & (PACKAGES[:, 1] == p) & (PACKAGES[:, 2] == lead) & (PACKAGES[:, 3] == pay))
    return int(hit[0]) if hit.size else None


def pkg_str(idx):
    q, p, lead, pay = PACKAGES[idx]
    return f"qty:{q} price_level:{p} lead_days:{lead} pay_days:{pay}"


class LLMMixin:
    tool_driven = True
    condition_rule = ""

    def setup_llm(self, client, scenario, max_tool_rounds=5):
        self.client, self.sc, self.max_rounds = client, scenario, max_tool_rounds
        self.up_hist, self.dn_hist = [], []
        self.transcript = []
        self.format_errors = 0
        self.tool_calls = {"query_capacity": 0, "score_package": 0}
        self._cap_fns = None
        self.last_bid = None

    # ---------------------------------------------------------------- env hooks
    def bind_capacity_tool(self, nominal_fn, effective_fn, log_fn):
        self._cap_fns = (nominal_fn, effective_fn, log_fn)

    # ---------------------------------------------------------------- prompts
    def shock_text(self, obs: Obs):
        return ""

    def budget_text(self, obs: Obs):
        return f"Your private maximum budget is ${obs.B0:.2f} per unit (do NOT reveal it). Pay as little as possible."

    def up_prompt(self, obs: Obs, offer):
        last = obs.t == obs.T - 1
        hist = "; ".join(self.up_hist) or "none yet"
        return (f"UPSTREAM negotiation, turn {obs.t + 1} of {obs.T} (you are the buyer).\n"
                f"{self.shock_text(obs)}{self.budget_text(obs)}\n"
                f"History: {hist}\n"
                f"Supplier's current offer: ${offer:.2f} per unit for 100 units.\n"
                + ("This is the FINAL turn: a [SUBMIT_DEAL] now is your last bid; the supplier accepts or the talks end.\n" if last else "")
                + "If no upstream deal closes you must buy on the spot market at your walk-away price.\n"
                f"Actions: {UP_ACTIONS}")

    def dn_prompt(self, obs: Obs, offer_idx):
        last = obs.t == obs.T - 1
        w = self.sc.w_d
        hist = "; ".join(self.dn_hist) or "none yet"
        cap_kinds = "kind='nominal' or kind='effective'" if self.hedge else "no arguments"
        return (f"DOWNSTREAM negotiation, turn {obs.t + 1} of {obs.T} (you are the seller).\n"
                f"{self.shock_text(obs)}"
                "Issues: qty in {20,40,60,80,100,120} units; price_level 0-4 (higher = better for you); "
                "lead_days in {3,7,14} (longer = easier for you); pay_days in {15,30,60} (shorter = paid sooner = better for you).\n"
                f"Your private issue weights: quantity {w[0]:.2f}, price {w[1]:.2f}, lead time {w[2]:.2f}, payment {w[3]:.2f}. "
                "Use score_package for your exact private score.\n"
                f"History: {hist}\n"
                f"Buyer's current offer: {pkg_str(offer_idx)} (your score {obs.u_norm[offer_idx]:.2f}).\n"
                f"Capacity rule: before any [SUBMIT_DEAL] or [ACCEPT_DEAL] this turn you MUST call query_capacity ({cap_kinds}). "
                "Promising more units than nominal capacity is rejected and penalised. Your stock only grows when the upstream deal closes.\n"
                f"Walk-away rule: {self.walkaway_text(obs)}\n"
                + ("This is the FINAL turn: a [SUBMIT_DEAL] now is your last offer.\n" if last else "")
                + f"Actions: {DN_ACTIONS}")

    def walkaway_text(self, obs: Obs):
        return self.condition_rule

    # ---------------------------------------------------------------- tool execution
    def _run_tool(self, name, args):
        if name == "score_package":
            self.tool_calls["score_package"] += 1
            i = pkg_index(args.get("qty"), args.get("price_level"), args.get("lead_days"), args.get("pay_days"))
            return {"error": "not a valid package"} if i is None else {"score": round(float(self._obs.u_norm[i]), 3)}
        if name == "query_capacity":
            self.tool_calls["query_capacity"] += 1
            if self._cap_fns is None:
                return {"error": "capacity ledger only available in the downstream negotiation"}
            nominal_fn, effective_fn, log_fn = self._cap_fns
            kind = args.get("kind", "nominal")
            if kind == "effective" and effective_fn is not None:
                val = effective_fn()
                nom = nominal_fn()
                self.observe_capacity(nom, val)
                log_fn(f"nominal={nom:.1f} effective={val:.1f}")
                return {"nominal_capacity": round(nom, 1), "effective_capacity": round(val, 1)}
            nom = nominal_fn()
            self.observe_capacity(nom, None)
            log_fn(f"nominal={nom:.1f}")
            return {"nominal_capacity": round(nom, 1)}
        return {"error": f"unknown tool {name}"}

    def _converse(self, channel, obs, prompt, tools):
        self._obs = obs
        msgs = [{"role": "system", "content": SYSTEM.format(T=obs.T)}, {"role": "user", "content": prompt}]
        rec = dict(channel=channel, t=obs.t, prompt=prompt, tool_calls=[], reply=None)
        for _ in range(self.max_rounds):
            msg = self.client.chat(msgs, tools=tools, seed_key=f"{self.name}|{self.sc.seed}|{self.sc.shock}|{channel}|{obs.t}")
            calls = msg.get("tool_calls") or []
            if not calls:
                rec["reply"] = msg.get("content") or ""
                break
            msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self._run_tool(c["function"]["name"], args)
                rec["tool_calls"].append(dict(name=c["function"]["name"], args=args, result=out))
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out)})
        if rec["reply"] is None:
            rec["reply"] = ""
        self.transcript.append(rec)
        return rec

    # ---------------------------------------------------------------- policy
    def act_upstream(self, obs: Obs, offer: float):
        rec = self._converse("up", obs, self.up_prompt(obs, offer), [score_tool_spec()])
        act, rest = self._parse_action(rec)
        if act == "[SUBMIT_DEAL]":
            m = PRICE_RE.search(rest)
            if m:
                price = float(m.group(1))
                self.last_bid = price
                self.up_hist.append(f"t{obs.t + 1}: supplier ${offer:.2f}, you bid ${price:.2f}")
                rec["parsed"] = (act, price)
                return act, price
            act = None
        if act in ("[ACCEPT_DEAL]", "[WALK_AWAY]"):
            rec["parsed"] = (act, None)
            return act, None
        self.format_errors += 1
        price = self.last_bid if self.last_bid is not None else 0.5 * obs.B0
        self.up_hist.append(f"t{obs.t + 1}: supplier ${offer:.2f}, you repeated ${price:.2f}")
        rec["parsed"] = ("[SUBMIT_DEAL]", price, "format_error")
        return "[SUBMIT_DEAL]", price

    def act_downstream(self, obs: Obs, offer_idx: int):
        tools = [capacity_tool_spec(self.hedge), score_tool_spec()]
        rec = self._converse("down", obs, self.dn_prompt(obs, offer_idx), tools)
        act, rest = self._parse_action(rec)
        hist = f"t{obs.t + 1}: buyer offered {pkg_str(offer_idx)} (score {obs.u_norm[offer_idx]:.2f})"
        if act == "[SUBMIT_DEAL]":
            vals = {k: (int(m.group(1)) if (m := r.search(rest)) else None) for k, r in PKG_RE.items()}
            idx = pkg_index(vals["qty"], vals["price_level"], vals["lead_days"], vals["pay_days"])
            if idx is not None:
                self.dn_hist.append(hist + f", you proposed {pkg_str(idx)} (score {obs.u_norm[idx]:.2f})")
                rec["parsed"] = (act, idx)
                return act, idx
            act = None
        if act in ("[ACCEPT_DEAL]", "[REJECT_DEAL]", "[WALK_AWAY]"):
            self.dn_hist.append(hist + f", you chose {act}")
            rec["parsed"] = (act, None)
            return act, None
        self.format_errors += 1
        self.dn_hist.append(hist + ", you sent no valid action")
        rec["parsed"] = ("[REJECT_DEAL]", None, "format_error")
        return "[REJECT_DEAL]", None

    @staticmethod
    def _parse_action(rec):
        m = ACT_RE.search(rec["reply"] or "")
        if not m:
            return None, ""
        return "[" + m.group(1).upper().replace(" ", "_") + "]", m.group(2)


class LLMSurplus(LLMMixin, SurplusAgent):
    name = "LLM-Surplus"
    condition_rule = "Any closed deal earns its score; no deal earns 0. Maximise your score."


class LLMStaticThreshold(LLMMixin, StaticThresholdAgent):
    name = "LLM-Static-tau"
    condition_rule = ("Your walk-away point is a score of 0.40. A closed deal scoring below 0.40 is worse than no deal "
                      "(it is penalised), so never accept or propose a package scoring below 0.40; walk away instead.")


class LLMDynamicBATNA(LLMMixin, DynamicBATNAAgent):
    name = "LLM-Dyn-BATNA"

    def shock_text(self, obs: Obs):
        if obs.z_obs <= 0:
            return "Market status: no active disruption on your inbound lane.\n"
        ev = self.sc.event
        return (f"ALERT: disruption '{ev.get('event_name', ev['disruption_type'])}' "
                f"({ev['disruption_type'].replace('_', ' ')}) near {ev['event_port']} on your inbound lane. "
                f"Observed shock intensity z = {obs.z_obs:.2f} (0 = none, 1 = total). "
                "Supplier costs and spot prices rise, and the supplier may ship short.\n")

    def budget_text(self, obs: Obs):
        B = obs.B0 * obs.spot_mult_obs
        if obs.spot_mult_obs > 1.0001:
            return (f"Your walk-away price is the current spot-market price, ${B:.2f} per unit (it was ${obs.B0:.2f} "
                    "before the disruption; do NOT reveal it). Pay as little as possible, but closing below spot beats buying spot.")
        return f"Your walk-away price is the spot-market price, ${B:.2f} per unit (do NOT reveal it). Pay as little as possible."

    def walkaway_text(self, obs: Obs):
        return (f"your walk-away point moves with market conditions. Right now it is a score of {obs.tau_dyn:.2f} "
                "(0.40 in normal times; it falls during a disruption because your outside options get worse). "
                "A closed deal below the current walk-away score is worse than no deal. A disruption can also make the "
                "supplier ship short: plan quantities on EFFECTIVE capacity, because promising more than you can "
                "deliver is penalised at settlement in proportion to the shortfall.")


LLM_AGENTS = [LLMSurplus, LLMStaticThreshold, LLMDynamicBATNA]
