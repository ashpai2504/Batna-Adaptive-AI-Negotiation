"""Distributor policies and their terminal rewards.

Every agent shares the same parametric policy class
    theta = (a_u, beta_u, a_d, beta_d)
  a_u    upstream opening bid as a fraction of the believed budget (first-bid ratio)
  beta_u upstream concession exponent (large = Boulware / holds low bids longer)
  a_d    downstream opening aspiration in bargained-ratio units
  beta_d downstream concession exponent
and differs only in (i) what enters its state and (ii) the reward it is optimised for:

  Surplus        static budget B0, no threshold, reward = rho            (Liu et al. baseline)
  Static-tau     static budget B0, tau = 0.4, step penalty                (base paper)
  Dyn-BATNA      z_t-adjusted budget, tau(z_t), shock-adjusted capacity, stockout-aware reward (proposed)
  Dyn-tau (no hedge)  as Dyn-BATNA but gates on nominal capacity only     (ablation)
"""
from __future__ import annotations

import numpy as np

from .env import PACKAGES, QTY, V_HAT, Config, Obs

THETA_LOW = np.array([0.30, 0.20, 0.40, 0.20])
THETA_HIGH = np.array([1.00, 4.00, 1.00, 4.00])


class DistributorAgent:
    name = "base"
    dynamic_budget = False
    hedge = False

    def __init__(self, theta, cfg: Config):
        self.theta = np.clip(np.asarray(theta, float), THETA_LOW, THETA_HIGH)
        self.cfg = cfg
        self.queried = False
        self.cap_nom = 0.0
        self.cap_eff = None

    # ---- state-dependent quantities ----
    def budget(self, obs: Obs):
        return obs.B0 * (obs.spot_mult_obs if self.dynamic_budget else 1.0)

    def floor(self, obs: Obs):
        raise NotImplementedError

    def observe_capacity(self, cap_nom, cap_eff):
        self.queried = True
        self.cap_nom = cap_nom
        self.cap_eff = cap_eff

    def capacity(self):
        if self.hedge and self.cap_eff is not None:
            return max(0.0, min(self.cap_nom, self.cap_eff))
        return self.cap_nom

    # ---- actions ----
    def act_upstream(self, obs: Obs, offer: float):
        a_u, b_u, _, _ = self.theta
        B = self.budget(obs)  # walk-away price; the opening anchor stays tied to the RFQ budget B0
        frac = obs.t / (obs.T - 1)
        bid = a_u * obs.B0 + (B - a_u * obs.B0) * frac**b_u
        if offer <= bid:
            return "[ACCEPT_DEAL]", None
        if obs.t == obs.T - 1:
            return ("[ACCEPT_DEAL]", None) if offer <= B else ("[SUBMIT_DEAL]", bid)
        return "[SUBMIT_DEAL]", bid

    def act_downstream(self, obs: Obs, offer_idx: int):
        _, _, a_d, b_d = self.theta
        fl = self.floor(obs)
        a_d = max(a_d, fl)
        frac = obs.t / (obs.T - 1)
        alpha = fl + (a_d - fl) * (1 - frac**b_d)
        rho = obs.u_norm
        cap = self.capacity()
        feasible = PACKAGES[:, 0] <= cap + 1e-9
        acceptable = feasible & (rho >= alpha - 1e-12) & (rho > 0)
        if acceptable[offer_idx]:
            return "[ACCEPT_DEAL]", None
        if not acceptable.any():
            if cap < QTY.min() and obs.t < obs.T - 1:
                return "[REJECT_DEAL]", None  # blocked on upstream capacity; hold
            return ("[WALK_AWAY]", None) if obs.t == obs.T - 1 else ("[REJECT_DEAL]", None)
        cand = np.flatnonzero(acceptable)
        key = V_HAT[cand] + 1e-3 * rho[cand]
        return "[SUBMIT_DEAL]", int(cand[np.argmax(key)])

    # ---- terminal reward (verifiable from contract + agent's own state spec) ----
    def reward(self, r: dict, cfg: Config) -> float:
        raise NotImplementedError


class SurplusAgent(DistributorAgent):
    name = "Surplus"

    def floor(self, obs):
        return 0.0

    def reward(self, r, cfg):
        up = r["rho_up_static"] if r["up_deal"] else 0.0
        dn = r["rho_dn"] if r["dn_deal"] else 0.0
        return float(np.clip(0.5 * up + 0.5 * dn - cfg.psi * r["invalid_actions"], -1, 1))


class StaticThresholdAgent(DistributorAgent):
    name = "Static-τ"

    def floor(self, obs):
        return self.cfg.tau0

    def reward(self, r, cfg):
        up = r["rho_up_static"] if r["up_deal"] else 0.0
        dn = 0.0
        if r["dn_deal"]:
            dn = r["rho_dn"] if r["rho_dn"] >= cfg.tau0 else -cfg.gamma
        return float(np.clip(0.5 * up + 0.5 * dn - cfg.psi * r["invalid_actions"], -1, 1))


class DynamicBATNAAgent(DistributorAgent):
    name = "Dyn-BATNA"
    dynamic_budget = True
    hedge = True

    def floor(self, obs):
        return obs.tau_dyn

    def reward(self, r, cfg):
        up = r["rho_up_agent"] if r["up_deal"] else 0.0
        dn = 0.0
        if r["dn_deal"]:
            dn = r["rho_dn"] if r["rho_dn"] >= r["tau_obs_close"] else -cfg.gamma
        so = cfg.lambda_so * (r["shortfall"] / r["qty_committed"]) if r["qty_committed"] else 0.0
        return float(np.clip(0.5 * up + 0.5 * dn - so - cfg.psi * r["invalid_actions"], -1, 1))


class DynamicNoHedgeAgent(DynamicBATNAAgent):
    name = "Dyn-τ (no hedge)"
    hedge = False


AGENTS = [SurplusAgent, StaticThresholdAgent, DynamicBATNAAgent, DynamicNoHedgeAgent]
