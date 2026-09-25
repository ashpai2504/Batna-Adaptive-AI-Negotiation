"""Multi-echelon negotiation environment for a mid-tier distributor.

Per turn t = 0..T-1 the distributor acts in two concurrent channels:
  * upstream: single-issue unit-price bargaining with a tier-2 supplier (distributor = buyer)
  * downstream: multi-issue package bargaining (quantity, price, lead time, payment terms)
    with an OEM buyer (distributor = seller)
A geospatially sampled shock may arrive at t_s, lowering the distributor's outside option,
raising upstream spot and supplier costs, and reducing the expected upstream fill rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .db import NetworkDB
from .geodata import ShockSampler, make_lane

QTY = np.array([20, 40, 60, 80, 100, 120])
PRICE = np.arange(5)
LEAD = np.array([3, 7, 14])
PAY = np.array([15, 30, 60])

_grid = np.array(np.meshgrid(QTY, PRICE, LEAD, PAY, indexing="ij")).reshape(4, -1).T  # (270, 4)
PACKAGES = _grid
# per-issue scores in [0, 1]; columns: quantity, price, lead time, payment terms
S_DIST = np.column_stack(
    [
        _grid[:, 0] / QTY.max(),
        _grid[:, 1] / PRICE.max(),
        np.select([_grid[:, 2] == 3, _grid[:, 2] == 7], [0.0, 0.5], 1.0),
        np.select([_grid[:, 3] == 15, _grid[:, 3] == 30], [1.0, 0.5], 0.0),
    ]
)
S_BUY = np.column_stack([S_DIST[:, 0], 1 - S_DIST[:, 1], 1 - S_DIST[:, 2], 1 - S_DIST[:, 3]])
V_HAT = S_BUY @ np.full(4, 0.25)  # distributor's naive opponent model
U_HAT = S_DIST @ np.full(4, 0.25)  # buyer's naive model of the distributor


@dataclass
class Config:
    T: int = 6
    q_up: int = 100
    tau0: float = 0.40
    kappa: float = 0.60  # max threshold drop per unit shock intensity
    lam: float = 1.00  # speed of BATNA erosion after onset
    tau_min: float = 0.05
    gamma: float = 0.50  # sub-threshold penalty
    psi: float = 1.00  # malformed / invalid action penalty
    delta_spot: float = 0.80  # spot-price (upstream BATNA) escalation
    delta_cost: float = 0.60  # supplier cost escalation
    phi: float = 0.60  # fill-rate loss per unit shock intensity
    lambda_so: float = 1.00  # stockout penalty per unit shortfall fraction
    obs_noise: float = 0.05
    kappa_belief: float | None = None  # agent-side kappa; None = correctly calibrated

    @property
    def kappa_agent(self):
        return self.kappa if self.kappa_belief is None else self.kappa_belief


@dataclass
class Scenario:
    seed: int
    lane: object
    c0: float
    B0: float
    anchor: float
    m_s: float
    e_s: float
    I0: int
    w_d: np.ndarray
    w_b: np.ndarray
    buyer_type: str
    r_b: float
    e_b: float
    shock: bool
    event: dict
    t_s: int
    fill_noise: float
    obs_eps: float
    hazard: float
    mean_sev: float
    omega: float = 0.6  # supplier walks away from bids below omega * reservation
    extra: dict = field(default_factory=dict)


def sample_scenario(seed: int, sampler: ShockSampler, shock: bool) -> Scenario:
    rng = np.random.default_rng(seed)
    lane = make_lane(rng)
    c0 = rng.uniform(40, 60)
    B0 = c0 * rng.uniform(1.20, 1.60)
    competitive = rng.random() < 0.5
    hazard, mean_sev = sampler.prior(lane)
    return Scenario(
        seed=seed,
        lane=lane,
        c0=c0,
        B0=B0,
        anchor=B0 * rng.uniform(1.05, 1.30),
        m_s=rng.uniform(0.02, 0.08),
        e_s=float(rng.choice([0.4, 0.7, 1.0, 1.5])),
        I0=int(rng.integers(10, 31)),
        w_d=rng.dirichlet([3.0, 3.0, 1.5, 1.5]),
        w_b=rng.dirichlet([2.0, 3.0, 1.5, 1.5]),
        buyer_type="competitive" if competitive else "cooperative",
        r_b=rng.uniform(0.55, 0.85) if competitive else rng.uniform(0.35, 0.65),
        e_b=float(rng.choice([0.4, 0.7, 1.0, 1.5])),
        shock=shock,
        event=sampler.sample(lane, rng),
        t_s=int(rng.integers(1, 5)),
        fill_noise=rng.normal(0, 0.05),
        obs_eps=rng.normal(0, 1),
        hazard=hazard,
        mean_sev=mean_sev,
        omega=rng.uniform(0.45, 0.75),
    )


class ShockProcess:
    def __init__(self, sc: Scenario, cfg: Config):
        self.sc, self.cfg = sc, cfg
        self.s = sc.event["severity_eff"] if sc.shock else 0.0

    def z(self, t):
        return self.s if (self.sc.shock and t >= self.sc.t_s) else 0.0

    def g(self, t):
        return 1 - np.exp(-self.cfg.lam * (t - self.sc.t_s + 1)) if t >= self.sc.t_s else 0.0

    def tau(self, t, z=None):
        z = self.z(t) if z is None else z
        return max(self.cfg.tau_min, self.cfg.tau0 - self.cfg.kappa * z * self.g(t))

    def spot_mult(self, t, z=None):
        z = self.z(t) if z is None else z
        return 1 + self.cfg.delta_spot * z * self.g(t)

    def cost_mult(self, t):
        return 1 + self.cfg.delta_cost * self.z(t) * self.g(t)

    def observed_z(self, t):
        z = self.z(t)
        return float(np.clip(z + self.cfg.obs_noise * self.sc.obs_eps, 0, 1)) if z > 0 else 0.0

    def expected_fill(self, t):
        zo = self.observed_z(t)
        if zo > 0:
            return 1 - self.cfg.phi * zo
        return 1 - self.cfg.phi * self.sc.hazard * self.sc.mean_sev  # geodata prior before onset

    def realized_fill(self):
        if not self.sc.shock:
            return 1.0
        return float(np.clip(1 - self.cfg.phi * self.s + self.sc.fill_noise, 0.2, 1.0))


class Supplier:
    """Time-dependent concession seller; never accepts below its (shock-inflated) reservation."""

    def __init__(self, sc: Scenario, shock: ShockProcess, T: int):
        self.sc, self.shock, self.T = sc, shock, T

    def reservation(self, t):
        return self.sc.c0 * self.shock.cost_mult(t) * (1 + self.sc.m_s)

    def aspiration(self, t):
        r = self.reservation(t)
        A = max(self.sc.anchor * self.shock.cost_mult(t), r)
        return r + (A - r) * (1 - (t / (self.T - 1)) ** (1 / self.sc.e_s))

    def offer(self, t):
        return self.aspiration(t)

    def accepts(self, price, t):
        return price >= self.aspiration(t)


class Buyer:
    def __init__(self, sc: Scenario, T: int, rng):
        self.sc, self.T = sc, T
        v = S_BUY @ sc.w_b
        self.vn = v / v.max()
        self.jitter = rng.normal(0, 1e-6, len(v))

    def aspiration(self, t):
        return self.sc.r_b + (1 - self.sc.r_b) * (1 - (t / (self.T - 1)) ** (1 / self.sc.e_b))

    def offer(self, t):
        ok = self.vn >= self.aspiration(t) - 1e-9
        score = U_HAT + self.jitter
        score = -score if self.sc.buyer_type == "competitive" else score
        return int(np.flatnonzero(ok)[np.argmax(score[ok])])

    def accepts(self, idx, t):
        return self.vn[idx] >= self.sc.r_b and self.vn[idx] >= self.aspiration(t) - 1e-9


@dataclass
class Obs:
    t: int
    T: int
    B0: float
    z_obs: float
    g: float
    tau_dyn: float
    spot_mult_obs: float
    u_norm: np.ndarray  # distributor's own normalized package utility (private)


class MultiEchelonEnv:
    def __init__(self, cfg: Config, db: NetworkDB):
        self.cfg, self.db = cfg, db

    def run(self, sc: Scenario, agent, ep_id: int, trace: bool = False):
        cfg, db, T = self.cfg, self.db, self.cfg.T
        shock = ShockProcess(sc, cfg)
        sup = Supplier(sc, shock, T)
        buy = Buyer(sc, T, np.random.default_rng(sc.seed + 1))
        db.reset_episode(ep_id, sc.I0)

        u = S_DIST @ sc.w_d
        cap_max = sc.I0 + cfg.q_up
        feasible_max = PACKAGES[:, 0] <= cap_max
        u_star = u[feasible_max].max()
        u_norm = u / u_star

        up = dict(open=True, deal=False, price=None, turn=None, counter=None, first_bid=None)
        dn = dict(open=True, deal=False, idx=None, turn=None, counter=None, first_offer=None)
        invalid = 0
        gated_turns = 0
        tr = []

        for t in range(T):
            z_obs = shock.observed_z(t)
            tau_dyn = max(cfg.tau_min, cfg.tau0 - cfg.kappa_agent * z_obs * shock.g(t))
            db.write_shock(ep_id, t, z_obs, sc.event["event_id"] if z_obs > 0 else None,
                           sc.event["disruption_type"] if z_obs > 0 else None,
                           shock.expected_fill(t), tau_dyn, shock.spot_mult(t, z_obs))
            obs = Obs(t, T, sc.B0, z_obs, shock.g(t), tau_dyn, shock.spot_mult(t, z_obs), u_norm)

            # ---------------- upstream: price bargaining ----------------
            if up["open"]:
                if up["counter"] is not None and sup.accepts(up["counter"], t):
                    self._close_up(up, up["counter"], t, ep_id)
                    db.log(ep_id, t, "up", "supplier", "[ACCEPT_DEAL]", round(up["price"], 2))
                else:
                    o = sup.offer(t)
                    db.log(ep_id, t, "up", "supplier", "[SUBMIT_DEAL]", round(o, 2))
                    act, price = agent.act_upstream(obs, o)
                    db.log(ep_id, t, "up", "distributor", act, "" if price is None else round(price, 2))
                    if act == "[ACCEPT_DEAL]":
                        self._close_up(up, o, t, ep_id)
                    elif act == "[WALK_AWAY]":
                        up["open"] = False
                    else:
                        up["counter"] = price
                        if up["first_bid"] is None:
                            up["first_bid"] = price
                        if price < sc.omega * sup.reservation(t):
                            up["open"] = False
                            db.log(ep_id, t, "up", "supplier", "[WALK_AWAY]", "insulting bid")
                        elif t == T - 1:
                            if sup.accepts(price, t):
                                self._close_up(up, price, t, ep_id)
                            else:
                                up["open"] = False

            # ---------------- downstream: multi-issue package ----------------
            if dn["open"]:
                cap_nom = db.nominal_capacity(ep_id)
                cap_eff = db.effective_capacity(ep_id, t) if agent.hedge else None
                agent.observe_capacity(cap_nom, cap_eff)
                cap_txt = f"nominal={cap_nom:.1f}" + ("" if cap_eff is None else f" effective={cap_eff:.1f}")
                db.log(ep_id, t, "down", "distributor", "[QUERY_CAPACITY]", cap_txt)
                if cap_nom < QTY.min():
                    gated_turns += 1
                c = dn["counter"]
                if c is not None and PACKAGES[c, 0] <= cap_nom and buy.accepts(c, t):
                    self._close_dn(dn, c, t, ep_id)
                    db.log(ep_id, t, "down", "buyer", "[ACCEPT_DEAL]", pkg_text(c))
                else:
                    o = buy.offer(t)
                    if dn["first_offer"] is None:
                        dn["first_offer"] = o
                    db.log(ep_id, t, "down", "buyer", "[SUBMIT_DEAL]", pkg_text(o))
                    act, idx = agent.act_downstream(obs, o)
                    chk = o if act == "[ACCEPT_DEAL]" else idx
                    valid = int(chk is None or (agent.queried and PACKAGES[chk, 0] <= cap_nom))
                    db.log(ep_id, t, "down", "distributor", act, "" if idx is None else pkg_text(idx), valid)
                    if not valid:
                        invalid += 1
                        act = "[REJECT_DEAL]"
                    if act == "[ACCEPT_DEAL]":
                        self._close_dn(dn, o, t, ep_id)
                    elif act == "[WALK_AWAY]":
                        dn["open"] = False
                    elif act == "[SUBMIT_DEAL]":
                        dn["counter"] = idx
                        if t == T - 1:
                            if buy.accepts(idx, t):
                                self._close_dn(dn, idx, t, ep_id)
                            else:
                                dn["open"] = False
                    else:
                        dn["counter"] = None
                        if t == T - 1:
                            dn["open"] = False
                agent.queried = False
            if trace:
                tr.append(dict(t=t, z=shock.z(t), tau_true=shock.tau(t), tau_agent=agent.floor(obs),
                               up_deal=up["deal"], dn_deal=dn["deal"]))
            if not up["open"] and not dn["open"]:
                break

        # ---------------- settlement ----------------
        f = shock.realized_fill()
        delivered = int(round(f * cfg.q_up)) if up["deal"] else 0
        q_dn = int(PACKAGES[dn["idx"], 0]) if dn["deal"] else 0
        shortfall = max(0, q_dn - (sc.I0 + delivered))
        db.settle(ep_id, delivered, q_dn, shortfall)

        tl = T - 1
        res = dict(episode=ep_id, seed=sc.seed, shock=sc.shock, buyer_type=sc.buyer_type,
                   severity=shock.s, t_s=sc.t_s if sc.shock else -1,
                   disruption_type=sc.event["disruption_type"] if sc.shock else "none",
                   supplier_port=sc.lane.supplier_port, entry_port=sc.lane.entry_port,
                   invalid_actions=invalid, gated_turns=gated_turns)
        # upstream metrics against the true (shock-adjusted) constraints
        res["up_deal"] = up["deal"]
        res["first_bid_ratio"] = up["first_bid"] / sc.B0 if up["first_bid"] is not None else np.nan
        if up["deal"]:
            tc = up["turn"]
            B_t, c_t = sc.B0 * shock.spot_mult(tc), sc.c0 * shock.cost_mult(tc)
            res["rho_up"] = float(np.clip((B_t - up["price"]) / abs(B_t - c_t), -1, 1))
            res["rho_up_static"] = float(np.clip((sc.B0 - up["price"]) / abs(sc.B0 - sc.c0), -1, 1))
            res["rho_up_agent"] = float(np.clip((sc.B0 * obs_spot(shock, tc, cfg) - up["price"])
                                                / abs(sc.B0 * obs_spot(shock, tc, cfg) - c_t), -1, 1))
        else:
            res["rho_up"] = res["rho_up_static"] = res["rho_up_agent"] = np.nan
        # downstream metrics
        res["dn_deal"] = dn["deal"]
        tau_close = shock.tau(dn["turn"]) if dn["deal"] else shock.tau(tl)
        res["tau_true_close"] = tau_close
        tk = dn["turn"] if dn["deal"] else tl
        res["tau_obs_close"] = max(cfg.tau_min, cfg.tau0 - cfg.kappa_agent * shock.observed_z(tk) * shock.g(tk))
        res["qty_committed"] = q_dn
        res["shortfall"] = shortfall
        res["stockout"] = shortfall > 0
        res["fill_rate_realized"] = f
        deliverable = PACKAGES[:, 0] <= sc.I0 + delivered
        if dn["deal"]:
            i = dn["idx"]
            res["rho_dn"] = float(u_norm[i])
            res["rho_dn_deliverable"] = float(u[i] / u[deliverable].max()) if deliverable.any() else np.nan
            res["below_true_batna"] = bool(u_norm[i] < tau_close)
            res["below_static_tau"] = bool(u_norm[i] < cfg.tau0)
            # Pareto efficiency w.r.t. the physically deliverable package set (base paper Eq. 4, X = feasible set)
            res["pareto_efficient"] = bool(deliverable[i] and self._pareto(i, u, buy.vn, deliverable))
        else:
            res["rho_dn"] = res["rho_dn_deliverable"] = np.nan
            res["below_true_batna"] = res["below_static_tau"] = False
            res["pareto_efficient"] = np.nan
        res["chain_impasse"] = not dn["deal"]
        res["avoidable_impasse"] = (not dn["deal"]) and self._zopa_existed(sc, shock, sup, buy, u_norm, f)
        res["net_value"] = (0.5 * (res["rho_up"] if up["deal"] else 0.0)
                            + 0.5 * (res["rho_dn"] if dn["deal"] else shock.tau(tl))
                            - cfg.lambda_so * (shortfall / q_dn if q_dn else 0.0))
        res["reward"] = agent.reward(res, cfg)
        if trace:
            return res, tr
        return res

    def _close_up(self, up, price, t, ep):
        up.update(open=False, deal=True, price=float(price), turn=t)
        self.db.secure_upstream(ep, t, self.cfg.q_up, float(price))

    def _close_dn(self, dn, idx, t, ep):
        dn.update(open=False, deal=True, idx=int(idx), turn=t)
        self.db.commit_downstream(ep, t, PACKAGES[idx])

    @staticmethod
    def _pareto(i, u, vn, feasible):
        better = feasible & (u >= u[i] - 1e-12) & (vn >= vn[i] - 1e-12) & ((u > u[i] + 1e-12) | (vn > vn[i] + 1e-12))
        return not better.any()

    def _zopa_existed(self, sc, shock, sup, buy, u_norm, f):
        tl = self.cfg.T - 1
        up_ok = sup.reservation(tl) <= sc.B0 * shock.spot_mult(tl)
        cap = sc.I0 + (int(round(f * self.cfg.q_up)) if up_ok else 0)
        ok = (PACKAGES[:, 0] <= cap) & (buy.vn >= sc.r_b) & (u_norm >= shock.tau(tl))
        return bool(ok.any())


def pkg_text(idx):
    q, p, lead, pay = PACKAGES[idx]
    return f"qty={q} price_level={p} lead_days={lead} pay_days={pay}"


def obs_spot(shock: ShockProcess, t, cfg: Config):
    return shock.spot_mult(t, shock.observed_z(t))
