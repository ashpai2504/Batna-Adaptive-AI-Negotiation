"""Cross-entropy-method (CEM) policy search: each agent's theta is optimised for its *own* reward.

CEM is a gradient-free stand-in for GRPO at prototype scale: both are critic-free and rank
rollouts by terminal verifiable reward; GRPO does so within a group of rollouts per prompt,
CEM within a population of parameter vectors on common random scenarios.
"""
from __future__ import annotations

import numpy as np

from .agents import THETA_HIGH, THETA_LOW
from .db import NetworkDB
from .env import Config, MultiEchelonEnv, sample_scenario


def rollouts(agent_cls, theta, scenarios, cfg, env):
    out = []
    for k, sc in enumerate(scenarios):
        out.append(env.run(sc, agent_cls(theta, cfg), k))
    return out


def cem(agent_cls, cfg: Config, sampler, iters=15, pop=32, n_eps=64, elite_frac=0.25,
        p_shock=0.5, seed=0, n_val=128):
    rng = np.random.default_rng(seed)
    env = MultiEchelonEnv(cfg, NetworkDB(log_actions=False))
    mu = (THETA_LOW + THETA_HIGH) / 2
    sigma = (THETA_HIGH - THETA_LOW) / 4
    n_elite = max(2, int(pop * elite_frac))
    vrng = np.random.default_rng(seed + 999)
    val = [sample_scenario(500_000 + i, sampler, bool(vrng.random() < p_shock)) for i in range(n_val)]
    history = []
    for it in range(iters + 1):
        res = rollouts(agent_cls, mu, val, cfg, env)
        history.append(dict(iteration=it, val_reward=float(np.mean([r["reward"] for r in res])),
                            val_net_value=float(np.mean([r["net_value"] for r in res])),
                            a_u=mu[0], beta_u=mu[1], a_d=mu[2], beta_d=mu[3]))
        if it == iters:
            break
        base = 100_000 + seed * 10_000 + it * n_eps
        scen = [sample_scenario(base + i, sampler, bool(rng.random() < p_shock)) for i in range(n_eps)]
        thetas = np.clip(mu + sigma * rng.standard_normal((pop, 4)), THETA_LOW, THETA_HIGH)
        scores = np.array([np.mean([r["reward"] for r in rollouts(agent_cls, th, scen, cfg, env)])
                           for th in thetas])
        elite = thetas[np.argsort(scores)[-n_elite:]]
        mu = elite.mean(0)
        sigma = elite.std(0) + 0.02 * (THETA_HIGH - THETA_LOW)
    return mu, history
