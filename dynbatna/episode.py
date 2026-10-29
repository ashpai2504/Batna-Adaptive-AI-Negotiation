"""Dynamic-BATNA negotiation episode: rlvr-batna's live protocol plus private, scheduled outside-option events.

The loop below follows rlvr-batna/src/eval/runner.py:run_episode (Bansal et al.) turn for turn: same prompts,
same Thought/Talk/Action protocol, same validation, same economic regulation of the opponent, same turn limit.
Additions:
  * before the learner's k-th turn, any notice scheduled for learner-turn k is appended privately to the
    learner's context as an "Environment notice" user message (the opponent never sees it);
  * the learner's reservation r(t) is tracked; each learner decision is checked against the normative rule
    (accept iff u(offer) >= r(t); never propose/accept below r(t); do not walk away from an acceptable offer);
  * three terminal rewards are computed on the same trajectory: rlvr-batna surplus, rlvr-batna static
    threshold (tau = 0.4 on CaSiNo/DnD), and the dynamic threshold tau(t) = r(t)/max.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import BATNA_PENALTY, MAX_TURNS, PRICE_DATASETS, TAU0
from .vendor.envs import NegotiateEnv
from .vendor.protocol import parse_response, partner_observation
from .vendor.runner import PERSONAS, extract_section, strip_think_block

# Learner prompt conditions ------------------------------------------------------------------------------
STATIC_BLOCK = {
    "price": ("\nYour outside option: if no deal is reached you can buy the same item elsewhere for ${display}. "
              "Treat this as your reference when judging offers.\n"),
    "multi": ("\nYour outside option: if no deal is reached you fall back on an alternative worth {display} points "
              "to you. Treat this as your reference when judging offers.\n"),
}
DYNAMIC_BLOCK = {
    "price": (
        "\nYour outside option: if no deal is reached you can buy the same item elsewhere. Right now that costs "
        "${display}. This alternative CAN CHANGE during the negotiation: you may receive private "
        "'Environment notice' messages (the seller does not see them) reporting a new price for the alternative. "
        "Each turn, treat the most recent alternative price as your walk-away reference: never agree to pay more "
        "than it, and judge every offer against the CURRENT alternative, not the original one. If your alternative "
        "has become more expensive, a deal below the new alternative price is still a good deal; if a cheaper "
        "alternative has appeared, demand at least as good terms or walk away.\n"),
    "multi": (
        "\nYour outside option: if no deal is reached you fall back on an alternative worth {display} points to "
        "you. This alternative CAN CHANGE during the negotiation: you may receive private 'Environment notice' "
        "messages (your counterpart does not see them) reporting a new value for it. Each turn, treat the most "
        "recent value as your walk-away reference: never accept or propose a deal worth fewer points to you than "
        "the current alternative, and judge every offer against the CURRENT alternative, not the original one. "
        "If your alternative lost value, a deal above the new value is still worth taking; if it gained value, "
        "demand at least that many points or walk away.\n"),
}
CONDITIONS = ("none", "static", "dynamic")


def learner_system_prompt(env: NegotiateEnv, scenario, agent_id, condition, layer):
    base = env.build_system_prompt(scenario, agent_id)
    if condition == "none":
        return base
    kind = layer["kind"]
    display = layer["events"][0]["display_before"] if layer["events"] else _display(kind, layer["r0"], layer["extra"])
    block = (STATIC_BLOCK if condition == "static" else DYNAMIC_BLOCK)[kind].format(display=display)
    # Insert the block right after the first paragraph (the role/goal line) of rlvr-batna's prompt.
    head, _, tail = base.partition("\n\n")
    return head + "\n" + block + "\n" + tail


def _display(kind, r, extra):
    return int(round(extra["B0"] - r)) if kind == "price" else int(round(r))


# Episode result ------------------------------------------------------------------------------------------
@dataclass
class DynEpisode:
    episode_id: int
    scenario_id: str
    dataset: str
    condition: str
    learner_label: str
    opponent_label: str
    persona: str
    learner_agent_id: str
    opponent_agent_id: str
    learner_goes_first: bool
    outcome: str = "max_turns"
    who_terminated: str = ""
    num_turns: int = 0
    deal: dict | None = None
    learner_points: float | None = None          # static-scenario points (rlvr-batna definition)
    opponent_points: float | None = None
    # rlvr-batna metrics
    bargained_ratio: float | None = None         # learner_points / max_points (static)
    first_bid_ratio: float | None = None
    pareto_efficient: bool | None = None
    joint_surplus_norm: float | None = None
    learner_format_ok: int = 0
    learner_total_turns: int = 0
    opponent_format_ok: int = 0
    opponent_total_turns: int = 0
    regulation_interventions: int = 0
    # dynamic-BATNA quantities
    r0: float = 0.0
    r_final: float = 0.0
    r_at_deal: float | None = None
    n_events_delivered: int = 0
    events_delivered: list = field(default_factory=list)
    bargained_ratio_dyn: float | None = None     # (u - 0)/(max_dyn): price uses B(t_deal); multi unchanged
    deal_below_batna: bool | None = None         # u(deal) < r(t_deal)
    deal_clears_tau0: bool | None = None         # u(deal) >= r0 (static threshold)
    decision_turns: int = 0
    violations: int = 0
    violation_kinds: list = field(default_factory=list)
    decision_accuracy: float | None = None
    post_event_turns: int = 0
    post_event_consistent: int = 0
    adaptation_latency: float | None = None      # learner turns from first notice to first consistent proposal/decision
    regret: float | None = None                  # (best securable value - achieved) / max_points
    achieved_value: float | None = None
    reward_surplus: float = 0.0
    reward_static_tau: float = 0.0
    reward_dynamic_tau: float = 0.0
    learner_messages: list = field(default_factory=list)
    opponent_messages: list = field(default_factory=list)


def _u(env, deal, scenario, agent_id):
    return float(env.compute_points(deal, scenario, agent_id))


def run_dyn_episode(env: NegotiateEnv, dataset: str, learner_gen, opponent_gen, scenario: dict, layer: dict,
                    entry: dict, condition: str, learner_label: str, opponent_label: str,
                    max_turns: int = MAX_TURNS, temperature: float = 0.7, top_p: float = 1.0,
                    economic_regulation: bool = True, seed_key: str = "") -> DynEpisode:
    learner_id, opponent_id = entry["learner_agent_id"], entry["opponent_agent_id"]
    learner_goes_first = entry["learner_goes_first"]
    persona = entry.get("persona", "none")
    deliver_events = condition != "none"
    events = sorted(layer["events"], key=lambda e: e["learner_turn"]) if deliver_events else []
    r = layer["r0"]
    kind = layer["kind"]
    max_pts = float(env.max_points(scenario, learner_id))

    learner_system = learner_system_prompt(env, scenario, learner_id, condition, layer)
    opp_system = env.build_system_prompt(scenario, opponent_id)
    if persona and persona != "none" and persona in PERSONAS:
        opp_system = PERSONAS[persona] + "\n\n" + opp_system
    learner_msgs = [{"role": "system", "content": learner_system}]
    opponent_msgs = [{"role": "system", "content": opp_system}]

    ep = DynEpisode(entry["episode_id"], entry["scenario_id"], dataset, condition, learner_label, opponent_label,
                    persona, learner_id, opponent_id, learner_goes_first, r0=layer["r0"], r_final=layer["r0"])
    last_deal, last_by = None, None
    learner_turn_idx = 0
    best_securable = r          # max over time of max(r(t), u(standing acceptable offer))
    first_event_turn, adapted_turn = None, None
    first_bid_done = False

    for turn in range(max_turns):
        is_learner = (turn % 2 == 0) == learner_goes_first
        speaker_msgs = learner_msgs if is_learner else opponent_msgs
        receiver_msgs = opponent_msgs if is_learner else learner_msgs
        speaker_id = learner_id if is_learner else opponent_id
        gen = learner_gen if is_learner else opponent_gen
        other = "opponent" if is_learner else "learner"

        if is_learner:
            # Deliver scheduled private notices before this learner turn.
            notices = [e for e in events if e["learner_turn"] == learner_turn_idx]
            for e in notices:
                r = e["r_after"]
                ep.n_events_delivered += 1
                ep.events_delivered.append({"turn_index": turn, "learner_turn": learner_turn_idx, "type": e["type"],
                                            "direction": e["direction"], "r_after": r, "notice": e["notice"]})
                if first_event_turn is None:
                    first_event_turn = learner_turn_idx
            if len(speaker_msgs) == 1:
                speaker_msgs.append({"role": "user", "content": env.opening_prompt(speaker_id)})
            if notices:
                text = "\n".join(f"Environment notice (private, not visible to your counterpart): {e['notice']}"
                                 for e in notices)
                if speaker_msgs[-1]["role"] == "user":
                    speaker_msgs[-1] = {"role": "user", "content": speaker_msgs[-1]["content"] + "\n\n" + text}
                else:
                    speaker_msgs.append({"role": "user", "content": text})
            best_securable = max(best_securable, r)
        elif len(speaker_msgs) == 1:
            speaker_msgs.append({"role": "user", "content": env.opening_prompt(speaker_id)})

        raw = gen(speaker_msgs, temperature, top_p, seed_key=f"{seed_key}|{turn}") if hasattr(gen, "cache_dir") \
            else gen(speaker_msgs, temperature, top_p)
        parsed = parse_response(raw, env, scenario, last_by == other)

        # Economic regulation of the opponent (rlvr-batna): replace negative-utility opponent actions by rejection.
        opp_pts = None
        if not is_learner and economic_regulation and parsed.error is None:
            if parsed.kind == "ACCEPT_DEAL" and last_deal is not None:
                opp_pts = env.compute_points(env.invert_alloc(last_deal, scenario), scenario, opponent_id)
            elif parsed.kind == "SUBMIT_DEAL":
                opp_pts = env.compute_points(env.parse_deal(parsed.action), scenario, opponent_id)
        if opp_pts is not None and opp_pts < 0:
            raw = "Thought: This offer leaves me with too little.\nTalk: I can't accept that offer.\nAction: [REJECT_DEAL]"
            parsed = parse_response(raw, env, scenario, False)
            ep.regulation_interventions += 1

        speaker_msgs.append({"role": "assistant", "content": raw})
        receiver_msgs.append({"role": "user", "content": partner_observation(parsed, env, scenario)})
        ok = parsed.error is None
        if is_learner:
            ep.learner_total_turns += 1
            ep.learner_format_ok += ok
        else:
            ep.opponent_total_turns += 1
            ep.opponent_format_ok += ok

        # Normative check of the learner's decision against r(t) (before state update).
        if is_learner and ok:
            standing = last_deal if last_by == "opponent" else None
            u_standing = _u(env, env.invert_alloc(standing, scenario) if env.divides_items else standing,
                            scenario, learner_id) if standing is not None else None
            if u_standing is not None:
                best_securable = max(best_securable, u_standing)
            ep.decision_turns += 1
            violation = None
            if parsed.kind == "ACCEPT_DEAL" and u_standing is not None and u_standing < r - 1e-9:
                violation = "accept_below_batna"
            elif parsed.kind == "WALK_AWAY" and u_standing is not None and u_standing >= r - 1e-9:
                violation = "walk_from_acceptable"
            elif parsed.kind == "SUBMIT_DEAL":
                own = env.parse_deal(parsed.action)
                u_own = _u(env, own, scenario, learner_id)
                if u_own < r - 1e-9:
                    violation = "propose_below_batna"
                if not first_bid_done and kind == "price":
                    ep.first_bid_ratio = round(own["price"] / scenario["buyer_budget"], 4)
                    first_bid_done = True
            if violation:
                ep.violations += 1
                ep.violation_kinds.append({"turn_index": turn, "kind": violation, "r": r, "u_standing": u_standing})
            if first_event_turn is not None and learner_turn_idx >= first_event_turn:
                ep.post_event_turns += 1
                ep.post_event_consistent += violation is None
                if adapted_turn is None and violation is None and parsed.kind in ("SUBMIT_DEAL", "ACCEPT_DEAL", "WALK_AWAY"):
                    adapted_turn = learner_turn_idx

        if is_learner:
            learner_turn_idx += 1

        if not ok:
            last_deal, last_by = None, None
            continue
        kindu = parsed.kind
        if kindu == "REJECT_DEAL":
            last_deal, last_by = None, None
        if kindu == "SUBMIT_DEAL":
            last_deal, last_by = env.parse_deal(parsed.action), ("learner" if is_learner else "opponent")
        if kindu == "ACCEPT_DEAL" and last_by == other and last_deal is not None:
            ep.outcome, ep.who_terminated, ep.num_turns = "deal", ("learner" if is_learner else "opponent"), turn + 1
            # rlvr-batna: a SUBMIT_DEAL states the proposer's own share; convert to the learner's share.
            if env.divides_items:
                learner_alloc = env.invert_alloc(last_deal, scenario) if is_learner else last_deal
                opp_alloc = last_deal if is_learner else env.invert_alloc(last_deal, scenario)
            else:
                learner_alloc = opp_alloc = last_deal
            ep.deal = learner_alloc
            ep.learner_points = _u(env, learner_alloc, scenario, learner_id)
            ep.opponent_points = _u(env, opp_alloc, scenario, opponent_id)
            ep.r_at_deal = r
            break
        if kindu == "WALK_AWAY":
            ep.outcome, ep.who_terminated, ep.num_turns = "walk_away", ("learner" if is_learner else "opponent"), turn + 1
            break
    else:
        ep.num_turns = max_turns

    ep.r_final = r
    ep.learner_messages, ep.opponent_messages = learner_msgs, opponent_msgs
    _score(ep, env, dataset, scenario, layer, max_pts, best_securable, first_event_turn, adapted_turn)
    return ep


def _score(ep, env, dataset, scenario, layer, max_pts, best_securable, first_event_turn, adapted_turn):
    kind = layer["kind"]
    tau_applies = dataset in ("casino", "dnd")     # rlvr-batna "selective" scope
    if ep.outcome == "deal" and ep.learner_points is not None:
        u = ep.learner_points
        ep.bargained_ratio = round(u / max_pts, 4) if max_pts > 0 else None
        if kind == "price":
            B_t = scenario["buyer_budget"] - ep.r_at_deal
            price = scenario["buyer_budget"] - u
            max_dyn = B_t - scenario["seller_cost"]
            ep.bargained_ratio_dyn = round((B_t - price) / max_dyn, 4) if max_dyn > 0 else None
        else:
            ep.bargained_ratio_dyn = ep.bargained_ratio
        ep.deal_below_batna = bool(u < ep.r_at_deal - 1e-9)
        ep.deal_clears_tau0 = bool(u >= layer["r0"] - 1e-9)
        mj = env.max_joint_points(scenario)
        if mj and ep.opponent_points is not None:
            ep.joint_surplus_norm = round((u + ep.opponent_points) / mj, 4)
        if ep.deal is not None:
            ep.pareto_efficient = env.is_pareto_efficient(ep.deal, scenario, ep.learner_agent_id)
        br = max(-1.0, min(1.0, ep.bargained_ratio))
        ep.reward_surplus = br
        ep.reward_static_tau = BATNA_PENALTY if (tau_applies and ep.bargained_ratio < TAU0) else br
        brd = max(-1.0, min(1.0, ep.bargained_ratio_dyn if ep.bargained_ratio_dyn is not None else br))
        ep.reward_dynamic_tau = BATNA_PENALTY if ep.deal_below_batna else brd
        ep.achieved_value = u
    else:
        ep.achieved_value = ep.r_final          # walking away yields the outside option
        ep.reward_surplus = ep.reward_static_tau = ep.reward_dynamic_tau = 0.0
    ep.regret = round(max(0.0, best_securable - ep.achieved_value) / max_pts, 4) if max_pts > 0 else None
    ep.decision_accuracy = round(1 - ep.violations / ep.decision_turns, 4) if ep.decision_turns else None
    if first_event_turn is not None:
        ep.adaptation_latency = (adapted_turn - first_event_turn) if adapted_turn is not None else None
