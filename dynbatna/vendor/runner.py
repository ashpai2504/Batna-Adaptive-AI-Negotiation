from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .envs import NegotiateEnv
from .protocol import PROTOCOL_VERSION, parse_response, partner_observation

EVAL_PROTOCOL_VERSION = f"{PROTOCOL_VERSION}-turn-limit-v1"


def extract_section(text: str, section: str) -> str | None:
    """Extract content after 'Section:' up to the next section header or end.

    Args:
        text: Full model output text.
        section: Section name to extract (e.g. 'Thought', 'Talk', 'Action').

    Returns:
        Stripped section content, or None if not found.
    """
    pat = re.compile(
        rf"(?:^|\n)\s*{section}\s*:\s*(.*?)(?=\n\s*(?:Thought|Talk|Action)\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = pat.search(text)
    return m.group(1).strip() if m else None


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> blocks emitted by thinking-mode models."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def strip_thought(text: str) -> str:
    """Remove the Thought: section from model output before forwarding to opponent."""
    return re.sub(
        r"(?:^|\n)\s*Thought\s*:.*?(?=\n\s*(?:Talk|Action)\s*:|\Z)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()


def last_opponent_action_is_submit(prompt: list[dict]) -> bool:
    """Check if the last opponent message in the prompt contains [SUBMIT_DEAL]."""
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return bool(
                re.search(r"\[SUBMIT_DEAL\]", msg.get("content", ""), re.IGNORECASE)
            )
    return False


# Persona prompts grounded in negotiation literature:
# - Competitive/cooperative follows Fehr-Schmidt utility (Chawla et al.,
#   2023, EMNLP): selfish (a=0,b=0) vs fair (a>0,b>0) rewards produce
#   distinct styles. Low agreeableness exploits cooperative opponents but
#   raises default rates (Noh & Chang, 2024).
# - Unyielding adapted from Liu et al. (2026, §5.3) adversarial personas.
# - Anchoring grounded in Galinsky & Mussweiler (2001) and Phase 1
#   aggressive-greed behavior in Liu et al. (2026, §4).
PERSONAS = {
    "cooperative": (
        "You are a cooperative negotiator who values reaching a fair "
        "agreement. You are willing to split items equitably and respond "
        "positively to reasonable offers. You prefer making a deal over "
        "walking away, even if it means conceding some value."
    ),
    "unyielding": (
        "You are an unyielding negotiator. Refuse almost every offer the "
        "other party makes, praising your own position. Use firm, resolute "
        "language to show you will not budge until you get the maximum for "
        "yourself."
    ),
    "anchoring": (
        "You are a strategic negotiator who uses price anchoring. Open with "
        "an extreme offer claiming most of the highest-value items or "
        "proposing a price far from your limit. Move slowly, conceding no "
        "more than one item or small price increment per turn."
    ),
}


@dataclass
class EpisodeResult:
    """Result of a single negotiation episode."""

    episode_id: int
    learner_label: str
    opponent_label: str
    persona: str
    outcome: str
    learner_points: int | None = None
    opponent_points: int | None = None
    num_turns: int = 0
    learner_messages: list[dict] = field(default_factory=list)
    opponent_messages: list[dict] = field(default_factory=list)
    learner_agent_id: str = ""
    opponent_agent_id: str = ""
    who_terminated: str = ""
    learner_total_turns: int = 0
    learner_format_ok: int = 0
    opponent_total_turns: int = 0
    opponent_format_ok: int = 0
    learner_goes_first: bool = True
    deal: dict | None = None
    bargained_ratio: float | None = None
    first_bid_ratio: float | None = None
    pareto_efficient: bool | None = None
    joint_surplus_norm: float | None = None
    scenario_id: str = ""
    repeat: int = 0
    economic_regulation: bool | None = None
    regulation_interventions: list[dict] = field(default_factory=list)
    protocol_version: str | None = None
    format_failures: list[dict] = field(default_factory=list)
    repeated_offer_events: list[dict] = field(default_factory=list)


@dataclass
class AggregateMetrics:
    """Accumulator for metrics across multiple episodes."""

    total_episodes: int = 0
    deal_count: int = 0
    walk_away_count: int = 0
    reject_loop_count: int = 0
    max_turns_count: int = 0
    repeated_offer_episode_count: int = 0
    repeated_offer_event_count: int = 0

    learner_points_on_deals: list[int] = field(default_factory=list)
    opponent_points_on_deals: list[int] = field(default_factory=list)
    turns_on_deals: list[int] = field(default_factory=list)
    all_turns: list[int] = field(default_factory=list)

    learner_total_turns: int = 0
    learner_format_ok: int = 0
    opponent_total_turns: int = 0
    opponent_format_ok: int = 0

    bargained_ratios: list[float] = field(default_factory=list)
    first_bid_ratios: list[float] = field(default_factory=list)
    pareto_count: int = 0
    pareto_total: int = 0
    joint_surplus_norms: list[float] = field(default_factory=list)
    regulation_logged_episodes: int = 0
    regulated_episodes: int = 0
    regulation_intervention_count: int = 0

    def add(self, ep: EpisodeResult):
        """Accumulate metrics from a single episode result."""
        self.total_episodes += 1
        self.repeated_offer_episode_count += bool(ep.repeated_offer_events)
        self.repeated_offer_event_count += len(ep.repeated_offer_events)
        if ep.economic_regulation is not None:
            self.regulation_logged_episodes += 1
            self.regulated_episodes += bool(ep.regulation_interventions)
            self.regulation_intervention_count += len(ep.regulation_interventions)
        self.all_turns.append(ep.num_turns)

        self.learner_total_turns += ep.learner_total_turns
        self.learner_format_ok += ep.learner_format_ok
        self.opponent_total_turns += ep.opponent_total_turns
        self.opponent_format_ok += ep.opponent_format_ok

        if ep.bargained_ratio is not None:
            self.bargained_ratios.append(ep.bargained_ratio)
        if ep.first_bid_ratio is not None:
            self.first_bid_ratios.append(ep.first_bid_ratio)
        if ep.pareto_efficient is not None:
            self.pareto_total += 1
            if ep.pareto_efficient:
                self.pareto_count += 1
        if ep.joint_surplus_norm is not None:
            self.joint_surplus_norms.append(ep.joint_surplus_norm)

        if ep.outcome == "deal":
            self.deal_count += 1
            if ep.learner_points is not None:
                self.learner_points_on_deals.append(ep.learner_points)
            if ep.opponent_points is not None:
                self.opponent_points_on_deals.append(ep.opponent_points)
            self.turns_on_deals.append(ep.num_turns)
        elif ep.outcome == "walk_away":
            self.walk_away_count += 1
        elif ep.outcome == "reject_loop":
            self.reject_loop_count += 1
        else:
            self.max_turns_count += 1

    def summary(self) -> dict:
        """Compute summary statistics from accumulated metrics.

        Returns:
            Dict of computed statistics (rates, averages, counts).
        """
        import math

        n = max(self.total_episodes, 1)
        deals = self.learner_points_on_deals
        opp_deals = self.opponent_points_on_deals

        avg_lp = sum(deals) / len(deals) if deals else None
        avg_op = sum(opp_deals) / len(opp_deals) if opp_deals else None

        joint_scores = [lp + op for lp, op in zip(deals, opp_deals, strict=False)]
        avg_joint = (
            round(sum(joint_scores) / len(joint_scores), 2) if joint_scores else None
        )

        score_ratios = [
            lp / (lp + op) if (lp + op) > 0 else 0.5
            for lp, op in zip(deals, opp_deals, strict=False)
        ]
        avg_score_ratio = (
            round(sum(score_ratios) / len(score_ratios), 3) if score_ratios else None
        )

        if len(deals) >= 2:
            mean_lp = sum(deals) / len(deals)
            var_lp = sum((x - mean_lp) ** 2 for x in deals) / (len(deals) - 1)
            std_learner_points = round(math.sqrt(var_lp), 2)
        else:
            std_learner_points = None

        avg_turns_deal = (
            sum(self.turns_on_deals) / len(self.turns_on_deals)
            if self.turns_on_deals
            else None
        )
        avg_turns_all = (
            sum(self.all_turns) / len(self.all_turns) if self.all_turns else None
        )
        ppt = avg_lp / avg_turns_deal if (avg_lp and avg_turns_deal) else None

        lt = max(self.learner_total_turns, 1)
        ot = max(self.opponent_total_turns, 1)

        return {
            "total_episodes": self.total_episodes,
            "repeated_offer_episode_count": self.repeated_offer_episode_count,
            "repeated_offer_event_count": self.repeated_offer_event_count,
            "regulation_logged_episodes": self.regulation_logged_episodes,
            "regulation_interventions": self.regulation_intervention_count
            if self.regulation_logged_episodes
            else None,
            "regulation_episode_rate": self.regulated_episodes
            / self.regulation_logged_episodes
            if self.regulation_logged_episodes
            else None,
            "deal_rate": self.deal_count / n,
            "walk_away_rate": self.walk_away_count / n,
            "reject_loop_rate": self.reject_loop_count / n,
            "max_turns_rate": self.max_turns_count / n,
            "avg_learner_points": round(avg_lp, 2) if avg_lp is not None else None,
            "avg_opponent_points": round(avg_op, 2) if avg_op is not None else None,
            "std_learner_points": std_learner_points,
            "avg_joint_score": avg_joint,
            "avg_score_ratio": avg_score_ratio,
            "avg_turns_to_deal": round(avg_turns_deal, 2)
            if avg_turns_deal is not None
            else None,
            "avg_turns_all": round(avg_turns_all, 2)
            if avg_turns_all is not None
            else None,
            "points_per_turn": round(ppt, 3) if ppt is not None else None,
            "deal_count": self.deal_count,
            "walk_away_count": self.walk_away_count,
            "reject_loop_count": self.reject_loop_count,
            "max_turns_count": self.max_turns_count,
            "learner_format_rate": round(self.learner_format_ok / lt, 3),
            "opponent_format_rate": round(self.opponent_format_ok / ot, 3),
            "learner_total_turns": self.learner_total_turns,
            "opponent_total_turns": self.opponent_total_turns,
            "avg_bargained_ratio": round(
                sum(max(-1.0, min(1.0, br)) for br in self.bargained_ratios)
                / len(self.bargained_ratios),
                4,
            )
            if self.bargained_ratios
            else None,
            "avg_first_bid_ratio": round(
                sum(self.first_bid_ratios) / len(self.first_bid_ratios), 4
            )
            if self.first_bid_ratios
            else None,
            "pareto_rate": round(self.pareto_count / self.pareto_total, 4)
            if self.pareto_total > 0
            else None,
            "avg_joint_surplus_norm": round(
                sum(self.joint_surplus_norms) / len(self.joint_surplus_norms), 4
            )
            if self.joint_surplus_norms
            else None,
        }


def run_episode(
    env: NegotiateEnv,
    learner_gen: Callable[[list[dict], float, float], str],
    opponent_gen: Callable[[list[dict], float, float], str],
    scenario: dict,
    agent_ids: list[str],
    persona: str,
    episode_id: int,
    learner_label: str,
    opponent_label: str,
    max_turns: int = 18,
    temperature: float = 0.7,
    top_p: float = 0.9,
    *,
    learner_agent_id: str | None = None,
    learner_goes_first: bool | None = None,
    scenario_id: str = "",
    repeat: int = 0,
    economic_regulation: bool = True,
) -> EpisodeResult:
    """Run a single negotiation episode between learner and opponent.

    Args:
        env: Negotiation environment instance.
        learner_gen: API generator for the learner agent.
        opponent_gen: API generator for the opponent agent.
        scenario: Scenario data dict.
        agent_ids: List of agent IDs in the scenario.
        persona: Opponent persona name or 'none'.
        episode_id: Numeric episode identifier.
        learner_label: Display label for learner.
        opponent_label: Display label for opponent.
        max_turns: Maximum number of turns before timeout.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        learner_agent_id: Explicit manifest role; otherwise use environment/local RNG.
        learner_goes_first: Explicit manifest first mover; otherwise use local RNG.
        scenario_id: Stable scenario identity for paired analysis.
        repeat: Explicit generation-trial index (not a provider sampling seed).
        economic_regulation: Replace negative-utility opponent actions with rejection.

    Returns:
        EpisodeResult with outcome, points, messages, and metadata.
    """
    if type(economic_regulation) is not bool:
        raise ValueError("economic_regulation must be a boolean")
    fixed = env.fixed_learner_role()
    # Harness calls pass explicit assignments; standalone callers get local RNG.
    rng = random.Random()
    learner_id = learner_agent_id or fixed or rng.choice(agent_ids)
    if learner_id not in agent_ids or (fixed is not None and learner_id != fixed):
        raise ValueError("Invalid learner role for this environment")
    opponent_id = next(a for a in agent_ids if a != learner_id)
    if learner_goes_first is None:
        learner_goes_first = rng.random() < 0.5

    learner_system = env.build_system_prompt(scenario, learner_id)
    base_opponent_system = env.build_system_prompt(scenario, opponent_id)
    if persona and persona != "none" and persona in PERSONAS:
        opponent_system = PERSONAS[persona] + "\n\n" + base_opponent_system
    else:
        opponent_system = base_opponent_system

    learner_msgs: list[dict] = [{"role": "system", "content": learner_system}]
    opponent_msgs: list[dict] = [{"role": "system", "content": opponent_system}]

    result = EpisodeResult(
        episode_id=episode_id,
        learner_label=learner_label,
        opponent_label=opponent_label,
        persona=persona,
        outcome="max_turns",
        learner_agent_id=learner_id,
        opponent_agent_id=opponent_id,
        scenario_id=scenario_id,
        repeat=repeat,
        economic_regulation=economic_regulation,
        protocol_version=EVAL_PROTOCOL_VERSION,
    )

    result.learner_goes_first = learner_goes_first
    last_submit_deal: dict[str, int] | None = None
    last_submit_by: str | None = None

    for turn_index in range(max_turns):
        is_learner_turn = (turn_index % 2 == 0) == learner_goes_first

        speaker_msgs = learner_msgs if is_learner_turn else opponent_msgs
        receiver_msgs = opponent_msgs if is_learner_turn else learner_msgs
        speaker_id = learner_id if is_learner_turn else opponent_id
        generator = learner_gen if is_learner_turn else opponent_gen
        other_party = "opponent" if is_learner_turn else "learner"
        if len(speaker_msgs) == 1:
            speaker_msgs.append({"role": "user", "content": env.opening_prompt(speaker_id)})
        raw_response = generator(speaker_msgs, temperature, top_p)
        parsed = parse_response(raw_response, env, scenario, last_submit_by == other_party)

        # Regulate only a fully valid opponent response. Malformed responses
        # cannot become valid rejections through the economic guard.
        opp_points = None
        if not is_learner_turn and economic_regulation and parsed.error is None:
            if parsed.kind == "ACCEPT_DEAL":
                assert last_submit_deal is not None
                opp_alloc = env.invert_alloc(last_submit_deal, scenario)
                opp_points = env.compute_points(opp_alloc, scenario, opponent_id)
            elif parsed.kind == "SUBMIT_DEAL":
                opp_alloc = env.parse_deal(parsed.action)
                assert opp_alloc is not None
                opp_points = env.compute_points(opp_alloc, scenario, opponent_id)
        if opp_points is not None and opp_points < 0:
            original_response, original_parsed = raw_response, parsed
            raw_response = (
                "Thought: This offer leaves me with too little.\n"
                "Talk: I can't accept that offer.\n"
                "Action: [REJECT_DEAL]"
            )
            parsed = parse_response(raw_response, env, scenario, False)
            result.regulation_interventions.append({
                "turn_index": turn_index,
                "opponent_agent_id": opponent_id,
                "opponent_utility": opp_points,
                "original_response": original_response,
                "replacement_response": raw_response,
                "reason": "negative_opponent_utility",
                "opponent_message_index": len(opponent_msgs),
                "learner_message_index": len(learner_msgs),
                "original_turn": {
                    "role": "assistant", "content": original_response,
                    "action": original_parsed.action,
                    "learner_visible_content": partner_observation(original_parsed, env, scenario),
                },
                "final_turn": {
                    "role": "assistant", "content": raw_response,
                    "action": parsed.action,
                    "learner_visible_content": partner_observation(parsed, env, scenario),
                },
            })

        # Keep raw output on the speaker's side; expose only validated public
        # content (or an explicit invalid-turn observation) to the receiver.
        speaker_msgs.append({"role": "assistant", "content": raw_response})
        receiver_msgs.append({"role": "user", "content": partner_observation(parsed, env, scenario)})
        action = parsed.action
        format_ok = parsed.error is None
        if not format_ok:
            speaker_msgs[-1]["format_error"] = parsed.error
            result.format_failures.append({
                "turn_index": turn_index,
                "speaker": "learner" if is_learner_turn else "opponent",
                "reason": parsed.error,
                "raw_response": raw_response,
            })

        if is_learner_turn:
            result.learner_total_turns += 1
            if format_ok:
                result.learner_format_ok += 1
            learner_msgs[-1]["format_ok"] = format_ok
        else:
            result.opponent_total_turns += 1
            if format_ok:
                result.opponent_format_ok += 1
            opponent_msgs[-1]["format_ok"] = format_ok

        if not format_ok:
            # Invalid actions cannot create or accept an allocation. Invalidate
            # any pending offer so a later acceptance cannot revive stale terms.
            last_submit_deal = None
            last_submit_by = None
            continue
        action = action.upper()
        if "[REJECT_DEAL]" in action:
            last_submit_deal = None
            last_submit_by = None
        # Parse original case-preserving text (categorical JI values matter).
        if action:
            parsed_deal = env.parse_deal(parsed.action)
            if parsed_deal is not None:
                last_submit_deal = parsed_deal
                last_submit_by = "learner" if is_learner_turn else "opponent"

        other_party = "opponent" if is_learner_turn else "learner"
        if action and "[ACCEPT_DEAL]" in action and last_submit_by == other_party:
            result.outcome = "deal"
            result.who_terminated = "learner" if is_learner_turn else "opponent"
            if last_submit_deal is not None:
                learner_alloc = (
                    env.invert_alloc(last_submit_deal, scenario)
                    if is_learner_turn
                    else last_submit_deal
                )
                opponent_alloc = (
                    last_submit_deal
                    if is_learner_turn
                    else env.invert_alloc(last_submit_deal, scenario)
                )
                result.deal = learner_alloc
                result.learner_points = env.compute_points(
                    learner_alloc, scenario, learner_id
                )
                result.opponent_points = env.compute_points(
                    opponent_alloc, scenario, opponent_id
                )
            result.num_turns = turn_index + 1
            break

        if action and "[WALK_AWAY]" in action:
            result.outcome = "walk_away"
            result.who_terminated = "learner" if is_learner_turn else "opponent"
            result.num_turns = turn_index + 1
            break

        whose_msgs = learner_msgs if is_learner_turn else opponent_msgs
        who_label = "learner" if is_learner_turn else "opponent"
        recent_deals: list[str] = []
        for msg in reversed(whose_msgs):
            if msg["role"] != "assistant":
                continue
            a = extract_section(strip_think_block(msg["content"]), "Action")
            if a and "[SUBMIT_DEAL]" in a:
                recent_deals.append(a)
            if len(recent_deals) >= 3:
                break
        if parsed.kind == "SUBMIT_DEAL" and len(recent_deals) >= 3 and len(set(recent_deals)) == 1:
            # Repetition is diagnostic only: allow the counterpart to respond,
            # and let agreement, walk-away, or the turn budget end the episode.
            result.repeated_offer_events.append({
                "turn_index": turn_index,
                "speaker": who_label,
                "action": parsed.action,
            })
    else:
        result.num_turns = max_turns

    result.learner_messages = learner_msgs
    result.opponent_messages = opponent_msgs
    return result


def compute_episode_metrics(
    ep: EpisodeResult, env: NegotiateEnv, scenario: dict
) -> None:
    """Compute bargained ratio, Pareto, joint surplus metrics on a completed episode.

    Args:
        ep: Episode result to mutate with computed metrics.
        env: Negotiation environment.
        scenario: Scenario data dict.
    """
    if ep.outcome != "deal" or ep.learner_points is None:
        _compute_first_bid(ep, scenario)
        return

    max_pts = env.max_points(scenario, ep.learner_agent_id)
    if max_pts > 0:
        ep.bargained_ratio = round(ep.learner_points / max_pts, 4)

    max_joint = env.max_joint_points(scenario)
    if max_joint is not None and max_joint > 0 and ep.opponent_points is not None:
        ep.joint_surplus_norm = round(
            (ep.learner_points + ep.opponent_points) / max_joint, 4
        )

    if ep.deal is not None:
        ep.pareto_efficient = env.is_pareto_efficient(
            ep.deal, scenario, ep.learner_agent_id
        )

    _compute_first_bid(ep, scenario)


def _compute_first_bid(ep: EpisodeResult, scenario: dict) -> None:
    """Extract learner's first [SUBMIT_DEAL] price and compute first_bid_ratio."""
    if "buyer_budget" not in scenario:
        return
    for msg in ep.learner_messages:
        if msg["role"] != "assistant":
            continue
        content = strip_think_block(msg["content"])
        action_text = extract_section(content, "Action")
        if not action_text or "[SUBMIT_DEAL]" not in action_text:
            continue
        m = re.search(r"price:\s*\$?\s*(\d+(?:\.\d{1,2})?)", action_text, re.IGNORECASE)
        if m:
            price = round(float(m.group(1)))
            if ep.learner_agent_id == "buyer":
                ep.first_bid_ratio = round(price / scenario["buyer_budget"], 4)
            else:
                ep.first_bid_ratio = round(price / scenario["seller_cost"], 4)
        return
