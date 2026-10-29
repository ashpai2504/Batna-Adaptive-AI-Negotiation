from __future__ import annotations

import ast
import csv
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from .prompts import (
    BUYER_SYSTEM_PROMPT,
    CASINO_SYSTEM_PROMPT,
    DND_SYSTEM_PROMPT,
    JI_SYSTEM_PROMPT,
    SELLER_SYSTEM_PROMPT,
)


class NegotiateEnv(ABC):
    """Interface that dataset-specific negotiation envs must implement."""

    divides_items = False

    def describe_proposal(self, deal: dict, scenario: dict) -> str:
        """Describe a proposer's deal to its receiver without changing speech."""
        def terms(allocation):
            return " ".join(f"{key}:{value}" for key, value in allocation.items())

        if self.divides_items:
            return (
                f"You receive: {terms(self.invert_alloc(deal, scenario))}\n"
                f"Partner receives: {terms(deal)}"
            )
        return f"Proposed terms: {terms(deal)}"

    @abstractmethod
    def load_scenarios(self, csv_path: str, /) -> list[dict]:
        """Load negotiation scenarios from a data file.

        Args:
            csv_path: Path to the dataset file (CSV, JSON, or directory).

        Returns:
            List of scenario dicts, each containing agent_ids and dataset-specific fields.
        """
        ...

    @abstractmethod
    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        """Build the system prompt for a given agent in a scenario.

        Args:
            scenario: Scenario data dict.
            agent_id: Which agent's perspective to build the prompt for.

        Returns:
            Formatted system prompt string.
        """
        ...

    @abstractmethod
    def parse_deal(self, action_text: str) -> dict | None:
        """Parse a complete [SUBMIT_DEAL] action into a structured allocation.

        Reject unconsumed text instead of accepting a numeric prefix (for
        example, interpreting an indivisible-item quantity of 0.5 as 0).

        Args:
            action_text: Raw action text from model output.

        Returns:
            Allocation dict if parseable, None otherwise.
        """
        ...

    @abstractmethod
    def flip_deal(self, text: str, scenario: dict) -> str:
        """Rewrite deal allocations from the proposer's to the receiver's perspective.

        Args:
            text: Message text potentially containing deal allocations.
            scenario: Scenario data dict.

        Returns:
            Text with allocations flipped for the other agent.
        """
        ...

    @abstractmethod
    def compute_points(self, deal: dict, scenario: dict, agent_id: str) -> int:
        """Compute point score for a given deal allocation.

        Args:
            deal: Allocation mapping.
            scenario: Full scenario data dict.
            agent_id: Which agent's perspective to score from.

        Returns:
            Numeric point score for the agent.
        """
        ...

    @abstractmethod
    def validate_deal(self, deal: dict, scenario: dict) -> bool:
        """Check whether a parsed deal is valid for the scenario.

        Args:
            deal: Parsed allocation dict.
            scenario: Scenario data dict.

        Returns:
            True if the deal is valid.
        """
        ...

    @abstractmethod
    def invert_alloc(self, deal: dict, scenario: dict) -> dict:
        """Invert an allocation to get the other agent's portion.

        Args:
            deal: Allocation from one agent's perspective.
            scenario: Scenario data dict.

        Returns:
            Allocation from the other agent's perspective.
        """
        ...

    @abstractmethod
    def max_points(self, scenario: dict, agent_id: str) -> int:
        """Maximum achievable points for a single agent.

        Args:
            scenario: Scenario data dict.
            agent_id: Which agent.

        Returns:
            Maximum possible points.
        """
        ...

    def max_joint_points(self, scenario: dict) -> int | None:
        """Maximum sum of both agents' points. None if not applicable."""
        return None

    def is_pareto_efficient(
        self, deal: dict, scenario: dict, agent_id: str
    ) -> bool | None:
        """Whether deal (from agent_id's perspective) is Pareto efficient."""
        return None

    def fixed_learner_role(self) -> str | None:
        """If non-None, the learner is always assigned this agent_id."""
        return None

    def opening_prompt(self, agent_id: str) -> str:
        """First user message for the agent who goes first."""
        return "Begin the negotiation."


def _max_joint(env: NegotiateEnv, scenario: dict, all_allocs) -> int:
    a, b = scenario["agent_ids"]
    best = 0
    for alloc in all_allocs:
        inv = env.invert_alloc(alloc, scenario)
        joint = env.compute_points(alloc, scenario, a) + env.compute_points(
            inv, scenario, b
        )
        if joint > best:
            best = joint
    return best


def _pareto_check(
    env: NegotiateEnv, deal: dict, scenario: dict, agent_id: str, all_allocs
) -> bool:
    other = [aid for aid in scenario["agent_ids"] if aid != agent_id][0]
    inv = env.invert_alloc(deal, scenario)
    my_pts = env.compute_points(deal, scenario, agent_id)
    opp_pts = env.compute_points(inv, scenario, other)
    for alt in all_allocs:
        alt_inv = env.invert_alloc(alt, scenario)
        alt_my = env.compute_points(alt, scenario, agent_id)
        alt_opp = env.compute_points(alt_inv, scenario, other)
        if (alt_my >= my_pts and alt_opp >= opp_pts) and (
            alt_my > my_pts or alt_opp > opp_pts
        ):
            return False
    return True


_CASINO_POINTS = {"High": 5, "Medium": 4, "Low": 3}

_CASINO_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*food:(\d+)\s*water:(\d+)\s*firewood:(\d+)", re.IGNORECASE
)


class CasinoEnv(NegotiateEnv):
    """CaSiNo: food/water/firewood, 3 packages each, High/Medium/Low priorities."""

    divides_items = True

    def load_scenarios(self, csv_path: str) -> list[dict]:
        scenarios = []
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pi = ast.literal_eval(row["participant_info"])
                scenarios.append({"participant_info": pi, "agent_ids": list(pi.keys())})
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        pi = scenario["participant_info"]
        v2i = pi[agent_id]["value2issue"]
        v2r = pi[agent_id]["value2reason"]
        items_block = "\n  ".join(
            f"{v2i[p]} ({_CASINO_POINTS[p]} points) - {v2r[p]}" for p in _CASINO_POINTS
        )
        return CASINO_SYSTEM_PROMPT.format(items_block=items_block)

    def parse_deal(self, action_text: str) -> dict[str, int] | None:
        m = _CASINO_DEAL_RE.fullmatch(action_text.strip())
        if m is None:
            return None
        return {
            "food": int(m.group(1)),
            "water": int(m.group(2)),
            "firewood": int(m.group(3)),
        }

    def flip_deal(self, text: str, scenario: dict) -> str:
        def _flip(m: re.Match) -> str:
            f, w, fw = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"[SUBMIT_DEAL] food:{3 - f} water:{3 - w} firewood:{3 - fw}"

        return _CASINO_DEAL_RE.sub(_flip, text)

    def compute_points(
        self, deal: dict[str, int], scenario: dict, agent_id: str
    ) -> int:
        pi = scenario["participant_info"]
        point_map = {
            pi[agent_id]["value2issue"][lv].lower(): pts
            for lv, pts in _CASINO_POINTS.items()
        }
        return sum(qty * point_map.get(item.lower(), 0) for item, qty in deal.items())

    def validate_deal(self, deal: dict[str, int], scenario: dict) -> bool:
        return deal is not None and all(0 <= v <= 3 for v in deal.values())

    def invert_alloc(self, deal: dict[str, int], scenario: dict) -> dict[str, int]:
        return {item: 3 - qty for item, qty in deal.items()}

    def max_points(self, scenario: dict, agent_id: str) -> int:
        return 3 * 5 + 3 * 4 + 3 * 3

    def _all_allocations(self):
        for f in range(4):
            for w in range(4):
                for fw in range(4):
                    yield {"food": f, "water": w, "firewood": fw}

    def max_joint_points(self, scenario: dict) -> int:
        return _max_joint(self, scenario, self._all_allocations())

    def is_pareto_efficient(self, deal: dict, scenario: dict, agent_id: str) -> bool:
        return _pareto_check(self, deal, scenario, agent_id, self._all_allocations())


_DND_ITEMS = ("book", "hat", "ball")

_DND_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*book:(\d+)\s*hat:(\d+)\s*ball:(\d+)", re.IGNORECASE
)


class DNDEnv(NegotiateEnv):
    """Deal or No Deal: book/hat/ball, variable counts, raw point values."""

    divides_items = True

    def load_scenarios(self, csv_path: str) -> list[dict]:
        """Load each distinct allocation context once, regardless of seat order.

        Rows are not guaranteed to be adjacent mirrored pairs. Read every row
        and deduplicate by item counts and the unordered pair of valuations.
        Preserve the first occurrence's agent ordering for prompts and scoring.
        This samples contexts uniformly instead of weighting repeated dialogues.
        """
        scenarios = []
        seen = set()
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row_number, row in enumerate(rows, start=2):
            my = json.loads(row["input"])
            partner = json.loads(row["partner_input"])
            if (
                len(my["count"]) != 3
                or my["count"] != partner["count"]
                or len(my["value"]) != 3
                or len(partner["value"]) != 3
            ):
                raise ValueError(
                    f"Invalid DnD counts or valuations at {csv_path}:{row_number}"
                )
            key = (
                tuple(my["count"]),
                tuple(sorted((tuple(my["value"]), tuple(partner["value"])))),
            )
            if key in seen:
                continue
            seen.add(key)
            counts = {
                "book": my["count"][0],
                "hat": my["count"][1],
                "ball": my["count"][2],
            }
            scenarios.append(
                {
                    "counts": counts,
                    "agents": {
                        "agent_0": {
                            "value": {
                                "book": my["value"][0],
                                "hat": my["value"][1],
                                "ball": my["value"][2],
                            }
                        },
                        "agent_1": {
                            "value": {
                                "book": partner["value"][0],
                                "hat": partner["value"][1],
                                "ball": partner["value"][2],
                            }
                        },
                    },
                    "agent_ids": ["agent_0", "agent_1"],
                }
            )
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        counts = scenario["counts"]
        values = scenario["agents"][agent_id]["value"]

        counts_desc = ", ".join(f"{counts[item]} {item}(s)" for item in _DND_ITEMS)
        items_block = "\n  ".join(
            f"{item}: {values[item]} points each (x{counts[item]} available)"
            for item in _DND_ITEMS
        )

        return DND_SYSTEM_PROMPT.format(
            counts_desc=counts_desc,
            items_block=items_block,
        )

    def parse_deal(self, action_text: str) -> dict[str, int] | None:
        m = _DND_DEAL_RE.fullmatch(action_text.strip())
        if m is None:
            return None
        return {
            "book": int(m.group(1)),
            "hat": int(m.group(2)),
            "ball": int(m.group(3)),
        }

    def flip_deal(self, text: str, scenario: dict) -> str:
        counts = scenario["counts"]

        def _flip(m: re.Match) -> str:
            b, h, ba = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return (
                f"[SUBMIT_DEAL] book:{counts['book'] - b} "
                f"hat:{counts['hat'] - h} ball:{counts['ball'] - ba}"
            )

        return _DND_DEAL_RE.sub(_flip, text)

    def compute_points(
        self, deal: dict[str, int], scenario: dict, agent_id: str
    ) -> int:
        values = scenario["agents"][agent_id]["value"]
        return sum(deal.get(item, 0) * values.get(item, 0) for item in _DND_ITEMS)

    def validate_deal(self, deal: dict[str, int], scenario: dict) -> bool:
        if deal is None:
            return False
        counts = scenario["counts"]
        return all(0 <= deal.get(item, -1) <= counts[item] for item in _DND_ITEMS)

    def invert_alloc(self, deal: dict[str, int], scenario: dict) -> dict[str, int]:
        counts = scenario["counts"]
        return {item: counts[item] - deal.get(item, 0) for item in _DND_ITEMS}

    def max_points(self, scenario: dict, agent_id: str) -> int:
        counts = scenario["counts"]
        values = scenario["agents"][agent_id]["value"]
        return sum(counts[item] * values[item] for item in _DND_ITEMS)

    def _all_allocations(self, scenario: dict):
        c = scenario["counts"]
        for b in range(c["book"] + 1):
            for h in range(c["hat"] + 1):
                for ba in range(c["ball"] + 1):
                    yield {"book": b, "hat": h, "ball": ba}

    def max_joint_points(self, scenario: dict) -> int:
        return _max_joint(self, scenario, self._all_allocations(scenario))

    def is_pareto_efficient(self, deal: dict, scenario: dict, agent_id: str) -> bool:
        return _pareto_check(
            self, deal, scenario, agent_id, self._all_allocations(scenario)
        )


_PRICE_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*price:\s*\$?\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE
)


def _parse_price(price_str: str) -> int:
    """Parse a price string like '$49.99' into an integer."""
    return round(float(price_str.replace("$", "").replace(",", "")))


class _PriceEnv(NegotiateEnv):
    """Base for single-issue price negotiation (Amazon, Craigslist)."""

    def parse_deal(self, action_text: str) -> dict | None:
        m = _PRICE_DEAL_RE.fullmatch(action_text.strip())
        if m is None:
            return None
        return {"price": round(float(m.group(1)))}

    def flip_deal(self, text: str, scenario: dict) -> str:
        return text

    def compute_points(self, deal: dict, scenario: dict, agent_id: str) -> int:
        price = deal["price"]
        if agent_id == "buyer":
            return scenario["buyer_budget"] - price
        return price - scenario["seller_cost"]

    def validate_deal(self, deal: dict, scenario: dict) -> bool:
        if deal is None:
            return False
        p = deal.get("price")
        return p is not None and p >= 0

    def invert_alloc(self, deal: dict, scenario: dict) -> dict:
        return deal

    def max_points(self, scenario: dict, agent_id: str) -> int:
        return scenario["buyer_budget"] - scenario["seller_cost"]

    def fixed_learner_role(self) -> str | None:
        return "buyer"

    def opening_prompt(self, agent_id: str) -> str:
        if agent_id == "buyer":
            return "You found this listing. Make your opening offer to the seller."
        return "A buyer is interested in your product. Make your opening offer."

    def _build_price_prompt(self, scenario: dict, agent_id: str) -> str:
        if agent_id == "buyer":
            return BUYER_SYSTEM_PROMPT.format(
                title=scenario["title"],
                category=scenario["category"],
                listing_price=scenario["listing_price"],
                description=scenario["description"],
                buyer_budget=scenario["buyer_budget"],
            )
        return SELLER_SYSTEM_PROMPT.format(
            title=scenario["title"],
            category=scenario["category"],
            listing_price=scenario["listing_price"],
            description=scenario["description"],
            seller_cost=scenario["seller_cost"],
        )


class AmazonPriceEnv(_PriceEnv):
    """AmazonHistoryPrice: buyer/seller negotiate a single product price.

    B = current_price (market price), C = lowest_price (historical low).
    """

    def load_scenarios(self, data_path: str) -> list[dict]:
        path = Path(data_path)
        products: list[dict] = []
        if path.is_dir():
            for json_file in sorted(path.glob("*.json")):
                with open(json_file, encoding="utf-8") as f:
                    products.extend(json.load(f))
        else:
            with open(path, encoding="utf-8") as f:
                products = json.load(f)

        scenarios = []
        for p in products:
            try:
                list_price = _parse_price(p["list_price"])
                current_price = _parse_price(p["current_price"])
                lowest_price = _parse_price(p["lowest_price"])
            except (ValueError, KeyError):
                continue
            if current_price <= lowest_price:
                continue

            desc_parts = []
            if p.get("description"):
                desc_parts.append(p["description"][:300])
            if p.get("features"):
                desc_parts.append("Features: " + p["features"][:200])
            description = "\n".join(desc_parts) if desc_parts else ""

            scenarios.append(
                {
                    "title": p.get("title", "Unknown Product"),
                    "category": p.get("category", ""),
                    "description": description,
                    "listing_price": list_price,
                    "buyer_budget": current_price,
                    "seller_cost": lowest_price,
                    "agent_ids": ["buyer", "seller"],
                }
            )
        return scenarios

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        return self._build_price_prompt(scenario, agent_id)


class CraigslistEnv(_PriceEnv):
    """Craigslist Bargains: buyer/seller negotiate a Craigslist listing price.

    B = buyer_target (from data), C = round(listing_price * 0.5).
    """

    def load_scenarios(self, data_path: str) -> list[dict]:
        with open(data_path, encoding="utf-8") as f:
            dialogues = json.load(f)

        from .craigslist_data import loaded_scenario

        return [
            scenario
            for row in dialogues
            if (scenario := loaded_scenario(row)) is not None
        ]

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        return self._build_price_prompt(scenario, agent_id)


# ─── Job Interview helpers (inlined from negotiation_ji module) ───────────────

_JI_POSITIONS = ("Engineer", "Manager", "Designer", "Sales")
_JI_COMPANIES = ("Google", "Facebook", "Apple", "Amazon")
_JI_WORKPLACES = ("Tokyo", "Seoul", "Beijing", "Sydney")

_JI_DEAL_RE = re.compile(
    r"\[SUBMIT_DEAL\]\s*salary:\s*\$?(\d+)\s+position:\s*(\w+)"
    r"\s+company:\s*(\w+)\s+workplace:\s*(\w+)\s+holiday:\s*(\d+)",
    re.IGNORECASE,
)


class _JIBid:
    """Minimal bid representation for JI scoring."""

    def __init__(self, options: dict[str, str | int]):
        self.options = options


class _JIUser:
    """Minimal user representation for JI scoring."""

    def __init__(self, context: dict):
        self.context = context

    def calc_score(self, bid: _JIBid) -> float:
        """Compute normalized utility score for a bid.

        Args:
            bid: Bid with options dict mapping issue names to chosen values.

        Returns:
            Score in [0, 1] representing total weighted utility.
        """
        score = 0.0
        role = self.context["role"]
        utilities = self.context["utilities"]
        for issue_name, option in bid.options.items():
            issue_utility = next(u for u in utilities if u["name"] == issue_name)
            weight = issue_utility["weight"]

            if issue_utility["type"] == "INTEGER":
                opt_max = issue_utility["max"]
                opt_min = issue_utility["min"]
                if role == "recruiter":
                    score += weight * (opt_max - option) / (opt_max - opt_min)
                else:
                    score += weight * (option - opt_min) / (opt_max - opt_min)
            elif "relatedTo" in issue_utility:
                related = issue_utility["relatedTo"]
                for o in issue_utility["options"]:
                    if (
                        o["names"][related] == bid.options[related]
                        and o["names"][issue_name] == option
                    ):
                        score += o["weight"] * weight
                        break
            else:
                for o in issue_utility["options"]:
                    if o["name"] == option:
                        score += o["weight"] * weight
                        break
        return score


def _read_ji_negotiations(file_name: str) -> list[dict]:
    """Read JI negotiation JSON and return list of scenario dicts with user contexts.

    Args:
        file_name: Path to the JI JSON data file.

    Returns:
        List of scenario dicts with 'users' and 'agent_ids' keys.
    """
    negos_raw = json.loads(Path(file_name).read_text())
    scenarios = []

    for nego_raw in negos_raw:
        users_raw = nego_raw["users"]
        users: dict[str, _JIUser] = {}
        for u in users_raw:
            role = u["role"]
            users[role] = _JIUser({"role": role, "utilities": u["utilities"]})

        if "worker" not in users or "recruiter" not in users:
            continue

        has_solution = any(s.get("accepted") for s in nego_raw.get("solutions", []))
        if nego_raw.get("status") != "completed" or not has_solution:
            continue

        scenarios.append({"users": users, "agent_ids": ["worker", "recruiter"]})

    return scenarios


def _ji_preferences_block(utilities: list[dict], role: str) -> str:
    lines = []
    for util in utilities:
        name = util["name"]
        pct = f"{util['weight'] * 100:.0f}%"

        if util["type"] == "INTEGER":
            if name == "Salary":
                direction = "higher" if role == "worker" else "lower"
                lines.append(
                    f"  Salary (importance: {pct}): you prefer {direction} salary"
                )
            else:
                direction = "more" if role == "worker" else "fewer"
                lines.append(
                    f"  Days off (importance: {pct}): you prefer {direction} days off"
                )
        elif "relatedTo" in util:
            related = util["relatedTo"]
            combos = sorted(
                util["options"], key=lambda o: o.get("weight", 0), reverse=True
            )
            top = ", ".join(
                f"{o['names'][name]}@{o['names'][related]}" for o in combos[:6]
            )
            lines.append(
                f"  {name} (importance: {pct}): depends on {related} — "
                f"best combos: {top}"
            )
        else:
            opts = sorted(
                util["options"], key=lambda o: o.get("weight", 0), reverse=True
            )
            ranked = " > ".join(o["name"] for o in opts)
            lines.append(f"  {name} (importance: {pct}): preference {ranked}")

    return "\n".join(lines)


class JobInterviewEnv(NegotiateEnv):
    """Job Interview: multi-attribute hybrid negotiation (eval only).

    5 issues: Salary, Position, Company, Workplace, Weekly holiday.
    Position x Company have interdependent utilities.
    """

    def load_scenarios(self, data_path: str) -> list[dict]:
        return _read_ji_negotiations(data_path)

    def build_system_prompt(self, scenario: dict, agent_id: str) -> str:
        user = scenario["users"][agent_id]
        role = user.context["role"]
        other_role = "recruiter" if role == "worker" else "applicant"
        role_desc = "job applicant" if role == "worker" else "hiring recruiter"
        preferences = _ji_preferences_block(user.context["utilities"], role)
        return JI_SYSTEM_PROMPT.format(
            role_desc=role_desc,
            other_role=other_role,
            preferences_block=preferences,
        )

    def parse_deal(self, action_text: str) -> dict | None:
        m = _JI_DEAL_RE.fullmatch(action_text.strip())
        if m is None:
            return None
        return {
            "Salary": int(m.group(1)),
            "Position": m.group(2).capitalize(),
            "Company": m.group(3).capitalize(),
            "Workplace": m.group(4).capitalize(),
            "Weekly holiday": int(m.group(5)),
        }

    def flip_deal(self, text: str, scenario: dict) -> str:
        return text

    def compute_points(self, deal: dict, scenario: dict, agent_id: str) -> int:
        user = scenario["users"][agent_id]
        bid = _JIBid(deal)
        return round(user.calc_score(bid) * 100)

    def validate_deal(self, deal: dict, scenario: dict) -> bool:
        if deal is None:
            return False
        s = deal.get("Salary")
        h = deal.get("Weekly holiday")
        if s is None or h is None:
            return False
        if not (20 <= s <= 50 and 2 <= h <= 6):
            return False
        if deal.get("Position") not in _JI_POSITIONS:
            return False
        if deal.get("Company") not in _JI_COMPANIES:
            return False
        return deal.get("Workplace") in _JI_WORKPLACES

    def invert_alloc(self, deal: dict, scenario: dict) -> dict:
        return deal

    def max_points(self, scenario: dict, agent_id: str) -> int:
        user = scenario["users"][agent_id]
        best = 0
        for s in range(20, 51):
            for p in _JI_POSITIONS:
                for c in _JI_COMPANIES:
                    for w in _JI_WORKPLACES:
                        for h in range(2, 7):
                            deal = {
                                "Salary": s,
                                "Position": p,
                                "Company": c,
                                "Workplace": w,
                                "Weekly holiday": h,
                            }
                            bid = _JIBid(deal)
                            pts = round(user.calc_score(bid) * 100)
                            if pts > best:
                                best = pts
        return best

    def fixed_learner_role(self) -> str | None:
        return "worker"

    def opening_prompt(self, agent_id: str) -> str:
        if agent_id == "worker":
            return "You are starting a job negotiation. State your initial terms to the recruiter."
        return "You have a candidate for this role. Make your opening offer to the applicant."


ENVS: dict[str, type[NegotiateEnv]] = {
    "casino": CasinoEnv,
    "dnd": DNDEnv,
    "amazon": AmazonPriceEnv,
    "craigslist": CraigslistEnv,
    "ji": JobInterviewEnv,
}


def get_env(dataset: str) -> NegotiateEnv:
    """Get a negotiation environment instance by dataset name.

    Args:
        dataset: One of 'casino', 'dnd', 'amazon', 'craigslist', 'ji'.

    Returns:
        Instantiated environment.

    Raises:
        ValueError: If dataset name is unknown.
    """
    cls = ENVS.get(dataset)
    if cls is None:
        raise ValueError(f"Unknown dataset {dataset!r}, expected one of {list(ENVS)}")
    return cls()
