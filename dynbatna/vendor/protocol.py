"""Live negotiation protocol shared by training and evaluation."""
import re
from dataclasses import dataclass

# v3 requires complete proposal parsing; fractional item counts cannot be truncated.
PROTOCOL_VERSION = "explicit-allocation-v3"
_HEADERS = re.compile(r"^(Thought|Talk|Action):[ \t]*", re.MULTILINE)
INVALID_RESPONSE_OBSERVATION = (
    "Environment: Your partner's response was invalid and did not create or "
    "accept a deal. This turn was consumed. There is no active offer to accept. "
    "Continue with a new proposal, rejection, or walk-away in the required format."
)


@dataclass(frozen=True)
class ParsedResponse:
    thought: str = ""
    talk: str = ""
    action: str = ""
    error: str | None = None

    @property
    def kind(self) -> str:
        return self.action.split("]", 1)[0].lstrip("[").upper() if not self.error else ""


def parse_response(text, env, scenario, has_opponent_offer):
    """Require exactly one nonempty section of each kind, in order.

    Complete native thinking blocks are private and separate from the required
    Thought field. Never salvage the last Action from a malformed response.
    """
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if "```" in text:
        return ParsedResponse(error="code_fence")
    headers = list(_HEADERS.finditer(text))
    names = [h.group(1) for h in headers]
    if any(names.count(name) > 1 for name in ("Thought", "Talk", "Action")):
        return ParsedResponse(error="duplicate_section")
    missing = [name for name in ("Thought", "Talk", "Action") if name not in names]
    if missing:
        return ParsedResponse(error="missing_section:" + ",".join(missing))
    if names != ["Thought", "Talk", "Action"]:
        return ParsedResponse(error="section_order")
    if text[:headers[0].start()].strip():
        return ParsedResponse(error="unexpected_preamble")
    sections = [
        text[h.end():headers[i + 1].start() if i < 2 else len(text)].strip()
        for i, h in enumerate(headers)
    ]
    empty = [name for name, content in zip(names, sections, strict=True) if not content]
    if empty:
        return ParsedResponse(*sections, error="empty_section:" + ",".join(empty))
    if "\n" in sections[2]:
        return ParsedResponse(*sections, error="multiline_action")
    if not valid_action(sections[2], env, scenario, has_opponent_offer):
        error = ("accept_without_offer"
                 if sections[2].upper() == "[ACCEPT_DEAL]" and not has_opponent_offer
                 else "invalid_action_or_allocation")
        return ParsedResponse(*sections, error=error)
    return ParsedResponse(*sections)


def partner_observation(response, env, scenario):
    """Expose validated public speech and explicitly labeled deal ownership."""
    if response.error:
        return INVALID_RESPONSE_OBSERVATION
    message = f"Partner says: {response.talk}"
    if response.kind == "SUBMIT_DEAL":
        deal = env.parse_deal(response.action)
        return message + "\n\nPartner action: [SUBMIT_DEAL]\n" + env.describe_proposal(deal, scenario)
    return message + f"\n\nPartner action: [{response.kind}]"


def valid_action(action, env, scenario, has_opponent_offer):
    tags = re.findall(r"\[([A-Z_]+)\]", action, re.IGNORECASE)
    if len(tags) != 1:
        return False
    kind = tags[0].upper()
    if kind == "SUBMIT_DEAL":
        if not re.match(r"^\s*\[SUBMIT_DEAL\]", action, re.IGNORECASE):
            return False
        deal = env.parse_deal(action)
        return deal is not None and env.validate_deal(deal, scenario)
    if kind not in {"ACCEPT_DEAL", "REJECT_DEAL", "WALK_AWAY"}:
        return False
    if not re.fullmatch(r"\s*\[" + kind + r"\]\s*", action, re.IGNORECASE):
        return False
    return kind != "ACCEPT_DEAL" or has_opponent_offer
