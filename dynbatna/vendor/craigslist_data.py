"""Craigslist scenario parsing and content-based identities for split repair."""

from __future__ import annotations

import hashlib
import json
import unicodedata


def content_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def buyer_kb(dialogue):
    return next(
        (kb for kb in dialogue["scenario"]["kbs"] if kb["personal"]["Role"] == "buyer"),
        None,
    )


def loaded_scenario(dialogue):
    """The exact scenario exposed by CraigslistEnv; None for filtered rows."""
    kb = buyer_kb(dialogue)
    if kb is None:
        return None
    item = kb["item"]
    listing_price = item["Price"]
    budget = kb["personal"]["Target"]
    cost = round(listing_price * 0.5)
    if budget <= cost:
        return None
    raw_description = item.get("Description", [])
    description = (
        " ".join(raw_description)
        if isinstance(raw_description, list)
        else raw_description
    )
    return {
        "title": item.get("Title", ""),
        "category": dialogue["scenario"].get("category", ""),
        "description": (description or "")[:300],
        "listing_price": listing_price,
        "buyer_budget": budget,
        "seller_cost": cost,
        "agent_ids": ["buyer", "seller"],
    }


def listing_content_id(dialogue):
    kb = buyer_kb(dialogue)
    if kb is None:
        return None
    item = kb["item"]

    def normalize(text):
        return " ".join(
            unicodedata.normalize("NFKC", str(text or "")).casefold().split()
        )

    description = item.get("Description", [])
    if isinstance(description, list):
        description = " ".join(description)
    title, description = normalize(item.get("Title", "")), normalize(description)
    # Avoid merging empty/generic unidentified rows by a blank content hash.
    if not title:
        return None
    return content_hash(
        [
            normalize(item.get("Category", dialogue["scenario"].get("category", ""))),
            title,
            description,
        ]
    )


def identity_keys(dialogue):
    """Exclude scenario UUIDs: the source reuses them for unrelated listings."""
    keys = []
    for kind, value in [
        ("dialogue_uuid", dialogue.get("uuid")),
        ("listing_post_id", dialogue["scenario"].get("post_id")),
        ("listing_content", listing_content_id(dialogue)),
    ]:
        if value is not None and value != "":
            keys.append((kind, str(value)))
    scenario = loaded_scenario(dialogue)
    if scenario is not None:
        keys.append(("loaded_context", content_hash(scenario)))
    return keys
