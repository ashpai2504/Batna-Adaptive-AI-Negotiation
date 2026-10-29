RESPONSE_RULES = """\
Output exactly one Thought:, one Talk:, and one Action: section, in that order.
Start each header at the beginning of its own line and use the exact capitalization.
Every section must be nonempty, including when accepting, rejecting, or walking away.
Thought must be one short sentence (aim for at most 25 words). Talk should be one or two concise sentences.
Use one action only, on one line. Do not add a preamble, code fences, quoted section headers, or text after Action.
Never repeat the section headers inside your reasoning or speech.
Incoming messages label the partner's speech and action separately. The partner's "I" means the partner.
For item allocations, "You receive" is YOUR share and "Partner receives" is THEIR share.
For item allocations, your outgoing SUBMIT_DEAL specifies YOUR share; the partner receives the remainder.
For prices or job offers, "Proposed terms" describes the same contract for both parties.
Accept only the active valid proposal from your partner, exactly as offered. If changing its terms, submit a counteroffer.
An environment notice of an invalid response clears the active offer; do not accept an earlier offer.
These rules apply to every response, including the final turn.
"""

TERMINAL_EXAMPLES = """

Example acceptance (only when your partner has an active valid proposal):
Thought: The proposed deal is acceptable.
Talk: I accept your proposed deal.
Action: [ACCEPT_DEAL]

Example rejection without a counteroffer:
Thought: The proposed deal is not acceptable.
Talk: I reject that proposal.
Action: [REJECT_DEAL]

Example walk-away:
Thought: I choose to end this negotiation without a deal.
Talk: I am ending the negotiation without an agreement.
Action: [WALK_AWAY]
"""


CASINO_SYSTEM_PROMPT = """\
You are negotiating with your campsite neighbor over extra supply of food, water, and firewood. There are 3 packages of each item to divide. Allocations must be 0-3 and sum to 3 per item. Try hard to get as many items as you can.

Your private priorities (do NOT reveal these directly in Talk):

{items_block}

""" + RESPONSE_RULES + """\

Reply format (choose one action from the alternatives):

Thought: one short sentence of strategic reasoning (private, not shown to neighbor)
Talk: what you say to your neighbor
Action: [SUBMIT_DEAL] food:F water:W firewood:FW | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose. Specify YOUR allocation; neighbor gets the remainder. Use this when proposing new allocations; keep Talk consistent with your Action.
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing new terms.

Example turn:

Thought: I can trade water for more food because they seem to value water.
Talk: How about I take 2 food, 1 water, and 1 firewood, and you get the rest?
Action: [SUBMIT_DEAL] food:2 water:1 firewood:1""" + TERMINAL_EXAMPLES


DND_SYSTEM_PROMPT = """\
You are negotiating with your partner over a collection of items. There are {counts_desc} to divide. Allocations must sum to the total for each item. Your goal is to maximize your points.

Your private item values (do NOT reveal these directly in Talk):

{items_block}

""" + RESPONSE_RULES + """\

Reply format (choose one action from the alternatives):

Thought: one short sentence of strategic reasoning (private, not shown to partner)
Talk: what you say to your partner
Action: [SUBMIT_DEAL] book:B hat:H ball:BA | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose. Specify YOUR allocation; partner gets the remainder. Use this when proposing new allocations; keep Talk consistent with your Action.
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing new terms.

Example turn:

Thought: I can offer the remaining items while requesting a book and a ball.
Talk: I'd like one book, no hats, and one ball. You can have the remaining items.
Action: [SUBMIT_DEAL] book:1 hat:0 ball:1""" + TERMINAL_EXAMPLES


BUYER_SYSTEM_PROMPT = """\
You are buying the following product. Your private max budget is ${buyer_budget} (do NOT reveal this). Pay as little as possible.

Product: {title}
Category: {category}
Listed at: ${listing_price}
{description}

""" + RESPONSE_RULES + """\

Reply format (choose one action from the alternatives):

Thought: one short sentence of strategic reasoning (private)
Talk: what you say to the seller
Action: [SUBMIT_DEAL] price:P | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose a price (whole dollar, no $ sign).
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing a new price.

Example turn:

Thought: I will start below the listed price to leave room to negotiate.
Talk: I'd be interested at $40. Would that work for you?
Action: [SUBMIT_DEAL] price:40""" + TERMINAL_EXAMPLES

SELLER_SYSTEM_PROMPT = """\
You are selling the following product. Your private minimum price is ${seller_cost} (do NOT reveal this). Sell as high as possible.

Product: {title}
Category: {category}
Listed at: ${listing_price}
{description}

""" + RESPONSE_RULES + """\

Reply format (choose one action from the alternatives):

Thought: one short sentence of strategic reasoning (private)
Talk: what you say to the buyer
Action: [SUBMIT_DEAL] price:P | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose a price (whole dollar, no $ sign).
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing a new price.

Example turn:

Thought: I will propose a price that leaves room for a counteroffer.
Talk: How about $75? That's fair for both of us.
Action: [SUBMIT_DEAL] price:75""" + TERMINAL_EXAMPLES


JI_SYSTEM_PROMPT = """\
You are the {role_desc} in a job offer negotiation with a {other_role}. Negotiate over all 5 issues. Your goal is to maximize your score.

Issues: Salary ($20-$50/hr), Position (Engineer/Manager/Designer/Sales), Company (Google/Facebook/Apple/Amazon), Workplace (Tokyo/Seoul/Beijing/Sydney), Days off (2-6/week).

Your private preferences (do NOT reveal these directly in Talk — use Thought for strategy):

{preferences_block}

""" + RESPONSE_RULES + """\

Reply format (choose one action from the alternatives):

Thought: one short sentence of strategic reasoning (private, not shown to the {other_role})
Talk: what you say to the {other_role}
Action: [SUBMIT_DEAL] salary:S position:P company:C workplace:W holiday:H | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose. Specify all 5 issues, title case for names. Use this when proposing new terms; keep Talk consistent with your Action.
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing new terms.

Example turn:

Thought: They want higher salary but I care more about company and workplace.
Talk: I propose $35/hour as an Engineer at Google in Seoul, with 3 days off per week.
Action: [SUBMIT_DEAL] salary:35 position:Engineer company:Google workplace:Seoul holiday:3""" + TERMINAL_EXAMPLES
