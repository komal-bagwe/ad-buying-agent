"""
The main loop - ties the simulator and a decision strategy together.

Each day:
1. Run the simulator with the campaign's current bid -> get performance
2. Feed recent history to a decision function -> get a decision (raise/lower/hold)
3. Apply the decision to the campaign's bid (guardrail-clamped)
4. Repeat for the next day, using the new bid

The decision function is pluggable (decide_fn), so the same loop can
drive the AI agent (app.agent.decide_bid_action), a rule-based baseline,
or any other strategy - see app/evaluate.py for how this is used to
compare strategies against each other.
"""

from app.simulator import Campaign, simulate_one_day, apply_bid_change
from app.agent import decide_bid_action


HISTORY_WINDOW = 3  # how many recent days a decision function gets to see at once


def run_simulation(campaign: Campaign, num_days: int = 14, decide_fn=decide_bid_action, verbose: bool = True):
    """
    Run the full loop for num_days using decide_fn as the decision
    strategy, logging every day's performance and decision.
    decide_fn must accept a list of recent day-result dicts and return
    a dict with at least {action, pct_change, reasoning}.
    Returns the full log as a list.
    """
    log = []

    for day in range(1, num_days + 1):
        # Step 1: simulate today using the campaign's CURRENT bid
        performance = simulate_one_day(campaign)

        # Step 2: ask the decision strategy what to do, showing it the
        # last few days (not just today) so it can spot trends
        recent_results = campaign.history[-HISTORY_WINDOW:]
        decision = decide_fn(recent_results)

        # Step 3: apply the decision to the bid for TOMORROW
        old_bid = campaign.bid
        new_bid = apply_bid_change(campaign, pct_change=decision["pct_change"])

        entry = {
            "day": day,
            "performance": performance,
            "decision": decision,
            "bid_before": old_bid,
            "bid_after": new_bid,
        }
        log.append(entry)

        if verbose:
            print(f"Day {day}: bid ${old_bid} -> ${new_bid} "
                  f"| action={decision['action']} | conversions={performance['conversions']} "
                  f"| cpa={performance['cpa']} | reasoning: {decision['reasoning']}")

    return log


if __name__ == "__main__":
    campaign = Campaign(
        name="Demo Campaign",
        budget=500,
        bid=1.50,
        market_price_mean=2.0,
        market_price_std=0.5,
        base_ctr=0.04,
        base_cvr=0.06,
    )

    run_simulation(campaign, num_days=14)