"""
Simulator for a fake ad campaign, based on real second-price auction mechanics
(the same mechanism real ad exchanges use - see RTB literature).

Core mechanic, in one sentence:
Each day, for each potential impression, there's a random "competing price"
(the highest bid from other advertisers). If our bid beats it, we win the
impression and pay that competing price (not our own bid) - this is exactly
how second-price auctions work in real ad platforms.

This does NOT call any AI model. It's plain math simulating auction outcomes,
so the agent has something safe to act on.
"""

import random
from dataclasses import dataclass, field


@dataclass
class Campaign:
    name: str
    budget: float              # max spend allowed per day
    bid: float                  # current bid the campaign is set to
    market_price_mean: float    # avg competing bid for this campaign's ad slots
    market_price_std: float     # how much competing bids vary
    base_ctr: float = 0.03      # click-through rate once an impression is won
    base_cvr: float = 0.05      # conversion rate once a click happens
    daily_auctions: int = 5000  # how many auctions this campaign competes in per day
    history: list = field(default_factory=list)


def simulate_one_day(campaign: Campaign) -> dict:
    """
    Simulate one day of a real-time bidding auction for this campaign.
    For each of the day's auctions, draw a random competing price.
    We win if our bid beats it, and if we win, we pay the competing
    price (second-price auction rule) - not our own bid.
    """
    bid = campaign.bid
    wins = 0
    total_cost = 0.0
    auctions_attempted = 0

    # Stage 1: the auction. Loop through each potential impression and
    # decide, auction by auction, whether we win it and what we pay.
    # This part needs per-auction randomness because each auction faces
    # a different random competing price.
    for _ in range(campaign.daily_auctions):
        auctions_attempted += 1
        competing_price = max(random.gauss(campaign.market_price_mean, campaign.market_price_std), 0)

        if bid > competing_price:
            wins += 1
            cost_this_auction = competing_price  # second-price rule: pay their price, not ours

            if total_cost + cost_this_auction > campaign.budget:
                break  # budget exhausted for the day, stop bidding (real platforms do this too)

            total_cost += cost_this_auction

    budget_exhausted = auctions_attempted < campaign.daily_auctions

    # Stage 2: engagement. Winning an auction only means the ad was shown.
    # Whether people click, and whether clickers convert, is a property of
    # the ad/audience match, not the bid - so we apply it as a direct rate
    # against however many impressions we actually won, once, after the
    # auction loop. This gives the expected click/conversion count without
    # needing a separate random roll per impression.
    clicks = wins * campaign.base_ctr
    conversions = clicks * campaign.base_cvr

    # Win rate among auctions actually attempted (not the full daily
    # capacity) - this is what tells us whether the bid is competitive.
    # If budget ran out early, win rate among attempted auctions can be
    # high even though few of the day's total potential auctions were
    # reached - that's a budget problem, not a competitiveness problem.
    win_rate_of_attempted = wins / auctions_attempted if auctions_attempted else 0
    cpa = (total_cost / conversions) if conversions > 0 else None

    result = {
        "bid": round(bid, 2),
        "auctions_available": campaign.daily_auctions,
        "auctions_attempted": auctions_attempted,
        "wins": wins,
        "win_rate": round(win_rate_of_attempted, 3),
        "budget_exhausted": budget_exhausted,
        "clicks": round(clicks),
        "conversions": round(conversions),
        "cost": round(total_cost, 2),
        "cpa": round(cpa, 2) if cpa else None,
    }

    campaign.history.append(result)
    return result


def apply_bid_change(campaign: Campaign, pct_change: float, max_pct: float = 0.20, min_bid: float = 0.10):
    """
    Adjust the campaign's bid by a percentage, clamped to a guardrail
    so the agent can't make wild swings in one step.

    Two safety limits are enforced:
    - max_pct: no single change can move the bid more than this percentage
      (protects against overreacting to one noisy day).
    - min_bid: the bid can never drop below this floor, even after many
      repeated cuts over multiple days (protects against the bid shrinking
      toward a meaningless near-zero value).
    """
    clamped = max(min(pct_change, max_pct), -max_pct)
    new_bid = round(campaign.bid * (1 + clamped), 2)
    campaign.bid = max(new_bid, min_bid)
    return campaign.bid