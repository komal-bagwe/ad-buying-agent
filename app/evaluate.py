"""
Evaluation suite for the bidding agent.

Answers four distinct questions, each requiring a different test design:

1. BASELINE COMPARISON - does the agent beat simple, non-AI strategies?
   (no-op / never change bid, and a fixed-rule heuristic)
2. ABLATION - does RAG retrieval actually change decision quality, or is
   it decorative? (agent with RAG vs. agent with RAG disabled)
3. ROBUSTNESS - is the agent's behavior a stable policy, or is it wildly
   sensitive to random noise in the simulated auctions? (same starting
   conditions, multiple random seeds)
4. GUARDRAIL STRESS TEST - do the safety limits (max bid change, min bid
   floor) actually hold under an extreme/adversarial scenario?

Run: python3 -m app.evaluate
"""

import random
import statistics
from functools import partial

from app.simulator import Campaign, apply_bid_change
from app.agent import decide_bid_action
from app.main import run_simulation

NUM_DAYS = 14
STARTING_CAMPAIGN_KWARGS = dict(
    name="Eval Campaign",
    budget=500,
    bid=1.50,
    market_price_mean=2.0,
    market_price_std=0.5,
    base_ctr=0.04,
    base_cvr=0.06,
)


def _fresh_campaign() -> Campaign:
    """A new Campaign with identical starting conditions every time,
    so every strategy/run is compared on a level playing field."""
    return Campaign(**STARTING_CAMPAIGN_KWARGS)


def _summarize(log: list[dict]) -> dict:
    """Roll a full run's day-by-day log up into headline numbers."""
    total_conversions = sum(day["performance"]["conversions"] for day in log)
    total_cost = sum(day["performance"]["cost"] for day in log)
    cpa = (total_cost / total_conversions) if total_conversions else None
    final_bid = log[-1]["bid_after"]
    bid_changes = [abs(day["bid_after"] - day["bid_before"]) for day in log]
    avg_daily_bid_change = statistics.mean(bid_changes) if bid_changes else 0
    return {
        "total_conversions": total_conversions,
        "total_cost": round(total_cost, 2),
        "cpa": round(cpa, 2) if cpa else None,
        "final_bid": final_bid,
        "avg_daily_bid_change": round(avg_daily_bid_change, 3),
    }


# =========================================================
# 1. BASELINE STRATEGIES (no LLM at all)
# =========================================================

def noop_strategy(recent_results: list[dict]) -> dict:
    """Never touch the bid. The simplest possible baseline - if the
    agent can't beat doing nothing, it isn't adding value."""
    return {"action": "hold", "pct_change": 0.0, "reasoning": "no-op baseline"}


def rule_based_strategy(recent_results: list[dict]) -> dict:
    """
    A simple, fixed-threshold heuristic with no LLM involved - the
    kind of rule a campaign manager might hardcode without any AI.
    Uses only the single most recent day (no trend-awareness, unlike
    the agent), to keep this a genuinely simple baseline.
    """
    latest = recent_results[-1]
    if latest["win_rate"] < 0.2 and not latest["budget_exhausted"]:
        return {"action": "raise", "pct_change": 0.10, "reasoning": "rule: low win rate, budget not exhausted"}
    if latest["budget_exhausted"] and latest["cpa"] and latest["cpa"] > 400:
        return {"action": "lower", "pct_change": -0.10, "reasoning": "rule: budget exhausted, high CPA"}
    return {"action": "hold", "pct_change": 0.0, "reasoning": "rule: no clear signal"}


def run_baseline_comparison():
    print("=" * 70)
    print("1. BASELINE COMPARISON")
    print("=" * 70)

    random.seed(42)  # same auction randomness across all three runs, for a fair comparison
    noop_log = run_simulation(_fresh_campaign(), NUM_DAYS, decide_fn=noop_strategy, verbose=False)

    random.seed(42)
    rule_log = run_simulation(_fresh_campaign(), NUM_DAYS, decide_fn=rule_based_strategy, verbose=False)

    random.seed(42)
    agent_log = run_simulation(_fresh_campaign(), NUM_DAYS, decide_fn=decide_bid_action, verbose=False)

    results = {
        "No-op (never change bid)": _summarize(noop_log),
        "Fixed-rule baseline (no LLM)": _summarize(rule_log),
        "RAG + LLM agent": _summarize(agent_log),
    }

    print(f"{'Strategy':<32}{'Conversions':<13}{'Total Cost':<13}{'CPA':<10}{'Final Bid':<10}")
    for name, r in results.items():
        print(f"{name:<32}{r['total_conversions']:<13}{r['total_cost']:<13}{str(r['cpa']):<10}{r['final_bid']:<10}")
    print()
    return results


# =========================================================
# 2. ABLATION - RAG on vs. RAG off
# =========================================================

def run_ablation():
    print("=" * 70)
    print("2. ABLATION - RAG enabled vs. RAG disabled")
    print("=" * 70)

    agent_no_rag = partial(decide_bid_action, use_rag=False)

    random.seed(42)
    with_rag_log = run_simulation(_fresh_campaign(), NUM_DAYS, decide_fn=decide_bid_action, verbose=False)

    random.seed(42)
    without_rag_log = run_simulation(_fresh_campaign(), NUM_DAYS, decide_fn=agent_no_rag, verbose=False)

    results = {
        "Agent WITH RAG": _summarize(with_rag_log),
        "Agent WITHOUT RAG": _summarize(without_rag_log),
    }

    print(f"{'Variant':<24}{'Conversions':<13}{'Total Cost':<13}{'CPA':<10}{'Final Bid':<10}")
    for name, r in results.items():
        print(f"{name:<24}{r['total_conversions']:<13}{r['total_cost']:<13}{str(r['cpa']):<10}{r['final_bid']:<10}")
    print()
    return results


# =========================================================
# 3. ROBUSTNESS - same conditions, different random seeds
# =========================================================

def run_robustness_check(num_seeds: int = 3):
    print("=" * 70)
    print(f"3. ROBUSTNESS - agent across {num_seeds} different random seeds")
    print("=" * 70)

    final_bids = []
    for seed in range(num_seeds):
        random.seed(seed)
        log = run_simulation(_fresh_campaign(), NUM_DAYS, decide_fn=decide_bid_action, verbose=False)
        summary = _summarize(log)
        final_bids.append(summary["final_bid"])
        print(f"Seed {seed}: final_bid=${summary['final_bid']}, "
              f"conversions={summary['total_conversions']}, cpa={summary['cpa']}")

    spread = max(final_bids) - min(final_bids)
    print(f"\nFinal bid spread across seeds: ${round(spread, 2)} "
          f"(range ${min(final_bids)} - ${max(final_bids)})")
    print("A small spread suggests a stable policy; a huge spread suggests "
          "the agent is overly sensitive to random noise rather than signal.\n")
    return final_bids


# =========================================================
# 4. GUARDRAIL STRESS TEST
# =========================================================

def run_guardrail_stress_test():
    print("=" * 70)
    print("4. GUARDRAIL STRESS TEST")
    print("=" * 70)

    campaign = _fresh_campaign()

    # simulate the agent requesting an extreme, adversarial change
    extreme_raise = apply_bid_change(campaign, pct_change=5.0)  # requests +500%
    print(f"Requested +500% change -> actual new bid: ${extreme_raise} "
          f"(expected max +20% from $1.50 -> $1.80)")
    assert extreme_raise <= 1.50 * 1.20 + 0.01, "GUARDRAIL FAILED: change exceeded +20% cap"

    campaign2 = _fresh_campaign()
    extreme_drop = apply_bid_change(campaign2, pct_change=-5.0)  # requests -500%
    print(f"Requested -500% change -> actual new bid: ${extreme_drop} "
          f"(expected max -20% from $1.50 -> $1.20)")
    assert extreme_drop >= 1.50 * 0.80 - 0.01, "GUARDRAIL FAILED: change exceeded -20% cap"

    # repeatedly cut a low bid to confirm the minimum floor holds
    campaign3 = _fresh_campaign()
    campaign3.bid = 0.15
    for _ in range(20):
        apply_bid_change(campaign3, pct_change=-0.20)
    print(f"After 20 consecutive -20% cuts from $0.15 -> ${campaign3.bid} "
          f"(expected floor: $0.10)")
    assert campaign3.bid >= 0.10, "GUARDRAIL FAILED: bid dropped below minimum floor"

    print("\nAll guardrail checks passed.\n")


if __name__ == "__main__":
    baseline_results = run_baseline_comparison()
    ablation_results = run_ablation()
    robustness_results = run_robustness_check()
    run_guardrail_stress_test()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("See sections above for: baseline comparison, RAG ablation, "
          "multi-seed robustness, and guardrail stress test results.")