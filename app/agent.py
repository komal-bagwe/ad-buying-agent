"""
The bidding agent.

Each simulated day, this module:
1. Takes the previous day's performance (from simulator.simulate_one_day)
2. Builds a query from that performance and retrieves relevant context
   from the RAG knowledge base (app.rag.search_context)
3. Sends both the numbers and the retrieved context to an LLM (Groq)
4. Parses the LLM's response into a structured action
5. Returns that decision so the caller can apply it via simulator.apply_bid_change
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from app.rag import search_context

load_dotenv()  # reads the .env file and loads GROQ_API_KEY into os.environ

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"


def _call_groq_with_retry(messages, max_retries: int = 5):
    """
    Call Groq's chat completion, retrying with exponential backoff if
    the rate limit (429) is hit. This matters when running many
    consecutive calls quickly, e.g. in app/evaluate.py's back-to-back
    baseline/ablation/robustness runs, which can exceed tokens-per-minute
    limits even though each individual call is small.
    """
    delay = 2
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
            )
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Rate limited, retrying in {delay}s... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
            delay *= 2  # exponential backoff: 2s, 4s, 8s, 16s...


SYSTEM_PROMPT = """You are an ad-bidding agent. Each day you see recent
campaign performance history, plus retrieved reference notes on bidding
strategy, and must decide one action: raise the bid, lower the bid, or
hold it steady.

Key fields to understand:
- win_rate: the fraction of ATTEMPTED auctions won (not the full daily
  capacity) - this tells you how competitive your bid is.
- budget_exhausted: true if the campaign ran out of budget before reaching
  its full daily auction capacity. If true, a high win_rate does NOT mean
  you should raise the bid further - it means you're already winning
  plenty and money is the limiting factor, not competitiveness.

You are shown the last few days, not just yesterday. This matters because
budget_exhausted can flip from day to day due to normal randomness in
which auctions get won - a single day's value is noisy. Look at the
pattern across the shown days, not just the most recent one.

Decision priority:
1. If the bid has been oscillating (raised then lowered then raised) over
   the recent days shown, with budget_exhausted flipping inconsistently,
   treat this as a sign the bid is already in a reasonable range - prefer
   HOLD over reacting to the latest day's noise.
2. If win_rate has been consistently low (under 20%) across MULTIPLE
   recent days, and budget_exhausted has been consistently false: RAISE
   the bid.
3. If budget_exhausted has been consistently true across multiple recent
   days AND CPA is consistently high: LOWER the bid.
4. If the signal is mixed or unclear across the recent days: HOLD rather
   than guessing from one noisy day.
5. Use the reference notes to inform reasoning, but numbers take priority.
6. Never suggest a change larger than 20% - it will be capped anyway.

Respond ONLY with valid JSON in this exact shape, nothing else:
{"action": "raise" | "lower" | "hold", "pct_change": <number between -0.20 and 0.20, use 0 for hold>, "reasoning": "<one or two sentences>"}
"""


def _build_query(recent_results: list[dict]) -> str:
    """
    Turn the most recent day's numbers into a short natural-language
    query, so the RAG search retrieves notes relevant to the current
    situation (e.g. high CPA, low win rate) rather than generic text.
    """
    latest = recent_results[-1]
    return (
        f"bid ${latest['bid']}, win rate {latest['win_rate']}, "
        f"budget exhausted {latest['budget_exhausted']}, "
        f"conversions {latest['conversions']}, cpa {latest['cpa']}"
    )


def decide_bid_action(recent_results: list[dict], use_rag: bool = True) -> dict:
    """
    Given a list of the last few days' simulate_one_day() outputs
    (most recent last), retrieve relevant context (unless use_rag is
    False) and ask the LLM to decide an action based on the trend, not
    just the latest day.
    use_rag=False is used for ablation testing (see app/evaluate.py) to
    measure whether retrieved context actually changes decision quality.
    Returns a dict: {action, pct_change, reasoning, retrieved_context}.
    """
    if use_rag:
        query = _build_query(recent_results)
        retrieved = search_context(query, top_k=2)
        context_block = f"Relevant reference notes:\n" + "\n".join(f"- {r['text']}" for r in retrieved) + "\n\n"
    else:
        retrieved = []
        context_block = ""

    history_text = "\n".join(
        f"Day -{len(recent_results) - i}: {json.dumps(day)}"
        for i, day in enumerate(recent_results)
    )

    user_prompt = (
        f"Recent performance history (oldest to most recent):\n{history_text}\n\n"
        f"{context_block}"
        f"Decide the action."
    )

    response = _call_groq_with_retry([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])

    raw = response.choices[0].message.content.strip()

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: if the model didn't return clean JSON, default to holding
        # rather than letting a bad response crash the loop
        decision = {"action": "hold", "pct_change": 0.0, "reasoning": f"Could not parse model output, defaulted to hold. Raw: {raw}"}

    decision["retrieved_context"] = retrieved  # keep for logging/transparency
    return decision