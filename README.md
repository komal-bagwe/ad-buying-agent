# RAG + Agentic Ad-Buying Assistant

An LLM agent that manages bidding for a simulated ad campaign over multiple days —
looking at recent performance, retrieving relevant bidding-strategy notes via RAG,
and deciding each day whether to raise, lower, or hold the bid. Built as a portfolio
project to demonstrate an agentic decision loop grounded in retrieval, evaluated
rigorously against simpler baselines rather than assumed to work.

## What this is (and isn't)

This project simulates a second-price ad auction rather than connecting to a real
ad platform. There is no real ad spend, no real account, and no live API access to
Google Ads / Meta Ads involved. This is a deliberate choice, not a limitation to
hide: letting an untested AI agent control real ad spend is a real financial and
reputational risk, and real ad platform APIs aren't accessible for a project like
this without an active advertiser account. The simulator implements real auction
mechanics (see below) so the agent faces a genuine decision problem, just in a safe,
reproducible environment.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  simulator  │────▶│    agent     │────▶│     rag      │
│  (auction)  │◀────│  (decision)  │◀────│ (retrieval)  │
└─────────────┘     └──────────────┘     └─────────────┘
       ▲                    │
       │                    ▼
       └──────────── main (daily loop) ────────────┐
                                                     ▼
                                              evaluate / frontend
```

- **`app/simulator.py`** — simulates one day of a second-price ad auction. For
  each of up to 5,000 daily auction opportunities, draws a random competing price
  and checks whether the campaign's bid beats it (Monte Carlo simulation of the
  win-probability integral used in real-time bidding literature). On a win, pays
  the competitor's price (second-price rule), not the campaign's own bid. Tracks
  budget exhaustion separately from win rate, since the two are easy to conflate
  (see Known Issues below). Includes bid-change guardrails: a ±20% cap per
  decision and a $0.10 minimum bid floor.

- **`app/agent.py`** — the decision-maker. Given the last 3 days of performance
  plus retrieved reference notes, calls an LLM (via Groq) to decide: raise, lower,
  or hold the bid, with reasoning. Includes retry-with-backoff for rate limits and
  a `use_rag` flag for ablation testing.

- **`app/rag.py`** — reads local `.txt` documents from `data/docs/`, chunks them
  into paragraphs, embeds them via Cohere, and stores/searches them in Qdrant
  Cloud. The knowledge base covers real-time bidding mechanics, Google Ads bid
  strategy types (Target CPA, Target ROAS, Smart Bidding), and portfolio-level
  bid strategies — paraphrased from public sources, not scraped verbatim.

- **`app/main.py`** — the daily loop: simulate a day → retrieve context → decide
  → apply the guardrail-clamped bid change → repeat. The decision function is
  pluggable, so the same loop drives the AI agent, a rule-based baseline, or a
  no-op baseline.

- **`app/evaluate.py`** — the evaluation suite (see Evaluation below).

- **`streamlit_app.py`** — a local UI for configuring a campaign, running the
  loop, and inspecting the day-by-day bid trajectory, metrics, and the agent's
  actual reasoning and retrieved context per day.

## Tech stack

| Purpose | Tool | Notes |
|---|---|---|
| LLM reasoning | Groq (`openai/gpt-oss-120b`) | Cloud API, no local model |
| Embeddings | Cohere (`embed-english-v3.0`) | Cloud API, no local model |
| Vector store | Qdrant Cloud | Managed, free tier |
| Backend | Python | No FastAPI layer — monolith by design for this project |
| Frontend | Streamlit | Imports backend functions directly, same process |

Nothing in this stack downloads or runs a model locally — every AI capability
(reasoning, embeddings) is a network call to a managed cloud service.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key
COHERE_API_KEY=your_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
```

Ingest the RAG knowledge base (run once, or whenever `data/docs/*.txt` changes):
```bash
python3 -m app.rag
```

Run a single 14-day simulation from the command line:
```bash
python3 -m app.main
```

Run the full evaluation suite (baseline comparison, RAG ablation, robustness
check, guardrail stress test):
```bash
python3 -m app.evaluate
```

Launch the interactive frontend:
```bash
streamlit run streamlit_app.py
```

## Evaluation

Rather than assume the agent works, four different questions were tested, each
requiring a different evaluation design:

**1. Baseline comparison** — does the agent beat simpler, non-AI strategies?
Compared against a no-op (never change the bid) and a fixed-rule heuristic (no
LLM), same starting conditions and random seed for a fair comparison.

| Strategy | Conversions | Total Cost | CPA | Final Bid |
|---|---|---|---|---|
| No-op | 14 | $6,989 | $499 | $1.50 |
| Fixed-rule baseline (no LLM) | 14 | $5,905 | $422 | $1.42 |
| RAG + LLM agent | 13 | $5,478 | $421 | $1.34 |

Honest finding: the agent essentially tied the simple heuristic on CPA while
achieving one fewer conversion. The added complexity of an LLM+RAG agent did not
clearly outperform a few lines of hardcoded logic in this scenario — a legitimate
result worth reporting rather than a clean win to claim.

**2. Ablation** — does RAG retrieval specifically add value, isolated from the
LLM's reasoning alone? Ran the agent with and without `search_context()` enabled.
Results were directionally inconsistent across repeated runs with the same seed
(RAG improved CPA in one run, worsened it in another) — evidence that a single
comparison isn't reliable when the LLM's own output has sampling variance, even
at low temperature. A statistically defensible answer would need averaging across
multiple seeds per arm, noted as a natural extension.

**3. Robustness check** — is the agent's policy stable across different random
market conditions, or driven by noise? Ran the agent across 3 different random
seeds (same starting conditions, different auction randomness).

| Seed | Final Bid | Conversions | CPA |
|---|---|---|---|
| 0 | $1.16 | 14 | $375 |
| 1 | $1.25 | 14 | $398 |
| 2 | $1.32 | 14 | $446 |

Final bids converged to a tight $0.16 range across all three seeds — evidence of
a stable policy responding to genuine signal rather than random noise.

**4. Guardrail stress test** — do the safety limits hold under deliberately
extreme, adversarial input, independent of whether the LLM "behaves"? Directly
called the bid-change function with a requested +500% and -500% change, and
applied 20 consecutive aggressive cuts to a low starting bid.

Result: all guardrails held exactly as designed — the ±20% cap and $0.10 floor
cannot be bypassed regardless of what the LLM requests.

## Known issues found and fixed during development

Three real bugs were found by running the system end-to-end and reading the
agent's own reasoning closely, not by code review alone:

1. **Contradictory decision rules** — two independent prompt rules ("lower if
   CPA high" and "raise if win rate low") could both fire on the same data,
   causing the agent to justify opposite actions on different days from
   effectively the same situation. Fixed with an explicit priority order instead
   of a flat rule list.

2. **A misleading metric caused runaway bid escalation** — win rate was
   originally computed as wins ÷ full daily auction capacity. In a second-price
   auction, cost per win depends on the competing price distribution, not the
   bid, so once the bid comfortably beats the market, the number of affordable
   wins is capped by budget ÷ average market price — not by the bid. This made
   win rate stay artificially low (~5%) no matter how high the bid climbed,
   which the agent misread as "still not competitive enough," escalating the
   bid from $1.50 to $10.60 over 14 days. Fixed by computing win rate against
   auctions actually attempted, and adding an explicit `budget_exhausted` flag
   so the agent can distinguish "losing auctions" from "winning plenty but out
   of money" — two situations needing opposite responses.

3. **No memory caused daily oscillation** — even after fixing #2, the bid
   flip-flopped every day in a borderline zone, because `budget_exhausted` is
   itself noisy day-to-day (whether the budget runs out before reaching all
   5,000 auction slots depends on random luck in which auctions are won). With
   no memory across days, the agent reacted fresh to each day's noise. Fixed by
   giving the agent a 3-day rolling window instead of a single day, with
   explicit instruction to treat recent oscillation as a signal to hold rather
   than react to the latest noisy day.

## Limitations / scope notes

- The simulator models one campaign competing against a simulated market price
  distribution, not multiple campaigns competing for identical shared inventory.
- CTR and conversion rate are fixed, campaign-level constants rather than varying
  with targeting, creative, or audience signals.
- The agent adjusts the bid once per simulated day; real-time bidding platforms
  adjust per-auction using pre-trained ML models. This project operates at the
  "daily strategic review" cadence rather than per-auction cadence.
- RAG knowledge base content is paraphrased from public sources (Google Ads
  documentation, a public podcast episode, a Coursera article, and a user-provided
  video summary on portfolio bid strategies) rather than scraped verbatim, per
  copyright practice — sources are cited in each document's header.
- No deployment for this project; it runs locally via the CLI or Streamlit.

## Possible future improvements

- Multi-seed ablation averaging for a statistically defensible RAG-effect estimate
- Multiple campaigns competing for shared auction inventory
- A pause action for campaigns with sustained poor performance
- Sub-daily (e.g. hourly) bid adjustment cadence
- RAGAS-based evaluation of retrieval quality (context precision/recall,
  faithfulness of reasoning to retrieved context)