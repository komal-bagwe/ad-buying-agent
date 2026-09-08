"""
Streamlit frontend for the ad-buying agent.

This imports app.main.run_simulation directly and runs it in the same
process - no separate API layer, matching the project's monolith
structure (simulator + agent + RAG all run locally, only the LLM,
embedding, and vector search calls go out to Groq/Cohere/Qdrant).

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd

from app.simulator import Campaign
from app.main import run_simulation

st.set_page_config(page_title="Ad-Buying Agent", layout="wide")

st.title("RAG + Agentic Ad-Buying Assistant")
st.caption(
    "A simulated second-price ad auction, managed by an LLM agent that reasons over "
    "recent performance and a small bidding-strategy knowledge base to decide daily bid changes."
)

# ---------------------------------------------------------
# Sidebar: campaign configuration
# ---------------------------------------------------------
st.sidebar.header("Campaign settings")

num_days = st.sidebar.slider("Days to simulate", min_value=3, max_value=30, value=14)
budget = st.sidebar.number_input("Daily budget ($)", min_value=50.0, value=500.0, step=50.0)
starting_bid = st.sidebar.number_input("Starting bid ($)", min_value=0.10, value=1.50, step=0.10)
market_price_mean = st.sidebar.number_input("Market avg. competing price ($)", min_value=0.10, value=2.0, step=0.10)
market_price_std = st.sidebar.number_input("Market price variability (std dev)", min_value=0.05, value=0.5, step=0.05)
base_ctr = st.sidebar.slider("Click-through rate", 0.01, 0.20, 0.04, step=0.01)
base_cvr = st.sidebar.slider("Conversion rate (of clicks)", 0.01, 0.20, 0.06, step=0.01)

run_button = st.sidebar.button("Run simulation", type="primary")

# ---------------------------------------------------------
# Run and display
# ---------------------------------------------------------
if run_button:
    campaign = Campaign(
        name="Streamlit Campaign",
        budget=budget,
        bid=starting_bid,
        market_price_mean=market_price_mean,
        market_price_std=market_price_std,
        base_ctr=base_ctr,
        base_cvr=base_cvr,
    )

    with st.spinner(f"Running {num_days}-day agent loop (calls the LLM once per day)..."):
        log = run_simulation(campaign, num_days=num_days, verbose=False)

    st.session_state["log"] = log

if "log" in st.session_state:
    log = st.session_state["log"]

    # Build a dataframe for charting
    rows = []
    for entry in log:
        perf = entry["performance"]
        rows.append({
            "Day": entry["day"],
            "Bid": entry["bid_after"],
            "Win Rate": perf["win_rate"],
            "Conversions": perf["conversions"],
            "CPA": perf["cpa"],
            "Cost": perf["cost"],
            "Budget Exhausted": perf["budget_exhausted"],
            "Action": entry["decision"]["action"],
        })
    df = pd.DataFrame(rows)

    total_conversions = df["Conversions"].sum()
    total_cost = df["Cost"].sum()
    overall_cpa = round(total_cost / total_conversions, 2) if total_conversions else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total conversions", int(total_conversions))
    col2.metric("Total spend", f"${total_cost:,.2f}")
    col3.metric("Overall CPA", f"${overall_cpa}" if overall_cpa else "N/A")
    col4.metric("Final bid", f"${log[-1]['bid_after']}")

    st.subheader("Bid over time")
    st.line_chart(df.set_index("Day")["Bid"])

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Conversions per day")
        st.bar_chart(df.set_index("Day")["Conversions"])
    with chart_col2:
        st.subheader("CPA per day")
        st.line_chart(df.set_index("Day")["CPA"])

    st.subheader("Day-by-day decision log")
    for entry in log:
        perf = entry["performance"]
        decision = entry["decision"]
        with st.expander(
            f"Day {entry['day']}: {decision['action'].upper()} "
            f"(${entry['bid_before']} \u2192 ${entry['bid_after']}) "
            f"\u2014 {perf['conversions']} conversions, CPA ${perf['cpa']}"
        ):
            st.markdown(f"**Reasoning:** {decision['reasoning']}")
            st.markdown(
                f"Win rate: {perf['win_rate']} · Budget exhausted: {perf['budget_exhausted']} · "
                f"Wins: {perf['wins']} · Clicks: {perf['clicks']}"
            )
            if decision.get("retrieved_context"):
                st.markdown("**Retrieved reference notes:**")
                for chunk in decision["retrieved_context"]:
                    st.markdown(f"- {chunk['text']}  \n  *(source: {chunk['source_url']})*")
else:
    st.info("Set your campaign parameters in the sidebar and click **Run simulation** to start.")