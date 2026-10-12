"""Shared Streamlit presentation and lifecycle helpers for GRAG."""

from __future__ import annotations

from typing import Literal

import streamlit as st

from terminal_interface import GRAGTerminalInterface, RouteResult, RoutingDecision
from terminal_interface_fast import FastStartGRAGTerminalInterface


Mode = Literal["fast", "normal"]


@st.cache_resource(show_spinner=False)
def load_interface(mode: Mode) -> GRAGTerminalInterface:
    """Build an indexed interface once per Streamlit server and mode."""
    if mode == "fast":
        return FastStartGRAGTerminalInterface(
            verbose=False,
            compare_mode=False,
            max_docs=2000,
            max_graph_nodes=5000,
            max_graph_edges=15000,
        )
    return GRAGTerminalInterface(verbose=False, compare_mode=False)


def _route_label(route: str) -> str:
    return {
        "hybrid_rag": "Hybrid RAG",
        "graph_rag": "GraphRAG",
        "optimized_graph_rag": "Optimized GraphRAG",
    }.get(route, route)


def _result_row(route: str, result: RouteResult, decision: RoutingDecision) -> dict[str, str]:
    return {
        "Route": _route_label(route),
        "Policy probability": f"{decision.policy_probabilities[route]:.1%}",
        "Latency": "Failed" if result.error else f"{result.latency_ms:.1f} ms",
        "Evidence units": str(result.retrieved_units),
        "Status": "Selected" if route == decision.selected_route else "Compared",
    }


def _render_results(decision: RoutingDecision, results: dict[str, RouteResult]) -> None:
    selected_result = results[decision.selected_route]
    selected_probability = decision.policy_probabilities[decision.selected_route]

    st.subheader("Routing decision")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Selected route", _route_label(decision.selected_route))
    metric_columns[1].metric("Policy confidence", f"{selected_probability:.1%}")
    metric_columns[2].metric(
        "Route latency",
        "Failed" if selected_result.error else f"{selected_result.latency_ms:.1f} ms",
    )
    st.caption(decision.reason)

    probability_rows = [
        {"Route": _route_label(route), "Probability": probability}
        for route, probability in decision.policy_probabilities.items()
    ]
    st.bar_chart(probability_rows, x="Route", y="Probability", horizontal=True)

    if selected_result.error:
        st.error(f"{_route_label(selected_result.route)} failed: {selected_result.error}")
    else:
        st.subheader("Answer")
        st.write(selected_result.answer)
        st.caption(f"Retrieved evidence units: {selected_result.retrieved_units}")
        with st.expander("Context preview"):
            st.code(selected_result.context_preview or "No context preview was returned.")

    with st.expander("Question features"):
        st.dataframe(
            [
                {"Feature": name, "Value": value}
                for name, value in decision.features.items()
            ],
            hide_index=True,
            use_container_width=True,
        )

    if len(results) > 1:
        st.subheader("All-route comparison")
        st.dataframe(
            [_result_row(route, results[route], decision) for route in decision.policy_probabilities],
            hide_index=True,
            use_container_width=True,
        )


def render_page(mode: Mode) -> None:
    """Render a terminal-flow-compatible question page."""
    is_fast = mode == "fast"
    title = "GRAG Router: Fast" if is_fast else "GRAG Router: Normal"
    st.title(title)
    st.caption(
        "Uses the trained contextual-bandit policy to select Hybrid RAG, GraphRAG, "
        "or Optimized GraphRAG before answering."
    )
    if is_fast:
        st.info("Fast mode indexes up to 2,000 documents, 5,000 graph nodes, and 15,000 graph edges.")
    else:
        st.info("Normal mode indexes the complete MetaQA test dataset and graph before answering.")

    question = (
        st.text_input("Question", placeholder="Ask a short factual question...")
        if is_fast
        else st.text_area("Question", placeholder="Ask a full multi-hop question...", height=130)
    )
    compare_all = st.checkbox("Compare all routes", value=False)
    submit = st.button("Ask GRAG", type="primary", use_container_width=True)

    if not submit:
        return
    if not question.strip():
        st.warning("Enter a question before running the router.")
        return

    mode_name = "fast" if is_fast else "normal"
    with st.spinner(f"Loading the {mode_name} RAG pipelines and trained policy..."):
        try:
            interface = load_interface(mode)
        except Exception as exc:
            st.error(f"Could not initialize GRAG: {exc}")
            st.stop()

    with st.spinner("Routing the question and generating an answer..."):
        decision, results = interface.answer_question(question, compare_mode=compare_all)
    _render_results(decision, results)
