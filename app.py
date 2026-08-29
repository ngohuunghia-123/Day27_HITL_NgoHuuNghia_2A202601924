"""Streamlit approval interface for the churn-risk HITL workflow."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from audit import AuditLogError, read_audit_entries
from graph import (
    ConfigurationError,
    EvaluationError,
    build_graph,
    create_initial_state,
)
from models import Action, GraphState, HumanDecision

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")

st.set_page_config(page_title="Churn Risk HITL", page_icon="🧑‍⚖️", layout="centered")

RUN_KEYS = ("run_config", "initial_state", "run_error", "reviewer_id")


def _ensure_workflow() -> Any:
    if "workflow" not in st.session_state:
        st.session_state.workflow = build_graph()
    return st.session_state.workflow


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (ConfigurationError, EvaluationError)):
        return str(exc)
    return f"Workflow failed safely ({type(exc).__name__}). Check the terminal log."


def _invoke_new_run(initial_state: GraphState, reviewer_id: str) -> None:
    workflow = _ensure_workflow()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.run_config = config
    st.session_state.initial_state = initial_state
    st.session_state.reviewer_id = reviewer_id
    st.session_state.run_error = None
    try:
        workflow.invoke(initial_state, config=config)
    except Exception as exc:
        st.session_state.run_error = _safe_error_message(exc)


def _resume_run(decision: HumanDecision, edited_action: Action | None = None) -> None:
    workflow = _ensure_workflow()
    config = st.session_state.run_config
    update: dict[str, object] = {
        "human_decision": decision,
        "reviewer_id": st.session_state.reviewer_id,
    }
    if edited_action is not None:
        # Keep the original high-risk proposal in place so update_state cannot
        # re-route an already reviewed edit into the auto-execute branch.
        update["edited_action"] = edited_action

    st.session_state.run_error = None
    try:
        workflow.update_state(config, update)
        workflow.invoke(None, config=config)
    except Exception as exc:
        st.session_state.run_error = _safe_error_message(exc)


def _clear_run() -> None:
    for key in RUN_KEYS:
        st.session_state.pop(key, None)


def _render_audit_history() -> None:
    with st.expander("Audit trail", expanded=False):
        try:
            entries = read_audit_entries()
        except AuditLogError as exc:
            st.error(str(exc))
            return

        if not entries:
            st.caption("Chưa có audit entry.")
            return
        st.dataframe(
            [entry.model_dump() for entry in entries[-10:]],
            use_container_width=True,
            hide_index=True,
        )


st.title("Churn Risk Agent — Human-in-the-Loop")
st.caption(
    "Gemini chỉ đề xuất. Hard policy và confidence gate được thực thi bằng code; "
    "mọi hành động high-risk cần con người phê duyệt."
)

with st.form("customer_form"):
    customer_id = st.text_input("Customer ID", value="CUST001")
    total_operating_income = st.number_input(
        "Total Operating Income (TOI)",
        min_value=0.0,
        value=50_000_000.0,
        step=1_000_000.0,
    )
    churn_probability = st.slider(
        "Churn probability",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01,
    )
    reviewer_id = st.text_input("Reviewer ID", value="operator_01")
    submitted = st.form_submit_button("Đánh giá khách hàng", type="primary")

if submitted:
    if not customer_id.strip() or not reviewer_id.strip():
        st.error("Customer ID và Reviewer ID là bắt buộc.")
    else:
        initial = create_initial_state(
            customer_id=customer_id,
            total_operating_income=float(total_operating_income),
            churn_probability=float(churn_probability),
        )
        _invoke_new_run(initial, reviewer_id.strip())

if st.session_state.get("run_error"):
    st.error(st.session_state.run_error)
    retry_col, cancel_col = st.columns(2)
    if retry_col.button("Retry", use_container_width=True):
        _invoke_new_run(
            st.session_state.initial_state,
            st.session_state.reviewer_id,
        )
        st.rerun()
    if cancel_col.button("Cancel", use_container_width=True):
        _clear_run()
        st.rerun()

if st.session_state.get("run_config") and not st.session_state.get("run_error"):
    workflow = _ensure_workflow()
    snapshot = workflow.get_state(st.session_state.run_config)
    values = snapshot.values

    if values.get("proposed_action"):
        st.subheader("Agent proposal")
        metric_col, model_col = st.columns(2)
        metric_col.metric("Confidence", f"{float(values['confidence_score']):.2%}")
        model_col.metric("Model", str(values.get("model_used") or "unknown"))
        st.markdown(f"**Customer:** `{values['customer_id']}`")
        st.markdown(f"**Proposed action:** `{values['proposed_action']}`")
        st.info(str(values.get("reasoning") or "No reasoning returned."))

    if "execute_high_risk_action" in snapshot.next:
        st.warning("Graph đang dừng trước high-risk action để chờ human review.")
        approve_col, reject_col = st.columns(2)
        if approve_col.button("Approve", type="primary", use_container_width=True):
            _resume_run("approve")
            st.rerun()
        if reject_col.button("Reject", use_container_width=True):
            _resume_run("reject")
            st.rerun()

        with st.expander("Edit action"):
            available_actions: list[Action] = ["send_email", "increase_credit_limit"]
            current_action = str(values["proposed_action"])
            current_index = (
                available_actions.index(current_action)
                if current_action in available_actions
                else 0
            )
            edited_action = st.selectbox(
                "Action sau chỉnh sửa",
                available_actions,
                index=current_index,
            )
            if st.button("Save edit and execute"):
                _resume_run("edit", edited_action)
                st.rerun()
    elif values.get("execution_result"):
        st.success(str(values["execution_result"]))
        if st.button("Bắt đầu run mới"):
            _clear_run()
            st.rerun()

_render_audit_history()
