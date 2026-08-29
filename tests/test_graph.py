from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from audit import read_audit_entries
from graph import build_graph, create_initial_state, route_action
from models import GraphState


def _initial_state() -> GraphState:
    return create_initial_state(
        customer_id="CUST001",
        total_operating_income=50_000_000,
        churn_probability=0.75,
    )


def _evaluated_state(action: str, confidence: float) -> GraphState:
    state = _initial_state()
    state["proposed_action"] = action  # type: ignore[typeddict-item]
    state["confidence_score"] = confidence
    state["reasoning"] = "Test reasoning"
    state["model_used"] = "test-model"
    return state


def _evaluator(action: str, confidence: float):  # type: ignore[no-untyped-def]
    def evaluate(state: GraphState) -> dict[str, object]:
        assert state["customer_id"] == "CUST001"
        return {
            "proposed_action": action,
            "confidence_score": confidence,
            "reasoning": "Test reasoning",
            "model_used": "test-model",
        }

    return evaluate


def _config() -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_hard_policy_overrides_high_confidence() -> None:
    assert route_action(_evaluated_state("increase_credit_limit", 0.99)) == "high_risk"


def test_high_confidence_low_risk_auto_routes() -> None:
    assert route_action(_evaluated_state("send_email", 0.90)) == "low_risk"


def test_low_confidence_low_risk_escalates() -> None:
    assert route_action(_evaluated_state("send_email", 0.82)) == "high_risk"


def test_low_risk_auto_executes_and_audits(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    workflow = build_graph(
        evaluator=_evaluator("send_email", 0.90),
        audit_path=audit_path,
    )

    result = workflow.invoke(_initial_state(), config=_config())

    assert "Auto-executed send_email" in result["execution_result"]
    entries = read_audit_entries(audit_path)
    assert len(entries) == 1
    assert entries[0].decision == "auto_execute"
    assert entries[0].reviewer_id == "system"


def test_high_risk_interrupts_before_execution_and_preserves_state(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    workflow = build_graph(
        evaluator=_evaluator("increase_credit_limit", 0.99),
        audit_path=audit_path,
    )
    config = _config()

    workflow.invoke(_initial_state(), config=config)
    snapshot = workflow.get_state(config)

    assert snapshot.next == ("execute_high_risk_action",)
    assert snapshot.values["customer_id"] == "CUST001"
    assert snapshot.values["confidence_score"] == 0.99
    assert read_audit_entries(audit_path) == []


@pytest.mark.parametrize(
    ("decision", "expected_text"),
    [("approve", "Human-approved"), ("reject", "no action executed")],
)
def test_human_decision_resumes_graph(
    tmp_path: Path,
    decision: str,
    expected_text: str,
) -> None:
    audit_path = tmp_path / "audit.json"
    workflow = build_graph(
        evaluator=_evaluator("increase_credit_limit", 0.99),
        audit_path=audit_path,
    )
    config = _config()
    workflow.invoke(_initial_state(), config=config)

    workflow.update_state(
        config,
        {"human_decision": decision, "reviewer_id": "operator_01"},
    )
    result = workflow.invoke(None, config=config)

    assert expected_text in result["execution_result"]
    entries = read_audit_entries(audit_path)
    assert len(entries) == 1
    assert entries[0].decision == decision


def test_edit_updates_action_before_resume(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    workflow = build_graph(
        evaluator=_evaluator("increase_credit_limit", 0.99),
        audit_path=audit_path,
    )
    config = _config()
    workflow.invoke(_initial_state(), config=config)

    workflow.update_state(
        config,
        {
            "human_decision": "edit",
            "reviewer_id": "operator_01",
            "edited_action": "send_email",
        },
    )
    result = workflow.invoke(None, config=config)

    assert "send_email" in result["execution_result"]
    entries = read_audit_entries(audit_path)
    assert entries[0].decision == "edit"
    assert entries[0].action == "send_email"
