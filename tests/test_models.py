from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import AgentEvaluation, AuditEntry


def test_agent_evaluation_accepts_valid_output() -> None:
    result = AgentEvaluation(
        proposed_action="send_email",
        confidence_score=0.9,
        reasoning="Customer has elevated churn risk.",
    )

    assert result.confidence_score == 0.9


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_agent_evaluation_rejects_out_of_range_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        AgentEvaluation(
            proposed_action="send_email",
            confidence_score=confidence,
            reasoning="Reason",
        )


def test_agent_evaluation_rejects_blank_reasoning() -> None:
    with pytest.raises(ValidationError):
        AgentEvaluation(
            proposed_action="send_email",
            confidence_score=0.9,
            reasoning="   ",
        )


def test_audit_entry_requires_valid_confidence() -> None:
    with pytest.raises(ValidationError):
        AuditEntry(
            timestamp="2026-08-29T09:00:00+00:00",
            agent_id="churn-risk-agent",
            action="send_email",
            confidence=2.0,
            reviewer_id="system",
            decision="auto_execute",
        )
