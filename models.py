"""Typed state and validated schemas for the HITL workflow."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

Action = Literal["send_email", "increase_credit_limit"]
HumanDecision = Literal["approve", "reject", "edit"]
Route = Literal["low_risk", "high_risk"]


class GraphState(TypedDict):
    """Persistent state shared by every LangGraph node."""

    customer_id: str
    total_operating_income: float
    churn_probability: float
    proposed_action: Action | None
    confidence_score: float | None
    reasoning: str | None
    model_used: str | None
    human_decision: HumanDecision | None
    edited_action: Action | None
    reviewer_id: str | None
    execution_result: str | None


class AgentEvaluation(BaseModel):
    """Strict contract for the action proposed by Gemini."""

    proposed_action: Action
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @field_validator("reasoning")
    @classmethod
    def reasoning_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("reasoning must not be blank")
        return cleaned


class AuditEntry(BaseModel):
    """Auditable record required by the lab rubric."""

    timestamp: str
    agent_id: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    reviewer_id: str
    decision: str
