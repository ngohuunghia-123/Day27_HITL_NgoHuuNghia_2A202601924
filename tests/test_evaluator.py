from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from graph import (
    ConfigurationError,
    EvaluationError,
    GeminiEvaluator,
    create_initial_state,
)


class FakeStructuredInvoker:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome

    def invoke(self, prompt: str) -> object:
        assert "CUSTOMER_DATA=" in prompt
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeChatModel:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome

    def with_structured_output(
        self,
        schema: dict[str, Any],
        *,
        method: str,
    ) -> FakeStructuredInvoker:
        assert schema["title"] == "AgentEvaluation"
        assert method == "json_schema"
        return FakeStructuredInvoker(self.outcome)


def _factory(
    outcomes: dict[str, object],
    calls: list[str],
) -> Callable[[str, str], FakeChatModel]:
    def create(model_name: str, api_key: str) -> FakeChatModel:
        assert api_key == "test-key"
        calls.append(model_name)
        return FakeChatModel(outcomes[model_name])

    return create


def _state():  # type: ignore[no-untyped-def]
    return create_initial_state(
        customer_id="CUST001",
        total_operating_income=50_000_000,
        churn_probability=0.75,
    )


def test_primary_success_does_not_call_fallback() -> None:
    calls: list[str] = []
    evaluator = GeminiEvaluator(
        api_key="test-key",
        primary_model="primary",
        fallback_model="fallback",
        model_factory=_factory(
            {
                "primary": {
                    "proposed_action": "send_email",
                    "confidence_score": 0.9,
                    "reasoning": "Valid proposal",
                },
                "fallback": AssertionError("Fallback should not be called"),
            },
            calls,
        ),
    )

    result = evaluator.evaluate(_state())

    assert result["model_used"] == "primary"
    assert calls == ["primary"]


def test_primary_failure_uses_fallback() -> None:
    calls: list[str] = []
    evaluator = GeminiEvaluator(
        api_key="test-key",
        primary_model="primary",
        fallback_model="fallback",
        model_factory=_factory(
            {
                "primary": TimeoutError("timeout"),
                "fallback": {
                    "proposed_action": "increase_credit_limit",
                    "confidence_score": 0.8,
                    "reasoning": "Fallback proposal",
                },
            },
            calls,
        ),
    )

    result = evaluator.evaluate(_state())

    assert result["model_used"] == "fallback"
    assert calls == ["primary", "fallback"]


def test_invalid_primary_schema_uses_fallback() -> None:
    calls: list[str] = []
    evaluator = GeminiEvaluator(
        api_key="test-key",
        primary_model="primary",
        fallback_model="fallback",
        model_factory=_factory(
            {
                "primary": {"proposed_action": "delete_account"},
                "fallback": {
                    "proposed_action": "send_email",
                    "confidence_score": 0.7,
                    "reasoning": "Escalate low confidence",
                },
            },
            calls,
        ),
    )

    result = evaluator.evaluate(_state())

    assert result["model_used"] == "fallback"


def test_both_models_fail_without_mock_result() -> None:
    evaluator = GeminiEvaluator(
        api_key="test-key",
        primary_model="primary",
        fallback_model="fallback",
        model_factory=_factory(
            {"primary": TimeoutError(), "fallback": RuntimeError()},
            [],
        ),
    )

    with pytest.raises(EvaluationError):
        evaluator.evaluate(_state())


def test_authentication_error_does_not_call_fallback() -> None:
    calls: list[str] = []
    evaluator = GeminiEvaluator(
        api_key="test-key",
        primary_model="primary",
        fallback_model="fallback",
        model_factory=_factory(
            {
                "primary": RuntimeError("401 API key not valid"),
                "fallback": {
                    "proposed_action": "send_email",
                    "confidence_score": 0.9,
                    "reasoning": "Should not run",
                },
            },
            calls,
        ),
    )

    with pytest.raises(ConfigurationError):
        evaluator.evaluate(_state())
    assert calls == ["primary"]


def test_missing_api_key_is_rejected_before_model_call() -> None:
    with pytest.raises(ConfigurationError):
        GeminiEvaluator(api_key="")
