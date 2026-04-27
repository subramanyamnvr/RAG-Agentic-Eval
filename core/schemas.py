"""Data models used across the evaluation dashboard.

In a project like this, the same data moves through several parts of the app:

- uploaded dataset rows
- evaluator inputs
- dashboard filters
- stored evaluation runs

Instead of passing around loosely structured dictionaries everywhere, this file
defines a few clear Python dataclasses. That makes the code easier to read and
helps new learners understand what each part of the application expects.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def current_timestamp() -> str:
    """Return the current UTC time in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def text_value(value: Any, default: str = "") -> str:
    """Return a string value with a safe default."""

    if value is None:
        return default
    return str(value)


def optional_float(value: Any) -> float | None:
    """Convert a value to float when possible."""

    if value in (None, ""):
        return None
    return float(value)


def optional_int(value: Any) -> int | None:
    """Convert a value to int when possible."""

    if value in (None, ""):
        return None
    return int(value)


def dict_value(value: Any) -> dict[str, Any]:
    """Return a dictionary value or an empty dictionary."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError("Expected a dictionary value.")


def string_list(value: Any) -> list[str]:
    """Convert a value into a list of strings.

    The uploaded data may contain:
    - a Python list
    - one single string
    - nothing at all
    """

    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


@dataclass
class DashboardSelections:
    """Values chosen by the user at the top of the dashboard."""

    application_type: str = "LLM"
    provider: str = "OpenAI"
    model: str = "gpt-4.1-mini"
    framework: str = "Custom"
    observability_tool: str = "None"
    storage_backend: str = "json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DashboardSelections":
        """Build dashboard selections from raw input data."""

        return cls(
            application_type=text_value(data.get("application_type"), "LLM"),
            provider=text_value(data.get("provider"), "OpenAI"),
            model=text_value(data.get("model"), "gpt-4.1-mini"),
            framework=text_value(data.get("framework"), "Custom"),
            observability_tool=text_value(data.get("observability_tool"), "None"),
            storage_backend=text_value(data.get("storage_backend"), "json"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass into a regular dictionary."""

        return asdict(self)


@dataclass
class AgentStep:
    """One step taken by an agent during a run.

    Agent evaluation is different from plain LLM evaluation because we care
    about the process, not only the final answer. A single run may include many
    steps such as planning, tool calls, retries, and final response generation.
    """

    step_number: int
    action: str
    tool_name: str = ""
    tool_input: str = ""
    tool_output: str = ""
    status: str = "completed"
    latency_ms: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentStep":
        """Build an agent step from raw data."""

        return cls(
            step_number=int(data.get("step_number", 1)),
            action=text_value(data.get("action"), "unknown"),
            tool_name=text_value(data.get("tool_name")),
            tool_input=text_value(data.get("tool_input")),
            tool_output=text_value(data.get("tool_output")),
            status=text_value(data.get("status"), "completed"),
            latency_ms=optional_float(data.get("latency_ms")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the step back into a dictionary."""

        return asdict(self)


@dataclass
class LLMEvaluationRow:
    """One uploaded row for plain LLM evaluation."""

    question: str
    expected_answer: str
    actual_answer: str
    model: str = ""
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMEvaluationRow":
        """Build an LLM evaluation row from uploaded data."""

        return cls(
            question=text_value(data.get("question")),
            expected_answer=text_value(data.get("expected_answer")),
            actual_answer=text_value(data.get("actual_answer")),
            model=text_value(data.get("model")),
            latency_ms=optional_float(data.get("latency_ms")),
            input_tokens=optional_int(data.get("input_tokens")),
            output_tokens=optional_int(data.get("output_tokens")),
            metadata=dict_value(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the row into a regular dictionary."""

        return asdict(self)


@dataclass
class RAGEvaluationRow:
    """One uploaded row for RAG evaluation."""

    question: str
    retrieved_contexts: list[str]
    expected_answer: str
    actual_answer: str
    model: str = ""
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RAGEvaluationRow":
        """Build a RAG evaluation row from uploaded data."""

        return cls(
            question=text_value(data.get("question")),
            retrieved_contexts=string_list(data.get("retrieved_contexts")),
            expected_answer=text_value(data.get("expected_answer")),
            actual_answer=text_value(data.get("actual_answer")),
            model=text_value(data.get("model")),
            latency_ms=optional_float(data.get("latency_ms")),
            input_tokens=optional_int(data.get("input_tokens")),
            output_tokens=optional_int(data.get("output_tokens")),
            metadata=dict_value(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the row into a regular dictionary."""

        return asdict(self)


@dataclass
class AgentEvaluationRow:
    """One uploaded row for agent evaluation."""

    task: str
    expected_result: str
    final_result: str
    tools_expected: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    retry_count: int = 0
    tool_errors: int = 0
    total_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEvaluationRow":
        """Build an agent evaluation row from uploaded data."""

        raw_steps = data.get("steps") or []
        parsed_steps = [AgentStep.from_dict(step) for step in raw_steps]

        return cls(
            task=text_value(data.get("task")),
            expected_result=text_value(data.get("expected_result")),
            final_result=text_value(data.get("final_result")),
            tools_expected=string_list(data.get("tools_expected")),
            tools_used=string_list(data.get("tools_used")),
            steps=parsed_steps,
            retry_count=optional_int(data.get("retry_count")) or 0,
            tool_errors=optional_int(data.get("tool_errors")) or 0,
            total_latency_ms=optional_float(data.get("total_latency_ms")),
            input_tokens=optional_int(data.get("input_tokens")),
            output_tokens=optional_int(data.get("output_tokens")),
            metadata=dict_value(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the row into a regular dictionary."""

        return {
            "task": self.task,
            "expected_result": self.expected_result,
            "final_result": self.final_result,
            "tools_expected": self.tools_expected,
            "tools_used": self.tools_used,
            "steps": [step.to_dict() for step in self.steps],
            "retry_count": self.retry_count,
            "tool_errors": self.tool_errors,
            "total_latency_ms": self.total_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "metadata": self.metadata,
        }


@dataclass
class RunRecord:
    """One saved dashboard run.

    This is the final record written by the storage layer after an evaluation
    has been completed.
    """

    run_id: str
    created_at: str
    app_type: str
    provider: str
    model: str
    framework: str
    observability_tool: str
    storage_backend: str
    dataset_name: str
    status: str
    overall_score: float | None
    latency_ms: float | None
    total_cost: float | None
    input_tokens: int | None
    output_tokens: int | None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        """Build a run record from raw data."""

        return cls(
            run_id=text_value(data.get("run_id"), str(uuid.uuid4())),
            created_at=text_value(data.get("created_at"), current_timestamp()),
            app_type=text_value(data.get("app_type"), "unknown"),
            provider=text_value(data.get("provider"), "unknown"),
            model=text_value(data.get("model"), "unknown"),
            framework=text_value(data.get("framework"), "unknown"),
            observability_tool=text_value(data.get("observability_tool"), "None"),
            storage_backend=text_value(data.get("storage_backend"), "json"),
            dataset_name=text_value(data.get("dataset_name"), "uploaded file"),
            status=text_value(data.get("status"), "completed"),
            overall_score=optional_float(data.get("overall_score")),
            latency_ms=optional_float(data.get("latency_ms")),
            total_cost=optional_float(data.get("total_cost")),
            input_tokens=optional_int(data.get("input_tokens")),
            output_tokens=optional_int(data.get("output_tokens")),
            metrics=dict_value(data.get("metrics")),
            metadata=dict_value(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the run record into a regular dictionary."""

        return asdict(self)
