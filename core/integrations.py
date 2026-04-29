"""Simple integration helpers for the evaluation dashboard.

This file answers a practical beginner question:

"How do the dropdown choices, uploaded file, evaluator, and storage layer all
connect together?"

Instead of calling real provider SDKs in version 1, this module focuses on the
flow of data through the project:

1. the user selects an application type and tools in the dashboard
2. the user uploads a JSON or CSV file
3. the file is converted into schema objects
4. the right evaluator is called
5. the results are saved using the selected storage format

That makes this file a useful bridge between the backend logic and the future
Streamlit interface.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.evaluators import EvaluationEngine
from core.schemas import AgentEvaluationRow
from core.schemas import DashboardSelections
from core.schemas import LLMEvaluationRow
from core.schemas import RAGEvaluationRow
from core.storage import EvaluationStorage


APPLICATION_TYPES = ["LLM", "RAG", "Agentic"]
EVALUATION_LAYERS = ["Model", "RAG", "Agent"]
PRIMARY_CONCERNS = [
    "Answer Quality",
    "Retrieval Quality",
    "Tool Use",
    "Safety",
    "Latency and Cost",
]
DEPLOYMENT_STAGES = ["Demo", "Staging", "Production"]
MODEL_PROVIDERS = ["OpenAI", "Anthropic", "Gemini", "Ollama", "OpenRouter", "Custom"]
FRAMEWORKS = [
    "Custom",
    "LangChain",
    "LlamaIndex",
    "CrewAI",
    "AutoGen",
    "OpenAI Agents SDK",
]
OBSERVABILITY_TOOLS = ["None", "Langfuse", "Phoenix", "OpenLIT", "Helicone", "LangSmith"]
STORAGE_BACKENDS = ["json", "csv"]


def read_uploaded_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Read uploaded data from a JSON or CSV file.

    The dashboard will eventually pass the uploaded file path to this function.
    We return a list of dictionaries because that is the simplest neutral format
    before converting rows into strongly named schema objects.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return read_json_file(file_path)
    if suffix == ".csv":
        return read_csv_file(file_path)

    raise ValueError("Only JSON and CSV files are supported in version 1.")


def read_json_file(file_path: Path) -> list[dict[str, Any]]:
    """Read rows from a JSON file.

    Accepted formats:
    - a list of dictionaries
    - one dictionary, which we wrap into a one-item list
    """

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    data = json.loads(content)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]

    raise ValueError("JSON input must contain either a dictionary or a list of dictionaries.")


def read_csv_file(file_path: Path) -> list[dict[str, Any]]:
    """Read rows from a CSV file."""

    with file_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def parse_dataset_rows(
    application_type: str,
    rows: list[dict[str, Any]],
) -> list[LLMEvaluationRow] | list[RAGEvaluationRow] | list[AgentEvaluationRow]:
    """Convert raw dictionaries into the correct schema objects."""

    application_type = application_type.strip()

    if application_type == "LLM":
        return [LLMEvaluationRow.from_dict(row) for row in rows]
    if application_type == "RAG":
        return [RAGEvaluationRow.from_dict(row) for row in rows]
    if application_type == "Agentic":
        return [AgentEvaluationRow.from_dict(row) for row in rows]

    raise ValueError("Application type must be one of: LLM, RAG, Agentic.")


def create_default_model_map() -> dict[str, list[str]]:
    """Return beginner-friendly model suggestions for each provider."""

    return {
        "OpenAI": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"],
        "Anthropic": ["claude-3-5-haiku", "claude-3-7-sonnet"],
        "Gemini": ["gemini-2.0-flash", "gemini-2.5-pro"],
        "Ollama": ["llama3", "mistral", "phi3"],
        "OpenRouter": ["openrouter-auto"],
        "Custom": ["custom-model"],
    }


def create_tool_catalog() -> dict[str, list[str]]:
    """Return the tool choices shown in the dashboard.

    These are not active SDK integrations yet. They are labels the dashboard
    can use to teach the ecosystem and let users describe their stack.
    """

    return {
        "evaluation_tools": ["RAGAS", "DeepEval", "TruLens", "Promptfoo", "LangSmith"],
        "observability_tools": OBSERVABILITY_TOOLS,
        "agent_frameworks": FRAMEWORKS,
        "storage_backends": STORAGE_BACKENDS,
    }


def create_dataset_column_guide() -> dict[str, list[str]]:
    """Return the expected columns for each dataset type.

    The dashboard can show this to learners before they upload a file.
    """

    return {
        "LLM": [
            "question",
            "expected_answer",
            "actual_answer",
            "model",
            "latency_ms",
            "input_tokens",
            "output_tokens",
        ],
        "RAG": [
            "question",
            "retrieved_contexts",
            "expected_answer",
            "actual_answer",
            "model",
            "latency_ms",
            "input_tokens",
            "output_tokens",
        ],
        "Agentic": [
            "task",
            "expected_result",
            "final_result",
            "tools_expected",
            "tools_used",
            "steps",
            "retry_count",
            "tool_errors",
            "total_latency_ms",
            "input_tokens",
            "output_tokens",
        ],
    }


class IntegrationService:
    """Small service that ties together file loading, evaluation, and storage."""

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        """Create the service with an evaluator instance."""

        self.evaluator = EvaluationEngine(thresholds=thresholds)

    def get_dashboard_options(self) -> dict[str, Any]:
        """Return the dropdown options used by the dashboard."""

        return {
            "application_types": APPLICATION_TYPES,
            "evaluation_layers": EVALUATION_LAYERS,
            "primary_concerns": PRIMARY_CONCERNS,
            "deployment_stages": DEPLOYMENT_STAGES,
            "model_providers": MODEL_PROVIDERS,
            "frameworks": FRAMEWORKS,
            "observability_tools": OBSERVABILITY_TOOLS,
            "storage_backends": STORAGE_BACKENDS,
            "models_by_provider": create_default_model_map(),
            "tool_catalog": create_tool_catalog(),
            "dataset_column_guide": create_dataset_column_guide(),
        }

    def evaluate_file(
        self,
        file_path: str | Path,
        selections: DashboardSelections,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Run the full evaluation flow for one uploaded file.

        This is the main entry point the Streamlit app will use later.
        """

        raw_rows = read_uploaded_file(file_path)
        parsed_rows = parse_dataset_rows(selections.application_type, raw_rows)
        evaluation_result = self.run_selected_evaluation(selections.application_type, parsed_rows)

        summary = evaluation_result["summary"]
        metrics = summary["metrics"]
        overall_score = metrics.get("overall_score")

        run_payload = {
            "app_type": selections.application_type,
            "provider": selections.provider,
            "model": selections.model,
            "framework": selections.framework,
            "observability_tool": selections.observability_tool,
            "storage_backend": selections.storage_backend,
            "dataset_name": dataset_name or Path(file_path).name,
            "status": self.calculate_run_status(summary),
            "overall_score": overall_score,
            "latency_ms": summary["average_latency_ms"],
            "total_cost": summary["average_cost"],
            "input_tokens": self.sum_numeric_field(evaluation_result["row_results"], "input_tokens"),
            "output_tokens": self.sum_numeric_field(evaluation_result["row_results"], "output_tokens"),
            "metrics": metrics,
            "metadata": {
                "row_count": summary["total_rows"],
                "passed_rows": summary["passed_rows"],
                "failed_rows": summary["failed_rows"],
                "file_name": Path(file_path).name,
                "selected_evaluation_tools": selections.evaluation_tools,
                "evaluation_layer": selections.evaluation_layer,
                "primary_concern": selections.primary_concern,
                "deployment_stage": selections.deployment_stage,
            },
        }

        storage = EvaluationStorage(selections.storage_backend)
        saved_run = storage.save_run(run_payload)

        return {
            "selections": selections.to_dict(),
            "evaluation_result": evaluation_result,
            "saved_run": saved_run.to_dict(),
            "storage_health": storage.healthcheck(),
        }

    def run_selected_evaluation(
        self,
        application_type: str,
        parsed_rows: list[LLMEvaluationRow] | list[RAGEvaluationRow] | list[AgentEvaluationRow],
    ) -> dict[str, Any]:
        """Call the matching evaluator for the selected application type."""

        if application_type == "LLM":
            llm_rows = [row for row in parsed_rows if isinstance(row, LLMEvaluationRow)]
            return self.evaluator.evaluate_llm_rows(llm_rows)
        if application_type == "RAG":
            rag_rows = [row for row in parsed_rows if isinstance(row, RAGEvaluationRow)]
            return self.evaluator.evaluate_rag_rows(rag_rows)
        if application_type == "Agentic":
            agent_rows = [row for row in parsed_rows if isinstance(row, AgentEvaluationRow)]
            return self.evaluator.evaluate_agent_rows(agent_rows)

        raise ValueError("Application type must be one of: LLM, RAG, Agentic.")

    def calculate_run_status(self, summary: dict[str, Any]) -> str:
        """Convert batch summary values into one run status.

        A run passes only when every row passes.
        """

        total_rows = summary.get("total_rows", 0)
        passed_rows = summary.get("passed_rows", 0)

        if total_rows == 0:
            return "failed"
        if total_rows == passed_rows:
            return "completed"
        return "failed"

    def sum_numeric_field(self, rows: list[dict[str, Any]], field_name: str) -> int:
        """Add up numeric values from row results.

        This is useful for total token counts across the uploaded dataset.
        """

        total = 0
        for row in rows:
            value = row.get(field_name)
            if isinstance(value, int):
                total += value
        return total
