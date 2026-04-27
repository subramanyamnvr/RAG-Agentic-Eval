"""Evaluation logic for LLM, RAG, and agentic applications.

This file focuses on teaching the ideas behind evaluation.

The scores here are simple heuristic scores, not production-grade academic
benchmarks. That is intentional for version 1 of the project:

- beginners can read the code and understand how each score is calculated
- the dashboard can demonstrate evaluation concepts without extra setup
- later, these methods can be replaced or extended with tools like RAGAS,
  DeepEval, or LangSmith

The module supports three evaluation styles:

1. LLM evaluation
   Compare a generated answer with an expected answer and the original question.

2. RAG evaluation
   Check whether the answer is relevant and whether it stays grounded in the
   retrieved context.

3. Agent evaluation
   Check the final result, tool usage, retries, and step efficiency.
"""

from __future__ import annotations

import re
from typing import Any

from core.schemas import AgentEvaluationRow
from core.schemas import AgentStep
from core.schemas import LLMEvaluationRow
from core.schemas import RAGEvaluationRow


DEFAULT_THRESHOLDS = {
    "correctness": 0.7,
    "relevancy": 0.7,
    "faithfulness": 0.7,
    "task_success": 0.7,
    "tool_selection_accuracy": 0.7,
}


def normalize_text(text: str) -> str:
    """Return a simplified version of text for matching.

    Evaluation is easier when we ignore differences like:
    - uppercase vs lowercase
    - punctuation
    - extra spaces

    This is not perfect, but it is enough for a beginner-friendly project.
    """

    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def split_words(text: str) -> list[str]:
    """Split text into normalized words."""

    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return cleaned.split()


def unique_words(text: str) -> set[str]:
    """Return unique normalized words from text."""

    return set(split_words(text))


def average(values: list[float]) -> float | None:
    """Return the average of a list, or `None` if the list is empty."""

    if not values:
        return None
    return sum(values) / len(values)


def clamp_score(value: float) -> float:
    """Keep scores between 0 and 1."""

    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return round(value, 4)


def overlap_ratio(reference_words: set[str], candidate_words: set[str]) -> float:
    """Return how much of the reference set appears in the candidate set.

    Example:
    - expected answer words are the reference
    - actual answer words are the candidate

    If most reference words appear in the candidate, the score is high.
    """

    if not reference_words:
        return 0.0

    matched_words = reference_words.intersection(candidate_words)
    return clamp_score(len(matched_words) / len(reference_words))


def jaccard_similarity(first_words: set[str], second_words: set[str]) -> float:
    """Return similarity between two word sets.

    Jaccard similarity looks at overlap compared with the total combined words.
    It is a simple way to compare whether two pieces of text talk about similar
    things.
    """

    if not first_words and not second_words:
        return 0.0

    union = first_words.union(second_words)
    if not union:
        return 0.0

    intersection = first_words.intersection(second_words)
    return clamp_score(len(intersection) / len(union))


def estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float:
    """Estimate cost using a very simple placeholder formula.

    The project needs a cost number for teaching dashboards, but real cost
    depends on the provider and model. For version 1 we use a transparent
    approximation:

    - input tokens cost $0.000001 each
    - output tokens cost $0.000002 each

    These numbers are not meant to match a specific vendor exactly.
    """

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0

    input_cost = input_tokens * 0.000001
    output_cost = output_tokens * 0.000002
    return round(input_cost + output_cost, 6)


def metric_status(score: float | None, threshold: float) -> str:
    """Convert a score into `pass` or `fail`."""

    if score is None:
        return "fail"
    return "pass" if score >= threshold else "fail"


def count_repeated_steps(steps: list[AgentStep]) -> int:
    """Count repeated actions as a simple loop signal.

    If an agent repeats the same action several times, it may be stuck.
    """

    if not steps:
        return 0

    repeated_count = 0
    previous_signature = None

    for step in steps:
        current_signature = (normalize_text(step.action), normalize_text(step.tool_name))
        if current_signature == previous_signature:
            repeated_count += 1
        previous_signature = current_signature

    return repeated_count


def build_metric_explanations() -> dict[str, str]:
    """Return short human-readable metric explanations for the UI."""

    return {
        "correctness": "How closely the generated answer matches the expected answer.",
        "relevancy": "Whether the answer actually addresses the user question.",
        "completeness": "How much of the expected answer seems to be covered.",
        "faithfulness": "Whether the answer stays grounded in the retrieved context.",
        "groundedness": "How much of the answer can be traced back to the source context.",
        "context_precision": "How useful the retrieved context chunks were for producing the answer.",
        "context_recall": "How much of the important expected information was present in the context.",
        "task_success": "How closely the final agent result matches the expected result.",
        "tool_selection_accuracy": "Whether the agent chose the tools it was expected to use.",
        "tool_precision": "Whether the tools the agent used were actually appropriate.",
        "step_efficiency": "Whether the agent used a reasonable number of steps.",
        "tool_failure_rate": "How often tool usage failed during the run.",
        "loop_risk": "Whether the agent repeated actions in a way that suggests it got stuck.",
    }


class EvaluationEngine:
    """Run simple evaluations for uploaded rows.

    The engine returns:
    - row-level metric details
    - aggregate summary metrics
    - metric explanations for the dashboard
    """

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        """Create the evaluation engine with optional custom thresholds."""

        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)

    def evaluate_llm_rows(self, rows: list[LLMEvaluationRow]) -> dict[str, Any]:
        """Evaluate a batch of plain LLM rows."""

        row_results: list[dict[str, Any]] = []

        for row in rows:
            row_results.append(self.evaluate_llm_row(row))

        summary = self.summarize_row_results(
            row_results,
            metric_names=["correctness", "relevancy", "completeness", "overall_score"],
        )

        return {
            "evaluation_type": "LLM",
            "row_results": row_results,
            "summary": summary,
            "metric_explanations": build_metric_explanations(),
        }

    def evaluate_llm_row(self, row: LLMEvaluationRow) -> dict[str, Any]:
        """Evaluate one plain LLM row.

        Metrics used here:
        - correctness: expected answer coverage
        - relevancy: overlap between question and answer
        - completeness: similarity between expected and actual answer
        """

        expected_words = unique_words(row.expected_answer)
        actual_words = unique_words(row.actual_answer)
        question_words = unique_words(row.question)

        correctness = overlap_ratio(expected_words, actual_words)
        relevancy = overlap_ratio(question_words, actual_words)
        completeness = jaccard_similarity(expected_words, actual_words)

        overall_score = clamp_score(average([correctness, relevancy, completeness]) or 0.0)

        return {
            "question": row.question,
            "expected_answer": row.expected_answer,
            "actual_answer": row.actual_answer,
            "metrics": {
                "correctness": correctness,
                "relevancy": relevancy,
                "completeness": completeness,
                "overall_score": overall_score,
            },
            "latency_ms": row.latency_ms,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "estimated_cost": estimate_cost(row.input_tokens, row.output_tokens),
            "status": metric_status(overall_score, self.thresholds["correctness"]),
        }

    def evaluate_rag_rows(self, rows: list[RAGEvaluationRow]) -> dict[str, Any]:
        """Evaluate a batch of RAG rows."""

        row_results: list[dict[str, Any]] = []

        for row in rows:
            row_results.append(self.evaluate_rag_row(row))

        summary = self.summarize_row_results(
            row_results,
            metric_names=[
                "faithfulness",
                "groundedness",
                "context_precision",
                "context_recall",
                "overall_score",
            ],
        )

        return {
            "evaluation_type": "RAG",
            "row_results": row_results,
            "summary": summary,
            "metric_explanations": build_metric_explanations(),
        }

    def evaluate_rag_row(self, row: RAGEvaluationRow) -> dict[str, Any]:
        """Evaluate one RAG row.

        We use simple educational heuristics:
        - faithfulness: how much of the answer appears in the context
        - groundedness: similarity between answer words and context words
        - context precision: how many retrieved chunks were actually useful
        - context recall: how much of the expected answer appears in the context
        """

        context_text = " ".join(row.retrieved_contexts)
        answer_words = unique_words(row.actual_answer)
        expected_words = unique_words(row.expected_answer)
        context_words = unique_words(context_text)

        faithfulness = overlap_ratio(answer_words, context_words)
        groundedness = jaccard_similarity(answer_words, context_words)
        context_precision = self.calculate_context_precision(row.retrieved_contexts, row.actual_answer)
        context_recall = overlap_ratio(expected_words, context_words)
        overall_score = clamp_score(
            average([faithfulness, groundedness, context_precision, context_recall]) or 0.0
        )

        return {
            "question": row.question,
            "expected_answer": row.expected_answer,
            "actual_answer": row.actual_answer,
            "retrieved_contexts": row.retrieved_contexts,
            "metrics": {
                "faithfulness": faithfulness,
                "groundedness": groundedness,
                "context_precision": context_precision,
                "context_recall": context_recall,
                "overall_score": overall_score,
            },
            "latency_ms": row.latency_ms,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "estimated_cost": estimate_cost(row.input_tokens, row.output_tokens),
            "status": metric_status(overall_score, self.thresholds["faithfulness"]),
        }

    def calculate_context_precision(self, contexts: list[str], answer: str) -> float:
        """Score how useful the retrieved chunks were.

        Each context chunk counts as useful if it shares at least one normalized
        word with the answer.
        """

        if not contexts:
            return 0.0

        answer_words = unique_words(answer)
        useful_chunk_count = 0

        for context in contexts:
            context_words = unique_words(context)
            if answer_words.intersection(context_words):
                useful_chunk_count += 1

        return clamp_score(useful_chunk_count / len(contexts))

    def evaluate_agent_rows(self, rows: list[AgentEvaluationRow]) -> dict[str, Any]:
        """Evaluate a batch of agent rows."""

        row_results: list[dict[str, Any]] = []

        for row in rows:
            row_results.append(self.evaluate_agent_row(row))

        summary = self.summarize_row_results(
            row_results,
            metric_names=[
                "task_success",
                "tool_selection_accuracy",
                "tool_precision",
                "step_efficiency",
                "overall_score",
            ],
        )

        return {
            "evaluation_type": "Agentic",
            "row_results": row_results,
            "summary": summary,
            "metric_explanations": build_metric_explanations(),
        }

    def evaluate_agent_row(self, row: AgentEvaluationRow) -> dict[str, Any]:
        """Evaluate one agent row.

        Agent evaluation looks at both:
        - the final result
        - the process used to get there
        """

        expected_words = unique_words(row.expected_result)
        final_words = unique_words(row.final_result)

        expected_tools = {normalize_text(tool) for tool in row.tools_expected if normalize_text(tool)}
        used_tools = {normalize_text(tool) for tool in row.tools_used if normalize_text(tool)}

        task_success = overlap_ratio(expected_words, final_words)
        tool_selection_accuracy = overlap_ratio(expected_tools, used_tools) if expected_tools else 1.0
        tool_precision = overlap_ratio(used_tools, expected_tools) if used_tools else 1.0
        step_efficiency = self.calculate_step_efficiency(row.steps, len(expected_tools))
        tool_failure_rate = self.calculate_tool_failure_rate(row.tool_errors, len(row.tools_used))
        loop_risk = self.calculate_loop_risk(row.steps)

        positive_loop_score = clamp_score(1 - loop_risk)
        positive_failure_score = clamp_score(1 - tool_failure_rate)
        overall_score = clamp_score(
            average(
                [
                    task_success,
                    tool_selection_accuracy,
                    tool_precision,
                    step_efficiency,
                    positive_failure_score,
                    positive_loop_score,
                ]
            )
            or 0.0
        )

        total_latency = row.total_latency_ms
        if total_latency is None and row.steps:
            step_latencies = [step.latency_ms for step in row.steps if step.latency_ms is not None]
            total_latency = sum(step_latencies) if step_latencies else None

        return {
            "task": row.task,
            "expected_result": row.expected_result,
            "final_result": row.final_result,
            "tools_expected": row.tools_expected,
            "tools_used": row.tools_used,
            "steps": [step.to_dict() for step in row.steps],
            "metrics": {
                "task_success": task_success,
                "tool_selection_accuracy": tool_selection_accuracy,
                "tool_precision": tool_precision,
                "step_efficiency": step_efficiency,
                "tool_failure_rate": tool_failure_rate,
                "loop_risk": loop_risk,
                "overall_score": overall_score,
            },
            "latency_ms": total_latency,
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "estimated_cost": estimate_cost(row.input_tokens, row.output_tokens),
            "status": metric_status(overall_score, self.thresholds["task_success"]),
        }

    def calculate_step_efficiency(self, steps: list[AgentStep], expected_tool_count: int) -> float:
        """Score whether the agent used a reasonable number of steps.

        The exact ideal number is application-specific, so we use a simple rule:
        - at least one step is expected
        - an agent usually needs around `expected tools + 1` steps
        - more steps than that reduce the score gradually
        """

        step_count = len(steps)
        if step_count == 0:
            return 0.0

        ideal_step_count = max(1, expected_tool_count + 1)
        if step_count <= ideal_step_count:
            return 1.0

        extra_steps = step_count - ideal_step_count
        score = 1 - (extra_steps / max(step_count, 1))
        return clamp_score(score)

    def calculate_tool_failure_rate(self, tool_errors: int, tool_count: int) -> float:
        """Return a simple tool failure rate."""

        if tool_count <= 0:
            return 0.0
        return clamp_score(tool_errors / tool_count)

    def calculate_loop_risk(self, steps: list[AgentStep]) -> float:
        """Return a score that estimates whether the agent may have looped."""

        if not steps:
            return 0.0

        repeated_count = count_repeated_steps(steps)
        return clamp_score(repeated_count / len(steps))

    def summarize_row_results(
        self,
        row_results: list[dict[str, Any]],
        metric_names: list[str],
    ) -> dict[str, Any]:
        """Summarize row-level results into dashboard-friendly numbers."""

        if not row_results:
            return {
                "total_rows": 0,
                "passed_rows": 0,
                "failed_rows": 0,
                "average_latency_ms": None,
                "average_cost": None,
                "metrics": {},
            }

        summary_metrics: dict[str, float | None] = {}
        for metric_name in metric_names:
            values = [result["metrics"].get(metric_name) for result in row_results]
            numeric_values = [value for value in values if isinstance(value, (float, int))]
            summary_metrics[metric_name] = average([float(value) for value in numeric_values])

        latencies = [result["latency_ms"] for result in row_results if result.get("latency_ms") is not None]
        costs = [
            result["estimated_cost"]
            for result in row_results
            if result.get("estimated_cost") is not None
        ]
        passed_rows = [result for result in row_results if result["status"] == "pass"]

        return {
            "total_rows": len(row_results),
            "passed_rows": len(passed_rows),
            "failed_rows": len(row_results) - len(passed_rows),
            "average_latency_ms": average([float(value) for value in latencies]),
            "average_cost": average([float(value) for value in costs]),
            "metrics": summary_metrics,
        }
