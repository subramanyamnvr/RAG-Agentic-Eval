"""Store and read evaluation runs for the dashboard.

This project is meant to be beginner friendly, so the storage layer only uses
two formats that most people already recognize:

- JSON
- CSV

JSON is the default because it is the easiest format to read in an editor.
CSV is included because many learners like opening files in spreadsheet tools.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from core.schemas import RunRecord


DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent / ".storage"
SUPPORTED_BACKENDS = {"json", "csv"}
CSV_COLUMNS = [
    "run_id",
    "created_at",
    "app_type",
    "provider",
    "model",
    "framework",
    "observability_tool",
    "storage_backend",
    "dataset_name",
    "status",
    "overall_score",
    "latency_ms",
    "total_cost",
    "input_tokens",
    "output_tokens",
    "metrics_json",
    "metadata_json",
]


def ensure_directory_exists(path: Path) -> None:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def to_json_string(value: Any) -> str:
    """Convert Python data into a JSON string."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def from_json_string(value: str | None) -> Any:
    """Convert a JSON string back into Python data."""

    if not value:
        return None
    return json.loads(value)


def dict_or_empty(value: Any) -> dict[str, Any]:
    """Return a dictionary value or an empty dictionary."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = from_json_string(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected a dictionary or a JSON string containing a dictionary.")


def calculate_average(values: list[float]) -> float | None:
    """Return the average of a list, or `None` if the list is empty."""

    if not values:
        return None
    return sum(values) / len(values)


class EvaluationStorage:
    """Save and load evaluation runs using JSON or CSV."""

    def __init__(self, backend: str = "json", base_dir: str | Path | None = None) -> None:
        """Create a storage object.

        Parameters
        ----------
        backend:
            The storage format to use. Supported values are `json` and `csv`.
        base_dir:
            The folder where local files should be saved.
        """

        backend = backend.lower().strip()
        if backend not in SUPPORTED_BACKENDS:
            available = ", ".join(sorted(SUPPORTED_BACKENDS))
            raise ValueError(f"Unsupported backend '{backend}'. Available options: {available}.")

        self.backend = backend
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_STORAGE_DIR
        self.json_path = self.base_dir / "eval_runs.json"
        self.csv_path = self.base_dir / "eval_runs.csv"

        ensure_directory_exists(self.base_dir)
        self.prepare_backend()

    def prepare_backend(self) -> None:
        """Create the required file for the selected backend."""

        if self.backend == "json":
            if not self.json_path.exists():
                self.json_path.write_text("[]", encoding="utf-8")
            return

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def save_run(self, run_data: dict[str, Any]) -> RunRecord:
        """Save one evaluation run."""

        record = RunRecord.from_dict(run_data)
        record.storage_backend = self.backend

        if self.backend == "json":
            self.save_run_to_json(record)
        else:
            self.save_run_to_csv(record)

        return record

    def list_runs(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return saved runs ordered from newest to oldest."""

        if self.backend == "json":
            records = self.read_runs_from_json(limit=limit, app_type=app_type, status=status)
        else:
            records = self.read_runs_from_csv(limit=limit, app_type=app_type, status=status)

        return [record.to_dict() for record in records]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run by ID."""

        for run in self.list_runs():
            if run["run_id"] == run_id:
                return run
        return None

    def summarize_runs(self) -> dict[str, Any]:
        """Return summary values used by the dashboard cards."""

        runs = self.list_runs()
        if not runs:
            return {
                "total_runs": 0,
                "average_score": None,
                "average_latency_ms": None,
                "average_cost": None,
                "success_rate": None,
                "last_run_at": None,
            }

        scores = [run["overall_score"] for run in runs if run["overall_score"] is not None]
        latencies = [run["latency_ms"] for run in runs if run["latency_ms"] is not None]
        costs = [run["total_cost"] for run in runs if run["total_cost"] is not None]
        completed_runs = [run for run in runs if run["status"] == "completed"]

        return {
            "total_runs": len(runs),
            "average_score": calculate_average(scores),
            "average_latency_ms": calculate_average(latencies),
            "average_cost": calculate_average(costs),
            "success_rate": len(completed_runs) / len(runs),
            "last_run_at": runs[0]["created_at"],
        }

    def healthcheck(self) -> dict[str, str]:
        """Return simple information that confirms storage is ready."""

        return {
            "backend": self.backend,
            "base_dir": str(self.base_dir),
            "status": "ready",
        }

    def save_run_to_json(self, record: RunRecord) -> None:
        """Save runs in one JSON file."""

        rows = self.load_json_rows()
        rows = [row for row in rows if row["run_id"] != record.run_id]
        rows.append(record.to_dict())
        rows.sort(key=lambda row: row["created_at"], reverse=True)

        self.json_path.write_text(to_json_string(rows), encoding="utf-8")

    def read_runs_from_json(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        """Read runs from the JSON file."""

        rows = self.filter_rows(self.load_json_rows(), app_type=app_type, status=status)
        if limit is not None:
            rows = rows[:limit]
        return [RunRecord.from_dict(row) for row in rows]

    def load_json_rows(self) -> list[dict[str, Any]]:
        """Load raw run dictionaries from the JSON file."""

        if not self.json_path.exists():
            return []

        content = self.json_path.read_text(encoding="utf-8").strip()
        if not content:
            return []

        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("The JSON storage file must contain a list of runs.")

        return data

    def save_run_to_csv(self, record: RunRecord) -> None:
        """Save runs in one CSV file.

        CSV supports only flat columns, so dictionaries are stored as JSON
        strings inside the `metrics_json` and `metadata_json` fields.
        """

        rows = self.load_csv_rows()
        rows = [row for row in rows if row["run_id"] != record.run_id]
        rows.append(self.record_to_csv_row(record))
        rows.sort(key=lambda row: row["created_at"], reverse=True)

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def read_runs_from_csv(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        """Read runs from the CSV file."""

        rows = self.filter_rows(self.load_csv_rows(), app_type=app_type, status=status)
        if limit is not None:
            rows = rows[:limit]

        records: list[RunRecord] = []
        for row in rows:
            records.append(RunRecord.from_dict(self.csv_row_to_record_data(row)))
        return records

    def load_csv_rows(self) -> list[dict[str, Any]]:
        """Load raw CSV rows as dictionaries."""

        if not self.csv_path.exists():
            return []

        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    def record_to_csv_row(self, record: RunRecord) -> dict[str, Any]:
        """Convert one `RunRecord` into one CSV row."""

        return {
            "run_id": record.run_id,
            "created_at": record.created_at,
            "app_type": record.app_type,
            "provider": record.provider,
            "model": record.model,
            "framework": record.framework,
            "observability_tool": record.observability_tool,
            "storage_backend": record.storage_backend,
            "dataset_name": record.dataset_name,
            "status": record.status,
            "overall_score": record.overall_score,
            "latency_ms": record.latency_ms,
            "total_cost": record.total_cost,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "metrics_json": to_json_string(record.metrics),
            "metadata_json": to_json_string(record.metadata),
        }

    def csv_row_to_record_data(self, row: dict[str, Any]) -> dict[str, Any]:
        """Convert one CSV row back into the dictionary shape used by `RunRecord`."""

        return {
            "run_id": row.get("run_id"),
            "created_at": row.get("created_at"),
            "app_type": row.get("app_type"),
            "provider": row.get("provider"),
            "model": row.get("model"),
            "framework": row.get("framework"),
            "observability_tool": row.get("observability_tool"),
            "storage_backend": row.get("storage_backend"),
            "dataset_name": row.get("dataset_name"),
            "status": row.get("status"),
            "overall_score": row.get("overall_score"),
            "latency_ms": row.get("latency_ms"),
            "total_cost": row.get("total_cost"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "metrics": dict_or_empty(row.get("metrics_json")),
            "metadata": dict_or_empty(row.get("metadata_json")),
        }

    def filter_rows(
        self,
        rows: list[dict[str, Any]],
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter rows in Python after reading the file."""

        filtered_rows = rows

        if app_type:
            filtered_rows = [row for row in filtered_rows if row.get("app_type") == app_type]

        if status:
            filtered_rows = [row for row in filtered_rows if row.get("status") == status]

        filtered_rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return filtered_rows
