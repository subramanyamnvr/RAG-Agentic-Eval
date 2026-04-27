"""Store and read evaluation runs for the dashboard.

This file is written to be easy to follow in a learning project.

The dashboard will produce evaluation results such as:
- app type: LLM, RAG, or Agentic
- model and framework used
- score, latency, token usage, and cost
- detailed metric breakdowns

We want to save those results locally so the dashboard can show:
- current run details
- previous runs
- trend charts
- summary cards

This module supports four local storage options:
- SQLite
- JSON
- CSV
- DuckDB

SQLite is the default because it is simple and works well for most local apps.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from core.schemas import RunRecord


DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent.parent / ".storage"
SUPPORTED_BACKENDS = {"sqlite", "json", "csv", "duckdb"}
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
    """Convert a JSON string back into Python data.

    Empty values return `None` because CSV and database fields can be blank.
    """

    if not value:
        return None
    return json.loads(value)
def optional_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary from either a dict or a JSON string."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = from_json_string(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected a dictionary or a JSON string containing a dictionary.")


class EvaluationStorage:
    """Save and load evaluation runs from a chosen local backend."""

    def __init__(self, backend: str = "sqlite", base_dir: str | Path | None = None) -> None:
        """Create a storage object.

        Parameters
        ----------
        backend:
            Which storage type to use. Supported values are `sqlite`, `json`,
            `csv`, and `duckdb`.
        base_dir:
            Where storage files should live. If not provided, the code creates a
            `.storage` folder in the project.
        """

        backend = backend.lower().strip()
        if backend not in SUPPORTED_BACKENDS:
            available = ", ".join(sorted(SUPPORTED_BACKENDS))
            raise ValueError(f"Unsupported backend '{backend}'. Available options: {available}.")

        self.backend = backend
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_STORAGE_DIR
        ensure_directory_exists(self.base_dir)

        self.sqlite_path = self.base_dir / "eval_runs.db"
        self.json_path = self.base_dir / "eval_runs.json"
        self.csv_path = self.base_dir / "eval_runs.csv"
        self.duckdb_path = self.base_dir / "eval_runs.duckdb"

        self.prepare_backend()

    def prepare_backend(self) -> None:
        """Create the required file or table for the selected backend."""

        if self.backend == "sqlite":
            self.prepare_sqlite()
            return

        if self.backend == "json":
            if not self.json_path.exists():
                self.json_path.write_text("[]", encoding="utf-8")
            return

        if self.backend == "csv":
            if not self.csv_path.exists():
                with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
            return

        if self.backend == "duckdb":
            self.prepare_duckdb()

    def save_run(self, run_data: dict[str, Any]) -> RunRecord:
        """Save one evaluation run.

        The caller passes a dictionary. We convert it into a `RunRecord` first,
        then store it using the selected backend.
        """

        record = RunRecord.from_dict(run_data)
        record.storage_backend = self.backend

        if self.backend == "sqlite":
            self.save_run_to_sqlite(record)
        elif self.backend == "json":
            self.save_run_to_json(record)
        elif self.backend == "csv":
            self.save_run_to_csv(record)
        elif self.backend == "duckdb":
            self.save_run_to_duckdb(record)

        return record

    def list_runs(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return stored runs ordered from newest to oldest."""

        if self.backend == "sqlite":
            records = self.read_runs_from_sqlite(limit=limit, app_type=app_type, status=status)
        elif self.backend == "json":
            records = self.read_runs_from_json(limit=limit, app_type=app_type, status=status)
        elif self.backend == "csv":
            records = self.read_runs_from_csv(limit=limit, app_type=app_type, status=status)
        else:
            records = self.read_runs_from_duckdb(limit=limit, app_type=app_type, status=status)

        return [record.to_dict() for record in records]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run by its ID."""

        for run in self.list_runs():
            if run["run_id"] == run_id:
                return run
        return None

    def summarize_runs(self) -> dict[str, Any]:
        """Return summary values used by dashboard cards.

        This avoids repeating the same aggregation logic in the Streamlit app.
        """

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

    def prepare_sqlite(self) -> None:
        """Create the SQLite table if it does not exist yet."""

        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    app_type TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    observability_tool TEXT NOT NULL,
                    storage_backend TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    overall_score REAL,
                    latency_ms REAL,
                    total_cost REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    metrics_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def save_run_to_sqlite(self, record: RunRecord) -> None:
        """Insert or update one run in SQLite."""

        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO eval_runs (
                    run_id,
                    created_at,
                    app_type,
                    provider,
                    model,
                    framework,
                    observability_tool,
                    storage_backend,
                    dataset_name,
                    status,
                    overall_score,
                    latency_ms,
                    total_cost,
                    input_tokens,
                    output_tokens,
                    metrics_json,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.created_at,
                    record.app_type,
                    record.provider,
                    record.model,
                    record.framework,
                    record.observability_tool,
                    record.storage_backend,
                    record.dataset_name,
                    record.status,
                    record.overall_score,
                    record.latency_ms,
                    record.total_cost,
                    record.input_tokens,
                    record.output_tokens,
                    to_json_string(record.metrics),
                    to_json_string(record.metadata),
                ),
            )

    def read_runs_from_sqlite(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        """Read runs from SQLite and convert them into `RunRecord` objects."""

        query = "SELECT * FROM eval_runs"
        conditions: list[str] = []
        parameters: list[Any] = []

        if app_type:
            conditions.append("app_type = ?")
            parameters.append(app_type)

        if status:
            conditions.append("status = ?")
            parameters.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)

        with sqlite3.connect(self.sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()

        records: list[RunRecord] = []
        for row in rows:
            records.append(
                RunRecord(
                    run_id=row["run_id"],
                    created_at=row["created_at"],
                    app_type=row["app_type"],
                    provider=row["provider"],
                    model=row["model"],
                    framework=row["framework"],
                    observability_tool=row["observability_tool"],
                    storage_backend=row["storage_backend"],
                    dataset_name=row["dataset_name"],
                    status=row["status"],
                    overall_score=row["overall_score"],
                    latency_ms=row["latency_ms"],
                    total_cost=row["total_cost"],
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    metrics=optional_dict(row["metrics_json"]),
                    metadata=optional_dict(row["metadata_json"]),
                )
            )

        return records

    def save_run_to_json(self, record: RunRecord) -> None:
        """Save all runs in a single JSON file.

        This option is nice for beginners because the stored data is easy to
        inspect directly in the editor.
        """

        runs = self.load_json_rows()
        runs = [run for run in runs if run["run_id"] != record.run_id]
        runs.append(record.to_dict())
        runs.sort(key=lambda run: run["created_at"], reverse=True)
        self.json_path.write_text(to_json_string(runs), encoding="utf-8")

    def read_runs_from_json(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        """Read runs from the JSON backend."""

        runs = self.filter_run_rows(self.load_json_rows(), app_type=app_type, status=status)
        if limit is not None:
            runs = runs[:limit]
        return [RunRecord.from_dict(run) for run in runs]

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
        """Save runs in CSV format.

        Dictionaries such as `metrics` and `metadata` are stored as JSON strings
        because CSV only supports flat columns.
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
        """Read runs from the CSV backend."""

        rows = self.filter_run_rows(self.load_csv_rows(), app_type=app_type, status=status)
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
        """Flatten a record into one CSV row."""

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
        """Convert a CSV row back into the dictionary shape used by `RunRecord`."""

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
            "metrics": from_json_string(row.get("metrics_json")),
            "metadata": from_json_string(row.get("metadata_json")),
        }

    def prepare_duckdb(self) -> None:
        """Create the DuckDB table if it does not exist yet."""

        connection = open_duckdb_connection(self.duckdb_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    run_id VARCHAR PRIMARY KEY,
                    created_at VARCHAR,
                    app_type VARCHAR,
                    provider VARCHAR,
                    model VARCHAR,
                    framework VARCHAR,
                    observability_tool VARCHAR,
                    storage_backend VARCHAR,
                    dataset_name VARCHAR,
                    status VARCHAR,
                    overall_score DOUBLE,
                    latency_ms DOUBLE,
                    total_cost DOUBLE,
                    input_tokens BIGINT,
                    output_tokens BIGINT,
                    metrics_json VARCHAR,
                    metadata_json VARCHAR
                )
                """
            )
        finally:
            connection.close()

    def save_run_to_duckdb(self, record: RunRecord) -> None:
        """Insert or update one run in DuckDB."""

        connection = open_duckdb_connection(self.duckdb_path)
        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO eval_runs VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.run_id,
                    record.created_at,
                    record.app_type,
                    record.provider,
                    record.model,
                    record.framework,
                    record.observability_tool,
                    record.storage_backend,
                    record.dataset_name,
                    record.status,
                    record.overall_score,
                    record.latency_ms,
                    record.total_cost,
                    record.input_tokens,
                    record.output_tokens,
                    to_json_string(record.metrics),
                    to_json_string(record.metadata),
                ],
            )
        finally:
            connection.close()

    def read_runs_from_duckdb(
        self,
        limit: int | None = None,
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[RunRecord]:
        """Read runs from DuckDB."""

        query = "SELECT * FROM eval_runs"
        conditions: list[str] = []
        parameters: list[Any] = []

        if app_type:
            conditions.append("app_type = ?")
            parameters.append(app_type)

        if status:
            conditions.append("status = ?")
            parameters.append(status)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)

        connection = open_duckdb_connection(self.duckdb_path)
        try:
            rows = connection.execute(query, parameters).fetchall()
        finally:
            connection.close()

        records: list[RunRecord] = []
        for row in rows:
            records.append(
                RunRecord(
                    run_id=row[0],
                    created_at=row[1],
                    app_type=row[2],
                    provider=row[3],
                    model=row[4],
                    framework=row[5],
                    observability_tool=row[6],
                    storage_backend=row[7],
                    dataset_name=row[8],
                    status=row[9],
                    overall_score=row[10],
                    latency_ms=row[11],
                    total_cost=row[12],
                    input_tokens=row[13],
                    output_tokens=row[14],
                    metrics=optional_dict(row[15]),
                    metadata=optional_dict(row[16]),
                )
            )

        return records

    def filter_run_rows(
        self,
        rows: list[dict[str, Any]],
        app_type: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Filter rows in memory.

        JSON and CSV do not support SQL queries directly, so we filter them in
        Python after reading the file.
        """

        filtered_rows = rows

        if app_type:
            filtered_rows = [row for row in filtered_rows if row.get("app_type") == app_type]

        if status:
            filtered_rows = [row for row in filtered_rows if row.get("status") == status]

        filtered_rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return filtered_rows


def calculate_average(values: list[float]) -> float | None:
    """Return the average of a list, or `None` if the list is empty."""

    if not values:
        return None
    return sum(values) / len(values)


def open_duckdb_connection(database_path: Path) -> Any:
    """Open a DuckDB connection.

    DuckDB is optional in this project, so we raise a clear message if the
    package has not been installed.
    """

    try:
        import duckdb  # type: ignore
    except ImportError as error:
        raise ImportError(
            "DuckDB support requires the 'duckdb' package. "
            "Install it or choose sqlite, json, or csv."
        ) from error

    return duckdb.connect(str(database_path))
