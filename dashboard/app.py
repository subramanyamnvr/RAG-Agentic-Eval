"""Streamlit dashboard for explaining and running RAG and agent evaluation.

This dashboard has two jobs:

1. teach the main evaluation ideas in a clear order
2. let the user run a small local evaluation on sample or uploaded data

The layout is intentionally simple and story-driven so it can support a blog
post or LinkedIn article, not only a developer demo.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from core.integrations import IntegrationService
from core.integrations import read_uploaded_file
from core.schemas import DashboardSelections
from core.storage import EvaluationStorage


UPLOAD_DIR = PROJECT_ROOT / ".storage" / "uploaded_files"
SAMPLE_FILE_MAP = {
    "LLM": PROJECT_ROOT / "sample_data" / "llm_eval.json",
    "RAG": PROJECT_ROOT / "sample_data" / "rag_eval.json",
    "Agentic": PROJECT_ROOT / "sample_data" / "agent_eval.json",
}
PAGE_VIEWS = [
    "All Sections",
    "Overview",
    "RAG Metrics",
    "Agent Metrics",
    "Model Signals",
    "Tools in 2026",
    "Interactive Lab",
]
EXPERIENCE_MODES = ["Beginner", "Detailed"]


def configure_page() -> None:
    """Configure the Streamlit page."""

    st.set_page_config(
        page_title="RAG and Agent Evaluation Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def apply_theme() -> None:
    """Add a custom theme so the dashboard feels more polished."""

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Fraunces:wght@600;700&display=swap');

        :root {
            --bg: #f6f1e8;
            --panel: #fffaf2;
            --panel-strong: #f2e6d6;
            --ink: #1f2933;
            --muted: #5a6573;
            --accent: #c96a3d;
            --accent-soft: #efd2c2;
            --line: #dbc8b1;
            --success: #2f7a52;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, #fff4df 0%, transparent 28%),
                linear-gradient(180deg, #f8f3ea 0%, #f4ede1 100%);
            color: var(--ink);
            font-family: "IBM Plex Sans", sans-serif;
        }

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div {
            color: var(--ink);
        }

        h1, h2, h3 {
            font-family: "Fraunces", serif;
            color: #21313c;
        }

        h4, h5, h6 {
            color: #263743;
        }

        a {
            color: #8f4524 !important;
        }

        strong, b {
            color: #1d2b34;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 250, 242, 0.95);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 12px;
        }

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #21313c !important;
        }

        div[data-testid="stMetricValue"] {
            color: #8f4524 !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #2f7a52 !important;
        }

        .hero-panel {
            background: linear-gradient(135deg, #fff8ef 0%, #f3e5d4 100%);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 26px;
            margin-bottom: 18px;
        }

        .hero-title {
            font-family: "Fraunces", serif;
            font-size: 2.4rem;
            line-height: 1.15;
            color: #22313b;
            margin-bottom: 0.6rem;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            max-width: 920px;
        }

        .note-strip {
            background: rgba(255, 255, 255, 0.78);
            border-left: 5px solid var(--accent);
            border-radius: 10px;
            padding: 12px 16px;
            margin-top: 14px;
            color: var(--ink);
        }

        .section-card {
            background: rgba(255, 250, 242, 0.95);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
            height: 100%;
        }

        .section-kicker {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .diagram-box {
            background: rgba(255, 250, 242, 0.96);
            color: var(--ink);
            border-radius: 18px;
            padding: 18px;
            font-family: "IBM Plex Sans", sans-serif;
            white-space: pre-wrap;
            line-height: 1.5;
            border: 1px solid var(--line);
        }

        .diagram-box,
        .diagram-box p,
        .diagram-box span,
        .diagram-box div,
        .diagram-box strong,
        .diagram-box code {
            color: var(--ink) !important;
        }

        .tool-card {
            background: rgba(255, 250, 242, 0.96);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 12px;
            color: var(--ink);
        }

        .filter-panel {
            background: rgba(255, 250, 242, 0.94);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 16px 16px 8px 16px;
            margin-bottom: 20px;
        }

        .stAlert {
            background: rgba(255, 250, 242, 0.95) !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }

        .stAlert * {
            color: var(--ink) !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: #fffaf2 !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="input"] span,
        div[data-baseweb="textarea"] span {
            color: var(--ink) !important;
        }

        .stSelectbox label,
        .stMultiSelect label,
        .stRadio label,
        .stSlider label,
        .stFileUploader label,
        .stTextInput label,
        .stTextArea label {
            color: #233641 !important;
            font-weight: 600 !important;
        }

        div[role="listbox"] *,
        ul[role="listbox"] * {
            color: var(--ink) !important;
            background: #fffaf2 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #f3e5d4;
            border-radius: 10px;
            color: var(--ink);
            border: 1px solid var(--line);
        }

        .stTabs [aria-selected="true"] {
            background: #fffaf2 !important;
            color: #8f4524 !important;
            border-color: #d6b594 !important;
        }

        .stExpander {
            background: rgba(255, 250, 242, 0.96);
            border: 1px solid var(--line);
            border-radius: 14px;
        }

        .stExpander summary,
        .stExpander details,
        .stExpander label,
        .stExpander p,
        .stExpander div {
            color: var(--ink) !important;
        }

        .stCodeBlock,
        pre,
        code {
            color: #21313c !important;
        }

        pre {
            background: #f7efe3 !important;
            border: 1px solid var(--line) !important;
        }

        .stDataFrame, .stTable {
            background: rgba(255, 250, 242, 0.96) !important;
            border-radius: 12px;
        }

        [data-testid="stDataFrame"] * {
            color: var(--ink) !important;
        }

        .stButton button,
        .stDownloadButton button {
            background: #c96a3d !important;
            color: #fffaf2 !important;
            border: 1px solid #b75c31 !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
        }

        .stButton button:hover,
        .stDownloadButton button:hover {
            background: #b95d31 !important;
            color: #fffaf2 !important;
        }

        .stCaption, [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_upload_directory() -> None:
    """Create the folder used for temporary uploaded files."""

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file: Any) -> Path:
    """Save the uploaded file locally and return its path."""

    ensure_upload_directory()
    saved_path = UPLOAD_DIR / uploaded_file.name
    saved_path.write_bytes(uploaded_file.getvalue())
    return saved_path


def rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    """Convert dataset rows into CSV text.

    JSON is more natural for nested values, but CSV is handy for quick editing
    in spreadsheet tools. Lists and dictionaries are stored as JSON strings.
    """

    if not rows:
        return ""

    normalized_rows: list[dict[str, Any]] = []
    all_columns: list[str] = []

    for row in rows:
        normalized_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                normalized_row[key] = json.dumps(value)
            else:
                normalized_row[key] = value
            if key not in all_columns:
                all_columns.append(key)
        normalized_rows.append(normalized_row)

    dataframe = pd.DataFrame(normalized_rows, columns=all_columns)
    return dataframe.to_csv(index=False)


def load_sample_rows(application_type: str) -> list[dict[str, Any]]:
    """Load the sample rows for the selected application type."""

    sample_path = SAMPLE_FILE_MAP[application_type]
    return read_uploaded_file(sample_path)


def make_preview_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a friendly preview table from raw rows."""

    preview_rows: list[dict[str, Any]] = []

    for row in rows:
        preview_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                preview_row[key] = json.dumps(value)
            else:
                preview_row[key] = value
        preview_rows.append(preview_row)

    return pd.DataFrame(preview_rows)


def flatten_row_results(row_results: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten row results so they are easy to show in a table."""

    flattened_rows: list[dict[str, Any]] = []

    for index, row in enumerate(row_results, start=1):
        flat_row: dict[str, Any] = {"row_number": index}
        for key, value in row.items():
            if key == "metrics" and isinstance(value, dict):
                for metric_name, metric_value in value.items():
                    flat_row[metric_name] = metric_value
            elif isinstance(value, (list, dict)):
                flat_row[key] = json.dumps(value)
            else:
                flat_row[key] = value
        flattened_rows.append(flat_row)

    return pd.DataFrame(flattened_rows)


def build_metric_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """Turn summary metrics into a small dataframe for charts."""

    rows: list[dict[str, Any]] = []
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            rows.append({"metric": metric_name, "score": float(metric_value)})
    return pd.DataFrame(rows)


def load_history_dataframe(storage_backend: str, application_type: str) -> pd.DataFrame:
    """Load saved run history for the selected backend and app type."""

    storage = EvaluationStorage(storage_backend)
    runs = storage.list_runs(limit=25, app_type=application_type)
    if not runs:
        return pd.DataFrame()

    history_rows: list[dict[str, Any]] = []
    for run in runs:
        history_rows.append(
            {
                "created_at": run["created_at"],
                "app_type": run["app_type"],
                "provider": run["provider"],
                "model": run["model"],
                "overall_score": run["overall_score"],
                "latency_ms": run["latency_ms"],
                "total_cost": run["total_cost"],
                "status": run["status"],
                "dataset_name": run["dataset_name"],
            }
        )

    history_df = pd.DataFrame(history_rows)
    history_df["created_at"] = pd.to_datetime(history_df["created_at"])
    history_df = history_df.sort_values("created_at")
    return history_df


def render_hero_section() -> None:
    """Render the introductory story section."""

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">LLM and Agentic Evaluation Dashboard</div>
            <div class="hero-subtitle">
                Use this dashboard to understand how evaluation works for plain LLM applications,
                RAG systems, and agentic workflows. The goal is to make the main ideas visible,
                practical, and easy to test with local sample data.
            </div>
            <div class="note-strip">
                This dashboard combines explanation, sample datasets, and interactive evaluation
                so readers can learn the concepts and then run them on their own data.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown(
            """
            <div class="diagram-box">
Demo version:
User -> Retriever -> Context -> Model -> Good answer

Production version:
User -> Retriever -> Context -> Planner -> Tools -> Model -> Final answer
                   |                |        |
                   |                |        -> wrong tool
                   |                -> loops or retries
                   -> missing or noisy context
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-kicker">Why Evaluation Matters</div>
                <strong>Good demos are not enough for production confidence.</strong>
                <p style="margin-top:10px;">
                Many issues appear only after you look at retrieval quality, answer grounding,
                tool selection, retry behavior, and trace-level workflow outcomes.
                </p>
                <p>
                That is why this dashboard separates <strong>model</strong>, <strong>RAG</strong>,
                and <strong>agent</strong> evaluation instead of reducing everything to one score.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_top_filters(options: dict[str, Any]) -> tuple[DashboardSelections, str, str]:
    """Render the top filter area used to shape the story and the evaluation run."""

    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    st.subheader("Choose What You Want To Focus On")

    application_types = options.get("application_types", ["LLM", "RAG", "Agentic"])
    evaluation_layers = options.get("evaluation_layers", ["Model", "RAG", "Agent"])
    primary_concerns = options.get(
        "primary_concerns",
        ["Answer Quality", "Retrieval Quality", "Tool Use", "Safety", "Latency and Cost"],
    )
    deployment_stages = options.get("deployment_stages", ["Demo", "Staging", "Production"])
    model_providers = options.get(
        "model_providers",
        ["OpenAI", "Anthropic", "Gemini", "Ollama", "OpenRouter", "Custom"],
    )
    frameworks = options.get(
        "frameworks",
        ["Custom", "LangChain", "LlamaIndex", "CrewAI", "AutoGen", "OpenAI Agents SDK"],
    )
    observability_tools = options.get(
        "observability_tools",
        ["None", "Langfuse", "Phoenix", "OpenLIT", "Helicone", "LangSmith"],
    )
    storage_backends = options.get("storage_backends", ["json", "csv"])

    col1, col2, col3, col4 = st.columns(4)
    experience_mode = col1.selectbox("Experience", EXPERIENCE_MODES)
    application_type = col2.selectbox("Application type", application_types, index=1)
    primary_concern = col3.selectbox("Primary concern", primary_concerns)
    page_view = col4.selectbox("Dashboard view", PAGE_VIEWS)

    if application_type == "LLM":
        default_layer_index = 0
    elif application_type == "RAG":
        default_layer_index = 1
    else:
        default_layer_index = 2

    evaluation_layer = evaluation_layers[default_layer_index]
    deployment_stage = "Production"
    provider = "OpenAI"
    framework = "Custom"
    observability_tool = "None"
    storage_backend = "json"
    evaluation_tools = ["RAGAS", "DeepEval"]

    models_by_provider = options.get("models_by_provider", {"Custom": ["custom-model"]})
    tool_catalog = options.get("tool_catalog", {"evaluation_tools": ["RAGAS", "DeepEval"]})
    model_options = models_by_provider.get(provider, ["custom-model"])
    model = model_options[0]

    with st.expander("Advanced settings", expanded=experience_mode == "Detailed"):
        col5, col6, col7 = st.columns(3)
        evaluation_layer = col5.selectbox(
            "Evaluation layer",
            evaluation_layers,
            index=default_layer_index,
        )
        deployment_stage = col6.selectbox("Deployment stage", deployment_stages, index=2)
        provider = col7.selectbox("Model provider", model_providers)

        model_options = models_by_provider.get(provider, ["custom-model"])

        col8, col9, col10 = st.columns(3)
        framework = col8.selectbox("Framework", frameworks)
        observability_tool = col9.selectbox("Observability tool", observability_tools)
        storage_backend = col10.selectbox("Storage backend", storage_backends)

        model = st.selectbox("Model", model_options)
        evaluation_tools = st.multiselect(
            "Tools and services highlighted in this run",
            tool_catalog.get("evaluation_tools", ["RAGAS", "DeepEval"]),
            default=tool_catalog.get("evaluation_tools", ["RAGAS", "DeepEval"])[:2],
        )
    st.markdown("</div>", unsafe_allow_html=True)

    selections = DashboardSelections(
        application_type=application_type,
        evaluation_layer=evaluation_layer,
        primary_concern=primary_concern,
        deployment_stage=deployment_stage,
        provider=provider,
        model=model,
        framework=framework,
        observability_tool=observability_tool,
        storage_backend=storage_backend,
        evaluation_tools=evaluation_tools,
    )
    return selections, page_view, experience_mode


def render_focus_summary(selections: DashboardSelections) -> None:
    """Show a short explanation of what the selected filters imply."""

    message = (
        f"You are looking at a {selections.application_type} workflow through the "
        f"{selections.evaluation_layer.lower()} evaluation lens, with emphasis on "
        f"{selections.primary_concern.lower()} during the {selections.deployment_stage.lower()} stage."
    )
    st.info(message)


def render_start_here(application_type: str) -> None:
    """Show a very simple beginner path."""

    st.subheader("Start Here")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Step 1</div>
            <strong>Pick one app type</strong>
            <p>Choose LLM, RAG, or Agentic based on the kind of system you want to understand.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Step 2</div>
            <strong>Run the sample dataset</strong>
            <p>Use the built-in sample first. It is easier to learn the metrics before using your own file.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        f"""
        <div class="section-card">
            <div class="section-kicker">Step 3</div>
            <strong>Read only the top three metrics</strong>
            <p>For {application_type}, focus on the first three recommended metrics before looking at anything else.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plain_english_intro(application_type: str) -> None:
    """Explain the selected evaluation type in very simple language."""

    explanations = {
        "LLM": {
            "what_it_is": "A plain LLM app gives an answer directly from the model.",
            "what_can_go_wrong": "The answer may be off-topic, incomplete, or factually wrong.",
            "what_to_measure": "So we mainly measure correctness, relevancy, and completeness.",
        },
        "RAG": {
            "what_it_is": "A RAG app first retrieves documents, then uses those documents to answer.",
            "what_can_go_wrong": "The system may retrieve the wrong context or answer with claims that are not supported by the context.",
            "what_to_measure": "So we mainly measure faithfulness, context precision, and context recall.",
        },
        "Agentic": {
            "what_it_is": "An agentic app can plan steps, use tools, and then produce a final result.",
            "what_can_go_wrong": "The agent may choose the wrong tool, repeat unnecessary steps, or fail to finish the task well.",
            "what_to_measure": "So we mainly measure task success, tool selection accuracy, and step efficiency.",
        },
    }

    content = explanations[application_type]
    st.subheader("In Plain English")
    st.write(f"**What it is:** {content['what_it_is']}")
    st.write(f"**What can go wrong:** {content['what_can_go_wrong']}")
    st.write(f"**What to measure first:** {content['what_to_measure']}")


def render_recommended_metrics(selections: DashboardSelections) -> None:
    """Show the first three metrics a reader should pay attention to."""

    recommendations = {
        "LLM": ["correctness", "relevancy", "completeness"],
        "RAG": ["faithfulness", "context_precision", "context_recall"],
        "Agentic": ["task_success", "tool_selection_accuracy", "step_efficiency"],
    }

    if selections.primary_concern == "Safety":
        recommended = ["faithfulness", "tool_failure_rate", "loop_risk"]
    elif selections.primary_concern == "Latency and Cost":
        recommended = ["overall_score", "average_latency_ms", "average_cost"]
    elif selections.primary_concern == "Tool Use":
        recommended = ["tool_selection_accuracy", "tool_precision", "step_efficiency"]
    elif selections.primary_concern == "Retrieval Quality":
        recommended = ["context_precision", "context_recall", "faithfulness"]
    else:
        recommended = recommendations[selections.application_type]

    st.subheader("The First Three Metrics I Would Put On The Dashboard")
    col1, col2, col3 = st.columns(3)
    for column, metric_name in zip([col1, col2, col3], recommended):
        column.markdown(
            f"""
            <div class="section-card">
                <div class="section-kicker">Priority Metric</div>
                <strong>{metric_name}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_reading_guide(application_type: str) -> None:
    """Explain how to read the graphs and how data changes the scores."""

    st.header("How To Read The Evaluation Graphs")
    st.write(
        "The charts in this dashboard are averages across the uploaded rows. That means the score does not describe one answer only. It describes the behavior of the whole dataset you tested."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-kicker">Bar Chart Meaning</div>
                <strong>Metric bars show average score from 0 to 1.</strong>
                <p>If a bar is close to 1, the dataset performed well on that metric overall.</p>
                <p>If a bar drops, at least some rows are weak on that dimension, even if other rows are strong.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
                <div class="section-kicker">History Chart Meaning</div>
                <strong>The history line shows how overall score changes across saved runs.</strong>
                <p>Use it to spot regressions after changing prompts, retrieval logic, tools, or models.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Why scores change when the data changes")
    if application_type == "LLM":
        st.write(
            "- If questions become more open-ended, correctness may fall even when relevancy stays decent.\n"
            "- If expected answers are short but actual answers are longer and still on-topic, completeness and correctness may move differently.\n"
            "- A tiny dataset makes scores swing more. One bad row can noticeably drop the average."
        )
    elif application_type == "RAG":
        st.write(
            "- If retrieved chunks are noisy, context precision drops even if the final answer still looks fluent.\n"
            "- If the right document never appears, context recall drops first and then faithfulness usually suffers.\n"
            "- If the answer sounds good but includes unsupported claims, groundedness and faithfulness drop even when relevancy looks high."
        )
    else:
        st.write(
            "- If the agent reaches the right answer but takes too many steps, task success may stay high while step efficiency falls.\n"
            "- If the wrong tool is chosen first and then corrected, tool selection accuracy drops even when the final result is acceptable.\n"
            "- If retries increase, loop risk and tool failure patterns become easier to see in the row-level trace."
        )

    st.info(
        "A good dashboard is useful because metrics can disagree. That disagreement is often the clue that tells you where the real problem lives."
    )


def render_factory_section() -> None:
    """Render the three levels of evaluation analogy."""

    st.header("Evaluation Layers")
    st.write(
        "It helps to evaluate the system in layers. Model metrics describe the base model, RAG metrics describe retrieval and grounding, and agent metrics describe multi-step workflow behavior."
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Model Level</div>
            <strong>The machine</strong>
            <p>Useful for model selection and background quality signals.</p>
            <p>Typical metrics: perplexity, MMLU, HellaSwag.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">RAG Level</div>
            <strong>The supply chain</strong>
            <p>Checks whether the right documents arrive and whether the answer uses them correctly.</p>
            <p>Typical metrics: faithfulness, context precision, context recall.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Agent Level</div>
            <strong>The assembly line</strong>
            <p>Checks whether planning, tools, retries, and policies lead to a safe successful outcome.</p>
            <p>Typical metrics: task success, tool selection accuracy, step efficiency.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rag_metrics_section() -> None:
    """Render the RAG metrics explanation section."""

    st.header("RAG Evaluation Concepts")
    st.write(
        "RAG evaluation usually combines answer-level quality checks with retrieval-level quality checks. A fluent answer can still be unreliable if the retrieval step was weak."
    )

    st.subheader("Answer-Level Metrics")
    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Courtroom Analogy</div>
            <strong>Answer relevancy</strong>
            <p>The answer should argue the same case as the question.</p>
            <p>High relevancy means the answer stays on topic and useful.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Courtroom Analogy</div>
            <strong>Faithfulness / groundedness</strong>
            <p>The answer should only cite facts that appear in evidence.</p>
            <p>High faithfulness means fewer hallucinated claims.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Ground Truth Check</div>
            <strong>Correctness / factuality</strong>
            <p>When labeled truth exists, compare the answer to that truth, not only to retrieved context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Retrieval-Level Metrics")
    col4, col5, col6, col7 = st.columns(4)
    col4.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Search and Rescue</div>
            <strong>Context precision</strong>
            <p>Of everything retrieved, how much was actually useful?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col5.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Search and Rescue</div>
            <strong>Context recall</strong>
            <p>Did the important document make it into the context window at all?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col6.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Quality Risk</div>
            <strong>Hallucination rate</strong>
            <p>How often does the answer contain claims that the context does not support?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col7.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Corpus Coverage</div>
            <strong>Coverage</strong>
            <p>Does the knowledge base even contain the information the user is asking for?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_metrics_section() -> None:
    """Render the agent metrics explanation section."""

    st.header("Agent Evaluation Concepts")
    st.write(
        "Agent evaluation looks at both the final outcome and the path the system followed to get there."
    )

    st.markdown(
        """
        <div class="diagram-box">
Task -> Planner -> Tool choice -> Tool arguments -> Retry / recovery -> Final answer
         |            |               |                 |
         |            |               |                 -> did it recover or stall?
         |            |               -> were the IDs, dates, and filters valid?
         |            -> was the right tool chosen?
         -> was the route sensible?
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Outcome</div>
            <strong>Task success / completion rate</strong>
            <p>Did the agent actually finish the task it was supposed to complete?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Tool Routing</div>
            <strong>Tool selection accuracy</strong>
            <p>Did the agent choose the correct API or tool instead of the wrong one?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Arguments</div>
            <strong>Argument correctness</strong>
            <p>Were the parameters valid and meaningful, such as the right ID, date, or filter?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col4, col5, col6 = st.columns(3)
    col4.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Efficiency</div>
            <strong>Step efficiency</strong>
            <p>Did the agent take a sensible route, or keep looping and calling tools without a reason?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col5.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Recovery</div>
            <strong>Error recovery rate</strong>
            <p>When tools fail, can the agent recover gracefully instead of stalling or hallucinating?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col6.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Safety</div>
            <strong>Safety and robustness</strong>
            <p>Can prompts or tool outputs push the agent into policy violations or unsafe behavior?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_metrics_section() -> None:
    """Render the model-centric metrics section."""

    st.header("Model-Level Signals")
    st.write(
        "Model-level metrics still matter for model selection, but they do not replace system-level evaluation."
    )

    col1, col2 = st.columns(2)
    col1.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Language Modeling</div>
            <strong>Perplexity</strong>
            <p>Perplexity measures how surprised the model is by real text. Lower perplexity usually means the model is better at predicting natural language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Benchmarks</div>
            <strong>MMLU, HellaSwag, and similar tests</strong>
            <p>These help compare base models before you plug them into RAG or agents, but they do not tell you whether your retrieval and workflow stack is reliable.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tools_section() -> None:
    """Render the 2026 tooling section."""

    st.header("Tooling Landscape")
    st.write(
        "Some tools are code-first and test-focused, while others are dashboard-first and trace-focused."
    )

    tool_cards = [
        {
            "name": "DeepEval",
            "type": "Code-first metrics",
            "description": "A testing framework for RAG, chat, and agents with Pytest-style evaluation and CI-friendly thresholds.",
            "link": "https://github.com/confident-ai/deepeval",
        },
        {
            "name": "RAGAS",
            "type": "RAG-specific evaluation",
            "description": "Focused on faithfulness, answer relevancy, context precision, context recall, and hallucination-aware scoring.",
            "link": "https://github.com/vibrantlabsai/ragas",
        },
        {
            "name": "Confident AI",
            "type": "Dashboard-first platform",
            "description": "Hosted UI for comparing prompts, models, and evaluation runs over time.",
            "link": "https://www.confident-ai.com",
        },
        {
            "name": "RAG observability platforms",
            "type": "Production monitoring",
            "description": "Platforms like Deepchecks, Phoenix, and others help teams log traces and watch live RAG quality signals.",
            "link": "",
        },
        {
            "name": "Agent frameworks",
            "type": "Orchestration layer",
            "description": "LangChain, LangGraph, CrewAI, AutoGen, and related frameworks orchestrate tool-using agents, but they still need evaluation on top.",
            "link": "",
        },
    ]

    for tool in tool_cards:
        link_text = (
            f'<a href="{tool["link"]}" target="_blank">Open link</a>'
            if tool["link"]
            else "No direct link included in this dashboard."
        )
        st.markdown(
            f"""
            <div class="tool-card">
                <div class="section-kicker">{tool['type']}</div>
                <strong>{tool['name']}</strong>
                <p style="margin-top:8px;">{tool['description']}</p>
                <p>{link_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_answer_generation_guide(selections: DashboardSelections) -> None:
    """Explain how different systems generate answers in real life.

    This project does not require API keys because version 1 evaluates answers
    already present in the dataset. The explanation below helps readers connect
    the offline dataset to a real system architecture.
    """

    st.header("How Answers Are Represented In This Demo")
    st.write(
        "This demo does not call live LLM APIs. Instead, it evaluates answers that are already stored in your dataset. That keeps the project simple and removes the need for API keys."
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f"""
        <div class="section-card">
            <div class="section-kicker">Plain LLM Flow</div>
            <strong>User question -> model -> answer</strong>
            <p>The answer is produced directly by a model such as <code>{selections.model}</code>.</p>
            <p>In this dashboard, that answer is represented by the <code>actual_answer</code> field in the dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col2.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">RAG Flow</div>
            <strong>User question -> retriever -> context -> model -> answer</strong>
            <p>The system first fetches relevant chunks, then gives both the question and context to the model.</p>
            <p>In this dashboard, the retrieved chunks are stored in <code>retrieved_contexts</code>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col3.markdown(
        """
        <div class="section-card">
            <div class="section-kicker">Agent Flow</div>
            <strong>Task -> planner -> tools -> model -> final result</strong>
            <p>The system may plan, call APIs, retry, and then produce a final result.</p>
            <p>In this dashboard, those actions are stored in the <code>steps</code> field.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="diagram-box">
Why the provider and model filters still matter in this demo:

- They show which stack the dataset came from or represents.
- They get saved with the evaluation run.
- They let you compare different runs later.

So even without API keys, you can still use this dashboard to reason about
model selection, prompt changes, retrieval quality, and tool behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_threshold_inputs(application_type: str) -> dict[str, float]:
    """Render threshold inputs and return the selected values."""

    st.subheader("Thresholds")
    st.caption("These thresholds decide whether a row is marked as pass or fail.")

    if application_type == "LLM":
        col1, col2 = st.columns(2)
        correctness = col1.slider("Correctness threshold", 0.0, 1.0, 0.7, 0.05)
        relevancy = col2.slider("Relevancy threshold", 0.0, 1.0, 0.7, 0.05)
        return {
            "correctness": correctness,
            "relevancy": relevancy,
        }

    if application_type == "RAG":
        col1, col2 = st.columns(2)
        faithfulness = col1.slider("Faithfulness threshold", 0.0, 1.0, 0.7, 0.05)
        relevancy = col2.slider("Relevancy threshold", 0.0, 1.0, 0.7, 0.05)
        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
        }

    col1, col2 = st.columns(2)
    task_success = col1.slider("Task success threshold", 0.0, 1.0, 0.7, 0.05)
    tool_accuracy = col2.slider("Tool selection threshold", 0.0, 1.0, 0.7, 0.05)
    return {
        "task_success": task_success,
        "tool_selection_accuracy": tool_accuracy,
    }


def render_dataset_section(
    application_type: str,
    dataset_column_guide: dict[str, list[str]],
) -> tuple[Path | None, str | None]:
    """Render dataset controls and return the chosen file path."""

    st.subheader("Choose a Dataset")
    st.caption("Use a sample file to learn fast, or upload your own JSON or CSV file.")

    st.markdown("**Expected columns for this dataset type**")
    fallback_columns = {
        "LLM": ["question", "expected_answer", "actual_answer"],
        "RAG": ["question", "retrieved_contexts", "expected_answer", "actual_answer"],
        "Agentic": ["task", "expected_result", "final_result", "tools_expected", "tools_used", "steps"],
    }
    columns_for_type = dataset_column_guide.get(application_type, fallback_columns[application_type])
    st.code(", ".join(columns_for_type))

    sample_rows = load_sample_rows(application_type)
    sample_json_bytes = json.dumps(sample_rows, indent=2).encode("utf-8")
    sample_csv_bytes = rows_to_csv_text(sample_rows).encode("utf-8")

    download_col1, download_col2 = st.columns(2)
    download_col1.download_button(
        "Download sample JSON",
        data=sample_json_bytes,
        file_name=f"{application_type.lower()}_sample.json",
        mime="application/json",
        use_container_width=True,
    )
    download_col2.download_button(
        "Download sample CSV",
        data=sample_csv_bytes,
        file_name=f"{application_type.lower()}_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "A simple workflow is: download a sample file, edit the answers or traces, then upload it back to see how the metrics change."
    )

    source_choice = st.radio(
        "Dataset source",
        ["Use sample dataset", "Upload file"],
        horizontal=True,
    )

    if source_choice == "Use sample dataset":
        sample_path = SAMPLE_FILE_MAP[application_type]
        st.info(f"Using sample file: `{sample_path.name}`")
        return sample_path, sample_path.name

    uploaded_file = st.file_uploader("Upload a JSON or CSV file", type=["json", "csv"])
    if uploaded_file is None:
        return None, None

    saved_path = save_uploaded_file(uploaded_file)
    st.success(f"Uploaded file saved as `{saved_path.name}`")
    return saved_path, uploaded_file.name


def render_dataset_preview(file_path: Path | None) -> None:
    """Show a preview of the selected file."""

    if file_path is None:
        st.warning("Choose a sample dataset or upload a file to preview it.")
        return

    rows = read_uploaded_file(file_path)
    preview_df = make_preview_dataframe(rows)

    st.subheader("Dataset Preview")
    st.dataframe(preview_df, use_container_width=True)


def render_summary_cards(summary: dict[str, Any]) -> None:
    """Render the top summary cards after evaluation."""

    total_rows = summary["total_rows"]
    pass_rate = 0.0
    if total_rows > 0:
        pass_rate = round((summary["passed_rows"] / total_rows) * 100, 2)

    metrics = summary["metrics"]
    overall_score = metrics.get("overall_score")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", total_rows)
    col2.metric("Pass rate", f"{pass_rate}%")
    col3.metric(
        "Overall score",
        round(overall_score, 4) if isinstance(overall_score, (int, float)) else "N/A",
    )
    col4.metric(
        "Avg latency (ms)",
        round(summary["average_latency_ms"], 2)
        if isinstance(summary["average_latency_ms"], (int, float))
        else "N/A",
    )
    col5.metric(
        "Avg cost ($)",
        round(summary["average_cost"], 6)
        if isinstance(summary["average_cost"], (int, float))
        else "N/A",
    )


def render_metric_section(summary: dict[str, Any], metric_explanations: dict[str, str]) -> None:
    """Render aggregate metrics and their explanations."""

    st.subheader("Evaluation Summary")
    render_summary_cards(summary)

    metrics_df = build_metric_dataframe(summary["metrics"])
    if not metrics_df.empty:
        st.bar_chart(metrics_df.set_index("metric"))

    with st.expander("Why these metrics exist"):
        for metric_name, explanation in metric_explanations.items():
            st.write(f"**{metric_name}**: {explanation}")


def render_row_results(row_results: list[dict[str, Any]], application_type: str) -> None:
    """Render row-level results and a row inspector."""

    st.subheader("Row-Level Results")
    row_df = flatten_row_results(row_results)
    st.dataframe(row_df, use_container_width=True)

    if not row_results:
        return

    inspector_options = [f"Row {index}" for index in range(1, len(row_results) + 1)]
    selected_label = st.selectbox("Inspect one row in detail", inspector_options)
    selected_index = inspector_options.index(selected_label)
    selected_row = row_results[selected_index]

    st.json(selected_row)

    if application_type == "Agentic" and "steps" in selected_row:
        steps = selected_row["steps"]
        if steps:
            st.markdown("**Agent steps**")
            st.dataframe(pd.DataFrame(steps), use_container_width=True)


def render_history_section(storage_backend: str, application_type: str) -> None:
    """Render saved run history from local storage."""

    st.subheader("Saved Run History")
    history_df = load_history_dataframe(storage_backend, application_type)

    if history_df.empty:
        st.info("No saved history yet for this backend and application type.")
        return

    st.line_chart(history_df.set_index("created_at")[["overall_score"]])
    st.dataframe(history_df, use_container_width=True)


def render_stack_explorer(options: dict[str, Any], selections: DashboardSelections) -> None:
    """Render a summary of the selected stack."""

    st.subheader("Selected Stack")
    st.write(f"- Application type: {selections.application_type}")
    st.write(f"- Evaluation layer: {selections.evaluation_layer}")
    st.write(f"- Primary concern: {selections.primary_concern}")
    st.write(f"- Deployment stage: {selections.deployment_stage}")
    st.write(f"- Provider: {selections.provider}")
    st.write(f"- Model: {selections.model}")
    st.write(f"- Framework: {selections.framework}")
    st.write(f"- Observability: {selections.observability_tool}")
    st.write(f"- Storage: {selections.storage_backend}")
    st.write(f"- Evaluation tools selected: {', '.join(selections.evaluation_tools) or 'None'}")

    with st.expander("See major tool categories"):
        catalog = options.get("tool_catalog", {})
        for category_name, tool_list in catalog.items():
            st.write(f"**{category_name.replace('_', ' ').title()}**")
            st.write(", ".join(tool_list))


def render_saved_run_details(saved_run: dict[str, Any], storage_health: dict[str, Any]) -> None:
    """Render details of the saved run record."""

    st.subheader("Saved Run Record")
    st.caption("This is the normalized run object written to local storage.")
    st.json(saved_run)
    st.caption(f"Storage status: {storage_health['status']} using `{storage_health['backend']}`")


def render_interactive_lab(options: dict[str, Any], selections: DashboardSelections) -> None:
    """Render the interactive evaluation lab."""

    st.header("Interactive Evaluation Lab")
    st.write(
        "This part turns the ideas above into something you can run. Start with the sample dataset, "
        "then replace it with your own JSON or CSV file."
    )

    with st.expander("Adjust thresholds", expanded=True):
        threshold_values = get_threshold_inputs(selections.application_type)

    selected_file_path, dataset_name = render_dataset_section(
        selections.application_type,
        options.get(
            "dataset_column_guide",
            {
                "LLM": ["question", "expected_answer", "actual_answer"],
                "RAG": ["question", "retrieved_contexts", "expected_answer", "actual_answer"],
                "Agentic": ["task", "expected_result", "final_result", "tools_expected", "tools_used", "steps"],
            },
        ),
    )
    render_dataset_preview(selected_file_path)

    if st.button("Run Evaluation", type="primary"):
        if selected_file_path is None:
            st.error("Please select a sample dataset or upload a file first.")
            return

        with st.spinner("Running evaluation..."):
            evaluation_service = IntegrationService(thresholds=threshold_values)
            result = evaluation_service.evaluate_file(
                selected_file_path,
                selections,
                dataset_name=dataset_name,
            )

        st.session_state["last_result"] = result
        st.success("Evaluation complete.")

    last_result = st.session_state.get("last_result")
    if not last_result:
        return

    evaluation_result = last_result["evaluation_result"]
    summary = evaluation_result["summary"]
    metric_explanations = evaluation_result["metric_explanations"]
    row_results = evaluation_result["row_results"]

    render_metric_section(summary, metric_explanations)
    render_row_results(row_results, selections.application_type)
    render_history_section(selections.storage_backend, selections.application_type)
    render_stack_explorer(options, selections)
    render_saved_run_details(last_result["saved_run"], last_result["storage_health"])


def should_render(page_view: str, section_name: str) -> bool:
    """Return whether a given section should be visible."""

    if page_view == "All Sections":
        return True
    return page_view == section_name


def render_deeper_explanations(page_view: str, selections: DashboardSelections) -> None:
    """Render advanced learning sections behind expanders for beginners."""

    with st.expander("Learn more: evaluation layers"):
        render_factory_section()

    with st.expander("Learn more: RAG concepts"):
        render_rag_metrics_section()

    with st.expander("Learn more: agent concepts"):
        render_agent_metrics_section()

    with st.expander("Learn more: model-level signals"):
        render_model_metrics_section()

    with st.expander("Learn more: tooling landscape"):
        render_tools_section()

    with st.expander("Learn more: how answers are represented in this demo"):
        render_answer_generation_guide(selections)


def main() -> None:
    """Run the Streamlit dashboard."""

    configure_page()
    apply_theme()

    service = IntegrationService()
    options = service.get_dashboard_options()

    render_hero_section()
    selections, page_view, experience_mode = render_top_filters(options)
    render_focus_summary(selections)
    render_start_here(selections.application_type)
    render_plain_english_intro(selections.application_type)
    render_recommended_metrics(selections)
    render_metric_reading_guide(selections.application_type)

    if experience_mode == "Beginner":
        render_interactive_lab(options, selections)
        render_deeper_explanations(page_view, selections)
        return

    if should_render(page_view, "Overview"):
        render_factory_section()
    if should_render(page_view, "RAG Metrics"):
        render_rag_metrics_section()
    if should_render(page_view, "Agent Metrics"):
        render_agent_metrics_section()
    if should_render(page_view, "Model Signals"):
        render_model_metrics_section()
    if should_render(page_view, "Tools in 2026"):
        render_tools_section()
    render_answer_generation_guide(selections)
    if should_render(page_view, "Interactive Lab"):
        render_interactive_lab(options, selections)


if __name__ == "__main__":
    main()
