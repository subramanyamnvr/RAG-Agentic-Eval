# LLM and Agentic Evaluation Dashboard

This project is a beginner-friendly dashboard for learning how to evaluate:

- LLM applications
- RAG applications
- Agentic applications

It is designed to explain evaluation concepts in a simple way instead of trying
to hide everything behind a complex framework.

## What This Project Shows

- how to upload an evaluation dataset
- how to choose a provider, model, framework, and observability tool
- how to run LLM, RAG, or agentic evaluation
- how to calculate simple metrics for each case
- how to save evaluation runs locally
- how to inspect score trends over time

## Why The Project Is Simple

This version keeps the code intentionally small and readable.

- storage uses `json` or `csv`
- evaluation metrics use simple Python heuristics
- the dashboard focuses on explaining concepts clearly
- external SDK integrations are not required to run version 1

Later, these simple evaluators can be replaced or extended with tools like:

- RAGAS
- DeepEval
- TruLens
- LangSmith

## Project Structure

```text
.
├── core/
│   ├── evaluators.py
│   ├── integrations.py
│   ├── schemas.py
│   └── storage.py
├── dashboard/
│   └── app.py
├── sample_data/
│   ├── llm_eval.json
│   ├── rag_eval.json
│   └── agent_eval.json
├── PROJECT_BLUEPRINT.md
├── README.md
└── requirements.txt
```

## What Each File Does

### `core/schemas.py`

Defines the main data structures used in the project:

- dashboard selections
- uploaded LLM rows
- uploaded RAG rows
- uploaded agent rows
- agent steps
- saved run records

### `core/evaluators.py`

Contains the evaluation logic.

For LLM evaluation it calculates:

- correctness
- relevancy
- completeness

For RAG evaluation it calculates:

- faithfulness
- groundedness
- context precision
- context recall

For agent evaluation it calculates:

- task success
- tool selection accuracy
- tool precision
- step efficiency
- tool failure rate
- loop risk

### `core/storage.py`

Saves and loads evaluation runs using:

- JSON
- CSV

JSON is the default because it is easiest to inspect in an editor.

### `core/integrations.py`

Connects everything together:

1. reads the uploaded file
2. converts it into schema objects
3. calls the right evaluator
4. saves the final run

### `dashboard/app.py`

This is the Streamlit interface.

It lets the user:

- choose app type
- choose tools and services
- use sample data or upload a file
- run evaluation
- inspect scores and row details
- view saved history

## How The Evaluation Flow Works

The full flow is:

1. user selects `LLM`, `RAG`, or `Agentic`
2. user chooses provider, model, framework, observability tool, and storage
3. user uploads a JSON or CSV file, or picks sample data
4. the app parses the rows into Python dataclasses
5. the evaluator calculates row-level metrics
6. the app creates summary scores
7. the run is saved locally
8. the dashboard shows current results and previous history

## Sample Dataset Formats

### LLM

Each row can contain:

- `question`
- `expected_answer`
- `actual_answer`
- `model`
- `latency_ms`
- `input_tokens`
- `output_tokens`

### RAG

Each row can contain:

- `question`
- `retrieved_contexts`
- `expected_answer`
- `actual_answer`
- `model`
- `latency_ms`
- `input_tokens`
- `output_tokens`

### Agentic

Each row can contain:

- `task`
- `expected_result`
- `final_result`
- `tools_expected`
- `tools_used`
- `steps`
- `retry_count`
- `tool_errors`
- `total_latency_ms`
- `input_tokens`
- `output_tokens`

## Installation

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

## Important Note About The Scores

The metrics in this project are intentionally simple.

They are useful for:

- learning
- demos
- portfolio projects
- understanding evaluation workflows

They are not a replacement for production-grade evaluation frameworks.

## Good Next Steps

Once you understand this version, you can extend it by adding:

- RAGAS-based faithfulness scoring
- DeepEval-based benchmark tests
- Langfuse or Phoenix tracing
- real provider pricing logic
- GitHub Actions quality gates
- experiment comparison across model versions
