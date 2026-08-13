"""
main.py
--------
FastAPI backend for a "DAX / Power BI Model" agent backend.

Flow:
  1. POST /models/upload   -> upload a .pbix file, it gets extracted and cached
  2. GET  /models          -> list uploaded models (model_id, file name, table count)
  3. GET  /models/{id}/... -> query tables, schema, measures, relationships, sample data
  4. GET  /models/{id}/full -> everything combined (JSON or Markdown) - best for LLM grounding
  5. DELETE /models/{id}   -> remove a model from memory

Auth: simple API key via header `x-api-key`, checked against API_KEY env var.
Swap for Azure AD / OAuth2 in production if calling from Copilot Studio in an
enterprise tenant (Copilot Studio custom connectors support both).

Run locally:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive OpenAPI UI, and
http://localhost:8000/openapi.json for the spec to import into Copilot Studio.
"""

import os
import shutil
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from extractor import PBIXExtractor

API_KEY = os.environ.get("PBIX_API_KEY", "dev-local-key")  # override in production
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploaded_models")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Power BI Model Extraction API",
    description=(
        "Extracts and exposes complete Power BI (.pbix) semantic model details — "
        "tables, columns, data types, relationships, DAX measures, calculated columns, "
        "Power Query M code, sample data, and statistics. Designed to be called by an "
        "AI agent (e.g., Microsoft Copilot Studio) so it can answer DAX/model questions "
        "grounded in the real model instead of guessing table or column names."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache: model_id -> PBIXExtractor
# NOTE: swap for Redis / DB-backed storage if you need persistence across restarts
# or multiple API instances (e.g., Azure App Service with >1 worker).
MODEL_STORE: Dict[str, Dict] = {}


def _check_api_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key (x-api-key header).")


def _get_extractor(model_id: str) -> PBIXExtractor:
    entry = MODEL_STORE.get(model_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found. Upload it first via /models/upload.")
    return entry["extractor"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/", summary="Root - basic service info")
def root():
    return {
        "service": "Power BI Model Extraction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "models_loaded": len(MODEL_STORE)}


@app.get("/console", response_class=HTMLResponse, summary="Serve the test console UI")
def console():
    """
    Serves console.html from the same origin as the API itself, so fetch()
    calls from the page are same-origin and never hit CORS restrictions.
    Opening console.html directly via file:// triggers browsers to block
    outgoing fetch requests entirely (silent failure, nothing reaches the
    server) — visiting this route instead sidesteps that completely.
    """
    console_path = os.path.join(os.path.dirname(__file__), "console.html")
    if not os.path.exists(console_path):
        raise HTTPException(status_code=404, detail="console.html not found next to main.py on the server.")
    with open(console_path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Upload / manage models
# ---------------------------------------------------------------------------

@app.post(
    "/models/upload",
    summary="Upload a .pbix file and extract its full model",
    description="Uploads a Power BI Desktop file, extracts all tables, columns, "
                "relationships, measures, and sample data, and caches it in memory "
                "under a model_id for subsequent queries.",
)
async def upload_model(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)

    if not file.filename.lower().endswith(".pbix"):
        raise HTTPException(status_code=400, detail="Only .pbix files are supported.")

    model_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{model_id}.pbix")

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        extractor = PBIXExtractor(dest_path)
        tables = extractor.get_tables()
    except Exception as e:
        os.remove(dest_path)
        raise HTTPException(status_code=422, detail=f"Failed to parse .pbix file: {e}")

    MODEL_STORE[model_id] = {
        "extractor": extractor,
        "file_name": file.filename,
        "path": dest_path,
    }

    return {
        "model_id": model_id,
        "file_name": file.filename,
        "table_count": len(tables),
        "tables": tables,
    }


@app.get("/models", summary="List all uploaded models currently cached")
def list_models(x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return [
        {
            "model_id": mid,
            "file_name": entry["file_name"],
            "table_count": len(entry["extractor"].get_tables()),
        }
        for mid, entry in MODEL_STORE.items()
    ]


@app.delete("/models/{model_id}", summary="Remove a model from memory/disk")
def delete_model(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    entry = MODEL_STORE.pop(model_id, None)
    if not entry:
        raise HTTPException(status_code=404, detail="Model not found.")
    if os.path.exists(entry["path"]):
        os.remove(entry["path"])
    return {"deleted": model_id}


# ---------------------------------------------------------------------------
# Query endpoints — the ones the agent will actually call most often
# ---------------------------------------------------------------------------

@app.get("/models/{model_id}/tables", summary="Get all table names in the model")
def tables(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"tables": _get_extractor(model_id).get_tables()}


@app.get("/models/{model_id}/schema", summary="Get full schema (all tables + columns + data types)")
def schema(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"schema": _get_extractor(model_id).get_schema_grouped()}


@app.get("/models/{model_id}/schema/{table_name}", summary="Get columns + data types for one specific table")
def schema_for_table(model_id: str, table_name: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    grouped = _get_extractor(model_id).get_schema_grouped()
    if table_name not in grouped:
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found in model.")
    return {"table": table_name, "columns": grouped[table_name]}


@app.get("/models/{model_id}/relationships", summary="Get all relationships between tables")
def relationships(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"relationships": _get_extractor(model_id).get_relationships()}


@app.get("/models/{model_id}/measures", summary="Get all existing DAX measures (name + table + DAX expression)")
def measures(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"measures": _get_extractor(model_id).get_measures()}


@app.get("/models/{model_id}/calculated-columns", summary="Get all calculated columns and their DAX expressions")
def calculated_columns(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"calculated_columns": _get_extractor(model_id).get_calculated_columns()}


@app.get("/models/{model_id}/power-query", summary="Get Power Query (M code) source definitions per table")
def power_query(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"power_query": _get_extractor(model_id).get_power_query()}


@app.get("/models/{model_id}/statistics", summary="Get row counts / size statistics per table and column")
def statistics(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"statistics": _get_extractor(model_id).get_statistics()}


@app.get("/models/{model_id}/rls", summary="Get Row-Level Security roles and filter expressions, if any")
def rls(model_id: str, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return {"row_level_security": _get_extractor(model_id).get_roles_rls()}


@app.get("/models/{model_id}/sample/{table_name}", summary="Get sample data rows from a specific table")
def sample_data(
    model_id: str,
    table_name: str,
    rows: int = Query(default=10, ge=1, le=200),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    data = _get_extractor(model_id).get_sample_data(table_name, rows=rows)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for table '{table_name}'.")
    return {"table": table_name, "row_count_returned": len(data), "rows": data}


@app.get(
    "/models/{model_id}/full",
    summary="Get the ENTIRE model extract in one call (schema, relationships, measures, samples, etc.)",
    description="This is the primary endpoint an agent should call first when it needs full "
                "context about the model before answering a DAX or data modeling question. "
                "Use format=markdown for a compact, LLM-friendly rendering, or format=json "
                "for structured data.",
)
def full_extract(
    model_id: str,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
    sample_rows: int = Query(default=5, ge=0, le=50),
    x_api_key: Optional[str] = Header(default=None),
):
    _check_api_key(x_api_key)
    extractor = _get_extractor(model_id)
    if format == "markdown":
        return PlainTextResponse(extractor.full_extract_markdown(sample_rows=sample_rows))
    return extractor.full_extract(sample_rows=sample_rows)
