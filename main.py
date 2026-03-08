import json
import logging
import os
import traceback
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import uvicorn

from app.agents.scheduler_graph import (
    run_langgraph_pipeline, run_langgraph_pipeline_until_executor, run_langgraph_revision,
)
from app.agents.langchain_pipeline import (
    run_langchain_pipeline, run_langchain_pipeline_until_executor, run_langchain_revision,
)
from app.agents.simple_pipeline import (
    run_simple_pipeline, run_simple_pipeline_until_executor, run_simple_revision,
)
from app.data.user_history import get_all_user_ids, get_user_profile

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("main")

# ── FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(title="Sensorless Cycling Coach Backend")

# ── Source file paths for the code viewer ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = {
    "langgraph": os.path.join(BASE_DIR, "app", "agents", "scheduler_graph.py"),
    "langchain": os.path.join(BASE_DIR, "app", "agents", "langchain_pipeline.py"),
    "none": os.path.join(BASE_DIR, "app", "agents", "simple_pipeline.py"),
}


class PlanRequest(BaseModel):
    user_id: str = "rider_001"
    query: str
    mode: str = "langgraph"  # "langgraph" | "langchain" | "none"
    manual_qc: bool = False


class ReviseRequest(BaseModel):
    user_id: str = "rider_001"
    query: str
    mode: str = "langgraph"
    pipeline_state: dict


# ── Streaming endpoint — routes to the chosen pipeline ──────────────────
@app.post("/plan/stream")
async def generate_plan_stream(request: PlanRequest):
    """
    Streaming endpoint that yields enriched Server-Sent Events as each
    pipeline node/step completes. Includes thinking logs, state snapshots,
    and telemetry. Supports three modes: langgraph, langchain, none.
    """

    def event_generator():
        start = time.time()

        try:
            # Select pipeline generator
            if request.manual_qc:
                # Manual QC mode — run until executor, then pause
                if request.mode == "langchain":
                    gen = run_langchain_pipeline_until_executor(request.user_id, request.query)
                elif request.mode == "none":
                    gen = run_simple_pipeline_until_executor(request.user_id, request.query)
                else:
                    gen = run_langgraph_pipeline_until_executor(request.user_id, request.query)
            else:
                if request.mode == "langchain":
                    gen = run_langchain_pipeline(request.user_id, request.query)
                elif request.mode == "none":
                    gen = run_simple_pipeline(request.user_id, request.query)
                else:
                    gen = run_langgraph_pipeline(request.user_id, request.query)

            for step_output in gen:
                elapsed = round(time.time() - start, 1)
                step_output["elapsed"] = elapsed
                yield f"data: {json.dumps(step_output)}\n\n"

        except Exception as e:
            logger.error("Stream error:\n%s", traceback.format_exc())
            yield f"data: {json.dumps({'node': 'error', 'status': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Revise endpoint (Manual QC) ────────────────────────────────────────
@app.post("/plan/revise")
async def revise_plan(request: ReviseRequest):
    """
    Re-runs Planner + Executor with revision context after a human REVISE
    decision in Manual QC mode.
    """
    def event_generator():
        start = time.time()
        try:
            if request.mode == "langchain":
                gen = run_langchain_revision(request.user_id, request.query, request.pipeline_state)
            elif request.mode == "none":
                gen = run_simple_revision(request.user_id, request.query, request.pipeline_state)
            else:
                gen = run_langgraph_revision(request.user_id, request.query, request.pipeline_state)

            for step_output in gen:
                elapsed = round(time.time() - start, 1)
                step_output["elapsed"] = elapsed
                yield f"data: {json.dumps(step_output)}\n\n"

        except Exception as e:
            logger.error("Revise stream error:\n%s", traceback.format_exc())
            yield f"data: {json.dumps({'node': 'error', 'status': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Source code endpoint ────────────────────────────────────────────────
@app.get("/source/{mode}")
def get_source_code(mode: str):
    """Return the raw Python source code for a given pipeline mode."""
    filepath = SOURCE_FILES.get(mode)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Unknown mode: {mode}")
    with open(filepath, "r") as f:
        return PlainTextResponse(f.read())


# ── Utility endpoints ───────────────────────────────────────────────────
@app.get("/users")
def list_users():
    user_ids = get_all_user_ids()
    return {
        uid: {
            "name": get_user_profile(uid)["name"],
            "level": get_user_profile(uid)["experience_level"],
            "ftp": get_user_profile(uid)["ftp_watts"],
            "location": get_user_profile(uid)["location"],
        }
        for uid in user_ids
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
