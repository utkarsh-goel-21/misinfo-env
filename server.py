"""
FastAPI server for the Misinformation Containment Environment.

Exposes the full OpenEnv HTTP interface:
  POST /reset        — start a new episode
  POST /step         — execute one action
  GET  /state        — get full internal state (ground truth)
  GET  /health       — health check for pre-validation
  GET  /tasks        — list all available tasks

Run locally:
  uvicorn server:app --host 0.0.0.0 --port 7860

Docker / HuggingFace Spaces:
  CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from environment.env import MisinfoEnv
from environment.models import (
    Action,
    ActionType,
    Observation,
    Reward,
    EnvironmentState,
)


# ─────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_detection"
    seed: int = 42


class StepRequest(BaseModel):
    action_type: str
    target_node_id: str
    reasoning: Optional[str] = None


class ResetResponse(BaseModel):
    observation: Observation
    task_id: str
    message: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class HealthResponse(BaseModel):
    status: str
    version: str
    env_name: str
    tasks: list[str]


# ─────────────────────────────────────────
# APP STATE
# Active environment instance (one per server)
# ─────────────────────────────────────────

_env: Optional[MisinfoEnv] = None


def get_env() -> MisinfoEnv:
    """Return current env instance or raise 400."""
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No active episode. "
                "Call POST /reset first."
            )
        )
    return _env


# ─────────────────────────────────────────
# LIFESPAN — warm up on startup
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm environment on startup."""
    global _env
    _env = MisinfoEnv(task_id="task1_detection", seed=42)
    _env.reset()
    print("[server] Environment warmed up. Ready.")
    yield
    if _env:
        _env.close()
    print("[server] Environment closed.")


# ─────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────

app = FastAPI(
    title="Misinformation Containment Environment",
    description=(
        "OpenEnv-compatible HTTP API for a social network "
        "misinformation containment simulation. Three tasks: "
        "detection (easy), tracing (medium), containment (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """
    Health check endpoint.
    Pre-validation script pings this to confirm server is up.
    """
    return HealthResponse(
        status="ok",
        version=MisinfoEnv.VERSION,
        env_name=MisinfoEnv.ENV_NAME,
        tasks=[
            "task1_detection",
            "task2_tracing",
            "task3_containment",
        ]
    )


@app.get("/tasks", tags=["environment"])
async def list_tasks():
    """
    List all available tasks with descriptions.
    """
    dummy = MisinfoEnv(task_id="task1_detection", seed=42)
    return {"tasks": dummy.list_tasks()}


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
async def reset(body: ResetRequest):
    """
    Reset environment and start a new episode.

    Args:
        task_id: One of task1_detection, task2_tracing, task3_containment
        seed: Random seed (default 42 for reproducibility)

    Returns:
        Initial observation and task info.
    """
    global _env

    valid_tasks = [
        "task1_detection",
        "task2_tracing",
        "task3_containment",
    ]
    if body.task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid task_id '{body.task_id}'. "
                f"Must be one of: {valid_tasks}"
            )
        )

    _env = MisinfoEnv(task_id=body.task_id, seed=body.seed)
    obs = _env.reset()

    return ResetResponse(
        observation=obs,
        task_id=body.task_id,
        message=(
            f"Episode started. Task: {body.task_id}. "
            f"Seed: {body.seed}. "
            f"Max steps: {obs.max_steps}."
        )
    )


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(body: StepRequest):
    """
    Execute one action in the environment.

    Args:
        action_type: One of inspect, trace, flag, quarantine, remove, restore
        target_node_id: Node ID to act upon (e.g. "node_0")
        reasoning: Optional agent explanation

    Returns:
        Updated observation, reward, done flag, info dict.
    """
    env = get_env()

    # Validate action_type
    valid_action_types = [a.value for a in ActionType]
    if body.action_type not in valid_action_types:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid action_type '{body.action_type}'. "
                f"Must be one of: {valid_action_types}"
            )
        )

    action = Action(
        action_type=ActionType(body.action_type),
        target_node_id=body.target_node_id,
        reasoning=body.reasoning,
    )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=EnvironmentState, tags=["environment"])
async def state():
    """
    Return full internal environment state.
    Includes ground truth hidden from agent (origin node, etc).
    Used by graders and evaluators.
    """
    env = get_env()
    return env.state()


# ─────────────────────────────────────────
# LOCAL ENTRYPOINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
