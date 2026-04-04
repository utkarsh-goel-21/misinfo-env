"""
FastAPI server for the Misinformation Containment Environment.
OpenEnv-compatible structure: server/app.py with main() entrypoint.

Exposes the full OpenEnv HTTP interface:
  POST /reset        — start a new episode
  POST /step         — execute one action
  GET  /state        — get full internal state (ground truth)
  GET  /health       — health check for pre-validation
  GET  /tasks        — list all available tasks

Entrypoints:
  uv run server                      (via pyproject.toml scripts)
  python -m server                   (via __main__.py)
  uvicorn server.app:app --port 7860 (direct)
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
# ─────────────────────────────────────────

_env: Optional[MisinfoEnv] = None


def get_env() -> MisinfoEnv:
    if _env is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    return _env


# ─────────────────────────────────────────
# LIFESPAN — warm up on startup
# ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        "misinformation containment simulation."
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
    return HealthResponse(
        status="ok",
        version=MisinfoEnv.VERSION,
        env_name=MisinfoEnv.ENV_NAME,
        tasks=["task1_detection", "task2_tracing", "task3_containment"],
    )


@app.get("/tasks", tags=["environment"])
async def list_tasks():
    dummy = MisinfoEnv(task_id="task1_detection", seed=42)
    return {"tasks": dummy.list_tasks()}


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
async def reset(body: ResetRequest):
    global _env
    valid_tasks = ["task1_detection", "task2_tracing", "task3_containment"]
    if body.task_id not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_id '{body.task_id}'. Must be one of: {valid_tasks}",
        )
    _env = MisinfoEnv(task_id=body.task_id, seed=body.seed)
    obs = _env.reset()
    return ResetResponse(
        observation=obs,
        task_id=body.task_id,
        message=(
            f"Episode started. Task: {body.task_id}. "
            f"Seed: {body.seed}. Max steps: {obs.max_steps}."
        ),
    )


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(body: StepRequest):
    env = get_env()
    valid_action_types = [a.value for a in ActionType]
    if body.action_type not in valid_action_types:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid action_type '{body.action_type}'. "
                f"Must be one of: {valid_action_types}"
            ),
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
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvironmentState, tags=["environment"])
async def state():
    env = get_env()
    return env.state()


# ─────────────────────────────────────────
# ENTRYPOINT — required by openenv validate
# ─────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = None):
    """
    Main entrypoint for uv run server / python -m server.
    Required by openenv validate: server/app.py must have main().
    """
    import uvicorn
    port = port or int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
