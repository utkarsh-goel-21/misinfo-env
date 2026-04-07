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

@app.get("/", tags=["system"])
async def root():
    """Premium landing page for the environment."""
    from fastapi.responses import HTMLResponse
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🛡️ Misinfo Containment | OpenEnv</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            :root {
                --primary: #3b82f6;
                --primary-glow: rgba(59, 130, 246, 0.5);
                --success: #10b981;
                --bg: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Inter', sans-serif; 
                background: radial-gradient(circle at top right, #1e293b, #0f172a 60%);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                overflow-x: hidden;
            }
            h1, h2, h3 { font-family: 'Outfit', sans-serif; }
            
            .container { 
                max-width: 900px; width: 90%; 
                margin: 40px auto; 
                animation: fadeIn 0.8s ease-out;
            }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            .hero-card {
                background: var(--card-bg);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 24px;
                padding: 60px 40px;
                text-align: center;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                position: relative;
            }
            .hero-card::before {
                content: ''; position: absolute; top: -1px; left: 50%; transform: translateX(-50%);
                width: 40%; height: 2px; background: linear-gradient(90deg, transparent, var(--primary), transparent);
            }

            .badge {
                display: inline-flex; align-items: center; gap: 8px;
                background: rgba(16, 185, 129, 0.1);
                color: var(--success);
                padding: 6px 16px; border-radius: 99px;
                font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 24px;
                border: 1px solid rgba(16, 185, 129, 0.2);
            }
            .dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; animation: pulse 2s infinite; }
            @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }

            h1 { font-size: 3rem; font-weight: 700; margin-bottom: 16px; letter-spacing: -1px; }
            .subtitle { color: var(--text-muted); font-size: 1.1rem; max-width: 600px; margin: 0 auto 40px; line-height: 1.6; }

            .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px; }
            .task-item {
                background: rgba(255, 255, 255, 0.03);
                padding: 24px; border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                transition: all 0.3s ease;
            }
            .task-item:hover { transform: translateY(-5px); border-color: var(--primary); background: rgba(59, 130, 246, 0.05); }
            .task-item i { color: var(--primary); margin-bottom: 12px; }
            .task-item h3 { font-size: 1rem; margin-bottom: 8px; }
            .task-item p { font-size: 0.8rem; color: var(--text-muted); }

            .actions { display: flex; gap: 16px; justify-content: center; }
            .btn {
                text-decoration: none; padding: 14px 28px; border-radius: 12px;
                font-weight: 600; transition: all 0.2s; display: flex; align-items: center; gap: 10px;
            }
            .btn-primary { 
                background: var(--primary); color: white; 
                box-shadow: 0 10px 15px -3px var(--primary-glow);
            }
            .btn-primary:hover { transform: scale(1.02); box-shadow: 0 20px 25px -5px var(--primary-glow); }
            
            .btn-secondary { 
                background: rgba(255, 255, 255, 0.05); color: white; 
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .btn-secondary:hover { background: rgba(255, 255, 255, 0.1); }

            footer { margin-top: 40px; color: var(--text-muted); font-size: 0.8rem; letter-spacing: 0.5px; }
            
            @media (max-width: 768px) {
                .grid { grid-template-columns: 1fr; }
                h1 { font-size: 2.2rem; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero-card">
                <div class="badge"><span class="dot"></span> Environment Ready</div>
                <h1>Misinfo Containment</h1>
                <p class="subtitle">A robust social network simulation for evaluating AI Agent reasoning in trust & safety environments.</p>
                
                <div class="grid">
                    <div class="task-item">
                        <i data-lucide="search"></i>
                        <h3>Detection</h3>
                        <p>Identifying infected users in frozen snapshots.</p>
                    </div>
                    <div class="task-item">
                        <i data-lucide="git-branch"></i>
                        <h3>Tracing</h3>
                        <p>Locating Patient Zero in active outbreaks.</p>
                    </div>
                    <div class="task-item">
                        <i data-lucide="shield-check"></i>
                        <h3>Containment</h3>
                        <p>Strategic isolation of high-impact nodes.</p>
                    </div>
                </div>

                <div class="actions">
                    <a href="/health" class="btn btn-primary">
                        <i data-lucide="activity"></i> Check Health
                    </a>
                    <a href="/tasks" class="btn btn-secondary">
                        <i data-lucide="list"></i> View API
                    </a>
                    <a href="/docs" class="btn btn-secondary">
                        <i data-lucide="book-open"></i> Docs
                    </a>
                </div>
            </div>
            <footer>POWERED BY OPENENV & LLAMA 3.3</footer>
        </div>
        <script>lucide.createIcons();</script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
