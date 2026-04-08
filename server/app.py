"""
SENTINEL-9 — FastAPI Server (OpenEnv Interface)

HTTP + WebSocket interface for the Misinformation Containment Environment.

Endpoints:
  GET  /          — Landing page
  GET  /health    — Health check (required by openenv validate)
  GET  /tasks     — List available tasks
  POST /reset     — Start a new episode
  POST /step      — Execute one action
  GET  /state     — Get full internal state (ground truth)
  WS   /ws        — WebSocket for low-latency persistent sessions
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment.env import MisinfoEnv
from environment.models import (
    Action, ActionType, Observation, Reward, EnvironmentState,
)


# ═══════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ═══════════════════════════════════════

class ResetRequest(BaseModel):
    task_id: str = "task1_detection"
    seed: int = 42


class StepRequest(BaseModel):
    action_type: str
    target_node_id: Optional[str] = None
    confidence: float = 0.5
    reasoning: Optional[str] = None
    causal_chain: Optional[list[dict[str, str]]] = None


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


# ═══════════════════════════════════════
# APP STATE
# ═══════════════════════════════════════

_env: Optional[MisinfoEnv] = None


def get_env() -> MisinfoEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    return _env


# ═══════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = MisinfoEnv(task_id="task1_detection", seed=42)
    _env.reset()
    print("[SENTINEL-9] Environment warmed up. Ready.")
    yield
    if _env:
        _env.close()
    print("[SENTINEL-9] Environment closed.")


# ═══════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════

app = FastAPI(
    title="SENTINEL-9 Misinformation Containment Environment",
    description="OpenEnv-compatible POMDP for adversarial misinformation containment.",
    version=MisinfoEnv.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_TASKS = ["task1_detection", "task2_tracing", "task3_containment"]


# ═══════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return HealthResponse(
        status="ok",
        version=MisinfoEnv.VERSION,
        env_name=MisinfoEnv.ENV_NAME,
        tasks=VALID_TASKS,
    )


@app.get("/tasks", tags=["environment"])
async def list_tasks():
    return {"tasks": VALID_TASKS}


@app.post("/reset", response_model=ResetResponse, tags=["environment"])
async def reset(body: Optional[ResetRequest] = None):
    global _env
    if body is None:
        body = ResetRequest()
    if body.task_id not in VALID_TASKS:
        raise HTTPException(status_code=422, detail=f"Invalid task_id. Must be one of: {VALID_TASKS}")
    _env = MisinfoEnv(task_id=body.task_id, seed=body.seed)
    obs = _env.reset()
    return ResetResponse(
        observation=obs,
        task_id=body.task_id,
        message=f"Episode started. Task: {body.task_id}. Seed: {body.seed}. Max steps: {obs.max_steps}.",
    )


@app.post("/step", response_model=StepResponse, tags=["environment"])
async def step(body: StepRequest):
    env = get_env()
    valid_types = [a.value for a in ActionType]
    if body.action_type not in valid_types:
        raise HTTPException(status_code=422, detail=f"Invalid action_type. Must be one of: {valid_types}")
    action = Action(
        action_type=ActionType(body.action_type),
        target_node_id=body.target_node_id,
        confidence=body.confidence,
        reasoning=body.reasoning,
        causal_chain=body.causal_chain,
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


# ═══════════════════════════════════════
# WEBSOCKET — Low-latency persistent sessions
# ═══════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    local_env: Optional[MisinfoEnv] = None

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            cmd = msg.get("command", "")

            if cmd == "reset":
                task_id = msg.get("task_id", "task1_detection")
                seed = msg.get("seed", 42)
                if task_id not in VALID_TASKS:
                    await ws.send_json({"error": f"Invalid task_id: {task_id}"})
                    continue
                local_env = MisinfoEnv(task_id=task_id, seed=seed)
                obs = local_env.reset()
                await ws.send_json({
                    "type": "reset",
                    "observation": obs.model_dump(),
                })

            elif cmd == "step":
                if not local_env:
                    await ws.send_json({"error": "No active episode. Send reset first."})
                    continue
                try:
                    action = Action(
                        action_type=ActionType(msg["action_type"]),
                        target_node_id=msg.get("target_node_id"),
                        confidence=msg.get("confidence", 0.5),
                        reasoning=msg.get("reasoning"),
                        causal_chain=msg.get("causal_chain"),
                    )
                    obs, reward, done, info = local_env.step(action)
                    await ws.send_json({
                        "type": "step",
                        "observation": obs.model_dump(),
                        "reward": reward.model_dump(),
                        "done": done,
                        "info": info,
                    })
                except Exception as e:
                    await ws.send_json({"error": str(e)})

            elif cmd == "state":
                if not local_env:
                    await ws.send_json({"error": "No active episode."})
                    continue
                state = local_env.state()
                await ws.send_json({
                    "type": "state",
                    "state": state.model_dump(),
                })

            else:
                await ws.send_json({"error": f"Unknown command: {cmd}"})

    except WebSocketDisconnect:
        if local_env:
            local_env.close()
    except Exception as e:
        await ws.send_json({"error": str(e)})


# ═══════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════

@app.get("/", tags=["system"])
async def root():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SENTINEL-9 | Misinformation Containment</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg:#020617; --card:rgba(15,23,42,.75); --border:rgba(255,255,255,.08); --blue:#3b82f6; --green:#10b981; --red:#ef4444; --text:#f8fafc; --muted:#94a3b8; }
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;align-items:center;justify-content:center;overflow:hidden}
        body::before{content:'';position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.015) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.015) 1px,transparent 1px);background-size:60px 60px;pointer-events:none}
        .container{max-width:880px;width:90%;animation:fadeIn .8s ease-out}
        @keyframes fadeIn{from{opacity:0;transform:translateY(15px)}to{opacity:1;transform:translateY(0)}}
        .hero{background:var(--card);backdrop-filter:blur(16px);border:1px solid var(--border);border-radius:24px;padding:56px 44px;text-align:center;box-shadow:0 30px 60px rgba(0,0,0,.5);position:relative}
        .hero::before{content:'';position:absolute;top:-1px;left:50%;transform:translateX(-50%);width:35%;height:2px;background:linear-gradient(90deg,transparent,var(--blue),transparent)}
        .badge{display:inline-flex;align-items:center;gap:8px;background:rgba(16,185,129,.1);color:var(--green);padding:6px 16px;border-radius:99px;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:28px;border:1px solid rgba(16,185,129,.2)}
        .dot{width:8px;height:8px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
        @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(16,185,129,.7)}70%{box-shadow:0 0 0 12px rgba(16,185,129,0)}100%{box-shadow:0 0 0 0 rgba(16,185,129,0)}}
        h1{font-size:2.8rem;font-weight:800;letter-spacing:-1px;margin-bottom:12px;background:linear-gradient(135deg,#fff,#94a3b8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .sub{color:var(--muted);font-size:1.05rem;max-width:580px;margin:0 auto 36px;line-height:1.7}
        .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:36px}
        .card{background:rgba(255,255,255,.02);padding:20px;border-radius:14px;border:1px solid rgba(255,255,255,.04);transition:all .3s}
        .card:hover{transform:translateY(-4px);border-color:var(--blue);background:rgba(59,130,246,.04)}
        .card h3{font-size:.95rem;margin-bottom:6px;color:#fff}
        .card p{font-size:.78rem;color:var(--muted);line-height:1.5}
        .card .tag{display:inline-block;font-size:.65rem;padding:3px 10px;border-radius:99px;margin-bottom:10px;font-weight:600;letter-spacing:.5px}
        .easy .tag{background:rgba(16,185,129,.15);color:var(--green)}
        .med .tag{background:rgba(234,179,8,.15);color:#eab308}
        .hard .tag{background:rgba(239,68,68,.15);color:var(--red)}
        .actions{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
        .btn{text-decoration:none;padding:12px 24px;border-radius:10px;font-weight:600;font-size:.9rem;transition:all .2s;display:inline-flex;align-items:center;gap:8px}
        .btn-p{background:var(--blue);color:#fff;box-shadow:0 8px 20px rgba(59,130,246,.4)}
        .btn-p:hover{transform:scale(1.03);box-shadow:0 12px 30px rgba(59,130,246,.5)}
        .btn-s{background:rgba(255,255,255,.04);color:#fff;border:1px solid rgba(255,255,255,.08)}
        .btn-s:hover{background:rgba(255,255,255,.08)}
        .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:32px;font-family:'JetBrains Mono',monospace}
        .stat{background:rgba(0,0,0,.3);padding:14px;border-radius:10px;border:1px solid rgba(255,255,255,.03)}
        .stat .label{font-size:.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
        .stat .val{font-size:1.3rem;font-weight:700}
        footer{margin-top:36px;color:var(--muted);font-size:.7rem;letter-spacing:.5px}
        @media(max-width:768px){.grid,.stats{grid-template-columns:1fr}h1{font-size:2rem}}
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <div class="badge"><span class="dot"></span> Environment Online</div>
            <h1>SENTINEL-9</h1>
            <p class="sub">Adversarial misinformation containment benchmark. Tests LLM causal reasoning, epistemic calibration, and resource management under adversarial pressure.</p>
            <div class="stats">
                <div class="stat"><div class="label">Version</div><div class="val" style="color:var(--blue)">2.0.0</div></div>
                <div class="stat"><div class="label">Tasks</div><div class="val">3</div></div>
                <div class="stat"><div class="label">Max Nodes</div><div class="val">150</div></div>
                <div class="stat"><div class="label">Interface</div><div class="val" style="color:var(--green)">POMDP</div></div>
            </div>
            <div class="grid">
                <div class="card easy"><span class="tag">EASY</span><h3>Detection</h3><p>Identify infected nodes in a frozen 40-node network using semantic analysis. 30% false positive noise.</p></div>
                <div class="card med"><span class="tag">MEDIUM</span><h3>Tracing</h3><p>Reconstruct causal chain in an 80-node scale-free network under active spread pressure.</p></div>
                <div class="card hard"><span class="tag">HARD</span><h3>Containment</h3><p>Manage budget, contain infection, and detect bot clusters in a 150-node adversarial network.</p></div>
            </div>
            <div class="actions">
                <a href="/docs" class="btn btn-p">📖 API Docs</a>
                <a href="/health" class="btn btn-s">♥ Health</a>
                <a href="/tasks" class="btn btn-s">📋 Tasks</a>
            </div>
        </div>
        <footer>SENTINEL-9 × OpenEnv Benchmark • Adversarial POMDP • Brier Calibration Scoring</footer>
    </div>
</body>
</html>"""
    return HTMLResponse(content=html)


# ═══════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════

def main(host: str = "0.0.0.0", port: int = None):
    import uvicorn
    port = port or int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
