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
    target_node_id: Optional[str] = None
    confidence: float = 1.0
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
                    <a href="/visualizer" class="btn btn-primary">
                        <i data-lucide="layout-dashboard"></i> View Visualizer
                    </a>
                    <a href="/health" class="btn btn-secondary">
                        <i data-lucide="activity"></i> Health
                    </a>
                    <a href="/tasks" class="btn btn-secondary">
                        <i data-lucide="list"></i> API
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
    return {"tasks": ["task1_detection", "task2_tracing", "task3_containment"]}


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


@app.post("/benchmark", tags=["system"])
async def benchmark():
    """Runs a deterministic episode to verify environment integrity."""
    test_env = MisinfoEnv(task_id="task1_detection", seed=42)
    test_env.reset()
    
    # Perform deterministic actions
    test_nodes = ["node_0", "node_1", "node_2", "node_3", "node_4"]
    for node_id in test_nodes:
        # Inspect
        test_env.step(Action(action_type=ActionType.inspect, target_node_id=node_id, confidence=1.0))
        # Quarantine
        test_env.step(Action(action_type=ActionType.quarantine, target_node_id=node_id, confidence=0.8))
    
    final_state = test_env.state()
    # Grader call for final score
    reward = test_env.grader.grade(test_env.task, test_env.cumulative_penalty)
    
    return {
        "status": "success",
        "task": "task1_detection",
        "seed": 42,
        "final_score": reward.score,
        "success": reward.success,
        "benchmark_hash": "det_42_t1_v2"
    }


@app.get("/visualizer", tags=["system"])
async def visualizer():
    """Real-time graph visualizer using Vis.js."""
    from fastapi.responses import HTMLResponse
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>S-9 Ops Center | War Room</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            :root {
                --bg-deep: #020617;
                --glass-bg: rgba(15, 23, 42, 0.75);
                --glass-border: rgba(255, 255, 255, 0.08);
                --accent-blue: #3b82f6;
                --accent-blue-glow: rgba(59, 130, 246, 0.5);
                --accent-red: #ef4444;
                --accent-red-glow: rgba(239, 68, 68, 0.6);
                --accent-green: #10b981;
                --accent-warn: #eab308;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
            }
            body { 
                margin: 0; padding: 0; background: var(--bg-deep); color: var(--text-main); 
                font-family: 'Outfit', sans-serif; overflow: hidden;
            }
            
            /* Background Grid Effect */
            body::before {
                content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
                background-image: 
                    linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
                background-size: 50px 50px; z-index: 0; pointer-events: none;
            }

            #mynetwork { width: 100vw; height: 100vh; position: absolute; top:0; left:0; z-index: 1; }
            
            .panel {
                background: var(--glass-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
                border: 1px solid var(--glass-border); border-radius: 16px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); z-index: 10;
            }

            /* LEFT SIDEBAR */
            .sidebar {
                position: absolute; top: 24px; left: 24px; width: 340px;
                padding: 24px; display: flex; flex-direction: column; gap: 20px;
            }
            .header { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
            .header h1 { margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.5px; text-transform: uppercase; background: linear-gradient(to right, #fff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
            
            .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
            .metric-card { background: rgba(0,0,0,0.3); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.03); }
            .metric-card.full { grid-column: 1 / -1; }
            .metric-card h3 { margin: 0 0 8px 0; font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; display: flex; align-items: center; gap: 6px;}
            .metric-val { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
            
            .budget-val { color: var(--accent-green); }
            .outrage-val { color: var(--accent-warn); }
            
            .btn {
                background: linear-gradient(135deg, var(--accent-blue), #2563eb);
                color: white; border: none; padding: 14px; border-radius: 8px;
                font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 1rem;
                cursor: pointer; transition: all 0.2s; display: flex; align-items: center; justify-content: center; gap: 8px;
                box-shadow: 0 0 15px var(--accent-blue-glow);
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 0 25px var(--accent-blue-glow); }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }

            /* RIGHT SIDEBAR (TERMINAL) */
            .terminal {
                position: absolute; top: 24px; right: 24px; bottom: 80px; width: 380px;
                display: flex; flex-direction: column; overflow: hidden; padding: 0;
            }
            .terminal-header { padding: 16px 20px; border-bottom: 1px solid var(--glass-border); font-size: 0.8rem; font-weight: 600; color: var(--text-muted); display: flex; align-items: center; gap: 8px; background: rgba(0,0,0,0.2) }
            .terminal-body { 
                padding: 20px; overflow-y: auto; flex: 1; 
                font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.5;
            }
            .log-entry { margin-bottom: 12px; opacity: 0; animation: slideIn 0.3s forwards; }
            .log-time { color: #64748b; margin-right: 8px; }
            .log-type { color: var(--accent-blue); font-weight: 700; }
            .log-warn { color: var(--accent-warn); font-weight: 700; }
            .log-crit { color: var(--accent-red); font-weight: 700; }

            @keyframes slideIn { from { opacity: 0; transform: translateX(10px); } to { opacity: 1; transform: translateX(0); } }

            /* BOTTOM BAR */
            .bottom-bar {
                position: absolute; bottom: 24px; left: 24px; right: 24px; height: 12px;
                background: rgba(0,0,0,0.5); border-radius: 6px; border: 1px solid var(--glass-border);
                overflow: hidden; z-index: 10;
            }
            #infection-fill {
                height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-red));
                transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); position: relative;
            }
            #infection-fill::after {
                content: ''; position: absolute; top: 0; right: 0; bottom: 0; left: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                animation: shimmer 2s infinite;
            }
            @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

            /* PULSE EFFECTS */
            @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
        </style>
    </head>
    <body>
        <div id="mynetwork"></div>

        <div class="sidebar panel">
            <div class="header">
                <i data-lucide="shield-alert" color="#3b82f6" size="28"></i>
                <h1>S-9 Ops Center</h1>
            </div>
            
            <div class="metric-card full">
                <h3><i data-lucide="activity" size="14"></i> Active Operation</h3>
                <div id="stat-task" class="metric-val" style="font-size: 1.1rem; color: #fff;">Awaiting Telemetry...</div>
            </div>

            <div class="metric-grid">
                <div class="metric-card">
                    <h3><i data-lucide="dollar-sign" size="14"></i> Budget</h3>
                    <div id="stat-budget" class="metric-val budget-val">$10,000</div>
                </div>
                <div class="metric-card">
                    <h3><i data-lucide="flame" size="14"></i> Outrage</h3>
                    <div id="stat-outrage" class="metric-val outrage-val">0.0</div>
                </div>
                <div class="metric-card">
                    <h3><i data-lucide="radio" size="14"></i> Infection</h3>
                    <div id="stat-rate" class="metric-val" style="color: #ef4444">0.0%</div>
                </div>
                <div class="metric-card">
                    <h3><i data-lucide="clock" size="14"></i> Step Cycle</h3>
                    <div id="stat-steps" class="metric-val">0</div>
                </div>
            </div>

            <button class="btn" id="btn-benchmark" onclick="runBenchmark()">
                <i data-lucide="play-circle"></i> Run Deterministic Sim
            </button>
            <div style="font-size: 0.7rem; color: var(--text-muted); text-align: center;">Triggers task1_detection verification sequence.</div>
        </div>

        <div class="terminal panel">
            <div class="terminal-header">
                <i data-lucide="terminal" size="16"></i> EVENT TELEMETRY
            </div>
            <div class="terminal-body" id="t-body">
                <div class="log-entry"><span class="log-time">[00:00:00]</span> <span class="log-type">SYS:</span> Connecting to OpenEnv backbone...</div>
            </div>
        </div>

        <div class="bottom-bar">
            <div id="infection-fill"></div>
        </div>

        <script>
            lucide.createIcons();
            let network = null;
            let nodes = new vis.DataSet();
            let edges = new vis.DataSet();
            let lastActionCount = 0;

            function formatTime() {
                const now = new Date();
                return `[${now.toTimeString().split(' ')[0]}]`;
            }

            function addLog(msg, type='type') {
                const term = document.getElementById('t-body');
                const div = document.createElement('div');
                div.className = 'log-entry';
                div.innerHTML = `<span class="log-time">${formatTime()}</span> <span class="log-${type}">${type.toUpperCase()}:</span> ${msg}`;
                term.appendChild(div);
                term.scrollTop = term.scrollHeight;
            }

            async function runBenchmark() {
                const btn = document.getElementById('btn-benchmark');
                btn.disabled = true;
                btn.innerHTML = '<i data-lucide="loader-2" class="lucide-spin"></i> Executing...';
                lucide.createIcons();
                addLog("Initiating deterministic benchmark sequence...", "warn");
                
                try {
                    await fetch('/benchmark', {method: 'POST'});
                    addLog("Benchmark sequence complete. Evaluating...", "crit");
                } catch (e) {
                    addLog("Benchmark execution failed", "crit");
                }
                setTimeout(() => {
                    btn.disabled = false;
                    btn.innerHTML = '<i data-lucide="play-circle"></i> Run Deterministic Sim';
                    lucide.createIcons();
                }, 2000);
            }

            async function fetchData() {
                try {
                    const response = await fetch('/state');
                    if (response.ok) {
                        const state = await response.json();
                        updateUI(state);
                    }
                } catch (e) {
                    console.error("Fetch failed", e);
                }
                setTimeout(fetchData, 1000); // 1s refresh for real-time feel
            }

            function updateUI(state) {
                document.getElementById('stat-task').innerText = state.task_id.toUpperCase().replace('_', ' | ');
                
                // Animate numbers smoothly (simplified for DOM updates)
                document.getElementById('stat-budget').innerText = '$' + state.financial_budget.toLocaleString();
                if(state.financial_budget < 2000) document.getElementById('stat-budget').style.color = '#ef4444';
                else document.getElementById('stat-budget').style.color = '#10b981';

                document.getElementById('stat-outrage').innerText = (state.public_outrage_index * 10).toFixed(1);
                if(state.public_outrage_index > 0.6) document.getElementById('stat-outrage').style.color = '#ef4444';

                const rate = (state.network.total_infected / Object.keys(state.network.nodes).length) * 100;
                document.getElementById('stat-rate').innerText = rate.toFixed(1) + '%';
                document.getElementById('stat-steps').innerText = state.step_number;
                
                const fill = document.getElementById('status-fill');
                const threshold = state.network.infection_threshold * 100;
                const barWidth = Math.min(100, (rate / (state.network.infection_threshold || 1)) * 100);
                document.getElementById('infection-fill').style.width = barWidth + '%';

                // Log New Actions
                // state doesn't track specific action history right now, but we can track steps
                if (state.step_number > lastActionCount) {
                    addLog(`Cycle advanced to STEP ${state.step_number}`, "warn");
                    addLog(`Infection spread calculation complete. Rate: ${rate.toFixed(1)}%`, "type");
                    lastActionCount = state.step_number;
                }

                // Update Graph with Premium Styling
                const updatedNodes = [];
                for (const node_id in state.network.nodes) {
                    const n = state.network.nodes[node_id];
                    let bgColor, borderCol, shadow;
                    if (n.status === 'clean') { bgColor = '#0f172a'; borderCol = '#334155'; }
                    else if (n.status === 'infected') { bgColor = '#dc2626'; borderCol = '#f87171'; shadow = {color:'#ef4444', size:15, x:0, y:0}; }
                    else if (n.status === 'quarantined') { bgColor = '#2563eb'; borderCol = '#60a5fa'; shadow = {color:'#3b82f6', size:15, x:0, y:0}; }
                    else { bgColor = '#1e293b'; borderCol = '#0f172a'; }

                    updatedNodes.push({
                        id: node_id, label: node_id.replace('node_',''),
                        color: { background: bgColor, border: borderCol, highlight: { background: bgColor, border: '#fff' } },
                        font: { color: '#ffffff', face: 'JetBrains Mono', size: 10 },
                        shadow: shadow || false,
                        title: `<div style="padding:8px; background:#0f172a; color:#fff; border-radius:4px; font-family:sans-serif;"><b>${n.user_persona}</b><br>Comm: ${n.community_id}<br>Bot: ${n.is_bot}<br>Skep: ${n.skepticism_score}</div>`
                    });
                }
                nodes.update(updatedNodes);

                // Update Edges (Handle Migration)
                const currentEdges = edges.getIds();
                const newEdgeMap = {};
                const edgePayload = [];
                state.network.edges.forEach((e, idx) => {
                    const eId = `${e.source}-${e.target}`;
                    newEdgeMap[eId] = true;
                    if(!edges.get(eId)) {
                        edgePayload.push({
                            id: eId, from: e.source, to: e.target,
                            color: { color: 'rgba(255,255,255,0.05)' }, width: e.weight * 2
                        });
                    }
                });
                if(edgePayload.length > 0) edges.add(edgePayload);
                // Remove severed edges
                currentEdges.forEach(id => {
                    if(!newEdgeMap[id]) edges.remove(id);
                });
            }

            const container = document.getElementById('mynetwork');
            const data = { nodes: nodes, edges: edges };
            const options = {
                nodes: { shape: 'dot', size: 14, borderWidth: 2 },
                edges: { smooth: { type: 'continuous' } },
                physics: {
                    solver: 'forceAtlas2Based',
                    forceAtlas2Based: { gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 }
                },
                interaction: { hover: true, tooltipDelay: 100 }
            };
            network = new vis.Network(container, data, options);
            
            setTimeout(() => { addLog("Uplink established. Receiving stream...", "type"); fetchData(); }, 1500);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


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
