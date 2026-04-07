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
        <title>Sentinel-9 | Live Visualizer</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body { 
                margin: 0; padding: 0; background: #0f172a; color: #f8fafc; 
                font-family: 'Outfit', sans-serif; overflow: hidden;
            }
            #mynetwork { width: 100vw; height: 100vh; }
            .sidebar {
                position: absolute; top: 20px; left: 20px; width: 300px;
                background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
                padding: 20px; z-index: 10;
            }
            .stat-card { background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; margin-bottom: 10px; }
            .stat-label { font-size: 0.8rem; color: #94a3b8; }
            .stat-value { font-size: 1.2rem; font-weight: 700; color: #3b82f6; }
            .legend { margin-top: 20px; font-size: 0.8rem; }
            .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
            .color-box { width: 12px; height: 12px; border-radius: 3px; }
            #status-bar {
                position: absolute; bottom: 20px; left: 20px; right: 20px;
                height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; border: 1px solid rgba(255,255,255,0.05);
            }
            #status-fill {
                height: 100%; width: 0%; background: #3b82f6; border-radius: 4px; transition: width 0.5s ease;
            }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2 style="margin-top: 0;">Sentinel-9</h2>
            <div class="stat-card">
                <div class="stat-label">Task</div>
                <div id="stat-task" class="stat-value">---</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Infection Rate</div>
                <div id="stat-rate" class="stat-value">0.0%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Step / Max</div>
                <div id="stat-steps" class="stat-value">0 / 0</div>
            </div>
            
            <div class="legend">
                <div class="legend-item"><div class="color-box" style="background:#f87171"></div> Infected</div>
                <div class="legend-item"><div class="color-box" style="background:#3b82f6"></div> Quarantined</div>
                <div class="legend-item"><div class="color-box" style="background:#10b981"></div> Clean</div>
                <div class="legend-item"><div class="color-box" style="background:#94a3b8"></div> Removed</div>
            </div>
        </div>
        
        <div id="status-bar"><div id="status-fill"></div></div>
        <div id="mynetwork"></div>

        <script>
            let network = null;
            let nodes = new vis.DataSet();
            let edges = new vis.DataSet();

            async function fetchData() {
                try {
                    const response = await fetch('/state');
                    const state = await response.json();
                    updateUI(state);
                } catch (e) {
                    console.error("Fetch failed", e);
                }
                setTimeout(fetchData, 2000);
            }

            function updateUI(state) {
                document.getElementById('stat-task').innerText = state.task_id.replace('task', 'Task ');
                const rate = (state.network.total_infected / Object.keys(state.network.nodes).length) * 100;
                document.getElementById('stat-rate').innerText = rate.toFixed(1) + '%';
                document.getElementById('stat-steps').innerText = state.step_number + ' / ' + state.network.nodes[Object.keys(state.network.nodes)[0]].infected_at_step; // wait, max steps?
                // actually better to just use state.step_number
                document.getElementById('stat-steps').innerText = state.step_number;
                
                const threshold = state.network.infection_threshold * 100;
                const fill = document.getElementById('status-fill');
                fill.style.width = Math.min(100, (rate / (state.network.infection_threshold || 1)) * 100) + '%';
                if (rate >= threshold) fill.style.background = '#ef4444';
                else fill.style.background = '#3b82f6';

                // Update Graph
                const statusColors = {
                    'clean': '#10b981',
                    'infected': '#f87171',
                    'quarantined': '#3b82f6',
                    'removed': '#94a3b8'
                };

                const updatedNodes = [];
                for (const node_id in state.network.nodes) {
                    const n = state.network.nodes[node_id];
                    updatedNodes.push({
                        id: node_id,
                        label: node_id,
                        color: {
                            background: statusColors[n.status] || '#10b981',
                            border: '#1e293b'
                        },
                        font: { color: '#ffffff' },
                        title: `Persona: ${n.user_persona}\\nPost: ${n.recent_post}\\nSkep: ${n.skepticism_score}`
                    });
                }
                nodes.update(updatedNodes);

                if (edges.length === 0) {
                    const edgeList = state.network.edges.map((e, idx) => ({
                        id: idx,
                        from: e.source,
                        to: e.target,
                        color: { color: 'rgba(255,255,255,0.1)' }
                    }));
                    edges.add(edgeList);
                }
            }

            const container = document.getElementById('mynetwork');
            const data = { nodes: nodes, edges: edges };
            const options = {
                nodes: { shape: 'dot', size: 16, borderWith: 2 },
                edges: { width: 1, smooth: false },
                physics: {
                    stabilization: true,
                    barnesHut: { gravitationalConstant: -2000, centralGravity: 0.3, springLength: 95 }
                }
            };
            network = new vis.Network(container, data, options);
            
            fetchData();
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
