---
title: SENTINEL-9 Misinfo Containment
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: Adversarial POMDP benchmark for misinformation containment.
tags:
  - openenv
---

# SENTINEL-9: Misinformation Containment Benchmark

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://openenv.ai)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-31%2F31%20Passing-brightgreen)]()

An adversarial POMDP benchmark where AI agents must detect, trace, and contain misinformation spreading through simulated social networks. Built for the [OpenEnv](https://openenv.ai) Global Hackathon.

---

## Why This Environment?

Misinformation containment is a **real-world, high-stakes problem** faced by platforms, governments, and civil society. This environment simulates the core challenge: an agent operates under **partial observability**, **limited budget**, and **adversarial pressure** from reactive bot networks that evolve in response to the agent's actions.

**What makes SENTINEL-9 genuinely hard:**
- 🔍 **Fog-of-war POMDP** — Agent only sees nodes it has inspected
- 📊 **5-tier deceptive content** — From blatant ALL CAPS to nearly undetectable stealth posts
- 🎯 **Brier calibration scoring** — Overconfidence on wrong actions = catastrophic quadratic penalty
- 🤖 **Adversarial bot network** — Bots evade detection when public outrage rises
- 💰 **Resource management** — Limited budget forces strategic action prioritization
- 📈 **SIR dynamics** — Nodes recover, creating temporal reasoning challenges
- 🌐 **Dynamic topology** — Network structure shifts as users migrate from quarantined nodes

---

## Three Tasks (Easy → Medium → Hard)

### Task 1 — Detection (Easy)
| Property | Value |
|----------|-------|
| Network | 40 nodes, Watts-Strogatz |
| Max steps | 10 |
| Spread | Frozen |
| Allowed actions | `inspect`, `quarantine` |
| Key challenge | ~30% false positive rate in stream reports |

**Grading:** `Score = (TPR × 0.50) − (FPR × 0.20) − (Brier × 0.15) + (Efficiency × 0.15)`

---

### Task 2 — Tracing (Medium)
| Property | Value |
|----------|-------|
| Network | 80 nodes, Barabási-Albert |
| Max steps | 15 |
| Spread | Active (advances every 3 agent actions) |
| Allowed actions | `inspect`, `trace`, `quarantine`, `submit_causal_chain` |
| Key challenge | Reconstruct causal chain under time pressure |

**Grading:** `Score = (Origin × 0.30) + (ChainF1 × 0.30) + (Containment × 0.20) + (Efficiency × 0.10) − (Brier × 0.10)`

---

### Task 3 — Containment (Hard)
| Property | Value |
|----------|-------|
| Network | 150 nodes, Mixed topology with planted bridges |
| Max steps | 20 |
| Actions per step | 5 |
| Budget | $10,000 |
| Allowed actions | All 7 action types |
| Key challenge | Simultaneous containment + bot detection + chain reconstruction + budget management |

**Grading:** `Score = (Containment × 0.25) + (CIB_F1 × 0.20) + (Chain × 0.15) + (Timing × 0.15) + (Budget × 0.10) + (Precision × 0.05) − (Brier × 0.10)`

---

## Observation Space (POMDP)

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Active task |
| `step_number` | int | Current step |
| `max_steps` | int | Maximum steps |
| `actions_remaining` | int? | Actions left this step (Task 3) |
| `stream_reports` | list[str] | Flagged node IDs (~30% false positives) |
| `revealed_nodes` | list[str] | Previously inspected node IDs (fog-of-war) |
| `inspection_results` | dict? | Results from last inspect/trace |
| `infection_rate` | float | Current network infection rate |
| `financial_budget` | float | Remaining budget |
| `public_outrage_index` | float | Streisand Effect metric (0-1) |
| `brier_score_running` | float | Running calibration score |
| `network_size` | int | Total nodes |

---

## Action Space

| Action | Cost | Task 1 | Task 2 | Task 3 | Description |
|--------|------|--------|--------|--------|-------------|
| `inspect` | $50 | ✓ | ✓ | ✓ | Read post content + demographics. Reveals fog-of-war. |
| `trace` | $200 | | ✓ | ✓ | Get centrality + neighbor infection timeline. |
| `quarantine` | $1,500 | ✓ | ✓ | ✓ | Isolate node. Wrong = Streisand Effect. |
| `remove` | $3,000 | | | ✓ | Permanently sever. Wrong = severe outrage. |
| `shadowban` | $500 | | | ✓ | Reduce influence 80%. Low risk. |
| `deploy_counter_narrative` | $4,000 | | | ✓ | Boost community resilience. |
| `submit_causal_chain` | Free | | ✓ | ✓ | Submit infection path. Ends episode. |

**Action format:**
```json
{
  "action_type": "inspect",
  "target_node_id": "node_7",
  "confidence": 0.75,
  "reasoning": "High centrality + infected neighbors suggest early infection"
}
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run tests (31 tests)
pytest tests/test_env.py -v

# Start server
uvicorn server.app:app --port 7860

# Run baseline inference
export HF_TOKEN=your_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
python inference.py
```

### Python API

```python
from environment.env import MisinfoEnv
from environment.models import Action, ActionType

env = MisinfoEnv(task_id="task1_detection", seed=42)
obs = env.reset()

# Inspect a flagged node
action = Action(
    action_type=ActionType.inspect,
    target_node_id=obs.stream_reports[0],
    confidence=0.6,
    reasoning="Checking flagged node"
)
obs, reward, done, info = env.step(action)
print(f"Brier: {info['brier_this_step']:.3f}")
```

### HTTP API

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detection", "seed": 42}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect", "target_node_id": "node_0", "confidence": 0.6}'
```

### WebSocket

```javascript
const ws = new WebSocket("ws://localhost:7860/ws");
ws.send(JSON.stringify({command: "reset", task_id: "task1_detection", seed: 42}));
ws.send(JSON.stringify({command: "step", action_type: "inspect", target_node_id: "node_0", confidence: 0.6}));
```

---

## Docker

```bash
docker build -t sentinel-9 .
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_xxx \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  sentinel-9
```

---

## Project Structure

```
misinfo-env/
├── openenv.yaml              # OpenEnv specification
├── Dockerfile                 # Multi-stage Docker build
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
├── inference.py               # Baseline LLM agent
├── .env                       # Environment variables
│
├── server/
│   ├── __main__.py            # python -m server
│   └── app.py                 # FastAPI server (HTTP + WebSocket)
│
├── environment/
│   ├── __init__.py
│   ├── env.py                 # Core environment (POMDP + Brier)
│   ├── models.py              # Pydantic data models
│   ├── graph.py               # Social network engine
│   ├── spread.py              # SIR + LTM spread engine
│   ├── tasks/
│   │   ├── task1_detection.py # Easy: frozen detection
│   │   ├── task2_tracing.py   # Medium: active tracing
│   │   └── task3_containment.py # Hard: adversarial containment
│   └── graders/
│       ├── grader1.py         # Multi-metric detection grader
│       ├── grader2.py         # GED-based tracing grader
│       └── grader3.py         # 7-dimensional containment grader
│
└── tests/
    └── test_env.py            # 31 comprehensive tests
```

---

## License

MIT License
