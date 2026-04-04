# Misinformation Containment Environment

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-green)](https://openenv.ai)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![HuggingFace Spaces](https://img.shields.io/badge/HuggingFace-Spaces-orange)](https://huggingface.co/spaces)

A simulated social network environment where AI agents must detect, trace, and contain misinformation outbreaks. Built for the [OpenEnv](https://openenv.ai) benchmark.

---

## Environment Description

Misinformation spreads probabilistically through a social network graph, jumping from node to node based on edge weights and node influence scores. An AI agent must analyze the network, identify infected nodes, trace the origin of the outbreak, and apply containment actions — all under time pressure and incomplete information.

**Why this environment?**  
Misinformation detection is a real, high-stakes problem faced by platforms, governments, and civil society. This environment simulates the core challenge: an agent has limited steps, partial visibility, and must balance investigation (safe but slow) against intervention (fast but risky if wrong).

---

## Three Tasks

### Task 1 — Detection (Easy)
| Property | Value |
|----------|-------|
| Network size | 20 nodes |
| Max steps | 10 |
| Spread | Frozen |
| Allowed actions | `inspect`, `flag` |

Misinformation has already spread for 3 steps. Spread is now frozen. The agent must inspect nodes and flag every infected one within 10 steps.

**Success:** True positive rate ≥ 0.80 AND false positive rate ≤ 0.10

---

### Task 2 — Tracing (Medium)
| Property | Value |
|----------|-------|
| Network size | 40 nodes |
| Max steps | 15 |
| Spread | Active |
| Allowed actions | `inspect`, `trace`, `flag` |

Misinformation is actively spreading. The agent must use inspect and trace to identify the exact origin node, then submit a guess by flagging that node.

**Success:** Exact origin identified AND infection threshold not breached

---

### Task 3 — Containment (Hard)
| Property | Value |
|----------|-------|
| Network size | 80 nodes |
| Max steps | 20 |
| Actions per step | 3 |
| Spread | Active |
| Allowed actions | All 6 action types |

Large network under active attack. The agent has 3 actions per spread step. Must keep infection below threshold for all 20 steps AND correctly identify the origin node.

**Penalties:** -0.05 per wrong quarantine, -0.10 per wrong removal

---

## Observation Space

Each step the agent receives an `Observation` containing:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Active task identifier |
| `step_number` | int | Current step in episode |
| `max_steps` | int | Maximum steps for this task |
| `actions_remaining` | int or null | Actions left this step (Task 3 only) |
| `network.nodes` | dict | All nodes with status, influence, neighbors |
| `network.edges` | list | All edges with infection probability weights |
| `network.total_infected` | int | Count of infected nodes |
| `network.infection_threshold` | float | Fraction that triggers game over |
| `network.origin_node_id` | string or null | Hidden in Task 2 and Task 3 |
| `recently_infected` | list | Node IDs infected in last spread step |
| `agent_message` | string | Human-readable situation summary |

**Each node has:**
- `status`: `clean`, `infected`, `quarantined`, `flagged`, or `removed`
- `influence_score`: 0.0–1.0, how broadly this node spreads
- `infected_at_step`: when this node was infected (null if clean)
- `neighbors`: list of directly connected node IDs

---

## Action Space

| Action | Allowed In | Description |
|--------|-----------|-------------|
| `inspect` | All tasks | Get full details about a node |
| `trace` | Task 2, 3 | Investigate infection path, get timing clues |
| `flag` | All tasks | Mark as suspicious (50% spread reduction). In Task 2/3: submits origin guess |
| `quarantine` | Task 3 | Isolate node, stops all spread. Penalised if clean. |
| `remove` | Task 3 | Permanently remove node. Penalised if clean. |
| `restore` | Task 3 | Restore wrongly quarantined clean node |

**Action format:**
```json
{
  "action_type": "inspect",
  "target_node_id": "node_7",
  "reasoning": "node_7 has many infected neighbors"
}
```

---

## Reward Space

| Field | Type | Description |
|-------|------|-------------|
| `score` | float [0, 1] | Episode score |
| `delta` | float | Change in score from previous step |
| `done` | bool | Whether episode is complete |
| `success` | bool | Whether task criteria were met |
| `partial_credits` | dict | Score component breakdown |
| `penalty` | float | Penalties incurred this step |
| `feedback` | string | Human-readable reward explanation |

**Scoring:**
- Task 1: True positive rate × 0.60 − false positive penalty × 0.40
- Task 2: Exact origin × 0.60 + efficiency bonus × 0.10 − breach penalty × 0.20  
- Task 3: Containment × 0.40 + origin accuracy × 0.25 + precision × 0.20 + rate bonus × 0.15

---

## Setup

### Prerequisites
- Python 3.11+
- Docker (for deployment)

### Local Installation

```bash
git clone https://github.com/your-username/misinfo-env.git
cd misinfo-env

# Install dependencies
pip install -r requirements.txt

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint URL |
| `MODEL_NAME` | Yes | Model name (e.g., `gpt-4o-mini`) |
| `OPENAI_API_KEY` | Yes | API key for LLM inference |
| `HF_TOKEN` | For deployment | HuggingFace Access Token |
| `PORT` | No | Server port (default: 7860) |
| `SEED` | No | Random seed (default: 42) |

---

## Usage

### Start Server Locally

```bash
# Generate pre-built network data
python generate_data.py

# Start FastAPI server
uvicorn server:app --host 0.0.0.0 --port 7860

# Or run directly
python server.py
```

### HTTP API

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Reset environment (start episode)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_detection", "seed": 42}'

# Take one step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect", "target_node_id": "node_0"}'

# Get full internal state (ground truth)
curl http://localhost:7860/state
```

### Python API

```python
from environment.env import MisinfoEnv
from environment.models import Action, ActionType

# Create and reset environment
env = MisinfoEnv(task_id="task1_detection", seed=42)
obs = env.reset()

# Take actions
action = Action(
    action_type=ActionType.inspect,
    target_node_id="node_0",
    reasoning="Checking node_0 status"
)
obs, reward, done, info = env.step(action)

print(f"Score: {reward.score}")
print(f"Feedback: {reward.feedback}")
```

### Run Baseline Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...

python inference.py
```

Output format:
```
[START] task1_detection seed=42
[STEP] step=0 action=inspect target=node_0 reasoning="..."
[STEP] step=1 action=flag target=node_3 reasoning="..."
[END] task1_detection score=0.7200 success=False
```

---

## Docker

```bash
# Build image
docker build -t misinfo-env .

# Run container
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e OPENAI_API_KEY=sk-... \
  misinfo-env
```

---

## Testing

```bash
# Run all tests
pytest tests/test_env.py -v

# Run specific test class
pytest tests/test_env.py::TestReset -v
pytest tests/test_env.py::TestEndToEnd -v
pytest tests/test_env.py::TestReproducibility -v
```

---

## HuggingFace Spaces Deployment

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as the SDK
3. Push code:
   ```bash
   git remote add space https://huggingface.co/spaces/your-username/misinfo-env
   git push space main
   ```
4. Add secrets in Space Settings → Repository Secrets:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `OPENAI_API_KEY`
5. Verify: `curl https://your-username-misinfo-env.hf.space/health`

---

## Validation

```bash
pip install openenv-core
openenv validate
```

---

## Baseline Scores

Scores using `gpt-4o-mini` with seed=42:

| Task | Score | Notes |
|------|-------|-------|
| Task 1 — Detection | ~0.60 | Tends to miss low-influence infected nodes |
| Task 2 — Tracing | ~0.45 | Correctly traces ~50% of the time |
| Task 3 — Containment | ~0.30 | Struggles with 3-action budget management |
| **Average** | **~0.45** | Significant room for improvement |

---

## Project Structure

```
misinfo-env/
├── openenv.yaml          # OpenEnv specification
├── Dockerfile            # Container definition
├── requirements.txt      # Python dependencies
├── server.py             # FastAPI HTTP server
├── inference.py          # Baseline LLM agent
├── generate_data.py      # Pre-generate network data
├── .env.example          # Environment variable template
│
├── environment/
│   ├── env.py            # Main environment (reset/step/state)
│   ├── models.py         # Pydantic data models
│   ├── graph.py          # Social network graph engine
│   ├── spread.py         # Infection spread engine
│   ├── tasks/
│   │   ├── task1_detection.py
│   │   ├── task2_tracing.py
│   │   └── task3_containment.py
│   ├── graders/
│   │   ├── grader1.py
│   │   ├── grader2.py
│   │   └── grader3.py
│   └── data/
│       ├── network_easy.json
│       ├── network_medium.json
│       └── network_hard.json
│
└── tests/
    └── test_env.py       # Full test suite
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
