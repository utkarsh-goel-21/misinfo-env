## SECTION 1: HACKATHON OVERVIEW
**What is the OpenEnv Global Hackathon exactly?**
A competition to build standardized environment "Gyms" where AI agents can connect and interact via automated benchmarking scripts.

**What are the official judging criteria?**
1. Working code with reproducibility (deterministic seeds).
2. OpenEnv 0.2.0 compliance.
3. Execution complexity that forces LLMs to utilize reasoning instead of basic graph-search algorithms.

**What does OpenEnv 0.2.0 compliance mean technically?**
An HTTP service (typically FastAPI) exposing `/reset`, `/step`, and `/state` routes. The `reset` route receives an episode initialization payload (Task ID + Seed) and returns an Observation. The `step` route receives an Agent Action and returns an Observation, a Reward, a done boolean, and info. The `state` route returns the hidden ground truth.

**What does a winning submission look like per the official rules?**
A Dockerized, containerized application deployed to Hugging Face Spaces on port 7860, implementing a mathematically sound and reproducible evaluator for LLM agent performance.

---

## SECTION 2: PROJECT SUMMARY
**What is Sentinel-9 trying to do in one clear paragraph?**
Sentinel-9 tests an AI Agent's ability to navigate a social network graph to detect infected nodes, forensically trace the origin of the network infection, and surgically quarantine bridge nodes to keep the total infection rate below 35% using a strictly limited action budget.

**What real-world problem does it simulate?**
Trust & Safety engineering and viral misinformation containment. 

**Current deployment status and Hugging Face Space URL?**
Deployed to Hugging Face Spaces: `https://huggingface.co/spaces/utkarsh-goel-21/misinfo-containment-env`.

---

## SECTION 3: EXACT FILE STRUCTURE
`/Dockerfile` — Container instructions exposing port 7860 using python:3.11-slim
`/README.md` — Project documentation
`/generate_data.py` — Script to pre-generate deterministic JSON networks
`/inference.py` — Baseline AI runner connecting to the environment
`/openenv.yaml` — Official specification of the environment observation and action spaces
`/pyproject.toml` — Python dependency and build configuration using hatchling
`/requirements.txt` — Frozen python dependencies
`/server/__main__.py` — Application entrypoint
`/server/app.py` — FastAPI application mounting `/reset`, `/step`, `/state`, and `/health` routes
`/environment/__init__.py` — Package init
`/environment/env.py` — The core orchestrator state-machine
`/environment/graph.py` — Watts-Strogatz random network generator and state container
`/environment/models.py` — Comprehensive Pydantic schema repository
`/environment/spread.py` — Probabilistic viral engine running infection events
`/environment/data/network_easy.json` — Pre-generated 20-node static network
`/environment/data/network_medium.json` — Pre-generated 40-node static network
`/environment/data/network_hard.json` — Pre-generated 80-node static network
`/environment/tasks/__init__.py` — Package init
`/environment/tasks/task1_detection.py` — Task 1 rules and state manager
`/environment/tasks/task2_tracing.py` — Task 2 rules and state manager
`/environment/tasks/task3_containment.py` — Task 3 rules and budget enforcement
`/environment/graders/__init__.py` — Package init
`/environment/graders/grader1.py` — Computes TPR / FPR detection scores
`/environment/graders/grader2.py` — Computes origin identification distance
`/environment/graders/grader3.py` — Multi-objective scorer for containment execution
`/tests/__init__.py` — Package init
`/tests/test_env.py` — Pytest test runner for state integrity

---

## SECTION 4: CORE TECHNICAL IMPLEMENTATION

**`openenv.yaml`**
- Contains the immutable API contract. Observation space is defined with `task_id`, `step_number`, `max_steps`, `actions_remaining`, `network` snapshot, and `recently_infected`. Action space restricts actions per task, allowing `inspect`, `trace`, `flag`, `quarantine`, `remove`, and `restore`.

**`environment/models.py`**
- Defines strictly typed enums `NodeStatus` (clean, infected, quarantined, flagged, removed) and `TaskID`. Defines Pydantic Models: `Node`, `Edge`, `Observation`, `Action`, `Reward`, `EnvironmentState`, `NetworkSnapshot`.

**`environment/env.py`**
- Contains `MisinfoEnv` class.
- `reset(request: ResetRequest)` loads the specific Task by ID and delegates reset to it.
- `step(action: Action)` checks if the action type is valid, calls `apply_action` on the Task, runs the Grader, builds the standard Response, and checks if `task.done` is True.

**`environment/graph.py`**
- Contains `MisinformationGraph` class.
- `build_from_config` dynamically generates a Watts-Strogatz Small World graph. It creates nodes with random influence scores (0.1 - 0.9) and random metadata (`followers`, `account_age_days`).
- Manages action execution primitives like `quarantine_node`, `flag_node`, `remove_node`, `inspect_node` (returns metadata), and `trace_node` (returns specifically infected neighbors and timestamps).

**`environment/spread.py`**
- Contains `SpreadEngine` class.
- Calculates simultanous spread via `step()`. Gathers all `current_infected` plus `current_flagged`. 
- For each infected node, it iterates over its connections. `prob = edge_weight * source_node.influence_score * spread_multiplier`. If target is `NodeStatus.clean`, it runs a uniform random check against `prob`.

**`environment/tasks/task1_detection.py`**
- State machine for Task 1. Graph is completely frozen (`self.engine.step()` is not called). Tracks agent flags. 

**`environment/tasks/task2_tracing.py`**
- State machine for Task 2. Spread is active. Tracks `self.origin_guess`.

**`environment/tasks/task3_containment.py`**
- Heaviest state machine. Tracks `self.actions_this_step`. If `actions_this_step >= 3`, it manually triggers `self.engine.step()` to advance the infection graph. 
- Automatically sets `done = True` if `infection_rate >= 0.35` (35%).

**`environment/graders/grader1.py`**
- Grades Task 1. `TPR = true_flags / max(1, total_infected)`. `FPR = false_flags / max(1, total_clean)`.
- Final Score: `(TPR * 0.6) + (max(0, 1 - (FPR * 4)) * 0.4)`.

**`environment/graders/grader2.py`**
- Grades Task 2. If `guess == actual_origin: 0.60`. If `guess in actual_origin_neighbors: 0.30`.
- Efficiency bonus subtracts `0.01` penalty per step taken.
- Breach penalty: if breached threshold, subtracts `-0.20`.

**`environment/graders/grader3.py`**
- Grades Task 3. 
- Containment score: `(steps_below_threshold / max_steps) * 0.40`.
- Origin Accuracy: Exact (0.25), Neighbor (0.12).
- Precision: Starts at 0.20, subtracts heavily for wrong actions.
- Rate bonus: `0.15 * max(0, 1.0 - (final_infection_rate / threshold))`.
- Action penalties: `-0.05` per wrong quarantine, `-0.10` per wrong removal.

**`server/app.py`**
- Implements FastAPI. Mounts `env = MisinfoEnv()`.
- Routes: `GET /` (Renders custom HTML string). `GET /health` (returns {"status": "ok"}). `POST /reset` (parses json body to ResetRequest, returns Observation dict). `POST /step` (parses json body to Action, returns tuple response). `GET /state` (returns `env.state()`).

---

## SECTION 5: OPENENV API CONTRACT

**/reset**
Request JSON:
`{ "task_id": "task3_containment", "seed": 42 }`
Response JSON: Must perfectly match the `ObservationSpace` in openenv.yaml.

**/step**
Request JSON (`Action`):
`{ "action_type": "quarantine", "target_node_id": "node_7", "reasoning": "High influence bridge" }`
Response JSON:
Returns an array of exactly 4 elements: `[Observation, Reward, Done, Info]`.

**/state**
Response JSON (`EnvironmentState`):
`{ "task_id": "task3_containment", "step": 5, "observation": {...}, "ground_truth": {"origin_node": "node_12", ...} }`

---

## SECTION 6: GRAPH TOPOLOGY
- **Model Used**: Watts-Strogatz Small World graph generation executed manually via looping iterators.
- **Node Attributes**: `node_id: str`, `status: NodeStatus`, `influence_score: float`, `infected_at_step: int | None`, `neighbors: list[str]`, `metadata: dict`.
- **Edge Attributes**: `source: str`, `target: str`, `weight: float`.
- **Reproducibility**: The exact `seed` (e.g., 42) is passed to a localized `random.Random(seed)` instance when initializing `MisinformationGraph` and `SpreadEngine`. This ensures perfectly identical node assignments and probability dice rolls.

---

## SECTION 7: SPREAD MODEL
- **Exact Formula**: `Probability = Edge_Weight * Source_Influence_Score * Spread_Multiplier`
- **Spread_Multiplier**: `1.0` if source is infected, `0.5` if source is flagged.
- **Node Status Effects**: Quarantined and Removed nodes bypass the inner loop; they cannot infect neighbors. Only nodes with status `clean` can be infected (i.e., a node cannot be re-infected or have its timer reset).
- **Mathematical limitations**: The spread is currently determined solely by the source node. The target node possesses no "defense" stat or media literacy. Highly connected nodes instantly overwhelm the network within 2 steps if their influence score is > 0.8.

---

## SECTION 8: TASK SPECIFICATIONS

**Task 1: Detection**
- Nodes: 20
- Action budget: 1 action per step (Total 10 steps permissible)
- Win threshold: TPR >= 0.80 and FPR <= 0.10.

**Task 2: Tracing**
- Nodes: 40
- Action budget: 1 action per step (Total 15 steps permissible)
- Win threshold: Exact origin node identified AND infection threshold (0.4) never breached.

**Task 3: Containment**
- Nodes: 80
- Action budget: 3 distinct actions per spread step (Total 20 max steps)
- Win threshold: Infection strictly stays below threshold (0.35) for all 20 steps AND exact origin identified.
- Grader measures: Containment length (40%), Origin accuracy (25%), Agent precision (20%), Final network preservation (15%).

---

## SECTION 9: GIT AND HUGGING FACE WORKFLOW
**Clone and Local Setup**
```bash
git clone https://huggingface.co/spaces/utkarsh-goel-21/misinfo-containment-env
cd misinfo-containment-env
```
**Pushing to Hugging Face Spaces**
```bash
huggingface-cli login
git remote add huggingface https://huggingface.co/spaces/utkarsh-goel-21/misinfo-containment-env
git pull huggingface main --rebase
git push huggingface main
```
- Workspace URL: `https://huggingface.co/spaces/utkarsh-goel-21/misinfo-containment-env`

---

## SECTION 10: CURRENT KNOWN GAPS / WHAT IS NOT BUILT YET
**NOT IMPLEMENTED YET:**
- Generative text/semantic properties. Nodes currently do not contain string posts or narrative content.
- Attacker/Defender adversarial mechanics. There are currently no bot capabilities mimicking "Spooking" or "Going dark" upon detection.
- Mathematical node immunity/skepticism. The Independent Cascade Model target resistance is absent.
- The `inference.py` script utilizes the `OpenAI` client in a rudimentary loop but lacks complex decision-tree heuristics suitable for Task 3's strict budget constraints. 
- There is no live `/visualizer` route utilizing D3.js.

---

## SECTION 11: DEPENDENCIES
- python >= 3.11
- pydantic >= 2.0.0
- fastapi >= 0.110.0
- uvicorn[standard] >= 0.29.0
- python-multipart >= 0.0.9
- openenv-core >= 0.2.0
- openai >= 1.0.0
- python-dotenv >= 1.0.0
- httpx >= 0.27.0
- pytest >= 8.0.0 (development only)
