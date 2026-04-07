# Sentinel-9 Meta-Hack Technical Blueprint (v2.0.0)
This document serves as the master context for the Sentinel-9 OpenEnv project. It provides precise, code-backed details on the project's architecture, API contract, graph topology, spread models, and task specifications.

## SECTION 1: HACKATHON OVERVIEW
**What is the OpenEnv Global Hackathon exactly?**
A competition to build standardized environment "Gyms" where AI agents can connect and interact via automated benchmarking scripts using the OpenEnv 0.2.0 standard.

**What are the official judging criteria?**
1. **Working code with reproducibility:** Deterministic seeds and baked JSON snapshots.
2. **OpenEnv 0.2.0 compliance:** Implementation of `/reset`, `/step`, and `/state` with correct schemas.
3. **Execution complexity:** Forcing LLMs to utilize reasoning (causal, semantic, and demographic) instead of basic graph-search or keyword matching.

**What does a winning submission look like?**
A Dockerized application deployed to Hugging Face Spaces on port 7860, implementing a mathematically sound and research-grade evaluator for LLM agent performance that produces a clear quality gradient across model tiers.

---

## SECTION 2: PROJECT SUMMARY
**What is Sentinel-9 trying to do in one clear paragraph?**
Sentinel-9 is a research-grade environment designed to evaluate an AI agent's ability to navigate **Partially Observable Hidden Markov Models (POMDP)**. Unlike simple graph puzzles, Sentinel-9 strips agents of global visibility, forcing them to reconstruct causal infection chains from unstructured report streams. Agents must reason about **Epistemic Uncertainty** (via Brier-scored confidence), **Semantic Nuance** (detecting debunkers and coordinated inauthentic behavior), and **Causal Propagation** to surgically contain misinformation in demographic echo chambers.

**What real-world problem does it simulate?**
Advanced Trust & Safety engineering, causal attribution in viral misinformation, and strategic policy intervention under information asymmetry.

---

## SECTION 3: EXACT FILE STRUCTURE
`/Dockerfile` — python:3.11-slim container exposing port 7860.
`/openenv.yaml` — Immutable API contract defining POMDP observation and causal action spaces.
`/generate_data.py` — Statically bakes demographic personas and 5-tier semantic posts into JSON.
`/environment/env.py` — The POMDP orchestrator (enforces the report stream vs. full graph).
`/environment/graph.py` — Advanced topology engine (Louvain communities, BA/WS graphs, Bot Spooking).
`/environment/spread.py` — **Linear Threshold Model (LTM)** spread engine.
`/environment/models.py` — Strictly typed Pydantic implementation of the causal-epistemic schema.
`/environment/graders/` — Ruthless graders implementing **Brier Scoring** and **Graph Edit Distance**.
`/server/app.py` — FastAPI application featuring live **Vis.js Visualizer** and **Benchmark** endpoints.

---

## SECTION 4: CORE TECHNICAL IMPLEMENTATION (V2.0)

**1. Epistemic Calibration (Brier Scoring)**
- Every action requires a `confidence: float [0.0 - 1.0]`.
- Graders apply a quadratic penalty: `(confidence - actual_correctness)^2`. 
- Overconfident models that hypothesize incorrectly are mathematically decimated in the final score.

**2. Partial Observability (POMDP)**
- Agents **DO NOT** receive the network graph in `Observation`.
- They receive a `stream_reports` list (noisy heuristic flags).
- They must use `inspect` (semantics) and `trace` (structural entropy) actions to build a local map of the infection.

**3. The 5-Tier Semantic Layer**
- **Tier 0 (Clean):** Mundane noise.
- **Tier 1 (Infected):** Blatant misinformation.
- **Tier 2 (Debunking):** Accurate content referencing the fake claim to refute it (The "Pattern Matcher Trap").
- **Tier 3 (CIB):** Coordinated Inauthentic Behavior (Syntactically unique, semantically identical narratives).
- **Tier 4 (Cowardice):** Deceptive framing without explicitly false state facts.

**4. Demographic Echo Chambers**
- Nodes possess `PoliticalLeaning`, `Occupation`, `Gender`, and `Age`.
- Innate `skepticism_score` is derived from demographics (e.g., higher for Engineers, lower for certain age brackets).
- **Louvain Communities:** Nodes are clustered into echo chambers where edge weights are 2.5x higher, simulating rapid viral spread within silos.

**5. Linear Threshold Model (LTM)**
- Ripped out ICP (independent cascade) for **LTM**.
- A node only becomes infected if the cumulative influence of infected neighbors exceeds their `skepticism_score`. 
- This creates "Herd Immunity" dynamics—high-skepticism clusters act as firebreaks.

---

## SECTION 5: API & TASKS

**/reset**
- Triggers a specific task topology (WS, BA, or Mixed).
- Returns the initial `stream_reports` but **zero** node status data.

**/step**
- `Action(action_type, target_node_id, confidence, reasoning, causal_chain)`
- Returns `[Observation, Reward, Done, Info]`.

**/visualizer** (NEW)
- A live dashboard using **Vis.js** that polls `/state`.
- Colors: Emerald (Clean), Red (Infected), Blue (Quarantined), Slate (Removed).
- Shows live Infection Rate bar and step counters.

**/benchmark** (NEW)
- Endpoint that runs a pre-defined 10-action sequence on a hardcoded seed (42).
- Returns a deterministic hex-hash to prove environment stability and scoring reproducibility.

---

## SECTION Task Specifications

| Task | Topology | Objective | Difficulty |
| :--- | :--- | :--- | :--- |
| **T1: Detection** | Watts-Strogatz | Identify semantic Tiers under budget | Easy |
| **T2: Tracing** | Barabasi-Albert | Reconstruct **Causal Chain** via Submit | Medium |
| **T3: Containment** | Mixed/Bridges | Block CIB and detect adversarial bots | Hard |

---

## SECTION 10: ADVANCED ADVERSARIAL MECHANICS
- **Adversarial Bots:** In Task 3, bots "spook" if the agent inspects their direct neighbors. They enter a `dormant_until` state, pausing spread to avoid detection by automated scanning.
- **Temporal Uncertainty:** The Hidden Markov layer ensures that a node classified as clean in Step 2 could be infected by Step 5 behind the agent's back. Re-inspection budget management is critical.

---

## SECTION 11: REPRODUCIBILITY & GIT
- **URL:** `https://huggingface.co/spaces/utkarsh-goel-21/misinfo-containment-env`
- **Seed:** 42 is the global benchmark seed.
- **Dependencies:** `networkx>=3.0.0`, `fastapi`, `pydantic`, `openai`, `vis.js` (frontend).

**Sentinel-9 is now the only OpenEnv submission testing TRUE causal attribution and epistemic calibration.**
