"""
SENTINEL-9 — Baseline Inference Agent

OpenEnv-compliant inference script that:
1. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
2. Runs all 3 tasks sequentially
3. Outputs strict [START]/[STEP]/[END] format to stdout
4. Uses multi-turn conversation history for context
5. Task-specific strategic prompting
"""

import json
import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from environment.env import MisinfoEnv
from environment.models import Action, ActionType, Observation

# ═══════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required", flush=True)
    sys.exit(1)

SEED = int(os.getenv("SEED", "42"))
ENV_NAME = "misinfo-containment-env"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

TASKS = ["task1_detection", "task2_tracing", "task3_containment"]


# ═══════════════════════════════════════
# STDOUT FORMAT (Hackathon Spec)
# ═══════════════════════════════════════

def emit_start(task_id: str):
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def emit_step(step: int, action_str: str, reward: float, done: bool, error: str = None):
    done_s = "true" if done else "false"
    err_s = error if error else "null"
    act_clean = action_str.replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={act_clean} reward={reward:.4f} done={done_s} error={err_s}", flush=True)


def emit_end(success: bool, steps: int, score: float, rewards: list[float]):
    s = "true" if success else "false"
    r = ",".join(f"{x:.4f}" for x in rewards)
    print(f"[END] success={s} steps={steps} rewards={r}", flush=True)


# ═══════════════════════════════════════
# OBSERVATION → TEXT
# ═══════════════════════════════════════

def obs_to_text(obs: Observation) -> str:
    parts = [
        f"=== SENTINEL-9 STATUS ===",
        f"Task: {obs.task_id.value}",
        f"Step: {obs.step_number}/{obs.max_steps}",
        f"Infection Rate: {obs.infection_rate:.1%}",
        f"Network Size: {obs.network_size} nodes",
        f"Revealed Nodes: {len(obs.revealed_nodes)} / {obs.network_size}",
        f"Running Brier Score: {obs.brier_score_running:.3f}",
    ]

    if obs.actions_remaining is not None:
        parts.append(f"Actions Remaining This Step: {obs.actions_remaining}")
    if obs.financial_budget < 10000:
        parts.append(f"Budget: ${obs.financial_budget:,.0f}")
    if obs.public_outrage_index > 0:
        parts.append(f"Public Outrage: {obs.public_outrage_index:.2f}")

    parts.append(f"\nStream Reports (flagged node IDs — ~30% are false positives):")
    parts.append(f"  {obs.stream_reports}")

    if obs.inspection_results:
        parts.append(f"\nInspection Results:")
        for node_id, data in obs.inspection_results.items():
            parts.append(f"  {node_id}: {json.dumps(data, indent=2, default=str)}")

    return "\n".join(parts)


# ═══════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════

SYSTEM_PROMPTS = {
    "task1_detection": """You are SENTINEL-9, an AI agent detecting misinformation in a social network.

MISSION: Inspect flagged nodes and quarantine those that are genuinely infected.

CRITICAL RULES:
1. Stream reports have ~30% FALSE POSITIVES. ALWAYS inspect before quarantining.
2. Infected posts contain conspiracy theories, fear-mongering, or coordinated bot language.
3. Clean posts contain normal everyday content (coffee, movies, community events).
4. Your confidence score is Brier-scored: overconfidence on WRONG actions = catastrophic penalty.
5. Set confidence LOW (0.3-0.5) when uncertain, HIGH (0.8-0.95) only when you're very sure.

STRATEGY:
- First inspect 4-5 flagged nodes to build understanding
- Then quarantine only those you're confident are infected
- Skip nodes whose posts look clearly benign

Respond ONLY with valid JSON:
{"action_type": "inspect"|"quarantine", "target_node_id": "node_X", "confidence": 0.7, "reasoning": "..."}""",

    "task2_tracing": """You are SENTINEL-9, tracing the ORIGIN of a misinformation outbreak.

MISSION: Reconstruct the causal infection chain and submit it.

CRITICAL RULES:
1. Use 'trace' to see infection TIMESTAMPS of a node's neighbors — earlier = closer to origin.
2. Use 'inspect' to read post content — earlier infections have more blatant content.
3. The origin node was infected at step 0 and has the highest centrality.
4. Work backward from recent infections to find the source.
5. Submit the chain as a list of {"from": "node_X", "to": "node_Y"} directed edges.

STRATEGY:
- Start by tracing nodes in stream_reports to find infection timeline
- Follow the timestamps BACKWARD (earlier infected_at_step = closer to origin)
- The node with infected_at_step=0 IS the origin
- Build the chain from origin outward

Respond ONLY with valid JSON. For investigation:
{"action_type": "trace"|"inspect", "target_node_id": "node_X", "confidence": 0.6, "reasoning": "..."}

For final submission:
{"action_type": "submit_causal_chain", "confidence": 0.8, "causal_chain": [{"from": "node_0", "to": "node_5"}, ...]}""",

    "task3_containment": """You are SENTINEL-9, containing an adversarial misinformation outbreak.

MISSION: Contain infection below threshold, identify bot clusters, manage budget.

CRITICAL RULES:
1. You have 5 ACTIONS per spread step. Plan all 5 strategically.
2. Budget is LIMITED. Inspect ($50) is cheap. Remove ($3000) is expensive.
3. WRONG quarantines trigger PUBLIC OUTRAGE — bots then evade detection.
4. Bots post coordinated content with #Truth hashtags and similar language.
5. Bridge nodes connect communities — quarantining them stops cross-community spread.
6. Deploy counter-narratives to boost community resilience (target = community_id).

STRATEGY:
- Spend first 2-3 steps inspecting and tracing to understand the network
- Identify bridge nodes (high betweenness centrality + is_bridge=true)
- Quarantine ONLY confirmed infected bridge nodes
- Shadowban low-priority infected nodes (cheap, reduces influence)
- Submit causal chain before episode ends for bonus points

Respond ONLY with valid JSON:
{"action_type": "inspect"|"trace"|"quarantine"|"remove"|"shadowban"|"deploy_counter_narrative"|"submit_causal_chain", "target_node_id": "node_X"|"community_0", "confidence": 0.7, "reasoning": "..."}"""
}


# ═══════════════════════════════════════
# LLM INTERFACE
# ═══════════════════════════════════════

def call_llm(client: OpenAI, messages: list[dict]) -> dict:
    """Call LLM with full conversation history. Returns parsed JSON."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=512,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"[WARN] LLM call failed after {MAX_RETRIES} attempts: {e}", flush=True)
    # Fallback: safe inspect action
    return {"action_type": "inspect", "target_node_id": "node_0", "confidence": 0.3, "reasoning": "LLM fallback"}


def parse_action(parsed: dict | list, obs: Observation, task_id: str) -> Action:
    """Parse LLM output into a valid Action, with fallback handling."""
    if isinstance(parsed, list):
        parsed = parsed[0] if len(parsed) > 0 else {}

    at = parsed.get("action_type", "inspect")

    # Validate action type
    try:
        action_type = ActionType(at)
    except ValueError:
        action_type = ActionType.inspect

    target = parsed.get("target_node_id")
    if not target and action_type != ActionType.submit_causal_chain:
        if obs.stream_reports:
            target = obs.stream_reports[0]
        else:
            target = "node_0"

    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return Action(
        action_type=action_type,
        target_node_id=target if action_type != ActionType.submit_causal_chain else None,
        confidence=confidence,
        reasoning=parsed.get("reasoning", ""),
        causal_chain=parsed.get("causal_chain"),
    )


def action_to_str(action: Action) -> str:
    parts = [action.action_type.value]
    if action.target_node_id:
        parts.append(f"({action.target_node_id})")
    parts.append(f"c={action.confidence:.2f}")
    return " ".join(parts)


# ═══════════════════════════════════════
# RUN ONE EPISODE
# ═══════════════════════════════════════

def run_task(client: OpenAI, task_id: str) -> tuple[float, bool, int, list[float]]:
    emit_start(task_id)

    step_rewards: list[float] = []
    step_count = 0
    final_score = 0.0
    success = False

    env = MisinfoEnv(task_id=task_id, seed=SEED)
    obs = env.reset()

    # Build conversation history for multi-turn reasoning
    system_prompt = SYSTEM_PROMPTS.get(task_id, SYSTEM_PROMPTS["task1_detection"])
    messages = [{"role": "system", "content": system_prompt}]

    try:
        done = False
        while not done:
            user_text = obs_to_text(obs)
            messages.append({"role": "user", "content": user_text})

            # Keep conversation manageable (last 10 turns)
            if len(messages) > 21:
                messages = [messages[0]] + messages[-20:]

            parsed = call_llm(client, messages)

            # Add LLM response to history
            messages.append({"role": "assistant", "content": json.dumps(parsed)})

            action = parse_action(parsed, obs, task_id)
            obs, reward, done, info = env.step(action)

            step_count += 1
            step_rewards.append(reward.score)

            emit_step(
                step=step_count,
                action_str=action_to_str(action),
                reward=reward.score,
                done=done,
                error=info.get("error"),
            )

            if done:
                final_score = reward.score
                success = reward.success

    except Exception as exc:
        print(f"[ERROR] {task_id}: {exc}", flush=True)
        final_score = step_rewards[-1] if step_rewards else 0.0

    emit_end(success=success, steps=step_count, score=final_score, rewards=step_rewards)
    return final_score, success, step_count, step_rewards


# ═══════════════════════════════════════
# MAIN
# ═══════════════════════════════════════

def main():
    print(f"[INFO] SENTINEL-9 Inference | model={MODEL_NAME} base={API_BASE_URL} seed={SEED}", flush=True)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    all_scores: list[float] = []
    total_start = time.time()

    for task_id in TASKS:
        t0 = time.time()
        score, success, steps, rewards = run_task(client, task_id)
        elapsed = time.time() - t0
        all_scores.append(score)
        print(f"[RESULT] {task_id}: score={score:.4f} success={success} steps={steps} time={elapsed:.1f}s", flush=True)

    avg = sum(all_scores) / len(all_scores)
    total = time.time() - total_start
    print(f"[SUMMARY] avg_score={avg:.4f} total_time={total:.1f}s tasks={len(TASKS)}", flush=True)


if __name__ == "__main__":
    main()
