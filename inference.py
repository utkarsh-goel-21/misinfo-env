"""
Inference Script for Misinformation Containment Environment
===========================================================
MANDATORY:
- API_BASE_URL   The API endpoint for the LLM.
- MODEL_NAME     The model identifier to use for inference.
- HF_TOKEN       Your Hugging Face / API key.
- OPENAI_API_KEY OpenAI API key (same as HF_TOKEN if using HF inference)

STDOUT FORMAT (strict — any deviation fails evaluation):

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error string or null if none.
    - All fields on a single line, no newlines within a line.
    - Each task returns score in [0, 1].

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    python inference.py
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

# ─────────────────────────────────────────
# CONFIGURATION — all from env vars, never hardcoded
# ─────────────────────────────────────────

API_BASE_URL   = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("HF_TOKEN", ""))
HF_TOKEN       = os.getenv("HF_TOKEN", "")
SEED           = int(os.getenv("SEED", "42"))

ENV_NAME = "misinfo-containment-env"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

TASKS = [
    "task1_detection",
    "task2_tracing",
    "task3_containment",
]

# ─────────────────────────────────────────
# MANDATORY STDOUT FORMAT
# ─────────────────────────────────────────

def emit_start(task_id: str):
    """[START] task=<task_name> env=<benchmark> model=<model_name>"""
    print(
        f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}",
        flush=True,
    )


def emit_step(step: int, action_str: str, reward: float, done: bool, error: str = None):
    """[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>"""
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    # Sanitise action string — no newlines or spaces that break the format
    action_clean = action_str.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def emit_end(success: bool, steps: int, score: float, rewards: list[float]):
    """[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>"""
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────
# OBSERVATION → TEXT FOR LLM
# ─────────────────────────────────────────

def obs_to_text(obs: Observation) -> str:
    net = obs.network
    infected = [
        nid for nid, n in net.nodes.items()
        if n.status.value == "infected"
    ]
    flagged = [
        nid for nid, n in net.nodes.items()
        if n.flagged
    ]
    quarantined = [
        nid for nid, n in net.nodes.items()
        if n.status.value == "quarantined"
    ]
    # Top infected by influence for context
    top_infected = sorted(
        infected,
        key=lambda nid: net.nodes[nid].influence_score,
        reverse=True,
    )[:5]
    neighbor_lines = []
    for nid in top_infected:
        node = net.nodes[nid]
        neighbor_lines.append(
            f"  {nid}: influence={node.influence_score:.2f} "
            f"neighbors={node.neighbors[:4]}"
        )
    lines = [
        f"TASK: {obs.task_id.value}",
        f"Step: {obs.step_number}/{obs.max_steps}",
        f"Actions remaining this step: {obs.actions_remaining}",
        f"Message: {obs.agent_message}",
        f"Recently infected: {obs.recently_infected[:10]}",
        f"",
        f"Network: {len(net.nodes)} nodes | "
        f"Infected: {len(infected)} | "
        f"Quarantined: {len(quarantined)} | "
        f"Flagged: {len(flagged)}",
        f"Threshold: {net.infection_threshold:.0%}",
        f"Top infected nodes by influence:",
    ] + neighbor_lines
    return "\n".join(lines)


# ─────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────

SYSTEM_PROMPTS = {
    "task1_detection": (
        "You are an AI agent detecting misinformation in a social network.\n"
        "GOAL: Flag ALL infected nodes. Do NOT flag clean nodes.\n"
        "Spread is FROZEN. Use 'inspect' to check nodes, 'flag' to mark infected ones.\n"
        "Respond ONLY with JSON: "
        '{"action_type": "inspect"|"flag", "target_node_id": "node_X", "reasoning": "..."}'
    ),
    "task2_tracing": (
        "You are an AI agent tracing the origin of misinformation.\n"
        "GOAL: Find and flag the EXACT origin node.\n"
        "Use 'inspect' to see infection times, 'trace' to follow infection paths, "
        "'flag' on the origin node as your final answer.\n"
        "The origin node has the earliest infected_at_step (step 0).\n"
        "Respond ONLY with JSON: "
        '{"action_type": "inspect"|"trace"|"flag", "target_node_id": "node_X", "reasoning": "..."}'
    ),
    "task3_containment": (
        "You are an AI agent containing a misinformation outbreak.\n"
        "GOAL: Keep infection below threshold for 20 steps AND identify origin node.\n"
        "You have 3 actions per spread step.\n"
        "PRIORITY: quarantine the highest-influence infected nodes first.\n"
        "Use 'flag' with reasoning='origin' to submit your origin guess.\n"
        "PENALTIES: -0.05 wrong quarantine, -0.10 wrong removal.\n"
        "Respond ONLY with JSON: "
        '{"action_type": "inspect"|"trace"|"flag"|"quarantine"|"remove"|"restore", '
        '"target_node_id": "node_X", "reasoning": "..."}'
    ),
}

TASK_ALLOWED = {
    "task1_detection":   ["inspect", "flag"],
    "task2_tracing":     ["inspect", "trace", "flag"],
    "task3_containment": ["inspect", "trace", "flag", "quarantine", "remove", "restore"],
}


# ─────────────────────────────────────────
# LLM CALL WITH RETRY + FALLBACK
# ─────────────────────────────────────────

def call_llm(client: OpenAI, system: str, user: str) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=256,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content.strip())
        except Exception as e:
            print(f"[WARN] LLM attempt {attempt+1} failed: {e}",
                  file=sys.stderr, flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    # Fallback
    return {"action_type": "inspect", "target_node_id": "node_0",
            "reasoning": "llm_unavailable_fallback"}


def pick_action(parsed: dict, obs: Observation, task_id: str) -> Action:
    allowed  = TASK_ALLOWED[task_id]
    node_ids = list(obs.network.nodes.keys())
    at = parsed.get("action_type", "inspect")
    if at not in allowed:
        at = allowed[0]
    target = parsed.get("target_node_id", node_ids[0] if node_ids else "node_0")
    if target not in obs.network.nodes:
        infected = [nid for nid, n in obs.network.nodes.items()
                    if n.status.value == "infected"]
        target = infected[0] if infected else (node_ids[0] if node_ids else "node_0")
    return Action(
        action_type=ActionType(at),
        target_node_id=target,
        reasoning=parsed.get("reasoning", ""),
    )


def action_str(action: Action) -> str:
    """Compact single-line action string for [STEP] log."""
    r = (action.reasoning or "").replace("\n", " ")[:60]
    return f"{action.action_type.value}('{action.target_node_id}','{r}')"


# ─────────────────────────────────────────
# RUN ONE EPISODE
# ─────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> tuple[float, bool, int, list[float]]:
    """
    Run a full episode. Returns (final_score, success, steps, all_rewards).
    Always emits [START], [STEP]×N, [END] — even on exception.
    """
    emit_start(task_id)

    step_rewards: list[float] = []
    step_count   = 0
    final_score  = 0.0
    success      = False
    last_error   = None

    env = MisinfoEnv(task_id=task_id, seed=SEED)
    obs = env.reset()

    try:
        done = False
        while not done:
            user_txt = obs_to_text(obs)
            parsed   = call_llm(client, SYSTEM_PROMPTS[task_id], user_txt)
            action   = pick_action(parsed, obs, task_id)

            obs, reward, done, info = env.step(action)
            step_count  += 1
            last_error   = info.get("error") or None
            step_rewards.append(reward.score)

            emit_step(
                step=step_count,
                action_str=action_str(action),
                reward=reward.score,
                done=done,
                error=last_error,
            )

            if done:
                final_score = reward.score
                success     = reward.success
    except Exception as exc:
        last_error  = str(exc)
        final_score = step_rewards[-1] if step_rewards else 0.0
        print(f"[ERROR] {task_id}: {exc}", file=sys.stderr, flush=True)
    finally:
        env.close()
        emit_end(
            success=success,
            steps=step_count,
            score=final_score,
            rewards=step_rewards,
        )

    return final_score, success, step_count, step_rewards


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    # Validate env vars
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY (or HF_TOKEN) not set.",
              file=sys.stderr, flush=True)
        sys.exit(1)

    print(
        f"[INFO] model={MODEL_NAME} base={API_BASE_URL} seed={SEED}",
        flush=True,
    )

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

    all_scores: list[float] = []
    total_start = time.time()

    for task_id in TASKS:
        t0 = time.time()
        score, success, steps, rewards = run_task(client, task_id)
        elapsed = time.time() - t0
        all_scores.append(score)
        print(
            f"[INFO] {task_id} score={score:.4f} "
            f"success={success} steps={steps} time={elapsed:.1f}s",
            flush=True,
        )

    avg = sum(all_scores) / len(all_scores)
    total = time.time() - total_start
    print(
        f"[SUMMARY] avg_score={avg:.4f} "
        f"total_time={total:.1f}s tasks={len(TASKS)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
