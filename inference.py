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
# CONFIGURATION
# ─────────────────────────────────────────

API_BASE_URL   = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN       = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

SEED           = int(os.getenv("SEED", "42"))

ENV_NAME = "misinfo-containment-env"
MAX_RETRIES = 3
RETRY_DELAY = 2.0

TASKS = ["task1_detection", "task2_tracing", "task3_containment"]

# ─────────────────────────────────────────
# MANDATORY STDOUT FORMAT
# ─────────────────────────────────────────

def emit_start(task_id: str):
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

def emit_step(step: int, action_str: str, reward: float, done: bool, error: str = None):
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    action_clean = action_str.replace("\n", " ").replace("\r", "")
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def emit_end(success: bool, steps: int, score: float, rewards: list[float]):
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)

# ─────────────────────────────────────────
# OBSERVATION → TEXT FOR LLM
# ─────────────────────────────────────────

def obs_to_text(obs: Observation) -> str:
    lines = [
        f"TASK: {obs.task_id.value}",
        f"Step: {obs.step_number}/{obs.max_steps}",
        f"Actions remaining this step: {obs.actions_remaining}",
        f"Message: {obs.agent_message}",
        f"Stream Reports (Recent flags): {obs.stream_reports}",
        f"Inspection Results: {json.dumps(obs.inspection_results) if obs.inspection_results else 'None'}"
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────

SYSTEM_PROMPTS = {
    "task1_detection": (
        "You are an AI detecting semantic misinformation.\n"
        "GOAL: Quarantine all infected nodes based on their posts.\n"
        "Use 'inspect' to read posts. Use 'quarantine' if infected. "
        "Brier-Score Warning: confidence must be strictly calibrated (0.0 - 1.0). Overconfidence on incorrect actions applies massive penalities.\n"
        "Respond ONLY with JSON: "
        '{"action_type": "inspect"|"quarantine", "target_node_id": "node_X", "confidence": 0.9, "reasoning": "..."}'
    ),
    "task2_tracing": (
        "You are tracing causal origin networks.\n"
        "GOAL: Reconstruct the causal chain.\n"
        "Use 'trace' to get betweenness centrality, 'inspect' for semantics. "
        "When certain, use 'submit_causal_chain' with an array of directed edges.\n"
        "Respond ONLY with JSON. Example for trace: "
        '{"action_type": "trace", "target_node_id": "node_X", "confidence": 0.8} '
        'Example for submit: '
        '{"action_type": "submit_causal_chain", "causal_chain": [{"from": "node_origin", "to": "node_next"}], "confidence": 0.9}'
    ),
    "task3_containment": (
        "You are containing a viral outbreak.\n"
        "GOAL: Isolate central bridges & identify Coordinated Inauthentic Behavior clusters.\n"
        "Use 'inspect', 'trace', 'quarantine', 'remove', 'submit_causal_chain'.\n"
        "Respond ONLY with JSON: "
        '{"action_type": "...", "target_node_id": "node_X", "confidence": 0.5, "causal_chain": null}'
    ),
}

TASK_ALLOWED = {
    "task1_detection":   ["inspect", "quarantine"],
    "task2_tracing":     ["inspect", "trace", "quarantine", "submit_causal_chain"],
    "task3_containment": ["inspect", "trace", "quarantine", "remove", "submit_causal_chain"],
}

# ─────────────────────────────────────────
# LLM CALL
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
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return {"action_type": "inspect", "target_node_id": "node_0", "confidence": 0.5}

def pick_action(parsed: dict, obs: Observation, task_id: str) -> Action:
    allowed  = TASK_ALLOWED[task_id]
    at = parsed.get("action_type", "inspect")
    if at not in allowed:
        at = allowed[0]
        
    target = parsed.get("target_node_id")
    # If target is missing, try picking from recent stream
    if not target and obs.stream_reports:
        target = obs.stream_reports[0]
    if not target:
        target = "node_0" # absolute fallback

    return Action(
        action_type=ActionType(at),
        target_node_id=target if at != "submit_causal_chain" else None,
        confidence=float(parsed.get("confidence", 0.5)),
        reasoning=parsed.get("reasoning", ""),
        causal_chain=parsed.get("causal_chain")
    )

def action_str(action: Action) -> str:
    return f"{action.action_type.value}('{action.target_node_id}' c={action.confidence})"

# ─────────────────────────────────────────
# RUN ONE EPISODE
# ─────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> tuple[float, bool, int, list[float]]:
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
                error=last_error
            )

            if done:
                final_score = reward.score
                success     = reward.success
    except Exception as exc:
        last_error  = str(exc)
        final_score = step_rewards[-1] if step_rewards else 0.0
    finally:
        emit_end(success=success, steps=step_count, score=final_score, rewards=step_rewards)

    return final_score, success, step_count, step_rewards

def main():
    if not HF_TOKEN:
        sys.exit(1)

    print(f"[INFO] model={MODEL_NAME} base={API_BASE_URL} seed={SEED}", flush=True)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    all_scores: list[float] = []
    total_start = time.time()

    for task_id in TASKS:
        t0 = time.time()
        score, success, steps, rewards = run_task(client, task_id)
        elapsed = time.time() - t0
        all_scores.append(score)

    avg = sum(all_scores) / len(all_scores)
    total = time.time() - total_start
    print(f"[SUMMARY] avg_score={avg:.4f} total_time={total:.1f}s tasks={len(TASKS)}", flush=True)

if __name__ == "__main__":
    main()
