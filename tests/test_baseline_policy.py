from baseline_policy import BaselinePolicy
from environment.env import MisinfoEnv
from environment.models import ActionType
from inference import run_task


def _run_episode(task_id: str, seed: int) -> float:
    env = MisinfoEnv(task_id=task_id, seed=seed)
    obs = env.reset()
    policy = BaselinePolicy(task_id)

    while not env.done:
        policy.observe(obs)
        action = policy.decide(obs)
        if action is None:
            target = obs.stream_reports[0] if obs.stream_reports else "node_0"
            action = policy._make_action(
                ActionType.inspect,
                target=target,
                confidence=0.3,
                reasoning="fallback",
            )
        policy.memory.note_action(action)
        obs, reward, done, info = env.step(action)

    return reward.score


def test_heuristic_policy_builds_non_empty_chain_for_tracing_task():
    env = MisinfoEnv(task_id="task2_tracing", seed=42)
    obs = env.reset()
    policy = BaselinePolicy("task2_tracing")

    for _ in range(6):
        policy.observe(obs)
        action = policy.decide(obs)
        policy.memory.note_action(action)
        obs, reward, done, info = env.step(action)
        if done:
            break

    chain = policy.build_causal_chain()
    assert chain
    assert all("from" in edge and "to" in edge for edge in chain)


def test_heuristic_policy_scores_are_reasonable_across_tasks():
    assert _run_episode("task1_detection", 42) >= 0.20
    assert _run_episode("task2_tracing", 42) >= 0.55
    assert _run_episode("task3_containment", 42) >= 0.25


def test_run_task_supports_heuristic_only_mode():
    score, success, steps, rewards = run_task(None, "task2_tracing")
    assert 0 < score < 1
    assert steps == len(rewards)
