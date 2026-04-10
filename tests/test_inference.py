from contextlib import redirect_stdout
from io import StringIO

from baseline_policy import BaselinePolicy
from environment.env import MisinfoEnv
from inference import choose_action, emit_end, emit_step


def _capture_stdout(func, *args, **kwargs) -> str:
    buf = StringIO()
    with redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue().strip()


def test_emit_step_matches_hackathon_stdout_contract():
    line = _capture_stdout(
        emit_step,
        3,
        "inspect(node_7) c=0.75",
        0.127,
        False,
        None,
    )
    assert line == "[STEP] step=3 action=inspect(node_7) c=0.75 reward=0.13 done=false error=null"


def test_emit_end_includes_score_and_two_decimal_rewards():
    line = _capture_stdout(
        emit_end,
        True,
        4,
        0.5671,
        [0.01, 0.127, 0.5671],
    )
    assert line == "[END] success=true steps=4 score=0.57 rewards=0.01,0.13,0.57"


def test_choose_action_reviews_heuristic_with_llm_when_client_present(monkeypatch):
    env = MisinfoEnv(task_id="task1_detection", seed=42)
    obs = env.reset()
    policy = BaselinePolicy("task1_detection")
    policy.observe(obs)
    messages = [{"role": "system", "content": "test"}]
    called = {"count": 0}

    def fake_call_llm(client, messages):
        called["count"] += 1
        return {"use_proposed": True, "confidence": 0.61, "reasoning": "Reviewed via proxy"}

    monkeypatch.setattr("inference.call_llm", fake_call_llm)

    action = choose_action(object(), "task1_detection", obs, policy, messages)

    assert called["count"] == 1
    assert action.action_type.value == "inspect"
    assert action.confidence == 0.61
