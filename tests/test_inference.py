from contextlib import redirect_stdout
from io import StringIO

from inference import emit_end, emit_step


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
