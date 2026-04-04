from environment.models import Reward
from environment.tasks.task2_tracing import Task2Tracing


class Grader2:
    """
    Grader for Task 2 — Origin Tracing

    Scoring breakdown:
    - Exact origin match:          0.60
    - Origin in neighborhood:      0.30
      (guess is direct neighbor of origin)
    - Efficiency bonus:            0.10
      (fewer trace steps = higher bonus)
    - Threshold breach penalty:   -0.20
      (if infection crossed threshold)

    Formula:
      base = 0.60 (exact) or 0.30 (neighbor) or 0.0
      efficiency = 0.10 * (1 - traces_used/max_traces)
      breach_penalty = 0.20 if threshold breached else 0.0
      score = clamp(base + efficiency - breach_penalty, 0.0, 1.0)
    """

    def grade(
        self,
        task: Task2Tracing,
        cumulative_penalties: float = 0.0
    ) -> Reward:

        ground_truth = task.get_ground_truth()
        origin = ground_truth["origin_node"]
        origin_neighbors = set(ground_truth["origin_neighbors"])
        agent_guess = ground_truth["agent_guess"]

        # Base score
        exact_match = agent_guess == origin
        neighbor_match = (
            agent_guess in origin_neighbors
            and not exact_match
        )

        if exact_match:
            base_score = 0.60
        elif neighbor_match:
            base_score = 0.30
        else:
            base_score = 0.0

        # Efficiency bonus
        traces_used = len(task.trace_history)
        max_traces = task.MAX_STEPS
        efficiency = 0.10 * (
            1.0 - min(traces_used, max_traces) / max_traces
        )

        # Threshold breach penalty
        breach_penalty = (
            0.20 if task.graph.threshold_breached() else 0.0
        )

        # Final score
        raw_score = base_score + efficiency - breach_penalty
        final_score = round(max(0.0, min(1.0, raw_score)), 4)

        success = exact_match and not task.graph.threshold_breached()

        partial_credits = {
            "exact_match": exact_match,
            "neighbor_match": neighbor_match,
            "base_score": base_score,
            "efficiency_bonus": round(efficiency, 4),
            "breach_penalty": breach_penalty,
            "traces_used": traces_used,
            "agent_guess": agent_guess,
            "correct_origin": origin
        }

        feedback = self._build_feedback(
            exact_match,
            neighbor_match,
            agent_guess,
            origin,
            task.graph.threshold_breached(),
            success
        )

        return Reward(
            score=final_score,
            delta=0.0,
            done=True,
            success=success,
            partial_credits=partial_credits,
            penalty=cumulative_penalties,
            feedback=feedback
        )

    def _build_feedback(
        self,
        exact: bool,
        neighbor: bool,
        guess: str,
        origin: str,
        breached: bool,
        success: bool
    ) -> str:
        breach_msg = (
            " WARNING: Infection threshold breached during investigation."
            if breached else ""
        )
        if success:
            return (
                f"SUCCESS: Correctly identified origin node {origin}."
                f"{breach_msg}"
            )
        if exact and breached:
            return (
                f"PARTIAL: Correct origin {origin} but threshold breached."
                f"{breach_msg}"
            )
        if neighbor:
            return (
                f"PARTIAL: Guessed {guess} which neighbors origin {origin}. "
                f"Close but not exact.{breach_msg}"
            )
        if not guess:
            return (
                f"FAIL: No origin guess submitted. "
                f"Correct origin was {origin}.{breach_msg}"
            )
        return (
            f"FAIL: Guessed {guess} but correct origin was {origin}."
            f"{breach_msg}"
        )