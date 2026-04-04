from environment.models import Reward
from environment.tasks.task3_containment import Task3Containment


class Grader3:
    """
    Grader for Task 3 — Full Containment

    Scoring breakdown:
    - Containment score:    0.40
      (fraction of steps infection stayed below threshold)
    - Origin accuracy:      0.25
      (exact=0.25, neighbor=0.12, wrong=0.0)
    - Precision score:      0.20
      (penalize wrong quarantines and removals)
    - Final rate bonus:     0.15
      (lower final infection rate = higher bonus)

    Penalties applied:
    - Each wrong quarantine: -0.05
    - Each wrong removal:    -0.10
    - Threshold breached:    containment_score suffers directly

    Formula:
      containment = steps_below_threshold / max_steps * 0.40
      origin = 0.25 or 0.12 or 0.0
      precision = 0.20 * (1 - wrong_actions/total_quarantines_removals)
      final_bonus = 0.15 * (1 - final_infection_rate/threshold)
      penalties = 0.05*wrong_q + 0.10*wrong_r
      score = clamp(containment+origin+precision+final_bonus-penalties, 0, 1)
    """

    def grade(
        self,
        task: Task3Containment
    ) -> Reward:

        ground_truth = task.get_ground_truth()
        origin = ground_truth["origin_node"]
        agent_guess = ground_truth["agent_origin_guess"]
        origin_neighbors = set(
            task.graph.nodes[origin].neighbors
        )
        wrong_quarantines = ground_truth["wrong_quarantines"]
        wrong_removals = ground_truth["wrong_removals"]
        steps_below = ground_truth["steps_below_threshold"]
        final_rate = ground_truth["final_infection_rate"]
        threshold = task.INFECTION_THRESHOLD

        # 1. Containment score
        containment = (
            steps_below / task.MAX_STEPS
        ) * 0.40

        # 2. Origin accuracy
        exact_origin = agent_guess == origin
        neighbor_origin = (
            agent_guess in origin_neighbors
            and not exact_origin
        )
        if exact_origin:
            origin_score = 0.25
        elif neighbor_origin:
            origin_score = 0.12
        else:
            origin_score = 0.0

        # 3. Precision score
        total_actions = len(wrong_quarantines) + len(wrong_removals)
        all_quarantines = task.graph.get_quarantined_nodes()
        all_removals = task.graph.get_removed_nodes()
        total_taken = len(all_quarantines) + len(all_removals)

        if total_taken == 0:
            precision = 0.20  # no actions = no wrong actions
        else:
            wrong_fraction = total_actions / total_taken
            precision = 0.20 * max(0.0, 1.0 - wrong_fraction)

        # 4. Final infection rate bonus
        if final_rate <= threshold:
            rate_bonus = 0.15 * (
                1.0 - final_rate / threshold
            )
        else:
            rate_bonus = 0.0

        # 5. Penalties
        penalty = (
            0.05 * len(wrong_quarantines)
            + 0.10 * len(wrong_removals)
        )

        # Final score
        raw_score = (
            containment
            + origin_score
            + precision
            + rate_bonus
            - penalty
        )
        final_score = round(max(0.0, min(1.0, raw_score)), 4)

        success = (
            exact_origin
            and ground_truth["threshold_breached_at"] == -1
            and final_rate < threshold
        )

        partial_credits = {
            "containment_score": round(containment, 4),
            "origin_score": round(origin_score, 4),
            "precision_score": round(precision, 4),
            "final_rate_bonus": round(rate_bonus, 4),
            "total_penalty": round(penalty, 4),
            "steps_below_threshold": steps_below,
            "wrong_quarantines": len(wrong_quarantines),
            "wrong_removals": len(wrong_removals),
            "final_infection_rate": round(final_rate, 4),
            "threshold": threshold,
            "exact_origin": exact_origin,
            "neighbor_origin": neighbor_origin
        }

        feedback = self._build_feedback(
            success,
            steps_below,
            task.MAX_STEPS,
            exact_origin,
            neighbor_origin,
            agent_guess,
            origin,
            len(wrong_quarantines),
            len(wrong_removals),
            final_rate,
            threshold
        )

        return Reward(
            score=final_score,
            delta=0.0,
            done=True,
            success=success,
            partial_credits=partial_credits,
            penalty=penalty,
            feedback=feedback
        )

    def _build_feedback(
        self,
        success: bool,
        steps_below: int,
        max_steps: int,
        exact: bool,
        neighbor: bool,
        guess: str,
        origin: str,
        wrong_q: int,
        wrong_r: int,
        final_rate: float,
        threshold: float
    ) -> str:
        lines = []
        if success:
            lines.append("SUCCESS: Full containment achieved.")
        else:
            lines.append("PARTIAL/FAIL: Containment incomplete.")

        lines.append(
            f"Containment: {steps_below}/{max_steps} steps "
            f"below threshold."
        )

        if exact:
            lines.append(f"Origin: Correctly identified {origin}.")
        elif neighbor:
            lines.append(
                f"Origin: Guessed {guess}, neighbor of {origin}."
            )
        else:
            lines.append(
                f"Origin: Wrong. Guessed {guess}, correct={origin}."
            )

        if wrong_q or wrong_r:
            lines.append(
                f"Precision: {wrong_q} wrong quarantines, "
                f"{wrong_r} wrong removals."
            )

        lines.append(
            f"Final infection rate: {final_rate:.2%} "
            f"(threshold: {threshold:.2%})"
        )

        return " | ".join(lines)