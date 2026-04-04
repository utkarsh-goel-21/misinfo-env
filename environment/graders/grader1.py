from environment.models import Reward
from environment.tasks.task1_detection import Task1Detection


class Grader1:
    """
    Grader for Task 1 — Detection

    Scoring breakdown:
    - True positive rate:  0.60 weight
      (correctly flagged infected nodes)
    - False positive rate: 0.40 weight
      (penalize flagging clean nodes)

    Formula:
      tpr = correctly_flagged / total_infected
      fpr = wrongly_flagged / total_clean
      score = (0.6 * tpr) - (0.4 * fpr)
      score = clamp(score, 0.0, 1.0)
    """

    def grade(
        self,
        task: Task1Detection,
        cumulative_penalties: float = 0.0
    ) -> Reward:

        ground_truth = task.get_ground_truth()
        infected_set = set(ground_truth["infected_nodes"])
        clean_set = set(task.graph.nodes.keys()) - infected_set
        flagged_set = set(task.agent_classifications.keys())

        # True positives — correctly identified infected
        true_positives = flagged_set & infected_set

        # False positives — flagged clean nodes
        false_positives = flagged_set - infected_set

        # True positive rate
        tpr = (
            len(true_positives) / len(infected_set)
            if infected_set else 0.0
        )

        # False positive rate
        fpr = (
            len(false_positives) / len(clean_set)
            if clean_set else 0.0
        )

        # Raw score
        raw_score = (0.6 * tpr) - (0.4 * fpr)

        # Clamp to 0.0 - 1.0
        final_score = round(max(0.0, min(1.0, raw_score)), 4)

        # Partial credits breakdown
        partial_credits = {
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "total_infected": len(infected_set),
            "total_flagged": len(flagged_set)
        }

        # Success threshold
        success = tpr >= 0.8 and fpr <= 0.1

        # Human readable feedback
        feedback = self._build_feedback(
            tpr, fpr,
            len(true_positives),
            len(false_positives),
            len(infected_set),
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
        tpr: float,
        fpr: float,
        tp: int,
        fp: int,
        total: int,
        success: bool
    ) -> str:
        if success:
            return (
                f"SUCCESS: Detected {tp}/{total} infected nodes "
                f"with only {fp} false positives. "
                f"TPR={tpr:.2f} FPR={fpr:.2f}"
            )
        if tpr < 0.8 and fpr <= 0.1:
            return (
                f"PARTIAL: Found {tp}/{total} infected nodes "
                f"but needed 80% detection rate. "
                f"TPR={tpr:.2f}"
            )
        if tpr >= 0.8 and fpr > 0.1:
            return (
                f"PARTIAL: Good detection but too many false "
                f"positives ({fp} clean nodes wrongly flagged). "
                f"FPR={fpr:.2f}"
            )
        return (
            f"FAIL: Only detected {tp}/{total} infected nodes "
            f"with {fp} false positives. "
            f"TPR={tpr:.2f} FPR={fpr:.2f}"
        )