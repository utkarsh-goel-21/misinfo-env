"""
Grader for Task 1: Detection
Multi-metric scoring with proper Brier calibration.

Score = (TPR × 0.50) − (FPR_penalty × 0.20) − (Brier × 0.15) + (Efficiency × 0.15)
"""

from environment.models import Reward, NodeStatus
from environment.scoring import clamp_openenv_score


class Grader1:
    def grade(self, task, cumulative_penalty: float, brier_scores: list[float]) -> Reward:
        gt = task.get_ground_truth()
        infected = set(gt["infected_nodes"])
        quarantined = set(
            nid for nid, node in task.graph.nodes.items()
            if node.status == NodeStatus.quarantined
        )

        total_infected = max(1, len(infected))

        # True Positives: infected nodes correctly quarantined
        true_positives = len(infected & quarantined)
        # False Positives: clean nodes wrongly quarantined
        false_positives = len(quarantined - infected)

        tpr = true_positives / total_infected

        # FPR penalty: exponential to punish scatter-shot quarantining
        fpr_penalty = min(1.0, (false_positives ** 1.5) * 0.08)

        # Brier calibration score (average of per-action Brier scores)
        avg_brier = sum(brier_scores) / max(1, len(brier_scores)) if brier_scores else 0.0

        # Efficiency: reward fast agents
        steps_used = task.step_count
        max_steps = task.MAX_STEPS
        efficiency = (max_steps - steps_used) / max_steps if max_steps > 0 else 0.0

        # Composite score must be strictly (0, 1) to pass validation
        score = clamp_openenv_score(
            (tpr * 0.50) - (fpr_penalty * 0.20) - (avg_brier * 0.15) + (efficiency * 0.15)
        )

        success = tpr >= 0.80 and false_positives <= 2

        return Reward(
            score=score,
            delta=0.0,
            done=True,
            success=success,
            partial_credits={
                "tpr": round(tpr, 4),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "fpr_penalty": round(fpr_penalty, 4),
                "avg_brier_score": round(avg_brier, 4),
                "efficiency": round(efficiency, 4),
                "total_infected": total_infected,
            },
            penalty=cumulative_penalty,
            feedback=(
                f"Detection: {true_positives}/{total_infected} infected quarantined (TPR={tpr:.2f}). "
                f"{false_positives} false positives. Brier={avg_brier:.3f}. Efficiency={efficiency:.2f}."
            ),
        )
