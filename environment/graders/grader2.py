"""
Grader for Task 2: Origin Tracing & Causal Chain Reconstruction
Uses Graph Edit Distance for chain accuracy.

Score = (Origin × 0.30) + (ChainAccuracy × 0.30) + (Containment × 0.20)
      + (Efficiency × 0.10) − (Brier × 0.10)
"""

from environment.models import Reward
from environment.scoring import clamp_openenv_score


class Grader2:
    def grade(self, task, cumulative_penalty: float, brier_scores: list[float]) -> Reward:
        truth = task.get_ground_truth()
        actual_origin = truth["origin_node"]
        submitted_chain = truth["submitted_chain"]
        actual_tree = truth["actual_causal_tree"]

        # ── 1. No submission penalty ──
        if not submitted_chain:
            return Reward(
                score=clamp_openenv_score(0.0),
                delta=0.0,
                done=True,
                success=False,
                partial_credits={"reason": "No causal chain submitted"},
                penalty=cumulative_penalty,
                feedback="Failed to submit causal chain before episode ended.",
            )

        # ── 2. Origin Score (0.30) ──
        agent_origin = submitted_chain[0].get("from", "") if submitted_chain else ""
        origin_correct = agent_origin == actual_origin
        origin_score = 0.30 if origin_correct else 0.0

        # ── 3. Chain Accuracy via edge overlap (GED proxy) (0.30) ──
        actual_edges = set()
        for edge in actual_tree:
            actual_edges.add((edge["from"], edge["to"]))

        submitted_edges = set()
        for edge in submitted_chain:
            f, t = edge.get("from", ""), edge.get("to", "")
            if f and t:
                submitted_edges.add((f, t))

        if not submitted_edges:
            chain_score = 0.0
        else:
            # Precision: what fraction of submitted edges are correct
            correct_edges = len(submitted_edges & actual_edges)
            precision = correct_edges / len(submitted_edges)
            # Recall: what fraction of actual edges were found
            recall = correct_edges / max(1, len(actual_edges))
            # F1 score
            if precision + recall > 0:
                f1 = 2.0 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            chain_score = f1 * 0.30

        # ── 4. Containment Score (0.20) ──
        infection_rate = truth["final_infection_rate"]
        threshold = task.INFECTION_THRESHOLD
        if infection_rate < threshold:
            containment_score = 0.20
        else:
            # Partial credit: how close to threshold
            overshoot = min(1.0, (infection_rate - threshold) / threshold)
            containment_score = 0.20 * max(0, 1.0 - overshoot)

        # ── 5. Efficiency (0.10) ──
        steps_taken = task.step_count
        max_steps = task.MAX_STEPS
        efficiency = 0.10 * ((1 - (steps_taken / max(1, max_steps))) ** 2)

        # ── 6. Brier penalty (0.10) ──
        avg_brier = sum(brier_scores) / max(1, len(brier_scores)) if brier_scores else 0.0
        brier_penalty = avg_brier * 0.10

        # ── Composite ──
        final_score = clamp_openenv_score(
            origin_score + chain_score + containment_score + efficiency - brier_penalty
        )

        success = origin_correct and chain_score >= 0.20

        return Reward(
            score=final_score,
            delta=0.0,
            done=True,
            success=success,
            partial_credits={
                "origin_correct": origin_correct,
                "origin_score": round(origin_score, 4),
                "chain_accuracy_score": round(chain_score, 4),
                "chain_precision": round(precision if submitted_edges else 0, 4),
                "chain_recall": round(recall if submitted_edges else 0, 4),
                "containment_score": round(containment_score, 4),
                "efficiency_score": round(efficiency, 4),
                "brier_penalty": round(brier_penalty, 4),
                "submitted_edges": len(submitted_edges),
                "actual_edges": len(actual_edges),
            },
            penalty=cumulative_penalty,
            feedback=(
                f"Origin: {'✓ Correct' if origin_correct else '✗ Wrong'}. "
                f"Chain: {len(submitted_edges & actual_edges)}/{len(actual_edges)} edges matched. "
                f"Infection: {infection_rate:.1%}. Brier={avg_brier:.3f}."
            ),
        )
