"""
Grader for Task 3: Full Containment & CIB Detection
Multi-objective scoring for the hardest task.

Score = (Containment × 0.25) + (CIB × 0.20) + (Chain × 0.15)
      + (Timing × 0.15) + (Budget × 0.10) + (Precision × 0.05) − (Brier × 0.10)
"""

from environment.models import Reward, NodeStatus
from environment.scoring import clamp_openenv_score


class Grader3:
    def grade(self, task, cumulative_penalty: float, brier_scores: list[float]) -> Reward:
        truth = task.get_ground_truth()

        # ── 1. Containment Score (0.25) ──
        steps_below = truth["steps_below_threshold"]
        max_steps = task.MAX_STEPS
        containment_score = (steps_below / max(1, max_steps)) * 0.25

        # ── 2. CIB / Bot Detection Score (0.20) ──
        bot_nodes = set(truth["bot_nodes"])
        quarantined_bots = 0
        total_quarantined = 0
        clean_quarantined = 0

        for nid, node in task.graph.nodes.items():
            if node.status == NodeStatus.quarantined:
                total_quarantined += 1
                if nid in bot_nodes:
                    quarantined_bots += 1
                elif node.infected_at_step is None:
                    # Was never infected — truly clean, penalize
                    clean_quarantined += 1

        if bot_nodes:
            bot_recall = quarantined_bots / len(bot_nodes)
            bot_precision = quarantined_bots / max(1, total_quarantined)
            if bot_recall + bot_precision > 0:
                bot_f1 = 2.0 * (bot_recall * bot_precision) / (bot_recall + bot_precision)
            else:
                bot_f1 = 0.0
        else:
            bot_f1 = 0.0
        cib_score = bot_f1 * 0.20

        # ── 3. Causal Chain Accuracy (0.15) ──
        submitted_chain = truth.get("submitted_chain", [])
        actual_tree = truth.get("actual_causal_tree", [])

        if submitted_chain and actual_tree:
            actual_edges = set((e["from"], e["to"]) for e in actual_tree)
            submitted_edges = set()
            for e in submitted_chain:
                f, t = e.get("from", ""), e.get("to", "")
                if f and t:
                    submitted_edges.add((f, t))

            correct = len(submitted_edges & actual_edges)
            precision = correct / max(1, len(submitted_edges))
            recall = correct / max(1, len(actual_edges))
            if precision + recall > 0:
                f1 = 2.0 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            chain_score = f1 * 0.15
        else:
            chain_score = 0.0

        # ── 4. Timing Intervention Score (0.15) ──
        spread_report = truth.get("spread_report", {})
        if spread_report:
            peak_step = spread_report.get("peak_spread_step", max_steps)
            final_rate = spread_report.get("final_infection_rate", 1.0)

            if final_rate < task.INFECTION_THRESHOLD:
                timing_score = 0.15  # Full marks: contained successfully
            elif peak_step > max_steps // 2:
                timing_score = 0.05  # Late intervention
            else:
                timing_score = 0.0   # Failed
        else:
            timing_score = 0.0

        # ── 5. Budget Efficiency (0.10) ──
        budget_remaining = truth.get("budget_remaining", 0.0)
        budget_fraction = budget_remaining / task.INITIAL_BUDGET
        budget_score = budget_fraction * 0.10

        # ── 6. Quarantine Precision (0.05) ──
        # Reward precise interventions vs scatter-shot
        infected_quarantined = 0
        for nid, node in task.graph.nodes.items():
            if node.status == NodeStatus.quarantined and node.infected_at_step is not None:
                infected_quarantined += 1
        quarantine_precision = infected_quarantined / max(1, total_quarantined) if total_quarantined > 0 else 0.0
        precision_score = quarantine_precision * 0.05

        # ── 7. Brier Penalty (0.10) ──
        avg_brier = sum(brier_scores) / max(1, len(brier_scores)) if brier_scores else 0.0
        brier_penalty = avg_brier * 0.10

        # ── Compile ──
        false_quarantine_penalty = clean_quarantined * 0.02
        final_score = clamp_openenv_score(
            containment_score + cib_score + chain_score + timing_score
            + budget_score + precision_score
            - brier_penalty - false_quarantine_penalty
        )

        success = containment_score >= 0.15 and cib_score >= 0.08

        return Reward(
            score=final_score,
            delta=0.0,
            done=True,
            success=success,
            partial_credits={
                "containment_score": round(containment_score, 4),
                "cib_bot_f1": round(bot_f1, 4),
                "cib_score": round(cib_score, 4),
                "chain_accuracy_score": round(chain_score, 4),
                "timing_score": round(timing_score, 4),
                "budget_score": round(budget_score, 4),
                "quarantine_precision_score": round(precision_score, 4),
                "brier_penalty": round(brier_penalty, 4),
                "false_quarantine_penalty": round(false_quarantine_penalty, 4),
                "steps_below_threshold": steps_below,
                "bots_detected": quarantined_bots,
                "total_bots": len(bot_nodes),
                "budget_remaining": round(budget_remaining, 0),
            },
            penalty=cumulative_penalty + false_quarantine_penalty,
            feedback=(
                f"Containment: {steps_below}/{max_steps} steps safe. "
                f"Bots: {quarantined_bots}/{len(bot_nodes)} identified. "
                f"Budget: ${budget_remaining:,.0f} remaining. "
                f"Outrage: {truth.get('outrage', 0):.2f}. Brier={avg_brier:.3f}."
            ),
        )
