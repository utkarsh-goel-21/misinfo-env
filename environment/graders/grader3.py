from environment.models import Reward, NodeStatus

class Grader3:
    """
    Grader for Task 3: Full Containment & CIB Detection
    Multi-objective scorer for the hardest task.
    """
    def grade(self, task, cumulative_penalty: float) -> Reward:
        truth = task.get_ground_truth()
        
        # 1. Containment Score (0.40)
        # Ratio of steps spent below the infection threshold
        steps_below = truth["steps_below_threshold"]
        max_steps = task.MAX_STEPS
        containment_score = (steps_below / max_steps) * 0.40
        
        # 2. CIB / Bridge Targeted Quarantine Score (0.30)
        # Did they quarantine nodes with high skepticism / bridges?
        strategic_quarantines = 0
        total_quarantines = 0
        quarantine_penalties = 0.0

        for node_id, node in task.graph.nodes.items():
            if node.status == NodeStatus.quarantined:
                total_quarantines += 1
                # Good quarantine: Infected node or high-risk bridge
                if node.is_bot or node.community_id != "none" or node.influence_score > 0.7:
                    strategic_quarantines += 1
                elif node.status == NodeStatus.clean:
                    # Penalize blindly shutting down clean, low-danger nodes
                    quarantine_penalties += 0.05
                    
        cib_score = (strategic_quarantines / max(1, total_quarantines)) * 0.30
        
        # 3. Timing Loss Counterfactual (0.15)
        # Evaluates if interventions happened proactively before the peak spread step
        spread_report = truth["spread_report"]
        peak_step = spread_report.get("peak_spread_step", max_steps)
        
        # If they successfully suppressed it so there was no massive peak, they get full points
        if spread_report.get("final_infection_rate", 1.0) < task.INFECTION_THRESHOLD:
            timing_score = 0.15
        else:
            # Timing loss: failed to act meaningfully before peak
            timing_score = 0.0

        # 4. Action Efficiency (0.15)
        actions_taken = task.step_count * task.ACTIONS_PER_STEP - task.actions_this_step
        efficiency = 0.15 * max(0, 1 - (actions_taken / (max_steps * task.ACTIONS_PER_STEP)))

        # Compile final score minus the running Brier calibration penalty from Env
        base_score = containment_score + cib_score + timing_score + efficiency
        
        final_score = base_score - quarantine_penalties - cumulative_penalty
        final_score = max(-1.0, min(1.0, final_score))
        
        success = (containment_score >= 0.35) and (cib_score > 0.15)

        return Reward(
            score=round(final_score, 4),
            delta=0.0,
            done=True,
            success=success,
            partial_credits={
                "containment_score": round(containment_score, 4),
                "cib_strategic_score": round(cib_score, 4),
                "timing_intervention_score": round(timing_score, 4),
                "efficiency_score": round(efficiency, 4),
                "false_quarantine_penalties": round(quarantine_penalties, 4)
            },
            penalty=quarantine_penalties,
            feedback=f"Containment: {steps_below}/{max_steps} steps. Strategic Interventions: {strategic_quarantines}/{total_quarantines}."
        )