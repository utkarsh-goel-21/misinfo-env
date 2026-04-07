from environment.models import Reward, NodeStatus

class Grader1:
    """
    Grader for Task 1: Detection
    Evaluates how effectively the agent identified and 
    quarantined infected nodes, integrating Brier score.
    """
    def grade(self, task, cumulative_penalty: float) -> Reward:
        infected = task.get_ground_truth()["infected_nodes"]
        quarantined = [
            n for n, node in task.graph.nodes.items() 
            if node.status == NodeStatus.quarantined
        ]

        total_infected = len(infected)
        if total_infected == 0:
            total_infected = 1

        # True Positives: Infected nodes that were quarantined
        true_positives = len(set(infected).intersection(set(quarantined)))
        
        # False Positives: Clean nodes that were wrongly quarantined
        false_positives = len(set(quarantined)) - true_positives

        tpr = true_positives / total_infected
        
        # Max 20 clean nodes penalized
        fpr_penalty = min(1.0, false_positives * 0.1) 

        base_score = (tpr * 0.7) - (fpr_penalty * 0.3)
        final_score = base_score - cumulative_penalty
        
        final_score = max(-1.0, min(1.0, final_score))
        
        success = (tpr >= 0.8) and (false_positives <= 2)

        return Reward(
            score=round(final_score, 4),
            delta=0.0,
            done=True,
            success=success,
            partial_credits={
                "tpr": round(tpr, 4),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "fpr_penalty": round(fpr_penalty, 4),
                "brier_calibration_penalty": round(cumulative_penalty, 4)
            },
            penalty=cumulative_penalty,
            feedback=f"Quarantined {true_positives}/{total_infected} infected nodes. {false_positives} wrong quarantines."
        )