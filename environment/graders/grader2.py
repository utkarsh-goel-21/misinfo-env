import networkx as nx
from environment.models import Reward

class Grader2:
    """
    Grader for Task 2: Origin Tracing & Causal Reconstruction
    Scores the agent based on identifying the true origin 
    node and the validity of the submitted causal chain.
    """
    def grade(self, task, cumulative_penalty: float) -> Reward:
        truth = task.get_ground_truth()
        actual_origin = truth["origin_node"]
        submitted_chain = truth["submitted_chain"]
        
        # 1. Did they submit anything?
        if not submitted_chain:
            return Reward(
                score=-1.0,
                delta=0.0,
                done=True,
                success=False,
                partial_credits={"brier_calibration_penalty": cumulative_penalty},
                penalty=cumulative_penalty,
                feedback="Failed to submit causal chain before episode ended."
            )
            
        # 2. Score Origin Node (First node in chain)
        agent_origin = submitted_chain[0].get("from", "")
        origin_score = 0.6 if agent_origin == actual_origin else 0.0

        # 3. Score Edge Validity (Graph Edit Distance proxy via Intersection)
        valid_edges = 0
        total_submitted_edges = len(submitted_chain)
        
        for edge in submitted_chain:
            u = edge.get("from")
            v = edge.get("to")
            
            # Check if this edge exists in graph and both nodes are infected
            if u in task.graph.adjacency and v in task.graph.adjacency[u]:
                if task.graph.nodes[u].status.value == "infected" and task.graph.nodes[v].status.value == "infected":
                    valid_edges += 1
                    
        edge_score = (valid_edges / max(1, total_submitted_edges)) * 0.4
        
        base_score = origin_score + edge_score
        
        # Efficiency Bonus (Exponential as per Claude's suggestion)
        steps_taken = task.step_count
        max_steps = task.MAX_STEPS
        efficiency_bonus = 0.15 * ((1 - (steps_taken / max_steps)) ** 2)
        
        final_score = base_score + efficiency_bonus - cumulative_penalty
        final_score = max(-1.0, min(1.0, final_score))
        
        success = (agent_origin == actual_origin) and (valid_edges == total_submitted_edges)

        return Reward(
            score=round(final_score, 4),
            delta=0.0,
            done=True,
            success=success,
            partial_credits={
                "origin_score": origin_score,
                "edge_validity_score": round(edge_score, 4),
                "efficiency_bonus": round(efficiency_bonus, 4),
                "brier_calibration_penalty": round(cumulative_penalty, 4)
            },
            penalty=cumulative_penalty,
            feedback=f"Origin guess: {'Correct' if origin_score > 0 else 'Incorrect'}. {valid_edges}/{total_submitted_edges} edges matched valid infected paths."
        )