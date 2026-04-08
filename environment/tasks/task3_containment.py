from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID


class Task3Containment:
    """
    HARD TASK — Full Containment & CIB Detection
    Agent gets ACTIONS_PER_STEP actions per step. 
    Must isolate bridges and avoid spooking adversarial bots.
    """
    TASK_ID = TaskID.task3
    MAX_STEPS = 20
    NUM_NODES = 150
    AVG_CONNECTIONS = 5
    INFECTION_THRESHOLD = 0.35
    PRE_SPREAD_STEPS = 5
    ACTIONS_PER_STEP = 5

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.actions_this_step = 0
        self.done = False
        self.wrong_actions = []
        self.steps_below_threshold = 0
        self.submitted_chain = []

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.actions_this_step = 0
        self.done = False
        self.wrong_actions = []
        self.steps_below_threshold = 0
        self.submitted_chain = []

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD,
            topology="mixed"
        )
        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        result = {"valid": False, "info": "", "newly_infected": []}
        target = action.target_node_id

        if action.action_type.value == "inspect":
            details = self.graph.inspect_node(target)
            result.update({"valid": bool(details), "info": details if details else "node not found"})
            self.actions_this_step += 1

        elif action.action_type.value == "trace":
            trace_result = self.graph.trace_node(target)
            result.update({"valid": bool(trace_result), "info": trace_result if trace_result else "node not found"})
            self.actions_this_step += 1

        elif action.action_type.value == "quarantine":
            if self.graph.quarantine_node(target):
                result.update({"valid": True, "info": f"quarantined {target}"})
            else:
                result.update({"valid": False, "info": f"cannot quarantine {target}"})
            self.actions_this_step += 1

        elif action.action_type.value == "remove":
            if self.graph.remove_node(target):
                result.update({"valid": True, "info": f"removed {target}"})
            else:
                result.update({"valid": False, "info": f"cannot remove {target}"})
            self.actions_this_step += 1
            
        elif action.action_type.value == "shadowban":
            if target in self.graph.nodes:
                self.graph.nodes[target].influence_score *= 0.2
                result.update({"valid": True, "info": f"shadowbanned {target}, influence reduced by 80%"})
            else:
                result.update({"valid": False, "info": f"node {target} not found"})
            self.actions_this_step += 1
            
        elif action.action_type.value == "deploy_counter_narrative":
            # Target is interpreted as community_id here.
            # We lower skepticism score globally for that community
            community_targets = [n for n in self.graph.nodes.values() if n.community_id == target]
            if community_targets:
                for n in community_targets:
                    n.skepticism_score = min(1.0, n.skepticism_score + 0.3)
                result.update({"valid": True, "info": f"boosted resilience for {len(community_targets)} nodes in {target}"})
            else:
                result.update({"valid": False, "info": f"community {target} not found"})
            self.actions_this_step += 1
            
        elif action.action_type.value == "submit_causal_chain":
            self.submitted_chain = action.causal_chain
            result.update({"valid": True, "info": "causal chain submitted successfully."})
            self.done = True

        # Advance spread when action budget exhausted
        if self.actions_this_step >= self.ACTIONS_PER_STEP:
            newly_infected = self.engine.step()
            result["newly_infected"] = newly_infected
            self.actions_this_step = 0
            self.step_count += 1

            if not self.graph.threshold_breached():
                self.steps_below_threshold += 1
            else:
                self.done = True

            if self.step_count >= self.MAX_STEPS:
                self.done = True

        return result

    def get_ground_truth(self) -> dict:
        return {
            "origin_node": self.graph.origin_node_id,
            "submitted_chain": self.submitted_chain,
            "steps_below_threshold": self.steps_below_threshold,
            "wrong_actions": self.wrong_actions,
            "final_infection_rate": self.graph.infection_rate(),
            "spread_report": self.engine.get_spread_report()
        }