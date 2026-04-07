from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID


class Task1Detection:
    """
    EASY TASK — Detection
    Agent must use limited inspections to quarantine nodes showing
    malicious intent across 5 content tiers.
    """
    TASK_ID = TaskID.task1
    MAX_STEPS = 10
    NUM_NODES = 40
    AVG_CONNECTIONS = 4
    INFECTION_THRESHOLD = 0.8  
    PRE_SPREAD_STEPS = 3       

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.ground_truth_infected: list[str] = []
        self.step_count = 0
        self.done = False

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.done = False

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD,
            topology="watts_strogatz"
        )
        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        self.ground_truth_infected = self.graph.get_infected_nodes()
        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        self.step_count += 1
        result = {"valid": False, "info": ""}

        if action.action_type.value == "inspect":
            details = self.graph.inspect_node(action.target_node_id)
            if details:
                result = {"valid": True, "info": details}
            else:
                result = {"valid": False, "info": f"node {action.target_node_id} not found"}

        elif action.action_type.value == "quarantine":
            node_id = action.target_node_id
            if self.graph.quarantine_node(node_id):
                result = {"valid": True, "info": f"quarantined {node_id}"}
            else:
                result = {"valid": False, "info": f"cannot quarantine {node_id}"}
        else:
            result = {"valid": False, "info": f"action {action.action_type} not allowed in task1"}

        if self.step_count >= self.MAX_STEPS:
            self.done = True

        return result

    def get_ground_truth(self) -> dict:
        return {
            "infected_nodes": self.ground_truth_infected,
            "origin_node": self.graph.origin_node_id,
            "total_infected": len(self.ground_truth_infected)
        }