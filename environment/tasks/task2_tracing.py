from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID


class Task2Tracing:
    """
    MEDIUM TASK — Origin Tracing (Causal Reconstruction)
    Agent must reconstruct the exact causal path of the 
    misinformation using submit_causal_chain.
    """
    TASK_ID = TaskID.task2
    MAX_STEPS = 15
    NUM_NODES = 80
    AVG_CONNECTIONS = 4
    INFECTION_THRESHOLD = 0.4
    PRE_SPREAD_STEPS = 5

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.done = False
        self.submitted_chain = []

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.done = False
        self.submitted_chain = []

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD,
            topology="barabasi_albert"
        )
        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        self.step_count += 1
        result = {"valid": False, "info": ""}

        if action.action_type.value == "inspect":
            details = self.graph.inspect_node(action.target_node_id)
            if details:
                result = {"valid": True, "info": details}
            else:
                result = {"valid": False, "info": "node not found"}
                
        elif action.action_type.value == "trace":
            details = self.graph.trace_node(action.target_node_id)
            if details:
                result = {"valid": True, "info": details}
            else:
                result = {"valid": False, "info": "node not found"}

        elif action.action_type.value == "quarantine":
            if self.graph.quarantine_node(action.target_node_id):
                result = {"valid": True, "info": f"quarantined {action.target_node_id}"}
            else:
                result = {"valid": False, "info": f"cannot quarantine {action.target_node_id}"}
                
        elif action.action_type.value == "submit_causal_chain":
            self.submitted_chain = action.causal_chain
            result = {"valid": True, "info": "causal chain submitted successfully."}
            self.done = True # Ends simulation

        if self.step_count >= self.MAX_STEPS:
            self.done = True

        return result

    def get_ground_truth(self) -> dict:
        return {
            "origin_node": self.graph.origin_node_id,
            "submitted_chain": self.submitted_chain,
            "actual_history": self.engine.spread_history
        }