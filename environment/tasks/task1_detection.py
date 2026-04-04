from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID


class Task1Detection:
    """
    EASY TASK — Detection

    Agent is given a network where misinformation
    has already spread for 3 steps.
    Agent must correctly identify ALL infected nodes.

    Agent gets 10 steps to inspect and classify nodes.
    No spread happens during this task — frozen snapshot.

    Success: correctly identify >= 80% of infected nodes
    with <= 10% false positives.
    """

    TASK_ID = TaskID.task1
    MAX_STEPS = 10
    NUM_NODES = 20
    AVG_CONNECTIONS = 3
    INFECTION_THRESHOLD = 0.8  # high — game wont end early
    PRE_SPREAD_STEPS = 3       # spread runs before agent starts

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.ground_truth_infected: list[str] = []
        self.agent_classifications: dict[str, str] = {}
        self.step_count = 0
        self.done = False

    def reset(self):
        """
        Build graph, spread for PRE_SPREAD_STEPS,
        freeze state. Agent now observes and classifies.
        """
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.agent_classifications = {}
        self.step_count = 0
        self.done = False

        # Build small network
        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD
        )

        # Pre spread — agent does not see this happening
        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        # Record ground truth
        self.ground_truth_infected = (
            self.graph.get_infected_nodes()
        )

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        """
        In task1 agent can only:
        - inspect: get node details
        - flag: classify node as infected

        No quarantine or removal allowed.
        Spread is frozen.
        """
        self.step_count += 1
        result = {"valid": False, "info": ""}

        if action.action_type.value == "inspect":
            details = self.graph.inspect_node(
                action.target_node_id
            )
            if details:
                result = {
                    "valid": True,
                    "info": details
                }
            else:
                result = {
                    "valid": False,
                    "info": f"node {action.target_node_id} not found"
                }

        elif action.action_type.value == "flag":
            node_id = action.target_node_id
            if node_id in self.graph.nodes:
                self.agent_classifications[node_id] = "infected"
                result = {
                    "valid": True,
                    "info": f"classified {node_id} as infected"
                }
            else:
                result = {
                    "valid": False,
                    "info": f"node {node_id} not found"
                }

        else:
            result = {
                "valid": False,
                "info": f"action {action.action_type} not allowed in task1"
            }

        if self.step_count >= self.MAX_STEPS:
            self.done = True

        return result

    def get_ground_truth(self) -> dict:
        return {
            "infected_nodes": self.ground_truth_infected,
            "origin_node": self.graph.origin_node_id,
            "total_infected": len(self.ground_truth_infected)
        }