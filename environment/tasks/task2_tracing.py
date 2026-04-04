from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID


class Task2Tracing:
    """
    MEDIUM TASK — Origin Tracing

    Misinformation has spread for 5 steps.
    Agent must trace back and identify the
    exact origin node using inspect and trace actions.

    Spread continues LIVE during this task.
    Agent has 15 steps to identify origin.

    Success: correctly identify origin node.
    Partial credit: identifying correct neighborhood.
    """

    TASK_ID = TaskID.task2
    MAX_STEPS = 15
    NUM_NODES = 40
    AVG_CONNECTIONS = 4
    INFECTION_THRESHOLD = 0.6
    PRE_SPREAD_STEPS = 5

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.agent_origin_guess: str = ""
        self.step_count = 0
        self.done = False
        self.trace_history: list[str] = []

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.agent_origin_guess = ""
        self.step_count = 0
        self.done = False
        self.trace_history = []

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD
        )

        # Pre spread before agent starts
        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        """
        Agent can:
        - inspect: get node details
        - trace: get infection path clues
        - flag: submit origin guess
          (flag action on a node = agent guessing it is origin)

        Spread continues every step.
        """
        self.step_count += 1
        result = {"valid": False, "info": "", "newly_infected": []}

        if action.action_type.value == "inspect":
            details = self.graph.inspect_node(
                action.target_node_id
            )
            result = {
                "valid": bool(details),
                "info": details if details else "node not found",
                "newly_infected": []
            }

        elif action.action_type.value == "trace":
            trace_result = self.graph.trace_node(
                action.target_node_id
            )
            self.trace_history.append(action.target_node_id)
            result = {
                "valid": bool(trace_result),
                "info": trace_result if trace_result else "node not found",
                "newly_infected": []
            }

        elif action.action_type.value == "flag":
            # Flag = agent submitting origin guess
            self.agent_origin_guess = action.target_node_id
            result = {
                "valid": True,
                "info": f"origin guess submitted: {action.target_node_id}",
                "newly_infected": []
            }
            self.done = True  # guess submitted = task ends

        else:
            result = {
                "valid": False,
                "info": f"action {action.action_type} not allowed in task2",
                "newly_infected": []
            }

        # Spread continues every step
        newly_infected = self.engine.step()
        result["newly_infected"] = newly_infected

        # Check threshold
        if (
            self.graph.threshold_breached()
            or self.step_count >= self.MAX_STEPS
        ):
            self.done = True

        return result

    def get_ground_truth(self) -> dict:
        origin = self.graph.origin_node_id
        origin_neighbors = self.graph.nodes[origin].neighbors
        return {
            "origin_node": origin,
            "origin_neighbors": origin_neighbors,
            "agent_guess": self.agent_origin_guess
        }