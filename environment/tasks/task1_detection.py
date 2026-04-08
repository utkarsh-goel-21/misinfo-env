"""
SENTINEL-9 — Task 1: Detection (Easy)

Scenario: Misinformation has spread for 3 steps through a Watts-Strogatz
small-world network. Spread is NOW FROZEN. Agent must use limited inspections
to identify and quarantine infected nodes.

Key Challenge:
- Stream reports contain ~30% false positives
- Agent must verify via inspect before quarantining
- Content tiers 1-3 (blatant → hedged) — tests semantic analysis
- Brier calibration: overconfident wrong quarantines are heavily penalized

Network: 40 nodes, Watts-Strogatz (k=4, p=0.1)
Max Steps: 10
Allowed Actions: inspect, quarantine
Success: TPR ≥ 0.80 AND FP ≤ 2
"""

from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID, ActionType


class Task1Detection:
    TASK_ID = TaskID.task1
    MAX_STEPS = 10
    NUM_NODES = 40
    AVG_CONNECTIONS = 4
    INFECTION_THRESHOLD = 0.8  # High threshold = won't end early
    PRE_SPREAD_STEPS = 3
    ALLOWED_ACTIONS = {ActionType.inspect, ActionType.quarantine}

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.ground_truth_infected: list[str] = []
        self.step_count = 0
        self.done = False
        self.actions_this_step = 0  # Not used in task1 but needed by env
        self.ACTIONS_PER_STEP = self.MAX_STEPS  # Effectively unlimited per step

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.done = False

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD,
            topology="watts_strogatz",
            bot_fraction=0.08,
            recovery_steps=999,  # No recovery in task1 (frozen)
        )

        # Pre-spread then freeze
        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        # Record ground truth AFTER spread (this is what we grade against)
        self.ground_truth_infected = self.graph.get_infected_nodes()
        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        self.step_count += 1
        result = {"valid": False, "info": "", "penalty": 0.0}

        if action.action_type not in self.ALLOWED_ACTIONS:
            result["info"] = f"Action {action.action_type.value} not allowed in task1. Use: inspect, quarantine."
            return result

        if action.action_type == ActionType.inspect:
            details = self.graph.inspect_node(action.target_node_id)
            if details:
                result = {"valid": True, "info": details, "penalty": 0.0}
            else:
                result = {"valid": False, "info": f"Node {action.target_node_id} not found", "penalty": 0.0}

        elif action.action_type == ActionType.quarantine:
            outcome = self.graph.quarantine_node(action.target_node_id)
            if outcome["success"]:
                result = {
                    "valid": True,
                    "info": outcome["info"],
                    "was_infected": outcome["was_infected"],
                    "was_bot": outcome.get("was_bot", False),
                    "penalty": 0.0 if outcome["was_infected"] else 0.05,
                }
            else:
                result = {"valid": False, "info": outcome["reason"], "penalty": 0.0}

        if self.step_count >= self.MAX_STEPS:
            self.done = True

        return result

    def get_ground_truth(self) -> dict:
        return {
            "infected_nodes": self.ground_truth_infected,
            "origin_node": self.graph.origin_node_id,
            "total_infected": len(self.ground_truth_infected),
            "bot_nodes": self.graph.get_bot_nodes(),
        }