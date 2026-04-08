"""
SENTINEL-9 — Task 2: Tracing (Medium)

Scenario: Misinformation is ACTIVELY SPREADING through a Barabási-Albert
scale-free network. Agent must reconstruct the causal infection chain by
analyzing infection timestamps, centrality metrics, and content semantics.

Key Challenge:
- Spread continues during agent actions (1 spread step per 3 agent actions)
- Must find origin node AND reconstruct the full causal path
- trace action reveals temporal ordering of neighbor infections
- Partial credit via Graph Edit Distance
- Content tiers 1-4 used (blatant → sophisticated)

Network: 80 nodes, Barabási-Albert (m=2)
Max Steps: 15
Allowed Actions: inspect, trace, quarantine, submit_causal_chain
Success: Exact origin identified AND GED < 3
"""

from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID, ActionType


class Task2Tracing:
    TASK_ID = TaskID.task2
    MAX_STEPS = 15
    NUM_NODES = 80
    AVG_CONNECTIONS = 4
    INFECTION_THRESHOLD = 0.5
    PRE_SPREAD_STEPS = 4
    SPREAD_EVERY_N_ACTIONS = 3  # Spread advances every 3 agent actions
    ALLOWED_ACTIONS = {ActionType.inspect, ActionType.trace, ActionType.quarantine, ActionType.submit_causal_chain}

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.action_count = 0
        self.done = False
        self.submitted_chain = []
        self.actions_this_step = 0
        self.ACTIONS_PER_STEP = self.MAX_STEPS

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.action_count = 0
        self.done = False
        self.submitted_chain = []

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD,
            topology="barabasi_albert",
            bot_fraction=0.05,
            recovery_steps=8,
        )

        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        self.action_count += 1
        result = {"valid": False, "info": "", "penalty": 0.0, "newly_infected": []}

        if action.action_type not in self.ALLOWED_ACTIONS:
            result["info"] = f"Action {action.action_type.value} not allowed in task2."
            return result

        if action.action_type == ActionType.inspect:
            details = self.graph.inspect_node(action.target_node_id)
            if details:
                result = {"valid": True, "info": details, "penalty": 0.0, "newly_infected": []}
            else:
                result = {"valid": False, "info": "Node not found", "penalty": 0.0, "newly_infected": []}

        elif action.action_type == ActionType.trace:
            details = self.graph.trace_node(action.target_node_id)
            if details:
                result = {"valid": True, "info": details, "penalty": 0.0, "newly_infected": []}
            else:
                result = {"valid": False, "info": "Node not found", "penalty": 0.0, "newly_infected": []}

        elif action.action_type == ActionType.quarantine:
            outcome = self.graph.quarantine_node(action.target_node_id)
            if outcome["success"]:
                result = {
                    "valid": True, "info": outcome["info"],
                    "was_infected": outcome["was_infected"],
                    "penalty": 0.0 if outcome["was_infected"] else 0.05,
                    "newly_infected": [],
                }
            else:
                result = {"valid": False, "info": outcome["reason"], "penalty": 0.0, "newly_infected": []}

        elif action.action_type == ActionType.submit_causal_chain:
            if action.causal_chain:
                self.submitted_chain = action.causal_chain
                result = {"valid": True, "info": "Causal chain submitted.", "penalty": 0.0, "newly_infected": []}
                self.done = True
            else:
                result = {"valid": False, "info": "Empty causal chain", "penalty": 0.0, "newly_infected": []}

        # Spread advances every N actions (active spread during investigation)
        if self.action_count % self.SPREAD_EVERY_N_ACTIONS == 0:
            newly_infected = self.engine.step()
            result["newly_infected"] = newly_infected
            self.step_count += 1

            if self.graph.threshold_breached():
                self.done = True

        if self.step_count >= self.MAX_STEPS:
            self.done = True

        return result

    def get_ground_truth(self) -> dict:
        return {
            "origin_node": self.graph.origin_node_id,
            "submitted_chain": self.submitted_chain,
            "actual_causal_tree": self.graph.get_causal_tree_as_dicts(),
            "actual_history": self.engine.spread_history,
            "final_infection_rate": self.graph.infection_rate(),
            "threshold_breached": self.graph.threshold_breached(),
        }