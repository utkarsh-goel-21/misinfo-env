from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID, NodeStatus


class Task3Containment:
    """
    HARD TASK — Full Containment

    Large network. Misinformation spreading live.
    Agent has limited actions per step (budget=3).
    Must contain spread below threshold for 20 steps.

    Agent can: inspect, trace, flag, quarantine,
               remove, restore

    Penalties:
    - Quarantining clean nodes: -0.05 per wrong node
    - Removing clean nodes: -0.10 per wrong node
    - Threshold breach: immediate failure

    Success: keep infection rate below threshold
             for all 20 steps AND correctly
             identify origin node.
    """

    TASK_ID = TaskID.task3
    MAX_STEPS = 20
    NUM_NODES = 80
    AVG_CONNECTIONS = 5
    INFECTION_THRESHOLD = 0.35
    PRE_SPREAD_STEPS = 3
    ACTIONS_PER_STEP = 3

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.actions_this_step = 0
        self.done = False
        self.origin_guess: str = ""
        self.wrong_quarantines: list[str] = []
        self.wrong_removals: list[str] = []
        self.threshold_breached_at: int = -1
        self.steps_below_threshold: int = 0

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.actions_this_step = 0
        self.done = False
        self.origin_guess = ""
        self.wrong_quarantines = []
        self.wrong_removals = []
        self.threshold_breached_at = -1
        self.steps_below_threshold = 0

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD
        )

        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        """
        Agent gets ACTIONS_PER_STEP actions per step.
        After budget exhausted spread advances.
        All 6 action types available.
        """
        result = {
            "valid": False,
            "info": "",
            "newly_infected": [],
            "penalty": 0.0,
            "actions_remaining": self.ACTIONS_PER_STEP - self.actions_this_step - 1
        }

        target = action.target_node_id

        if action.action_type.value == "inspect":
            details = self.graph.inspect_node(target)
            result.update({
                "valid": bool(details),
                "info": details if details else "node not found"
            })
            self.actions_this_step += 1

        elif action.action_type.value == "trace":
            trace_result = self.graph.trace_node(target)
            result.update({
                "valid": bool(trace_result),
                "info": trace_result if trace_result else "node not found"
            })
            self.actions_this_step += 1

        elif action.action_type.value == "flag":
            if target in self.graph.nodes:
                node = self.graph.nodes[target]
                # Check if submitting origin guess
                if action.reasoning and "origin" in action.reasoning.lower():
                    self.origin_guess = target
                    result.update({
                        "valid": True,
                        "info": f"origin guess recorded: {target}"
                    })
                else:
                    success = self.graph.flag_node(target)
                    result.update({
                        "valid": success,
                        "info": f"flagged {target}" if success else f"cannot flag {target}"
                    })
                self.actions_this_step += 1

        elif action.action_type.value == "quarantine":
            if target in self.graph.nodes:
                node = self.graph.nodes[target]
                was_clean = node.status == NodeStatus.clean
                success = self.graph.quarantine_node(target)
                if success and was_clean:
                    self.wrong_quarantines.append(target)
                    result.update({
                        "valid": True,
                        "info": f"quarantined {target} (warning: was clean)",
                        "penalty": 0.05
                    })
                else:
                    result.update({
                        "valid": success,
                        "info": f"quarantined {target}" if success else f"cannot quarantine {target}"
                    })
                self.actions_this_step += 1

        elif action.action_type.value == "remove":
            if target in self.graph.nodes:
                node = self.graph.nodes[target]
                was_clean = node.status == NodeStatus.clean
                success = self.graph.remove_node(target)
                if success and was_clean:
                    self.wrong_removals.append(target)
                    result.update({
                        "valid": True,
                        "info": f"removed {target} (warning: was clean)",
                        "penalty": 0.10
                    })
                else:
                    result.update({
                        "valid": success,
                        "info": f"removed {target}" if success else f"cannot remove {target}"
                    })
                self.actions_this_step += 1

        elif action.action_type.value == "restore":
            success = self.graph.restore_node(target)
            result.update({
                "valid": success,
                "info": f"restored {target}" if success else f"cannot restore {target}"
            })
            self.actions_this_step += 1

        # Advance spread when action budget exhausted
        if self.actions_this_step >= self.ACTIONS_PER_STEP:
            newly_infected = self.engine.step()
            result["newly_infected"] = newly_infected
            self.actions_this_step = 0
            self.step_count += 1

            if not self.graph.threshold_breached():
                self.steps_below_threshold += 1
            else:
                if self.threshold_breached_at == -1:
                    self.threshold_breached_at = self.step_count
                self.done = True

            if self.step_count >= self.MAX_STEPS:
                self.done = True

        return result

    def get_ground_truth(self) -> dict:
        origin = self.graph.origin_node_id
        return {
            "origin_node": origin,
            "agent_origin_guess": self.origin_guess,
            "wrong_quarantines": self.wrong_quarantines,
            "wrong_removals": self.wrong_removals,
            "threshold_breached_at": self.threshold_breached_at,
            "steps_below_threshold": self.steps_below_threshold,
            "final_infection_rate": self.graph.infection_rate(),
            "spread_report": self.engine.get_spread_report()
        }