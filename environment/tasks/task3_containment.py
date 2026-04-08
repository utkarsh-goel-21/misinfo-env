"""
SENTINEL-9 — Task 3: Containment (Hard)

Scenario: Large mixed-topology network under active adversarial attack.
Agent must simultaneously contain infection, identify bot clusters,
reconstruct causal chains, and manage limited resources — all while
an adversarial bot network reacts to the agent's actions.

Key Challenges:
- 5 actions per spread step — must prioritize strategically
- Full action space: inspect, trace, quarantine, remove, shadowban,
  deploy_counter_narrative, submit_causal_chain
- Budget system: each action costs money, episode ends at $0
- Streisand Effect: wrong quarantines trigger public outrage which
  makes bots harder to detect
- Content tiers 1-5 used, with stealth content appearing later
- Dynamic topology: users migrate away from quarantined nodes
- Bot evasion: inspecting near bots causes them to go dormant

Network: 150 nodes, Mixed (Watts-Strogatz + planted bridges)
Max Steps: 20
Actions Per Step: 5
Success: Containment ≥ 0.7 AND CIB precision ≥ 0.5
"""

from environment.graph import MisinformationGraph
from environment.spread import SpreadEngine
from environment.models import TaskID, ActionType


class Task3Containment:
    TASK_ID = TaskID.task3
    MAX_STEPS = 20
    NUM_NODES = 150
    AVG_CONNECTIONS = 5
    INFECTION_THRESHOLD = 0.35
    PRE_SPREAD_STEPS = 4
    ACTIONS_PER_STEP = 5
    INITIAL_BUDGET = 10000.0
    ALLOWED_ACTIONS = set(ActionType)  # All actions allowed

    # Action costs
    ACTION_COSTS = {
        ActionType.inspect: 50.0,
        ActionType.trace: 200.0,
        ActionType.shadowban: 500.0,
        ActionType.quarantine: 1500.0,
        ActionType.remove: 3000.0,
        ActionType.deploy_counter_narrative: 4000.0,
        ActionType.submit_causal_chain: 0.0,
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = MisinformationGraph(seed=seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.actions_this_step = 0
        self.done = False
        self.wrong_quarantines = 0
        self.wrong_removals = 0
        self.steps_below_threshold = 0
        self.submitted_chain = []
        self.budget = self.INITIAL_BUDGET
        self.outrage = 0.0
        self.action_history: list[dict] = []

    def reset(self):
        self.graph = MisinformationGraph(seed=self.seed)
        self.engine = SpreadEngine(self.graph)
        self.step_count = 0
        self.actions_this_step = 0
        self.done = False
        self.wrong_quarantines = 0
        self.wrong_removals = 0
        self.steps_below_threshold = 0
        self.submitted_chain = []
        self.budget = self.INITIAL_BUDGET
        self.outrage = 0.0
        self.action_history = []

        self.graph.build_from_config(
            num_nodes=self.NUM_NODES,
            avg_connections=self.AVG_CONNECTIONS,
            infection_threshold=self.INFECTION_THRESHOLD,
            topology="mixed",
            bot_fraction=0.12,
            recovery_steps=6,
        )

        for _ in range(self.PRE_SPREAD_STEPS):
            self.engine.step()

        return self.graph.snapshot(hide_origin=True)

    def apply_action(self, action) -> dict:
        result = {"valid": False, "info": "", "penalty": 0.0, "newly_infected": [],
                  "was_infected": None, "was_bot": None, "budget_remaining": self.budget}

        target = action.target_node_id

        # Budget check
        cost = self.ACTION_COSTS.get(action.action_type, 100.0)
        if self.budget < cost:
            self.done = True
            result["info"] = f"Budget depleted. Need ${cost:.0f}, have ${self.budget:.0f}"
            return result

        self.budget -= cost

        # ── Execute Action ──
        if action.action_type == ActionType.inspect:
            details = self.graph.inspect_node(target)
            if details:
                result.update({"valid": True, "info": details})
            else:
                result.update({"info": f"Node {target} not found"})
                self.budget += cost  # Refund if invalid

        elif action.action_type == ActionType.trace:
            details = self.graph.trace_node(target)
            if details:
                result.update({"valid": True, "info": details})
            else:
                result.update({"info": f"Node {target} not found"})
                self.budget += cost

        elif action.action_type == ActionType.quarantine:
            outcome = self.graph.quarantine_node(target)
            if outcome["success"]:
                result.update({
                    "valid": True, "info": outcome["info"],
                    "was_infected": outcome["was_infected"],
                    "was_bot": outcome.get("was_bot", False),
                })
                if not outcome["was_infected"]:
                    self.wrong_quarantines += 1
                    result["penalty"] = 0.05
                    # Streisand Effect
                    self.outrage = min(1.0, self.outrage + 0.15)
            else:
                result.update({"info": outcome["reason"]})
                self.budget += cost

        elif action.action_type == ActionType.remove:
            outcome = self.graph.remove_node(target)
            if outcome["success"]:
                result.update({
                    "valid": True, "info": outcome["info"],
                    "was_infected": outcome["was_infected"],
                    "was_bot": outcome.get("was_bot", False),
                })
                if not outcome["was_infected"]:
                    self.wrong_removals += 1
                    result["penalty"] = 0.10
                    self.outrage = min(1.0, self.outrage + 0.30)
            else:
                result.update({"info": outcome["reason"]})
                self.budget += cost

        elif action.action_type == ActionType.shadowban:
            outcome = self.graph.shadowban_node(target)
            if outcome["success"]:
                result.update({"valid": True, "info": outcome["info"]})
            else:
                result.update({"info": outcome["reason"]})
                self.budget += cost

        elif action.action_type == ActionType.deploy_counter_narrative:
            outcome = self.graph.deploy_counter_narrative(target)  # target = community_id
            if outcome["success"]:
                result.update({"valid": True, "info": outcome["info"]})
            else:
                result.update({"info": outcome["reason"]})
                self.budget += cost

        elif action.action_type == ActionType.submit_causal_chain:
            if action.causal_chain:
                self.submitted_chain = action.causal_chain
                result.update({"valid": True, "info": "Causal chain submitted."})
                self.done = True
            else:
                result.update({"info": "Empty causal chain"})

        # Track action
        self.action_history.append({
            "step": self.step_count,
            "action": action.action_type.value,
            "target": target,
            "valid": result["valid"],
        })

        result["budget_remaining"] = self.budget
        self.actions_this_step += 1

        # ── Adversarial: Bot evasion when outrage is high ──
        if self.outrage > 0.5:
            self.graph.activate_bot_evasion(self.outrage)

        # ── Advance Spread when action budget exhausted ──
        if self.actions_this_step >= self.ACTIONS_PER_STEP:
            # Dynamic topology shift every 3 spread steps
            if self.step_count > 0 and self.step_count % 3 == 0:
                self.graph.remap_edges(migration_rate=0.1)

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
            "actual_causal_tree": self.graph.get_causal_tree_as_dicts(),
            "steps_below_threshold": self.steps_below_threshold,
            "wrong_quarantines": self.wrong_quarantines,
            "wrong_removals": self.wrong_removals,
            "final_infection_rate": self.graph.infection_rate(),
            "spread_report": self.engine.get_spread_report(),
            "bot_nodes": self.graph.get_bot_nodes(),
            "budget_remaining": self.budget,
            "outrage": self.outrage,
            "total_actions": len(self.action_history),
        }