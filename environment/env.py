"""
SENTINEL-9 — Core Environment (OpenEnv Interface)

Implements the full OpenEnv interface: reset(), step(), state(), close()
with proper POMDP mechanics:
- Fog-of-war: agent only sees nodes it has inspected
- Noisy stream reports: 30% false positive rate
- Proper Brier scoring: (confidence - actual_outcome)²
- Financial budget tracking
- Streisand effect / public outrage
- Dynamic topology remapping
"""

from environment.models import (
    Action, ActionType, EnvironmentState, Observation,
    Reward, TaskID, NodeStatus
)
from environment.scoring import clamp_openenv_score
from environment.tasks.task1_detection import Task1Detection
from environment.tasks.task2_tracing import Task2Tracing
from environment.tasks.task3_containment import Task3Containment
from environment.graders.grader1 import Grader1
from environment.graders.grader2 import Grader2
from environment.graders.grader3 import Grader3

import random


class MisinfoEnv:
    """
    Misinformation Containment Network Environment
    OpenEnv-compatible POMDP with adversarial dynamics.
    """

    VERSION = "2.0.0"
    ENV_NAME = "misinfo-containment-env"

    TASK_ALLOWED_ACTIONS = {
        TaskID.task1: {ActionType.inspect, ActionType.quarantine},
        TaskID.task2: {ActionType.inspect, ActionType.trace, ActionType.quarantine, ActionType.submit_causal_chain},
        TaskID.task3: set(ActionType),  # All actions
    }

    ACTION_COSTS = {
        ActionType.inspect: 50.0,
        ActionType.trace: 200.0,
        ActionType.shadowban: 500.0,
        ActionType.quarantine: 1500.0,
        ActionType.remove: 3000.0,
        ActionType.deploy_counter_narrative: 4000.0,
        ActionType.submit_causal_chain: 0.0,
    }

    def __init__(self, task_id: str = "task1_detection", seed: int = 42):
        self.seed = seed
        self.task_id = TaskID(task_id)
        self.task = None
        self.grader = None
        self.current_observation = None
        self.cumulative_penalty = 0.0
        self.cumulative_score = 0.0
        self.prev_score = 0.0
        self.financial_budget = 10000.0
        self.public_outrage_index = 0.0
        self.step_rewards: list[float] = []
        self.brier_scores: list[float] = []
        self.revealed_nodes: set[str] = set()  # Fog-of-war
        self.actions_taken: list[dict] = []
        self.done = False
        self.rng = random.Random(seed)
        self._setup_task()

    def _setup_task(self):
        if self.task_id == TaskID.task1:
            self.task = Task1Detection(seed=self.seed)
            self.grader = Grader1()
        elif self.task_id == TaskID.task2:
            self.task = Task2Tracing(seed=self.seed)
            self.grader = Grader2()
        elif self.task_id == TaskID.task3:
            self.task = Task3Containment(seed=self.seed)
            self.grader = Grader3()
        else:
            raise ValueError(f"Unknown task_id: {self.task_id}")

    # ═══════════════════════════════════════
    # OPENENV INTERFACE
    # ═══════════════════════════════════════

    def reset(self) -> Observation:
        self.cumulative_penalty = 0.0
        self.cumulative_score = 0.0
        self.prev_score = 0.0
        self.financial_budget = 10000.0
        self.public_outrage_index = 0.0
        self.step_rewards = []
        self.brier_scores = []
        self.revealed_nodes = set()
        self.actions_taken = []
        self.done = False

        self._setup_task()
        self.task.reset()

        # Generate initial stream reports with noise
        stream = self._generate_stream_reports()

        self.current_observation = self._build_observation(
            stream_reports=stream,
            inspection_results=None,
            message=self._initial_message(),
        )
        return self.current_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # ── Validate ──
        validation_error = self._validate_action(action)
        if validation_error:
            return (
                self.current_observation,
                self._error_reward(validation_error),
                self.done,
                {"error": validation_error},
            )

        # ── Budget (only for Task 3) ──
        if self.task_id == TaskID.task3:
            cost = self.ACTION_COSTS.get(action.action_type, 100.0)
            if self.financial_budget < cost:
                self.done = True
                return (
                    self.current_observation,
                    self._error_reward(f"Budget depleted. Need ${cost:.0f}, have ${self.financial_budget:.0f}"),
                    True,
                    {"error": "budget_depleted"},
                )
            self.financial_budget -= cost

        # ── Execute ──
        result = self.task.apply_action(action)
        penalty = result.get("penalty", 0.0)
        self.cumulative_penalty += penalty

        # ── Brier Scoring ──
        brier = self._compute_brier(action, result)
        self.brier_scores.append(brier)

        # ── Fog-of-war update ──
        if action.action_type in (ActionType.inspect, ActionType.trace):
            if action.target_node_id:
                self.revealed_nodes.add(action.target_node_id)
                # Inspecting also reveals neighbor IDs
                node = self.task.graph.nodes.get(action.target_node_id)
                if node:
                    for nb in node.neighbors:
                        self.revealed_nodes.add(nb)

        # ── Streisand Effect (Task 3) ──
        if self.task_id == TaskID.task3:
            if action.action_type == ActionType.quarantine and not result.get("was_infected", True):
                self.public_outrage_index = min(1.0, self.public_outrage_index + 0.15)
            elif action.action_type == ActionType.remove and not result.get("was_infected", True):
                self.public_outrage_index = min(1.0, self.public_outrage_index + 0.30)

            if self.public_outrage_index > 0.5:
                self.task.graph.activate_bot_evasion(self.public_outrage_index)

        # ── Track ──
        self.actions_taken.append({
            "action": action.action_type.value,
            "target": action.target_node_id,
            "confidence": action.confidence,
            "valid": result.get("valid", False),
            "brier": brier,
        })

        self.done = self.task.done

        # ── Reward ──
        if self.done:
            reward = self._compute_final_reward()
        else:
            reward = self._compute_step_reward(action, result)

        self.step_rewards.append(reward.score)
        self.prev_score = self.cumulative_score
        self.cumulative_score = reward.score

        # ── Build next observation ──
        inspection_results = None
        if action.action_type in (ActionType.inspect, ActionType.trace) and result.get("valid"):
            inspection_results = {action.target_node_id: result.get("info")}

        self.current_observation = self._build_observation(
            stream_reports=self._generate_stream_reports(),
            inspection_results=inspection_results,
            message=self._build_step_message(action, result),
        )

        info = {
            "action_valid": result.get("valid", False),
            "newly_infected": result.get("newly_infected", []),
            "infection_rate": self.task.graph.infection_rate(),
            "step": self.task.step_count,
            "brier_this_step": brier,
            "budget": self.financial_budget,
            "outrage": self.public_outrage_index,
        }

        return (self.current_observation, reward, self.done, info)

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_id=self.task_id,
            step_number=self.task.step_count,
            network=self.task.graph.snapshot(hide_origin=False),
            origin_node_id=self.task.graph.origin_node_id,
            infected_nodes=self.task.graph.get_infected_nodes(),
            quarantined_nodes=self.task.graph.get_quarantined_nodes(),
            recovered_nodes=self.task.graph.get_recovered_nodes(),
            removed_nodes=self.task.graph.get_nodes_by_status(NodeStatus.removed),
            bot_nodes=self.task.graph.get_bot_nodes(),
            causal_tree=self.task.graph.get_causal_tree_as_dicts(),
            actions_taken=self.actions_taken,
            cumulative_score=self.cumulative_score,
            financial_budget=self.financial_budget,
            public_outrage_index=self.public_outrage_index,
            brier_scores=self.brier_scores,
            done=self.done,
        )

    def close(self):
        """Clean up resources. Required by OpenEnv lifespan."""
        self.task = None
        self.grader = None
        self.current_observation = None

    # ═══════════════════════════════════════
    # POMDP: PARTIAL OBSERVABILITY
    # ═══════════════════════════════════════

    def _generate_stream_reports(self) -> list[str]:
        """
        Platform heuristic reports with ~30% false positive rate.
        Returns a mix of truly infected + some clean nodes.
        """
        infected = self.task.graph.get_infected_nodes()
        clean = [
            nid for nid, n in self.task.graph.nodes.items()
            if n.status == NodeStatus.clean
        ]

        # Pick some truly infected nodes
        n_real = min(len(infected), max(2, len(infected) // 2))
        real_flags = self.rng.sample(infected, min(n_real, len(infected))) if infected else []

        # Add false positives (~30% of total reports)
        n_false = max(1, int(len(real_flags) * 0.43))  # 30% of total = 43% of real
        false_flags = self.rng.sample(clean, min(n_false, len(clean))) if clean else []

        reports = real_flags + false_flags
        self.rng.shuffle(reports)
        return reports

    def _build_observation(
        self, stream_reports: list[str],
        inspection_results: dict | None, message: str
    ) -> Observation:
        actions_remaining = None
        if self.task_id == TaskID.task3:
            actions_remaining = self.task.ACTIONS_PER_STEP - self.task.actions_this_step

        return Observation(
            task_id=self.task_id,
            step_number=self.task.step_count,
            max_steps=self.task.MAX_STEPS,
            actions_remaining=actions_remaining,
            stream_reports=stream_reports,
            revealed_nodes=sorted(self.revealed_nodes),
            inspection_results=inspection_results,
            agent_message=message,
            financial_budget=self.financial_budget,
            public_outrage_index=self.public_outrage_index,
            infection_rate=self.task.graph.infection_rate(),
            brier_score_running=sum(self.brier_scores) / max(1, len(self.brier_scores)),
            network_size=len(self.task.graph.nodes),
        )

    # ═══════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════

    def _validate_action(self, action: Action) -> str:
        # Confidence check
        if action.confidence is None or not (0.0 <= action.confidence <= 1.0):
            return "Confidence must be between 0.0 and 1.0 (required for Brier scoring)."

        # Task-specific action restriction
        allowed = self.TASK_ALLOWED_ACTIONS.get(self.task_id, set())
        if action.action_type not in allowed:
            return (
                f"Action '{action.action_type.value}' not allowed in "
                f"{self.task_id.value}. Allowed: {[a.value for a in allowed]}"
            )

        # Target validation
        if action.action_type == ActionType.submit_causal_chain:
            if not action.causal_chain:
                return "submit_causal_chain requires 'causal_chain' parameter."
        elif action.action_type == ActionType.deploy_counter_narrative:
            if not action.target_node_id:
                return "deploy_counter_narrative requires target community_id."
        else:
            if not action.target_node_id:
                return "target_node_id is required."
            if action.target_node_id not in self.task.graph.nodes:
                return f"Node '{action.target_node_id}' does not exist in the network."

        return ""

    # ═══════════════════════════════════════
    # BRIER SCORING
    # ═══════════════════════════════════════

    def _compute_brier(self, action: Action, result: dict) -> float:
        """
        Proper Brier score: (confidence - actual_outcome)²
        actual_outcome = 1.0 if the action was correct, 0.0 if wrong.
        Lower = better calibrated.
        """
        if action.action_type in (ActionType.inspect, ActionType.trace):
            # Information-gathering actions: outcome = 1.0 if valid
            outcome = 1.0 if result.get("valid", False) else 0.0
        elif action.action_type in (ActionType.quarantine, ActionType.remove):
            # Intervention actions: outcome = 1.0 if target was infected
            outcome = 1.0 if result.get("was_infected", False) else 0.0
        elif action.action_type == ActionType.submit_causal_chain:
            return 0.0  # Don't Brier-score chain submissions
        else:
            outcome = 1.0 if result.get("valid", False) else 0.0

        brier = (action.confidence - outcome) ** 2
        return round(brier, 4)

    # ═══════════════════════════════════════
    # REWARDS
    # ═══════════════════════════════════════

    def _compute_final_reward(self) -> Reward:
        return self.grader.grade(self.task, self.cumulative_penalty, self.brier_scores)

    def _compute_step_reward(self, action: Action, result: dict) -> Reward:
        """Informative intermediate reward signal."""
        valid = result.get("valid", False)
        infection_rate = self.task.graph.infection_rate()

        # Base: small reward for valid actions, penalty for invalid
        base = 0.02 if valid else -0.02

        # Bonus for correct interventions
        if action.action_type in (ActionType.quarantine, ActionType.remove):
            if result.get("was_infected", False):
                base += 0.05  # Correct intervention
            else:
                base -= 0.08  # Wrong target

        # Brier penalty for this step
        brier = self.brier_scores[-1] if self.brier_scores else 0.0
        brier_penalty = brier * 0.1

        # Infection pressure signal
        threshold = self.task.graph.infection_threshold
        pressure = max(0, infection_rate / threshold - 0.5) * 0.1

        step_score = clamp_openenv_score(base - brier_penalty - pressure)

        return Reward(
            score=step_score,
            delta=round(step_score - self.prev_score, 4),
            done=False,
            success=False,
            partial_credits={
                "action_valid": valid,
                "brier_this_step": brier,
                "infection_rate": infection_rate,
            },
            penalty=brier_penalty,
            feedback=f"Infection: {infection_rate:.1%}. Brier: {brier:.3f}.",
        )

    def _error_reward(self, error: str) -> Reward:
        return Reward(
            score=clamp_openenv_score(0.0),
            delta=0.0,
            done=False,
            success=False,
            partial_credits={},
            penalty=0.1,
            feedback=f"Invalid: {error}",
        )

    # ═══════════════════════════════════════
    # MESSAGES
    # ═══════════════════════════════════════

    def _initial_message(self) -> str:
        n_infected = len(self.task.graph.get_infected_nodes())
        n_total = len(self.task.graph.nodes)
        return (
            f"SENTINEL-9 initialized. Network: {n_total} nodes. "
            f"Initial infection detected in {n_infected} nodes ({n_infected/n_total:.0%}). "
            f"Stream reports incoming — WARNING: ~30% false positive rate. "
            f"Verify via inspect before taking costly actions."
        )

    def _build_step_message(self, action: Action, result: dict) -> str:
        valid = "✓" if result.get("valid") else "✗"
        rate = self.task.graph.infection_rate()
        msg = f"[{valid}] {action.action_type.value}"
        if action.target_node_id:
            msg += f"({action.target_node_id})"
        msg += f" | Infection: {rate:.1%}"
        if self.task_id == TaskID.task3:
            msg += f" | Budget: ${self.financial_budget:,.0f} | Outrage: {self.public_outrage_index:.2f}"
        newly = result.get("newly_infected", [])
        if newly:
            msg += f" | ⚠ {len(newly)} new infections this cycle"
        return msg
