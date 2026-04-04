from environment.models import (
    Action,
    ActionType,
    EnvironmentState,
    Observation,
    Reward,
    TaskID,
    NodeStatus
)
from environment.tasks.task1_detection import Task1Detection
from environment.tasks.task2_tracing import Task2Tracing
from environment.tasks.task3_containment import Task3Containment
from environment.graders.grader1 import Grader1
from environment.graders.grader2 import Grader2
from environment.graders.grader3 import Grader3


class MisinfoEnv:
    """
    Misinformation Containment Network Environment

    Implements the full OpenEnv interface:
    - reset() → Observation
    - step(action) → tuple[Observation, Reward, bool, dict]
    - state() → EnvironmentState

    Three tasks available:
    - task1_detection  (easy)
    - task2_tracing    (medium)
    - task3_containment (hard)
    """

    VERSION = "1.0.0"
    ENV_NAME = "misinfo-containment-env"

    def __init__(
        self,
        task_id: str = "task1_detection",
        seed: int = 42
    ):
        self.seed = seed
        self.task_id = TaskID(task_id)
        self.task = None
        self.grader = None
        self.current_observation = None
        self.cumulative_penalty = 0.0
        self.cumulative_score = 0.0
        self.step_rewards: list[float] = []
        self.done = False
        self._setup_task()

    # ─────────────────────────────────────────
    # TASK SETUP
    # ─────────────────────────────────────────

    def _setup_task(self):
        """Initialize correct task and grader."""
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
            raise ValueError(
                f"Unknown task_id: {self.task_id}. "
                f"Must be one of: task1_detection, "
                f"task2_tracing, task3_containment"
            )

    # ─────────────────────────────────────────
    # OPENENV INTERFACE
    # ─────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset environment to initial state.
        Returns initial Observation.
        """
        self.cumulative_penalty = 0.0
        self.cumulative_score = 0.0
        self.step_rewards = []
        self.done = False

        self._setup_task()
        network_snapshot = self.task.reset()

        self.current_observation = self._build_observation(
            network_snapshot=network_snapshot,
            recently_infected=[network_snapshot.origin_node_id]
            if network_snapshot.origin_node_id
            else [],
            message=self._initial_message()
        )

        return self.current_observation

    def step(
        self,
        action: Action
    ) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one action in the environment.

        Returns:
            observation: updated network state
            reward: score, partial credits, feedback
            done: whether episode is complete
            info: additional metadata
        """
        if self.done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )

        # Validate action
        validation_error = self._validate_action(action)
        if validation_error:
            return (
                self.current_observation,
                self._invalid_action_reward(validation_error),
                self.done,
                {"error": validation_error}
            )

        # Apply action to task
        result = self.task.apply_action(action)

        # Update cumulative penalty
        self.cumulative_penalty += result.get("penalty", 0.0)

        # Check if task is done
        self.done = self.task.done

        # Build reward
        if self.done:
            reward = self._compute_final_reward()
        else:
            reward = self._compute_step_reward(result)

        # Track rewards
        self.step_rewards.append(reward.score)
        self.cumulative_score = reward.score

        # Build new observation
        network_snapshot = self.task.graph.snapshot(
            hide_origin=True
        )
        self.current_observation = self._build_observation(
            network_snapshot=network_snapshot,
            recently_infected=result.get("newly_infected", []),
            message=self._build_step_message(action, result)
        )

        info = {
            "action_valid": result.get("valid", False),
            "action_info": result.get("info", ""),
            "newly_infected": result.get("newly_infected", []),
            "infection_rate": self.task.graph.infection_rate(),
            "step": self.task.step_count,
            "cumulative_penalty": self.cumulative_penalty
        }

        return (
            self.current_observation,
            reward,
            self.done,
            info
        )

    def state(self) -> EnvironmentState:
        """
        Returns full internal environment state.
        Includes ground truth hidden from agent.
        """
        return EnvironmentState(
            task_id=self.task_id,
            step_number=self.task.step_count,
            network=self.task.graph.snapshot(hide_origin=False),
            origin_node_id=self.task.graph.origin_node_id,
            infected_nodes=self.task.graph.get_infected_nodes(),
            quarantined_nodes=self.task.graph.get_quarantined_nodes(),
            flagged_nodes=self.task.graph.get_flagged_nodes(),
            removed_nodes=self.task.graph.get_removed_nodes(),
            actions_taken=getattr(
                self.task, "actions_taken", []
            ),
            cumulative_score=self.cumulative_score,
            done=self.done
        )

    # ─────────────────────────────────────────
    # REWARD COMPUTATION
    # ─────────────────────────────────────────

    def _compute_final_reward(self) -> Reward:
        """
        Compute final reward at episode end.
        Delegates to task specific grader.
        """
        if self.task_id == TaskID.task1:
            return self.grader.grade(
                self.task,
                self.cumulative_penalty
            )
        elif self.task_id == TaskID.task2:
            return self.grader.grade(
                self.task,
                self.cumulative_penalty
            )
        elif self.task_id == TaskID.task3:
            return self.grader.grade(self.task)

    def _compute_step_reward(self, result: dict) -> Reward:
        """
        Compute intermediate reward during episode.
        Provides partial progress signal to agent.
        """
        infected = self.task.graph.get_infected_nodes()
        total = len(self.task.graph.nodes)
        infection_rate = self.task.graph.infection_rate()
        threshold = self.task.graph.infection_threshold

        # Base signal — lower infection rate = better
        containment_signal = max(
            0.0,
            1.0 - (infection_rate / threshold)
        )

        # Action quality signal
        action_bonus = 0.05 if result.get("valid") else -0.05

        # Penalty from this step
        step_penalty = result.get("penalty", 0.0)

        raw = containment_signal * 0.5 + action_bonus - step_penalty
        step_score = round(max(0.0, min(1.0, raw)), 4)

        return Reward(
            score=step_score,
            delta=step_score - (
                self.step_rewards[-1]
                if self.step_rewards else 0.0
            ),
            done=False,
            success=False,
            partial_credits={
                "containment_signal": round(containment_signal, 4),
                "action_bonus": action_bonus,
                "step_penalty": step_penalty,
                "infection_rate": round(infection_rate, 4),
                "infected_count": len(infected),
                "total_nodes": total
            },
            penalty=step_penalty,
            feedback=(
                f"Step reward. Infection rate: "
                f"{infection_rate:.2%} / threshold: "
                f"{threshold:.2%}. "
                f"Action {'valid' if result.get('valid') else 'invalid'}."
            )
        )

    def _invalid_action_reward(
        self,
        error: str
    ) -> Reward:
        """Reward for invalid action."""
        return Reward(
            score=0.0,
            delta=0.0,
            done=False,
            success=False,
            partial_credits={},
            penalty=0.05,
            feedback=f"Invalid action: {error}"
        )

    # ─────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────

    def _validate_action(self, action: Action) -> str:
        """
        Validate action before applying.
        Returns error string if invalid, empty string if valid.
        """
        # Check target node exists
        if action.target_node_id not in self.task.graph.nodes:
            return (
                f"Node {action.target_node_id} does not exist "
                f"in the network."
            )

        # Check action type is valid enum
        valid_types = [a.value for a in ActionType]
        if action.action_type.value not in valid_types:
            return (
                f"Action type {action.action_type} is not valid. "
                f"Must be one of: {valid_types}"
            )

        # Task 1 only allows inspect and flag
        if self.task_id == TaskID.task1:
            if action.action_type.value not in ["inspect", "flag"]:
                return (
                    f"Task 1 only allows inspect and flag actions. "
                    f"Got: {action.action_type.value}"
                )

        # Task 2 only allows inspect, trace, flag
        if self.task_id == TaskID.task2:
            if action.action_type.value not in [
                "inspect", "trace", "flag"
            ]:
                return (
                    f"Task 2 only allows inspect, trace, flag. "
                    f"Got: {action.action_type.value}"
                )

        return ""

    # ─────────────────────────────────────────
    # OBSERVATION BUILDER
    # ─────────────────────────────────────────

    def _build_observation(
        self,
        network_snapshot,
        recently_infected: list[str],
        message: str
    ) -> Observation:
        """Build Observation from current state."""
        actions_remaining = None
        if self.task_id == TaskID.task3:
            actions_remaining = (
                Task3Containment.ACTIONS_PER_STEP
                - self.task.actions_this_step
            )

        max_steps = self.task.MAX_STEPS

        return Observation(
            task_id=self.task_id,
            step_number=self.task.step_count,
            max_steps=max_steps,
            actions_remaining=actions_remaining,
            network=network_snapshot,
            recently_infected=recently_infected,
            agent_message=message
        )

    # ─────────────────────────────────────────
    # MESSAGES
    # ─────────────────────────────────────────

    def _initial_message(self) -> str:
        messages = {
            TaskID.task1: (
                "TASK 1 — DETECTION: Misinformation has spread "
                "through this network. Inspect nodes and flag "
                "all infected ones. You have 10 steps. "
                "Spread is frozen — focus on detection."
            ),
            TaskID.task2: (
                "TASK 2 — TRACING: Misinformation is actively "
                "spreading. Use inspect and trace actions to "
                "find the origin node, then submit your guess "
                "using the flag action. You have 15 steps."
            ),
            TaskID.task3: (
                "TASK 3 — CONTAINMENT: Large network under active "
                "attack. You have 3 actions per step before spread "
                "advances. Contain infection below threshold for "
                "20 steps AND identify origin node. "
                "Penalized for wrong quarantines and removals."
            )
        }
        return messages[self.task_id]

    def _build_step_message(
        self,
        action: Action,
        result: dict
    ) -> str:
        rate = self.task.graph.infection_rate()
        threshold = self.task.graph.infection_threshold
        newly = result.get("newly_infected", [])
        return (
            f"Action: {action.action_type.value} on "
            f"{action.target_node_id}. "
            f"Result: {result.get('info', '')}. "
            f"Infection rate: {rate:.2%}/{threshold:.2%}. "
            f"Newly infected this step: {len(newly)} nodes."
        )

    # ─────────────────────────────────────────
    # UTILITY
    # ─────────────────────────────────────────

    def list_tasks(self) -> list[dict]:
        """List all available tasks."""
        return [
            {
                "task_id": "task1_detection",
                "difficulty": "easy",
                "description": "Identify all infected nodes",
                "max_steps": 10,
                "network_size": 20
            },
            {
                "task_id": "task2_tracing",
                "difficulty": "medium",
                "description": "Trace and identify origin node",
                "max_steps": 15,
                "network_size": 40
            },
            {
                "task_id": "task3_containment",
                "difficulty": "hard",
                "description": (
                    "Contain spread below threshold "
                    "and identify origin"
                ),
                "max_steps": 20,
                "network_size": 80
            }
        ]

    def close(self):
        """Clean up environment resources."""
        self.task = None
        self.grader = None
        self.current_observation = None