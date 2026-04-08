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
    Misinformation Containment Network Environment - V2
    (Causal Inference & Epistemic Uncertainty POMDP)
    """

    VERSION = "2.0.0"
    ENV_NAME = "misinfo-containment-env"

    def __init__(self, task_id: str = "task1_detection", seed: int = 42):
        self.seed = seed
        self.task_id = TaskID(task_id)
        self.task = None
        self.grader = None
        self.current_observation = None
        self.cumulative_penalty = 0.0
        self.cumulative_score = 0.0
        self.financial_budget = 10000.0
        self.public_outrage_index = 0.0
        self.step_rewards: list[float] = []
        self.done = False
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

    # ─────────────────────────────────────────
    # OPENENV INTERFACE
    # ─────────────────────────────────────────

    def reset(self) -> Observation:
        self.cumulative_penalty = 0.0
        self.cumulative_score = 0.0
        self.financial_budget = 10000.0
        self.public_outrage_index = 0.0
        self.step_rewards = []
        self.done = False

        self._setup_task()
        self.task.reset()

        self.current_observation = self._build_observation(
            stream_reports=self._generate_stream_reports(),
            inspection_results=None,
            message=self._initial_message()
        )
        return self.current_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        validation_error = self._validate_action(action)
        if validation_error:
            return (
                self.current_observation,
                self._invalid_action_reward(validation_error),
                self.done,
                {"error": validation_error}
            )

        # ACTION COSTS
        costs = {
            "inspect": 50.0,
            "trace": 200.0,
            "shadowban": 500.0,
            "quarantine": 1500.0,
            "remove": 3000.0,
            "deploy_counter_narrative": 4000.0,
            "submit_causal_chain": 0.0
        }
        cost = costs.get(action.action_type.value, 100.0)
        self.financial_budget -= cost
        
        if self.financial_budget < 0:
            self.done = True
            result = {"valid": False, "penalty": 1.0, "info": "Budget depleted."}
        else:
            # Custom Action execution logic mapped over Task engine
            result = self.task.apply_action(action)
            self.cumulative_penalty += result.get("penalty", 0.0)
            
            # Adv. Streisand Mechanics
            if action.action_type == ActionType.quarantine and not result.get("was_infected", True):
                self.public_outrage_index = min(1.0, self.public_outrage_index + 0.15)
            elif action.action_type == ActionType.remove and not result.get("was_infected", True):
                self.public_outrage_index = min(1.0, self.public_outrage_index + 0.3)
                
            # If outrage high, mutate bots (Adversarial Reaction)
            if self.public_outrage_index > 0.6:
                for n in self.task.graph.nodes.values():
                    if n.is_bot:
                        n.community_id = "hidden_" + str(self.task.graph.step)
                        n.user_persona = "Evading detection. Persona shifted."

            # Every 3 steps, dynamic topology shifts (Users migrate)
            if self.task.step_count > 0 and self.task.step_count % 3 == 0:
                self.task.graph.remap_edges(migration_rate=0.1)
                
            self.done = self.task.done

        # If they inspected or traced, pass that to next obs
        inspection_results = None
        if action.action_type in ["inspect", "trace"] and result.get("valid"):
            inspection_results = {action.target_node_id: result.get("info")}

        if self.done:
            reward = self._compute_final_reward()
        else:
            reward = self._compute_step_reward(action, result)

        self.step_rewards.append(reward.score)
        self.cumulative_score = reward.score

        self.current_observation = self._build_observation(
            stream_reports=self._generate_stream_reports(),
            inspection_results=inspection_results,
            message=self._build_step_message(action, result)
        )

        info = {
            "action_valid": result.get("valid", False),
            "newly_infected": result.get("newly_infected", []),
            "infection_rate": self.task.graph.infection_rate(),
            "step": self.task.step_count
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
            removed_nodes=[],  # Could implement if needed
            actions_taken=[],
            cumulative_score=self.cumulative_score,
            financial_budget=self.financial_budget,
            public_outrage_index=self.public_outrage_index,
            done=self.done
        )

    # ─────────────────────────────────────────
    # OBSERVATION (POMDP)
    # ─────────────────────────────────────────

    def _generate_stream_reports(self) -> list[str]:
        """
        Partial Observability logic:
        Return recently infected nodes, simulating a platform heuristics report.
        """
        if hasattr(self.task, "engine") and self.task.engine.spread_history:
            return self.task.engine.spread_history[-1]["newly_infected"]
        return [self.task.graph.origin_node_id]

    def _build_observation(self, stream_reports: list[str], inspection_results: dict, message: str) -> Observation:
        actions_remaining = None
        if self.task_id == TaskID.task3:
            actions_remaining = self.task.ACTIONS_PER_STEP - self.task.actions_this_step

        return Observation(
            task_id=self.task_id,
            step_number=self.task.step_count,
            max_steps=self.task.MAX_STEPS,
            actions_remaining=actions_remaining,
            stream_reports=stream_reports,
            inspection_results=inspection_results,
            agent_message=message,
            financial_budget=self.financial_budget,
            public_outrage_index=self.public_outrage_index
        )

    # ─────────────────────────────────────────
    # VALIDATION
    # ─────────────────────────────────────────

    def _validate_action(self, action: Action) -> str:
        if action.confidence is None or not (0.0 <= action.confidence <= 1.0):
            return "confidence score between 0.0 and 1.0 is required for Brier scoring."

        if action.action_type == ActionType.submit_causal_chain:
            if not action.causal_chain:
                return "submit_causal_chain requires 'causal_chain' parameter."
        elif action.action_type == ActionType.deploy_counter_narrative:
            if not action.target_node_id:
                return "deploy_counter_narrative requires 'target_node_id' specifying community_id."
        else:
            if not action.target_node_id or action.target_node_id not in self.task.graph.nodes:
                return f"Node {action.target_node_id} does not exist."

        # Task 1: inspect, quarantine
        if self.task_id == TaskID.task1:
            if action.action_type.value not in ["inspect", "quarantine"]:
                return "Task 1 only allows inspect and quarantine."
        # Task 2: inspect, trace, quarantine, submit_causal_chain
        if self.task_id == TaskID.task2:
            if action.action_type.value not in ["inspect", "trace", "quarantine", "submit_causal_chain"]:
                return "Task 2 allows inspect, trace, quarantine, submit."
        # Task 3: inspect, trace, quarantine, remove, shadowban, deploy_counter_narrative, submit_causal_chain
        if self.task_id == TaskID.task3:
            pass # All permitted

        return ""

    # ─────────────────────────────────────────
    # REWARDS & MESSAGES
    # ─────────────────────────────────────────

    def _compute_final_reward(self) -> Reward:
        if self.task_id == TaskID.task1:
            return self.grader.grade(self.task, self.cumulative_penalty)
        elif self.task_id == TaskID.task2:
            return self.grader.grade(self.task, self.cumulative_penalty)
        elif self.task_id == TaskID.task3:
            return self.grader.grade(self.task, self.cumulative_penalty)

    def _compute_step_reward(self, action: Action, result: dict) -> Reward:
        infection_rate = self.task.graph.infection_rate()
        action_bonus = 0.05 if result.get("valid") else -0.05
        
        # Immediate Brier Calibration Penalty for invalid actions guessed with high confidence
        if not result.get("valid") and action.confidence > 0.5:
            penalty = (action.confidence - 0.0) ** 2  # Brier penalty
        else:
            penalty = 0.0

        step_score = max(-1.0, min(1.0, action_bonus - penalty))

        return Reward(
            score=step_score,
            delta=0.0,
            done=False,
            success=False,
            partial_credits={"action_bonus": action_bonus, "brier_penalty": penalty},
            penalty=penalty,
            feedback=f"Infection rate {infection_rate:.2%}. Calibration penalty: {penalty:.2f}"
        )

    def _invalid_action_reward(self, error: str) -> Reward:
        return Reward(
            score=-1.0,
            delta=-1.0,
            done=False,
            success=False,
            partial_credits={},
            penalty=1.0,
            feedback=f"Invalid action: {error}"
        )

    def _initial_message(self) -> str:
        return "System initialized. Processing stream_reports feed."

    def _build_step_message(self, action: Action, result: dict) -> str:
        return f"Processed {action.action_type.value}. Output inside inspection_results."