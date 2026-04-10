"""
SENTINEL-9 Misinformation Containment Environment
Pydantic Data Models — Full POMDP + Adversarial Specification

All typed models for observations, actions, rewards, and internal state.
Designed for rigorous RL post-training evaluation of frontier LLMs.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, List, Dict
from enum import Enum


# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────

class NodeStatus(str, Enum):
    clean = "clean"
    infected = "infected"
    quarantined = "quarantined"
    removed = "removed"
    recovered = "recovered"        # SIR: was infected, now immune
    shadowbanned = "shadowbanned"  # Influence reduced, still active


class ContentTier(int, Enum):
    """Deception difficulty levels — higher = harder to detect."""
    BLATANT = 1        # ALL CAPS conspiracy, obvious
    SENSATIONAL = 2    # Emotional, plausible-sounding
    HEDGED = 3         # "Just asking questions"
    SOPHISTICATED = 4  # Miscontextualized real data
    STEALTH = 5        # Nearly indistinguishable from clean


class ActionType(str, Enum):
    inspect = "inspect"
    trace = "trace"
    quarantine = "quarantine"
    remove = "remove"
    shadowban = "shadowban"
    deploy_counter_narrative = "deploy_counter_narrative"
    submit_causal_chain = "submit_causal_chain"


class TaskID(str, Enum):
    task1 = "task1_detection"
    task2 = "task2_tracing"
    task3 = "task3_containment"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"


class PoliticalLeaning(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    RADICAL = "radical"


# ─────────────────────────────────────────
# GRAPH COMPONENTS
# ─────────────────────────────────────────

class Demographics(BaseModel):
    age: int = Field(ge=13, le=99)
    gender: Gender
    political_leaning: PoliticalLeaning
    occupation: str


class Node(BaseModel):
    node_id: str
    status: NodeStatus = NodeStatus.clean
    influence_score: float = Field(ge=0.0, le=1.0)
    skepticism_score: float = Field(ge=0.0, le=1.0)

    # ── Semantic & Demographic Layer ──
    demographics: Demographics
    recent_post: str
    user_persona: str
    community_id: Optional[str] = None
    content_tier: ContentTier = ContentTier.BLATANT

    # ── Adversarial Layer ──
    is_bot: bool = False
    dormant_until: Optional[int] = None
    bot_generation: int = 0           # 0 = original, 1+ = spawned
    evasion_active: bool = False      # Bot entered evasion mode

    # ── SIR Dynamics ──
    infected_at_step: Optional[int] = None
    recovery_timer: Optional[int] = None   # Steps until recovery
    infected_by: Optional[str] = None      # Causal parent for chain

    # ── Graph Memory ──
    neighbors: List[str] = []
    times_inspected: int = 0         # Track inspection frequency


class Edge(BaseModel):
    source: str
    target: str
    weight: float = Field(ge=0.0, le=1.0)
    is_bridge: bool = False


class CausalEdge(BaseModel):
    """A directed edge in the ground-truth causal infection tree."""
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    step: int
    content_tier: ContentTier = ContentTier.BLATANT

    model_config = {"populate_by_name": True}


class NetworkSnapshot(BaseModel):
    step: int
    nodes: Dict[str, Node]
    edges: List[Edge]
    total_infected: int
    total_quarantined: int
    total_recovered: int = 0
    infection_threshold: float
    origin_node_id: Optional[str] = None


# ─────────────────────────────────────────
# OBSERVATION — What agent sees each step
# (Partial Observability enforced here)
# ─────────────────────────────────────────

class Observation(BaseModel):
    task_id: TaskID
    step_number: int
    max_steps: int
    actions_remaining: Optional[int] = None

    # ── Partial Observability (POMDP Core) ──
    stream_reports: List[str] = Field(
        default_factory=list,
        description=(
            "Node IDs flagged by platform heuristics this step. "
            "WARNING: ~30% are false positives. Verify via inspect."
        )
    )
    revealed_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs the agent has previously inspected (fog-of-war)."
    )
    inspection_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from last inspect/trace action."
    )
    agent_message: str = Field(description="System summary of current situation.")

    # ── Macro Dynamics ──
    financial_budget: float = Field(default=10000.0)
    public_outrage_index: float = Field(default=0.0, ge=0.0, le=1.0)
    infection_rate: float = Field(
        default=0.0,
        description="Current fraction of network that is infected."
    )
    brier_score_running: float = Field(
        default=0.0,
        description="Agent's running Brier calibration score (lower = better calibrated)."
    )
    network_size: int = Field(default=0, description="Total nodes in network.")


# ─────────────────────────────────────────
# ACTION — What agent sends each step
# ─────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType

    target_node_id: Optional[str] = Field(
        default=None,
        description="Node to act upon. Required for all except submit_causal_chain."
    )

    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "MANDATORY calibration score (0.0-1.0). "
            "This is your probability estimate that this action is correct. "
            "Scored via Brier calibration — overconfidence on incorrect actions "
            "incurs catastrophic quadratic penalties."
        )
    )

    reasoning: Optional[str] = Field(
        default=None,
        description="Free-text reasoning for this action."
    )

    causal_chain: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Used with submit_causal_chain. "
            "List of {'from': 'node_x', 'to': 'node_y'} directed edges "
            "representing the inferred infection propagation path."
        )
    )


# ─────────────────────────────────────────
# REWARD — What environment returns each step
# ─────────────────────────────────────────

class Reward(BaseModel):
    score: float = Field(gt=0.0, lt=1.0)
    delta: float = Field(description="Change in score from previous step")
    done: bool
    success: bool
    partial_credits: Dict[str, Any] = Field(
        default_factory=dict,
        description="Multi-dimensional score breakdown."
    )
    penalty: float = Field(default=0.0)
    feedback: str


# ─────────────────────────────────────────
# STATE — Full internal environment state
# (Ground truth, not exposed to agent)
# ─────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id: TaskID
    step_number: int
    network: NetworkSnapshot
    origin_node_id: str
    infected_nodes: List[str]
    quarantined_nodes: List[str]
    recovered_nodes: List[str] = []
    removed_nodes: List[str] = []
    bot_nodes: List[str] = []
    causal_tree: List[Dict[str, str]] = []
    actions_taken: List[Dict[str, Any]] = []
    cumulative_score: float = 0.0
    financial_budget: float = 10000.0
    public_outrage_index: float = 0.0
    brier_scores: List[float] = []
    done: bool = False