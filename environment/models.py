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


class ActionType(str, Enum):
    quarantine = "quarantine"
    remove = "remove"
    trace = "trace"
    inspect = "inspect"
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
    
    # ── Adversarial Layer ──
    is_bot: bool = False
    dormant_until: Optional[int] = None
    
    # ── Graph Memory ──
    infected_at_step: Optional[int] = None
    neighbors: List[str] = []


class Edge(BaseModel):
    source: str
    target: str
    weight: float = Field(ge=0.0, le=1.0)


class NetworkSnapshot(BaseModel):
    step: int
    nodes: Dict[str, Node]
    edges: List[Edge]
    total_infected: int
    total_quarantined: int
    infection_threshold: float
    origin_node_id: Optional[str] = None


# ─────────────────────────────────────────
# OBSERVATION — What agent sees each step
# ─────────────────────────────────────────

class Observation(BaseModel):
    task_id: TaskID
    step_number: int
    max_steps: int
    actions_remaining: Optional[int] = None
    
    # ── Partial Observability ──
    stream_reports: List[str] = Field(
        description="List of node IDs flagged by heuristic alerts this step."
    )
    inspection_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from 'inspect' or 'trace' actions."
    )
    agent_message: str = Field(description="System summary")


# ─────────────────────────────────────────
# ACTION — What agent sends each step
# ─────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType
    
    target_node_id: Optional[str] = Field(
        default=None,
        description="Node to act upon. Null for submit_causal_chain."
    )
    
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="MANDATORY calibration score. Overconfidence heavily penalized."
    )
    
    reasoning: Optional[str] = None
    
    causal_chain: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Used with submit_causal_chain. List of {'from': 'node_x', 'to': 'node_y'} edges."
    )


# ─────────────────────────────────────────
# REWARD — What environment returns each step
# ─────────────────────────────────────────

class Reward(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    delta: float = Field(description="Change in score from previous step")
    done: bool
    success: bool
    partial_credits: Dict[str, Any] = Field(
        description="Breakdown including Brier score calibration."
    )
    penalty: float = Field(default=0.0)
    feedback: str


# ─────────────────────────────────────────
# STATE — Full internal environment state
# ─────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id: TaskID
    step_number: int
    network: NetworkSnapshot
    origin_node_id: str
    infected_nodes: List[str]
    quarantined_nodes: List[str]
    removed_nodes: List[str]
    actions_taken: List[Action]
    cumulative_score: float = 0.0
    done: bool = False