from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────

class NodeStatus(str, Enum):
    clean = "clean"
    infected = "infected"
    quarantined = "quarantined"
    flagged = "flagged"
    removed = "removed"


class ActionType(str, Enum):
    flag = "flag"           # flag a node as suspicious
    quarantine = "quarantine"  # isolate node from network
    remove = "remove"       # permanently remove node
    trace = "trace"         # investigate node's origin
    inspect = "inspect"     # get detailed info about node
    restore = "restore"     # restore a wrongly quarantined node


class TaskID(str, Enum):
    task1 = "task1_detection"
    task2 = "task2_tracing"
    task3 = "task3_containment"


# ─────────────────────────────────────────
# GRAPH COMPONENTS
# ─────────────────────────────────────────

class Node(BaseModel):
    node_id: str
    status: NodeStatus = NodeStatus.clean
    influence_score: float = Field(
        ge=0.0, le=1.0,
        description="How many neighbors this node can infect per step"
    )
    infected_at_step: Optional[int] = None
    flagged: bool = False
    neighbors: list[str] = []
    metadata: dict = {}


class Edge(BaseModel):
    source: str
    target: str
    weight: float = Field(
        ge=0.0, le=1.0,
        description="Probability of infection spreading across this edge"
    )


class NetworkSnapshot(BaseModel):
    step: int
    nodes: dict[str, Node]
    edges: list[Edge]
    total_infected: int
    total_quarantined: int
    infection_threshold: float = Field(
        description="Fraction of network infected = game over"
    )
    origin_node_id: Optional[str] = None  # hidden from agent in task2/3


# ─────────────────────────────────────────
# OBSERVATION — What agent sees each step
# ─────────────────────────────────────────

class Observation(BaseModel):
    task_id: TaskID
    step_number: int
    max_steps: int
    actions_remaining: Optional[int] = None  # only for task3
    network: NetworkSnapshot
    recently_infected: list[str] = Field(
        description="Nodes that got infected this step"
    )
    agent_message: str = Field(
        description="Human readable situation summary"
    )


# ─────────────────────────────────────────
# ACTION — What agent sends each step
# ─────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType
    target_node_id: str
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent explains why it is taking this action"
    )


# ─────────────────────────────────────────
# REWARD — What environment returns each step
# ─────────────────────────────────────────

class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    delta: float = Field(
        description="Change in score from previous step"
    )
    done: bool
    success: bool
    partial_credits: dict[str, float] = Field(
        description="Breakdown of what contributed to score"
    )
    penalty: float = Field(
        default=0.0,
        description="Penalties incurred this step"
    )
    feedback: str = Field(
        description="Human readable reward explanation"
    )


# ─────────────────────────────────────────
# STATE — Full internal environment state
# ─────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id: TaskID
    step_number: int
    network: NetworkSnapshot
    origin_node_id: str
    infected_nodes: list[str]
    quarantined_nodes: list[str]
    flagged_nodes: list[str]
    removed_nodes: list[str]
    actions_taken: list[Action]
    cumulative_score: float = 0.0
    done: bool = False