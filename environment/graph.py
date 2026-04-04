import json
import random
from pathlib import Path
from environment.models import Node, Edge, NetworkSnapshot, NodeStatus


class MisinformationGraph:
    """
    Manages the social network graph.
    Handles graph construction, node access,
    and network state snapshots.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.adjacency: dict[str, list[str]] = {}
        self.origin_node_id: str = ""
        self.infection_threshold: float = 0.4
        self.step: int = 0

    # ─────────────────────────────────────────
    # GRAPH CONSTRUCTION
    # ─────────────────────────────────────────

    def build_from_config(
        self,
        num_nodes: int,
        avg_connections: int,
        infection_threshold: float,
        origin_node_id: str = None
    ):
        """
        Builds a random social network graph.
        num_nodes: total users in network
        avg_connections: average friends per user
        infection_threshold: fraction infected = game over
        """
        self.rng = random.Random(self.seed)
        self.nodes = {}
        self.edges = []
        self.adjacency = {}
        self.infection_threshold = infection_threshold
        self.step = 0

        # Create nodes
        for i in range(num_nodes):
            node_id = f"node_{i}"
            self.nodes[node_id] = Node(
                node_id=node_id,
                status=NodeStatus.clean,
                influence_score=round(self.rng.uniform(0.1, 0.9), 2),
                neighbors=[],
                metadata={
                    "username": f"user_{i}",
                    "followers": self.rng.randint(10, 10000),
                    "account_age_days": self.rng.randint(1, 3650)
                }
            )
            self.adjacency[node_id] = []

        # Create edges
        node_ids = list(self.nodes.keys())
        for i, node_id in enumerate(node_ids):
            num_connections = self.rng.randint(
                max(1, avg_connections - 2),
                avg_connections + 2
            )
            candidates = [
                n for n in node_ids
                if n != node_id
                and n not in self.adjacency[node_id]
            ]
            connections = self.rng.sample(
                candidates,
                min(num_connections, len(candidates))
            )
            for target in connections:
                weight = round(self.rng.uniform(0.1, 0.8), 2)
                edge = Edge(
                    source=node_id,
                    target=target,
                    weight=weight
                )
                self.edges.append(edge)
                self.adjacency[node_id].append(target)
                self.adjacency[target].append(node_id)
                self.nodes[node_id].neighbors.append(target)
                self.nodes[target].neighbors.append(node_id)

        # Set origin node
        if origin_node_id:
            self.origin_node_id = origin_node_id
        else:
            # Pick high influence node as origin
            self.origin_node_id = max(
                node_ids,
                key=lambda n: self.nodes[n].influence_score
            )

        # Infect origin
        self._infect_node(self.origin_node_id, step=0)

    def build_from_file(self, filepath: str):
        """Load a pre-built network from JSON file."""
        data = json.loads(Path(filepath).read_text())
        self.infection_threshold = data["infection_threshold"]
        self.origin_node_id = data["origin_node_id"]
        self.seed = data.get("seed", 42)
        self.step = 0
        self.nodes = {}
        self.edges = []
        self.adjacency = {}

        for node_data in data["nodes"]:
            node = Node(**node_data)
            self.nodes[node.node_id] = node
            self.adjacency[node.node_id] = node.neighbors.copy()

        for edge_data in data["edges"]:
            self.edges.append(Edge(**edge_data))

        # Infect origin
        self._infect_node(self.origin_node_id, step=0)

    # ─────────────────────────────────────────
    # NODE OPERATIONS
    # ─────────────────────────────────────────

    def _infect_node(self, node_id: str, step: int):
        """Mark node as infected."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.status in [
            NodeStatus.clean,
            NodeStatus.flagged
        ]:
            node.status = NodeStatus.infected
            node.infected_at_step = step

    def quarantine_node(self, node_id: str) -> bool:
        """
        Isolate node. Returns True if valid action.
        Quarantined nodes cannot spread infection.
        """
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.status in [
            NodeStatus.infected,
            NodeStatus.flagged,
            NodeStatus.clean
        ]:
            node.status = NodeStatus.quarantined
            return True
        return False

    def flag_node(self, node_id: str) -> bool:
        """
        Flag node as suspicious.
        Flagged nodes spread at 50% reduced rate.
        Returns True if valid action.
        """
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.status == NodeStatus.clean:
            node.flagged = True
            node.status = NodeStatus.flagged
            return True
        return False

    def remove_node(self, node_id: str) -> bool:
        """
        Permanently remove node from network.
        High cost action — use carefully.
        Returns True if valid action.
        """
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.status != NodeStatus.removed:
            node.status = NodeStatus.removed
            return True
        return False

    def restore_node(self, node_id: str) -> bool:
        """
        Restore wrongly quarantined clean node.
        Returns True if valid action.
        """
        if node_id not in self.nodes:
            return False
        node = self.nodes[node_id]
        if node.status == NodeStatus.quarantined:
            node.status = NodeStatus.clean
            node.flagged = False
            return True
        return False

    def inspect_node(self, node_id: str) -> dict:
        """
        Get full details about a node.
        Returns empty dict if invalid.
        """
        if node_id not in self.nodes:
            return {}
        node = self.nodes[node_id]
        return {
            "node_id": node.node_id,
            "status": node.status,
            "influence_score": node.influence_score,
            "infected_at_step": node.infected_at_step,
            "neighbor_count": len(node.neighbors),
            "neighbors": node.neighbors,
            "metadata": node.metadata
        }

    def trace_node(self, node_id: str) -> dict:
        """
        Investigate infection path of a node.
        Returns infected neighbors and timing clues.
        """
        if node_id not in self.nodes:
            return {}
        node = self.nodes[node_id]
        infected_neighbors = [
            {
                "node_id": nb,
                "infected_at_step": self.nodes[nb].infected_at_step,
                "influence_score": self.nodes[nb].influence_score
            }
            for nb in node.neighbors
            if nb in self.nodes
            and self.nodes[nb].status == NodeStatus.infected
        ]
        return {
            "node_id": node_id,
            "infected_at_step": node.infected_at_step,
            "infected_neighbors": infected_neighbors,
            "hint": "origin node has no infected neighbors" if not infected_neighbors else "trace the earliest infected neighbor"
        }

    # ─────────────────────────────────────────
    # NETWORK METRICS
    # ─────────────────────────────────────────

    def get_infected_nodes(self) -> list[str]:
        return [
            n for n, node in self.nodes.items()
            if node.status == NodeStatus.infected
        ]

    def get_quarantined_nodes(self) -> list[str]:
        return [
            n for n, node in self.nodes.items()
            if node.status == NodeStatus.quarantined
        ]

    def get_flagged_nodes(self) -> list[str]:
        return [
            n for n, node in self.nodes.items()
            if node.status == NodeStatus.flagged
        ]

    def get_removed_nodes(self) -> list[str]:
        return [
            n for n, node in self.nodes.items()
            if node.status == NodeStatus.removed
        ]

    def infection_rate(self) -> float:
        """Fraction of total network that is infected."""
        total = len(self.nodes)
        if total == 0:
            return 0.0
        infected = len(self.get_infected_nodes())
        return round(infected / total, 4)

    def threshold_breached(self) -> bool:
        return self.infection_rate() >= self.infection_threshold

    # ─────────────────────────────────────────
    # SNAPSHOT
    # ─────────────────────────────────────────

    def snapshot(self, hide_origin: bool = True) -> NetworkSnapshot:
        """
        Returns current network state.
        hide_origin: hides origin from agent in task2/3.
        """
        return NetworkSnapshot(
            step=self.step,
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            total_infected=len(self.get_infected_nodes()),
            total_quarantined=len(self.get_quarantined_nodes()),
            infection_threshold=self.infection_threshold,
            origin_node_id=None if hide_origin else self.origin_node_id
        )

    # ─────────────────────────────────────────
    # SERIALIZATION
    # ─────────────────────────────────────────

    def save_to_file(self, filepath: str):
        """Save current network to JSON for reproducibility."""
        data = {
            "seed": self.seed,
            "infection_threshold": self.infection_threshold,
            "origin_node_id": self.origin_node_id,
            "nodes": [
                node.model_dump()
                for node in self.nodes.values()
            ],
            "edges": [
                edge.model_dump()
                for edge in self.edges
            ]
        }
        Path(filepath).write_text(json.dumps(data, indent=2))