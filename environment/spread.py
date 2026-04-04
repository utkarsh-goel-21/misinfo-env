import random
from environment.models import NodeStatus
from environment.graph import MisinformationGraph


class SpreadEngine:
    """
    Controls how misinformation spreads
    through the network every step.

    Spread is deterministic given the same seed.
    Agent cannot stop spread — only slow it down
    through quarantine, flagging, and removal.
    """

    def __init__(self, graph: MisinformationGraph):
        self.graph = graph
        self.rng = random.Random(graph.seed)
        self.spread_history: list[dict] = []

    # ─────────────────────────────────────────
    # CORE SPREAD LOGIC
    # ─────────────────────────────────────────

    def step(self) -> list[str]:
        """
        Execute one spread step across the network.
        Returns list of newly infected node IDs.

        Spread Rules:
        - infected node attempts to infect each neighbor
        - probability = edge_weight * node_influence_score
        - quarantined nodes do NOT spread
        - removed nodes do NOT spread
        - flagged nodes spread at 50% reduced rate
        - clean and flagged nodes CAN be infected
        """
        newly_infected = []
        current_infected = self.graph.get_infected_nodes()
        current_flagged = self.graph.get_flagged_nodes()

        # Collect all spread attempts this step
        # before applying — simultaneous spread
        infection_attempts: dict[str, float] = {}

        for node_id in current_infected + current_flagged:
            node = self.graph.nodes[node_id]

            # Quarantined or removed nodes dont spread
            if node.status in [
                NodeStatus.quarantined,
                NodeStatus.removed
            ]:
                continue

            # Flagged nodes spread at half rate
            spread_multiplier = 0.5 if node.status == NodeStatus.flagged else 1.0

            for neighbor_id in node.neighbors:
                neighbor = self.graph.nodes.get(neighbor_id)
                if neighbor is None:
                    continue

                # Only clean nodes can be infected
                if neighbor.status != NodeStatus.clean:
                    continue

                # Find edge weight between these nodes
                edge_weight = self._get_edge_weight(
                    node_id,
                    neighbor_id
                )

                # Infection probability
                prob = (
                    edge_weight
                    * node.influence_score
                    * spread_multiplier
                )

                # Keep highest probability attempt per node
                if neighbor_id not in infection_attempts:
                    infection_attempts[neighbor_id] = prob
                else:
                    infection_attempts[neighbor_id] = max(
                        infection_attempts[neighbor_id],
                        prob
                    )

        # Apply infection attempts
        for target_id, prob in infection_attempts.items():
            roll = self.rng.random()
            if roll < prob:
                self.graph._infect_node(
                    target_id,
                    step=self.graph.step
                )
                newly_infected.append(target_id)

        # Increment graph step
        self.graph.step += 1

        # Record history
        self.spread_history.append({
            "step": self.graph.step,
            "newly_infected": newly_infected,
            "total_infected": len(self.graph.get_infected_nodes()),
            "infection_rate": self.graph.infection_rate()
        })

        return newly_infected

    # ─────────────────────────────────────────
    # EDGE WEIGHT LOOKUP
    # ─────────────────────────────────────────

    def _get_edge_weight(
        self,
        source: str,
        target: str
    ) -> float:
        """
        Find weight of edge between two nodes.
        Returns 0.1 as default if edge not found.
        """
        for edge in self.graph.edges:
            if (
                (edge.source == source and edge.target == target)
                or
                (edge.source == target and edge.target == source)
            ):
                return edge.weight
        return 0.1

    # ─────────────────────────────────────────
    # SPREAD METRICS
    # ─────────────────────────────────────────

    def get_spread_velocity(self) -> float:
        """
        How fast is infection growing?
        Returns average new infections per step.
        """
        if len(self.spread_history) < 2:
            return 0.0
        recent = self.spread_history[-3:]
        avg = sum(
            len(s["newly_infected"])
            for s in recent
        ) / len(recent)
        return round(avg, 2)

    def get_peak_spread_step(self) -> int:
        """
        Which step had the most new infections?
        """
        if not self.spread_history:
            return 0
        return max(
            self.spread_history,
            key=lambda s: len(s["newly_infected"])
        )["step"]

    def get_spread_report(self) -> dict:
        """
        Full spread analytics for graders.
        """
        if not self.spread_history:
            return {}
        return {
            "total_steps": self.graph.step,
            "final_infection_rate": self.graph.infection_rate(),
            "peak_spread_step": self.get_peak_spread_step(),
            "spread_velocity": self.get_spread_velocity(),
            "threshold_breached": self.graph.threshold_breached(),
            "history": self.spread_history
        }

    # ─────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────

    def reset(self):
        """
        Reset spread engine alongside graph reset.
        """
        self.rng = random.Random(self.graph.seed)
        self.spread_history = []