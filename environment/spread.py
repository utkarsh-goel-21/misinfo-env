"""
SENTINEL-9 — Spread Engine (SIR + Linear Threshold Model)

Combines two epidemiological models:
1. Linear Threshold Model (LTM): Infection pressure from neighbors must
   exceed a node's skepticism score for infection to occur.
2. SIR Recovery: Infected nodes recover after a timer, becoming immune.

Additional mechanics:
- Content tier mutation: Infected posts evolve to harder-to-detect versions
- Community amplification: Echo chambers accelerate intra-community spread
- Bot super-spreading: Bots have 2x influence within their community
- Causal tree tracking: Every infection records its parent for chain grading
- Stochastic noise: Sub-threshold infection with 8% probability
"""

import random
from environment.models import NodeStatus, ContentTier
from environment.graph import MisinformationGraph


class SpreadEngine:
    """
    Deterministic (seeded) spread simulation with SIR dynamics.
    Each step:
      1. Compute infection pressure for all clean/susceptible nodes
      2. Apply LTM thresholding + stochastic noise
      3. Mutate 15% of infected posts to harder tiers
      4. Process SIR recovery timers
      5. Record history for grading
    """

    def __init__(self, graph: MisinformationGraph):
        self.graph = graph
        self.rng = random.Random(graph.seed)
        self.spread_history: list[dict] = []
        self._edge_cache: dict[tuple[str, str], float] = {}
        self._build_edge_cache()

    def _build_edge_cache(self):
        """Pre-compute edge weights for O(1) lookup."""
        self._edge_cache = {}
        for edge in self.graph.edges:
            self._edge_cache[(edge.source, edge.target)] = edge.weight
            self._edge_cache[(edge.target, edge.source)] = edge.weight

    def step(self) -> list[str]:
        """Execute one full spread cycle. Returns list of newly infected node IDs."""
        newly_infected = []
        newly_recovered = []

        current_infected = set(self.graph.get_infected_nodes())
        infection_attempts: dict[str, tuple[float, str]] = {}  # target → (pressure, strongest_parent)

        # ── Phase 1: Compute Infection Pressure ──
        for node_id in current_infected:
            node = self.graph.nodes[node_id]

            # Skip nodes that can't spread
            if node.status in (NodeStatus.quarantined, NodeStatus.removed):
                continue
            if node.is_bot and node.dormant_until and node.dormant_until > self.graph.step:
                continue

            # Bot super-spreading: 2x influence within same community
            effective_influence = node.influence_score
            if node.is_bot:
                effective_influence = min(0.95, effective_influence * 2.0)

            for neighbor_id in node.neighbors:
                neighbor = self.graph.nodes.get(neighbor_id)
                if not neighbor or neighbor.status != NodeStatus.clean:
                    continue

                edge_weight = self._get_edge_weight(node_id, neighbor_id)

                # Community amplification: same community = 1.5x pressure
                community_mult = 1.0
                if node.community_id and node.community_id == neighbor.community_id:
                    community_mult = 1.5

                pressure = edge_weight * effective_influence * community_mult

                if neighbor_id not in infection_attempts:
                    infection_attempts[neighbor_id] = (pressure, node_id)
                else:
                    existing_pressure, existing_parent = infection_attempts[neighbor_id]
                    new_total = existing_pressure + pressure
                    # Track strongest parent for causal chain
                    best_parent = node_id if pressure > existing_pressure else existing_parent
                    infection_attempts[neighbor_id] = (new_total, best_parent)

        # ── Phase 2: Apply LTM Thresholding ──
        for target_id, (total_pressure, parent_id) in infection_attempts.items():
            target_node = self.graph.nodes[target_id]

            # Determine content tier based on step (later = harder to detect)
            if self.graph.step <= 2:
                tier = ContentTier.BLATANT
            elif self.graph.step <= 5:
                tier = ContentTier.SENSATIONAL
            elif self.graph.step <= 8:
                tier = ContentTier.HEDGED
            elif self.graph.step <= 12:
                tier = ContentTier.SOPHISTICATED
            else:
                tier = ContentTier.STEALTH

            if total_pressure >= target_node.skepticism_score:
                # Deterministic infection
                self.graph._infect_node(target_id, step=self.graph.step, tier=tier, parent=parent_id)
                newly_infected.append(target_id)
            elif total_pressure > (target_node.skepticism_score * 0.5):
                # Stochastic sub-threshold infection (8% chance)
                if self.rng.random() < 0.08:
                    self.graph._infect_node(target_id, step=self.graph.step, tier=tier, parent=parent_id)
                    newly_infected.append(target_id)

        # ── Phase 3: Content Mutation (15% of infected posts evolve) ──
        for node_id in current_infected:
            if self.rng.random() < 0.15:
                self.graph.mutate_content(node_id)

        # ── Phase 4: SIR Recovery ──
        for node_id in list(current_infected):
            node = self.graph.nodes[node_id]
            if node.is_bot:
                continue  # Bots don't recover
            if node.recovery_timer is not None:
                steps_infected = self.graph.step - (node.infected_at_step or 0)
                if steps_infected >= node.recovery_timer:
                    self.graph.recover_node(node_id)
                    newly_recovered.append(node_id)

        # ── Phase 5: Advance Step & Record ──
        self.graph.step += 1

        # Reactivate dormant bots whose timer expired
        for bot_id in self.graph.bot_ids:
            bot = self.graph.nodes.get(bot_id)
            if bot and bot.dormant_until and bot.dormant_until <= self.graph.step:
                bot.dormant_until = None
                bot.evasion_active = False

        self.spread_history.append({
            "step": self.graph.step,
            "newly_infected": newly_infected.copy(),
            "newly_recovered": newly_recovered.copy(),
            "total_infected": len(self.graph.get_infected_nodes()),
            "total_recovered": len(self.graph.get_recovered_nodes()),
            "infection_rate": self.graph.infection_rate(),
        })

        # Rebuild edge cache if topology changed
        if len(self.graph.edges) != len(self._edge_cache) // 2:
            self._build_edge_cache()

        return newly_infected

    def _get_edge_weight(self, source: str, target: str) -> float:
        return self._edge_cache.get((source, target), 0.05)

    def get_spread_velocity(self) -> float:
        """Average new infections per step over last 3 steps."""
        if len(self.spread_history) < 2:
            return 0.0
        recent = self.spread_history[-3:]
        avg = sum(len(s["newly_infected"]) for s in recent) / len(recent)
        return round(avg, 2)

    def get_peak_spread_step(self) -> int:
        if not self.spread_history:
            return 0
        return max(self.spread_history, key=lambda s: len(s["newly_infected"]))["step"]

    def get_spread_report(self) -> dict:
        if not self.spread_history:
            return {}
        return {
            "total_steps": self.graph.step,
            "final_infection_rate": self.graph.infection_rate(),
            "peak_spread_step": self.get_peak_spread_step(),
            "spread_velocity": self.get_spread_velocity(),
            "threshold_breached": self.graph.threshold_breached(),
            "total_recovered": len(self.graph.get_recovered_nodes()),
            "history": self.spread_history,
        }

    def reset(self):
        self.rng = random.Random(self.graph.seed)
        self.spread_history = []
        self._edge_cache = {}