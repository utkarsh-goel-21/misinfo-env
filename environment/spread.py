import random
from environment.models import NodeStatus
from environment.graph import MisinformationGraph


class SpreadEngine:
    """
    Linear Threshold Model (LTM) Spread Engine.
    
    A clean node checks all of its infected neighbors. 
    It sums up (edge_weight * neighbor_influence_score).
    If this cumulative 'infection pressure' exceeds the target
    node's skepticism_score, the target node becomes infected.
    """

    def __init__(self, graph: MisinformationGraph):
        self.graph = graph
        self.rng = random.Random(graph.seed)
        self.spread_history: list[dict] = []

    def step(self) -> list[str]:
        newly_infected = []
        current_infected = set(self.graph.get_infected_nodes())
        
        infection_attempts: dict[str, float] = {}

        for node_id in current_infected:
            node = self.graph.nodes[node_id]

            if node.status in [NodeStatus.quarantined, NodeStatus.removed]:
                continue
                
            # If it's a bot and currently dormant, it doesn't spread
            if node.is_bot and node.dormant_until and node.dormant_until > self.graph.step:
                continue

            for neighbor_id in node.neighbors:
                neighbor = self.graph.nodes.get(neighbor_id)
                if not neighbor or neighbor.status != NodeStatus.clean:
                    continue

                edge_weight = self._get_edge_weight(node_id, neighbor_id)
                pressure = edge_weight * node.influence_score
                
                # Accumulate pressure from all infected neighbors
                if neighbor_id not in infection_attempts:
                    infection_attempts[neighbor_id] = pressure
                else:
                    infection_attempts[neighbor_id] += pressure

        # Apply deterministic LTM thresholding
        for target_id, total_pressure in infection_attempts.items():
            target_node = self.graph.nodes[target_id]
            
            # If the cumulative pressure exceeds their innate skepticism, they fall
            if total_pressure >= target_node.skepticism_score:
                self.graph._infect_node_with_semantic(
                    target_id, 
                    step=self.graph.step,
                    tier=1  # Default fallback, dynamic generation already handled origin
                )
                newly_infected.append(target_id)
            else:
                # 10% chance to still fall if pressure is > half skepticism (adds slight stochastic noise)
                if total_pressure > (target_node.skepticism_score / 2.0):
                    if self.rng.random() < 0.1:
                        self.graph._infect_node_with_semantic(
                            target_id, 
                            step=self.graph.step,
                            tier=1
                        )
                        newly_infected.append(target_id)

        self.graph.step += 1

        self.spread_history.append({
            "step": self.graph.step,
            "newly_infected": newly_infected.copy(),
            "total_infected": len(self.graph.get_infected_nodes()),
            "infection_rate": self.graph.infection_rate()
        })

        return newly_infected

    def _get_edge_weight(self, source: str, target: str) -> float:
        for edge in self.graph.edges:
            if (
                (edge.source == source and edge.target == target)
                or (edge.source == target and edge.target == source)
            ):
                return edge.weight
        return 0.1

    def get_spread_velocity(self) -> float:
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
            "history": self.spread_history
        }

    def reset(self):
        self.rng = random.Random(self.graph.seed)
        self.spread_history = []