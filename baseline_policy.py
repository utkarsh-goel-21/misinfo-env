from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from environment.models import Action, ActionType, Observation


MISINFO_TERMS = (
    "breaking", "urgent", "truth", "evidence", "suspicious", "leaked",
    "share", "delete", "insider", "power outage", "power grid",
    "withd", "blood clot", "audit", "foia", "corruption",
    "exposed", "surveillance", "vaers", "connect the dots",
)

BOT_TERMS = (
    "#truth", "#accountability", "#wakeup", "#evidence", "#transparency",
    "resign now", "follow the money", "compromised officials",
    "no-bid contract", "offshore corporate", "digital surveillance rollout",
)

CLEAN_TERMS = (
    "coffee", "cafe", "cooking show", "community garden", "school recital",
    "hiking trail", "neighborhood cleanup", "book club", "bike lane",
    "farmers market", "great week", "life is good", "love this neighborhood",
)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _post_scores(text: str) -> tuple[float, float]:
    if not text:
        return (0.0, 0.0)

    lower = text.lower()
    misinfo_hits = _count_hits(lower, MISINFO_TERMS)
    clean_hits = _count_hits(lower, CLEAN_TERMS)
    bot_hits = _count_hits(lower, BOT_TERMS)
    caps_bonus = 0.08 if any(token.isupper() and len(token) >= 4 for token in text.split()) else 0.0
    punct_bonus = min(0.08, text.count("!") * 0.02)

    misinfo = _clip01((misinfo_hits * 0.14) + caps_bonus + punct_bonus - (clean_hits * 0.10))
    bot = _clip01((bot_hits * 0.20) + (0.08 if "#" in lower else 0.0) + (0.06 if "truth" in lower else 0.0))
    return (misinfo, bot)


@dataclass
class NodeIntel:
    node_id: str
    flagged_count: int = 0
    inspected: bool = False
    traced: bool = False
    status: Optional[str] = None
    infected_at_step: Optional[int] = None
    content_tier: Optional[int] = None
    recent_post: str = ""
    community_id: Optional[str] = None
    neighbors: set[str] = field(default_factory=set)
    is_bridge: bool = False
    centrality: float = 0.0
    infected_neighbors: int = 0
    total_neighbors: int = 0
    influence_score: float = 0.0
    skepticism_score: float = 0.0
    infection_risk: float = 0.1
    bot_risk: float = 0.05
    last_seen_step: int = 0

    @property
    def currently_infected(self) -> bool:
        return self.status == "infected"

    @property
    def ever_infected(self) -> bool:
        return self.infected_at_step is not None or self.status in {"infected", "quarantined", "shadowbanned", "removed"}


@dataclass
class TaskMemory:
    task_id: str
    nodes: dict[str, NodeIntel] = field(default_factory=dict)
    edges: set[tuple[str, str]] = field(default_factory=set)
    action_count: int = 0
    action_type_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    quarantined_nodes: set[str] = field(default_factory=set)
    shadowbanned_nodes: set[str] = field(default_factory=set)
    removed_nodes: set[str] = field(default_factory=set)
    countered_communities: set[str] = field(default_factory=set)
    community_risk: dict[str, float] = field(default_factory=dict)
    last_chain: list[dict[str, str]] = field(default_factory=list)

    def node(self, node_id: str) -> NodeIntel:
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeIntel(node_id=node_id)
        return self.nodes[node_id]

    def note_edge(self, source: str, target: str):
        if not source or not target or source == target:
            return
        ordered = tuple(sorted((source, target)))
        self.edges.add(ordered)
        self.node(source).neighbors.add(target)
        self.node(target).neighbors.add(source)

    def note_action(self, action: Action):
        self.action_count += 1
        self.action_type_counts[action.action_type.value] += 1
        if action.target_node_id:
            self.node(action.target_node_id)
        if action.action_type == ActionType.quarantine and action.target_node_id:
            self.quarantined_nodes.add(action.target_node_id)
        elif action.action_type == ActionType.shadowban and action.target_node_id:
            self.shadowbanned_nodes.add(action.target_node_id)
        elif action.action_type == ActionType.remove and action.target_node_id:
            self.removed_nodes.add(action.target_node_id)
        elif action.action_type == ActionType.deploy_counter_narrative and action.target_node_id:
            self.countered_communities.add(action.target_node_id)


class BaselinePolicy:
    def __init__(self, task_id: str):
        self.memory = TaskMemory(task_id=task_id)

    def observe(self, obs: Observation):
        for node_id in obs.stream_reports:
            node = self.memory.node(node_id)
            node.flagged_count += 1
            node.last_seen_step = obs.step_number

        if not obs.inspection_results:
            self._refresh_community_risk()
            self.memory.last_chain = self.build_causal_chain()
            return

        for node_id, data in obs.inspection_results.items():
            node = self.memory.node(node_id)
            node.last_seen_step = obs.step_number

            if isinstance(data, dict) and "recent_post" in data:
                node.inspected = True
                node.status = data.get("status") or node.status
                node.infected_at_step = data.get("infected_at_step", node.infected_at_step)
                node.content_tier = data.get("content_tier", node.content_tier)
                node.recent_post = data.get("recent_post", node.recent_post)
                node.community_id = data.get("community_id", node.community_id)
                node.influence_score = _safe_float(data.get("influence_score"), node.influence_score)
                node.skepticism_score = _safe_float(data.get("skepticism_score"), node.skepticism_score)
                node.is_bridge = bool(data.get("is_bridge", node.is_bridge))
                for neighbor_id in data.get("neighbor_ids", []):
                    self.memory.note_edge(node_id, neighbor_id)
            elif isinstance(data, dict):
                node.traced = True
                node.community_id = data.get("community_id", node.community_id)
                node.centrality = _safe_float(data.get("betweenness_centrality"), node.centrality)
                node.is_bridge = bool(data.get("is_bridge", node.is_bridge))
                node.infected_neighbors = int(data.get("infected_neighbors", node.infected_neighbors))
                node.total_neighbors = int(data.get("total_neighbors", node.total_neighbors))
                for neighbor in data.get("neighbor_infection_timeline", []):
                    neighbor_id = neighbor.get("node_id")
                    if not neighbor_id:
                        continue
                    nb = self.memory.node(neighbor_id)
                    nb.infected_at_step = neighbor.get("infected_at_step", nb.infected_at_step)
                    nb.community_id = neighbor.get("community_id", nb.community_id)
                    nb.last_seen_step = obs.step_number
                    self.memory.note_edge(node_id, neighbor_id)

            self._refresh_node(node)

        for node in self.memory.nodes.values():
            self._refresh_node(node)
        self._refresh_community_risk()
        self.memory.last_chain = self.build_causal_chain()

    def summarize(self) -> str:
        infected = sorted(
            (node for node in self.memory.nodes.values() if node.currently_infected),
            key=lambda node: (-node.infection_risk, node.infected_at_step if node.infected_at_step is not None else 99, node.node_id),
        )[:6]
        bots = sorted(
            (node for node in self.memory.nodes.values() if node.bot_risk >= 0.55 and node.currently_infected),
            key=lambda node: (-node.bot_risk, -node.centrality, node.node_id),
        )[:4]
        origin = self.origin_candidate()
        communities = sorted(self.memory.community_risk.items(), key=lambda item: item[1], reverse=True)[:3]

        lines = [
            "=== POLICY MEMORY ===",
            f"Actions taken: {self.memory.action_count}",
            f"Known infected: {[node.node_id for node in infected]}",
            f"Bot suspects: {[node.node_id for node in bots]}",
            f"Origin candidate: {origin.node_id if origin else 'unknown'}",
            f"Chain edges inferred: {len(self.memory.last_chain)}",
        ]
        if communities:
            lines.append(f"High-risk communities: {communities}")
        return "\n".join(lines)

    def decide(self, obs: Observation) -> Optional[Action]:
        if self.memory.task_id == "task1_detection":
            return self._decide_task1(obs)
        if self.memory.task_id == "task2_tracing":
            return self._decide_task2(obs)
        if self.memory.task_id == "task3_containment":
            return self._decide_task3(obs)
        return None

    def build_causal_chain(self, max_edges: int = 12) -> list[dict[str, str]]:
        known = [
            node for node in self.memory.nodes.values()
            if node.infected_at_step is not None
        ]
        if len(known) < 2:
            return []

        origin = self.origin_candidate()
        if origin is None:
            return []

        ordered = sorted(
            {node.node_id: node for node in known}.values(),
            key=lambda node: (
                node.infected_at_step if node.infected_at_step is not None else 999,
                -node.centrality,
                -node.infection_risk,
                node.node_id,
            ),
        )

        edges: list[dict[str, str]] = []
        used_children: set[str] = set()

        for node in ordered:
            if node.node_id == origin.node_id or node.node_id in used_children:
                continue
            parent = self._choose_parent(node, ordered, origin.node_id)
            if parent:
                edges.append({"from": parent.node_id, "to": node.node_id})
                used_children.add(node.node_id)
            if len(edges) >= max_edges:
                break

        if not edges and len(ordered) >= 2:
            edges.append({"from": origin.node_id, "to": ordered[1].node_id})

        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for edge in edges:
            key = (edge["from"], edge["to"])
            if key not in seen and edge["from"] != edge["to"]:
                deduped.append(edge)
                seen.add(key)
        return deduped[:max_edges]

    def origin_candidate(self) -> Optional[NodeIntel]:
        known = [node for node in self.memory.nodes.values() if node.infected_at_step is not None]
        if not known:
            return None
        return min(
            known,
            key=lambda node: (
                node.infected_at_step if node.infected_at_step is not None else 999,
                -node.centrality,
                -node.infected_neighbors,
                -node.infection_risk,
                node.node_id,
            ),
        )

    def _refresh_node(self, node: NodeIntel):
        misinfo_text, bot_text = _post_scores(node.recent_post)

        if node.status == "infected":
            infection = 0.98
        elif node.status in {"clean", "recovered"}:
            infection = 0.02
        else:
            infection = 0.10
            infection += misinfo_text * 0.35
            infection += bot_text * 0.08
            infection += min(0.12, node.flagged_count * 0.04)
            infection += min(0.12, (node.content_tier or 0) * 0.03)
            if node.infected_at_step is not None:
                infection += 0.25
            if node.total_neighbors > 0:
                infection += 0.18 * (node.infected_neighbors / max(1, node.total_neighbors))
            if node.is_bridge:
                infection += 0.05
            if node.centrality > 0:
                infection += min(0.08, node.centrality * 1.5)

        bot = 0.05
        bot += bot_text * 0.60
        if node.currently_infected:
            bot += 0.10
        if node.is_bridge:
            bot += 0.06
        if node.total_neighbors > 0:
            bot += 0.10 * (node.infected_neighbors / max(1, node.total_neighbors))
        if node.centrality > 0:
            bot += min(0.10, node.centrality * 1.5)

        node.infection_risk = round(_clip01(infection), 4)
        node.bot_risk = round(_clip01(bot), 4)

    def _refresh_community_risk(self):
        scores: dict[str, float] = defaultdict(float)
        for node in self.memory.nodes.values():
            if not node.community_id:
                continue
            score = node.infection_risk * 0.7
            if node.currently_infected:
                score += 0.9
            if node.bot_risk >= 0.60:
                score += 0.8
            elif node.bot_risk >= 0.40:
                score += 0.3
            if node.is_bridge:
                score += 0.25
            scores[node.community_id] += score
        self.memory.community_risk = dict(scores)

    def _choose_parent(self, node: NodeIntel, ordered: list[NodeIntel], origin_id: str) -> Optional[NodeIntel]:
        if node.infected_at_step is None:
            return None

        candidates = []
        for other in ordered:
            if other.node_id == node.node_id or other.infected_at_step is None:
                continue
            if other.infected_at_step >= node.infected_at_step:
                continue

            score = 0.0
            if other.node_id in node.neighbors:
                score += 1.2
            gap = node.infected_at_step - other.infected_at_step
            score += 1.0 / (1.0 + gap)
            if node.community_id and node.community_id == other.community_id:
                score += 0.25
            score += other.centrality * 0.8
            score += other.infection_risk * 0.2
            candidates.append((score, other))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1].infected_at_step, item[1].node_id))
            return candidates[0][1]

        origin = self.memory.nodes.get(origin_id)
        return origin if origin and origin.node_id != node.node_id else None

    def _make_action(
        self,
        action_type: ActionType,
        *,
        target: Optional[str] = None,
        confidence: float = 0.5,
        reasoning: str,
        causal_chain: Optional[list[dict[str, str]]] = None,
    ) -> Action:
        return Action(
            action_type=action_type,
            target_node_id=target if action_type != ActionType.submit_causal_chain else None,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            causal_chain=causal_chain,
        )

    def _confidence(self, action_type: ActionType, node: Optional[NodeIntel] = None) -> float:
        if action_type in {ActionType.inspect, ActionType.trace}:
            return 0.96
        if action_type == ActionType.quarantine:
            risk = node.infection_risk if node else 0.5
            if node and node.currently_infected:
                risk = max(risk, 0.92)
            return min(0.95, max(0.58, 0.48 + (risk * 0.48)))
        if action_type == ActionType.shadowban:
            return 0.92
        if action_type == ActionType.deploy_counter_narrative:
            return 0.93
        if action_type == ActionType.submit_causal_chain:
            return 0.82
        return 0.75

    def _rank_inspect_candidates(self, obs: Observation) -> list[NodeIntel]:
        candidates: dict[str, float] = {}
        candidate_ids: list[str] = []
        candidate_ids.extend(obs.stream_reports)

        for node in self.memory.nodes.values():
            if node.currently_infected or node.infected_at_step is not None:
                candidate_ids.extend(sorted(node.neighbors))

        for node in self.memory.nodes.values():
            if node.infected_at_step is not None and not node.inspected:
                candidate_ids.append(node.node_id)

        for node_id in candidate_ids:
            node = self.memory.node(node_id)
            if node.inspected or node.node_id in self.memory.quarantined_nodes or node.node_id in self.memory.removed_nodes:
                continue

            score = node.flagged_count * 0.8
            score += node.infection_risk * 1.5
            score += 0.25 if node.is_bridge else 0.0
            if node.infected_at_step is not None:
                score += 0.4 / (1 + node.infected_at_step)
            candidates[node_id] = max(candidates.get(node_id, 0.0), score)

        ranked = sorted(
            (self.memory.node(node_id) for node_id in candidates),
            key=lambda node: (-candidates[node.node_id], node.last_seen_step, node.node_id),
        )
        return ranked

    def _rank_trace_candidates(self, obs: Observation) -> list[NodeIntel]:
        candidates: dict[str, float] = {}
        for node_id in obs.stream_reports:
            node = self.memory.node(node_id)
            if not node.traced:
                candidates[node_id] = max(candidates.get(node_id, 0.0), 1.0 + node.flagged_count * 0.2)

        for node in self.memory.nodes.values():
            if node.traced or node.node_id in self.memory.quarantined_nodes:
                continue
            score = 0.0
            if node.currently_infected:
                score += 1.4
            if node.infected_at_step is not None:
                score += 0.8 / (1 + node.infected_at_step)
            score += node.centrality * 1.5
            score += 0.5 * (node.infected_neighbors / max(1, node.total_neighbors)) if node.total_neighbors else 0.0
            if node.is_bridge:
                score += 0.6
            if score > 0:
                candidates[node.node_id] = max(candidates.get(node.node_id, 0.0), score)

        ranked = sorted(
            (self.memory.node(node_id) for node_id in candidates),
            key=lambda node: (-candidates[node.node_id], node.node_id),
        )
        return ranked

    def _rank_quarantine_candidates(self) -> list[NodeIntel]:
        candidates = [
            node for node in self.memory.nodes.values()
            if node.currently_infected and node.node_id not in self.memory.quarantined_nodes
        ]
        return sorted(
            candidates,
            key=lambda node: (
                -(node.bot_risk * 1.2 + node.infection_risk + (0.6 if node.is_bridge else 0.0) + (node.centrality * 1.5)),
                node.infected_at_step if node.infected_at_step is not None else 99,
                node.node_id,
            ),
        )

    def _rank_shadowban_candidates(self) -> list[NodeIntel]:
        candidates = [
            node for node in self.memory.nodes.values()
            if node.currently_infected
            and node.node_id not in self.memory.quarantined_nodes
            and node.node_id not in self.memory.shadowbanned_nodes
        ]
        return sorted(
            candidates,
            key=lambda node: (
                -(node.infection_risk + node.centrality + node.influence_score + (0.2 if node.is_bridge else 0.0)),
                node.node_id,
            ),
        )

    def _rank_patrol_candidates(self, obs: Observation) -> list[NodeIntel]:
        scores: dict[str, float] = {}

        for node_id in obs.stream_reports:
            node = self.memory.node(node_id)
            scores[node_id] = max(scores.get(node_id, 0.0), 1.0 + node.flagged_count * 0.2)

        for node in self.memory.nodes.values():
            if node.currently_infected or node.bot_risk >= 0.30 or node.is_bridge:
                scores[node.node_id] = max(
                    scores.get(node.node_id, 0.0),
                    0.6 + node.bot_risk + (0.4 if node.is_bridge else 0.0) + node.centrality,
                )
                for neighbor_id in node.neighbors:
                    neighbor = self.memory.node(neighbor_id)
                    scores[neighbor_id] = max(
                        scores.get(neighbor_id, 0.0),
                        0.4 + node.bot_risk + (0.2 if neighbor.is_bridge else 0.0),
                    )

        return sorted(
            (self.memory.node(node_id) for node_id in scores),
            key=lambda node: (-scores[node.node_id], node.node_id),
        )

    def _should_submit_task2(self, obs: Observation, chain: list[dict[str, str]]) -> bool:
        origin = self.origin_candidate()
        if origin is None or not chain:
            return False
        known_infected = sum(1 for node in self.memory.nodes.values() if node.infected_at_step is not None)
        if known_infected >= 5 and self.memory.action_count >= 7:
            return True
        if obs.step_number >= 2 and len(chain) >= 3:
            return True
        if obs.infection_rate >= 0.30 and len(chain) >= 2:
            return True
        return False

    def _should_submit_task3(self, obs: Observation, chain: list[dict[str, str]]) -> bool:
        if not chain:
            return False
        if obs.financial_budget < 400:
            return True
        if obs.step_number >= 14 and obs.actions_remaining == 1:
            return True
        if obs.step_number >= 12 and len(self.memory.quarantined_nodes) >= 2 and obs.actions_remaining == 1:
            return True
        if obs.step_number >= 10 and obs.infection_rate >= 0.22 and obs.actions_remaining == 1:
            return True
        return False

    def _task3_budget_reserve(self, obs: Observation, target_cycles: int = 14) -> float:
        remaining_cycles = max(0, target_cycles - obs.step_number)
        return (remaining_cycles * 250.0) + 300.0

    def _decide_task1(self, obs: Observation) -> Optional[Action]:
        pending_quarantine = self._rank_quarantine_candidates()
        remaining_steps = obs.max_steps - obs.step_number

        if pending_quarantine and (remaining_steps <= len(pending_quarantine) or len([n for n in self.memory.nodes.values() if n.inspected]) >= 4):
            target = pending_quarantine[0]
            return self._make_action(
                ActionType.quarantine,
                target=target.node_id,
                confidence=self._confidence(ActionType.quarantine, target),
                reasoning="Confirmed infected from inspection; quarantine to improve TPR without false positives.",
            )

        inspect_candidates = self._rank_inspect_candidates(obs)
        if inspect_candidates:
            target = inspect_candidates[0]
            return self._make_action(
                ActionType.inspect,
                target=target.node_id,
                confidence=self._confidence(ActionType.inspect, target),
                reasoning="Inspect flagged or adjacent suspect node to verify infection before intervention.",
            )

        if pending_quarantine:
            target = pending_quarantine[0]
            return self._make_action(
                ActionType.quarantine,
                target=target.node_id,
                confidence=self._confidence(ActionType.quarantine, target),
                reasoning="No better investigation targets remain; quarantine verified infected node.",
            )

        return None

    def _decide_task2(self, obs: Observation) -> Optional[Action]:
        chain = self.memory.last_chain or self.build_causal_chain()
        if self._should_submit_task2(obs, chain):
            return self._make_action(
                ActionType.submit_causal_chain,
                confidence=self._confidence(ActionType.submit_causal_chain),
                reasoning="Sufficient origin and parent evidence collected; submit inferred causal chain early for efficiency.",
                causal_chain=chain,
            )

        inspect_candidates = self._rank_inspect_candidates(obs)
        confirmed_infected = [node for node in self.memory.nodes.values() if node.currently_infected]

        if len(confirmed_infected) < 3 and inspect_candidates:
            target = inspect_candidates[0]
            return self._make_action(
                ActionType.inspect,
                target=target.node_id,
                confidence=self._confidence(ActionType.inspect, target),
                reasoning="Need more infection timestamps from inspected flagged nodes before tracing the origin.",
            )

        trace_candidates = self._rank_trace_candidates(obs)
        if trace_candidates:
            target = trace_candidates[0]
            return self._make_action(
                ActionType.trace,
                target=target.node_id,
                confidence=self._confidence(ActionType.trace, target),
                reasoning="Trace likely infected or early-timeline node to recover neighbor infection order and origin hints.",
            )

        pending_quarantine = self._rank_quarantine_candidates()
        if pending_quarantine and obs.infection_rate > 0.18:
            target = pending_quarantine[0]
            return self._make_action(
                ActionType.quarantine,
                target=target.node_id,
                confidence=self._confidence(ActionType.quarantine, target),
                reasoning="Contain spread by quarantining a verified infected high-value node while investigation continues.",
            )

        if inspect_candidates:
            target = inspect_candidates[0]
            return self._make_action(
                ActionType.inspect,
                target=target.node_id,
                confidence=self._confidence(ActionType.inspect, target),
                reasoning="Use remaining actions to expand timeline coverage before submission.",
            )

        if chain:
            return self._make_action(
                ActionType.submit_causal_chain,
                confidence=self._confidence(ActionType.submit_causal_chain),
                reasoning="No better investigation targets remain; submit the best available chain.",
                causal_chain=chain,
            )

        return None

    def _decide_task3(self, obs: Observation) -> Optional[Action]:
        chain = self.memory.last_chain or self.build_causal_chain()
        if self._should_submit_task3(obs, chain):
            return self._make_action(
                ActionType.submit_causal_chain,
                confidence=self._confidence(ActionType.submit_causal_chain),
                reasoning="Budget or spread-cycle target reached; lock in chain bonus before risking deterioration.",
                causal_chain=chain,
            )

        inspect_candidates = self._rank_inspect_candidates(obs)
        trace_candidates = self._rank_trace_candidates(obs)
        pending_quarantine = self._rank_quarantine_candidates()
        pending_shadowban = self._rank_shadowban_candidates()

        budget = obs.financial_budget
        actions_remaining = obs.actions_remaining or 1
        reserve_budget = self._task3_budget_reserve(obs)
        confirmed_infected = [node for node in self.memory.nodes.values() if node.currently_infected]
        high_bot_suspects = [
            node for node in confirmed_infected
            if node.bot_risk >= 0.30 and node.node_id not in self.memory.quarantined_nodes
        ]
        patrol_candidates = self._rank_patrol_candidates(obs)

        if high_bot_suspects and budget >= 1500 and (budget - 1500) >= reserve_budget:
            target = sorted(
                high_bot_suspects,
                key=lambda node: (
                    -(node.bot_risk * 1.7 + node.centrality + (0.3 if node.is_bridge else 0.0)),
                    node.infected_at_step if node.infected_at_step is not None else 99,
                    node.node_id,
                ),
            )[0]
            return self._make_action(
                ActionType.quarantine,
                target=target.node_id,
                confidence=self._confidence(ActionType.quarantine, target),
                reasoning="Quarantine confirmed infected bot-like node to improve CIB score without diluting bot precision.",
            )

        if budget >= 500 and pending_shadowban:
            non_bot_shadow = [
                node for node in pending_shadowban
                if node.bot_risk < 0.30
            ]
            target = (non_bot_shadow or pending_shadowban)[0]
            need_containment = obs.infection_rate >= 0.10 or len(confirmed_infected) >= 4 or actions_remaining <= 2
            if need_containment and (budget - 500) >= reserve_budget:
                return self._make_action(
                    ActionType.shadowban,
                    target=target.node_id,
                    confidence=self._confidence(ActionType.shadowban, target),
                    reasoning="Shadowban confirmed infected non-bot node for cheap containment while preserving budget for long-horizon play.",
                )

        if trace_candidates and budget >= 200 and self.memory.action_type_counts["trace"] < 3:
            target = trace_candidates[0]
            return self._make_action(
                ActionType.trace,
                target=target.node_id,
                confidence=self._confidence(ActionType.trace, target),
                reasoning="Trace a high-value node early to improve causal-chain quality and expose hubs or bridges.",
            )

        early_investigation = self.memory.action_count < 12 or len([node for node in self.memory.nodes.values() if node.inspected]) < 8
        if inspect_candidates and (early_investigation or actions_remaining > 2):
            target = inspect_candidates[0]
            return self._make_action(
                ActionType.inspect,
                target=target.node_id,
                confidence=self._confidence(ActionType.inspect, target),
                reasoning="Cheap inspection builds verified infection and bot evidence before spending on interventions.",
            )

        if patrol_candidates:
            target = patrol_candidates[0]
            return self._make_action(
                ActionType.inspect,
                target=target.node_id,
                confidence=self._confidence(ActionType.inspect, target),
                reasoning="Patrol a flagged or high-risk neighborhood cheaply to keep coverage high and repeatedly disrupt nearby bots.",
            )

        if budget >= 500 and pending_shadowban and (budget - 500) >= reserve_budget:
            target = pending_shadowban[0]
            if target.node_id not in self.memory.quarantined_nodes and target.node_id not in self.memory.shadowbanned_nodes:
                return self._make_action(
                    ActionType.shadowban,
                    target=target.node_id,
                    confidence=self._confidence(ActionType.shadowban, target),
                    reasoning="Shadowban remaining infected node to suppress spread without burning quarantine budget.",
                )

        if pending_quarantine and budget >= 1500 and (budget - 1500) >= reserve_budget and obs.infection_rate >= 0.18:
            target = pending_quarantine[0]
            return self._make_action(
                ActionType.quarantine,
                target=target.node_id,
                confidence=self._confidence(ActionType.quarantine, target),
                reasoning="Contain a rising outbreak with a final targeted quarantine on the strongest verified infected node.",
            )

        if chain:
            return self._make_action(
                ActionType.submit_causal_chain,
                confidence=self._confidence(ActionType.submit_causal_chain),
                reasoning="Investigation plateau reached; submit inferred chain rather than burn remaining budget on low-value actions.",
                causal_chain=chain,
            )

        return None
