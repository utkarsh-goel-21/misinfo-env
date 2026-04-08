import json
import random
from pathlib import Path
import networkx as nx
import copy

from environment.models import (
    Node, Edge, NetworkSnapshot, NodeStatus, 
    Demographics, Gender, PoliticalLeaning
)


# Dynamic NLP Combinatorics Matricies
FRAGMENTS = {
    "clean": {
        "prefix": ["Just finished", "Looking at", "Reviewing", "Has anyone tried", "Finally got around to"],
        "subject": ["my morning coffee", "the Q3 earnings report", "the new pasta place", "that new movie", "the weekend game"],
        "suffix": ["Beautiful day!", "Looks solid.", "7/10.", "Can't wait.", "Highly recommend it."]
    },
    "infected": {
        "prefix": ["THEY ARE HIDING THE TRUTH!", "Evidence proves", "Secret leaked documents show", "URGENT:", "DO NOT COMPLY."],
        "subject": ["the banks are collapsing tomorrow", "the voting machines were breached", "the new shot causes collapses", "the mayor is taking bribes", "the power grid is going down next week"],
        "suffix": ["withdraw your cash NOW!", "Don't let them steal it!", "Wake up sheeple!", "stock up on supplies immediately.", "Share before they delete this!"]
    },
    "debunking": {
        "prefix": ["There's a rumor going around about", "Stop sharing the post about", "I investigated the claims about", "Experts have addressed", "It is 100% false that"],
        "subject": ["the banks collapsing", "the voting machines", "the new shot causing issues", "the mayor's leaked documents", "the power grid failing"],
        "suffix": ["It was debunked by three independent agencies.", "Look at the actual data.", "Stop spreading lies.", "It's a known hoax from 2018.", "Check your sources."]
    },
    "cib": { # Coordinated Inauthentic Behavior
        "prefix": ["Just saw the truth about", "It's obvious that", "Clear conflict of interest regarding", "We need transparency right now on", "Look closely at"],
        "subject": ["the offshore corp funding.", "the compromised mayor.", "the secret pipeline contract.", "the overseas ballot printing.", "the digital ID rollout."],
        "suffix": ["Resign. #Truth", "Corruption. #Truth", "The people know. #Truth", "Exposed. #Truth", "Follow the money. #Truth"]
    },
    "cowardice": {
        "prefix": ["I'm not saying it's true, but", "Just asking questions:", "A lot of people are experiencing", "Interesting timing on", "I'd highly suggest looking into"],
        "subject": ["how many bank branches are closing lately.", "why they won't release the full logs.", "weird symptoms after the rollout.", "the mayor's new real estate purchase.", "when the grid might 'accidentally' fail."],
        "suffix": ["Probably just a coincidence though...", "You never know.", "Make up your own mind.", "Just my two cents.", "Stay prepared."]
    }
}

PERSONAS = [
    ("Teacher", PoliticalLeaning.LEFT, Gender.FEMALE, 35),
    ("Software Engineer", PoliticalLeaning.CENTER, Gender.MALE, 28),
    ("Retired Veteran", PoliticalLeaning.RIGHT, Gender.MALE, 68),
    ("Student Activist", PoliticalLeaning.RADICAL, Gender.NON_BINARY, 21),
    ("Small Business Owner", PoliticalLeaning.RIGHT, Gender.FEMALE, 52),
    ("Healthcare Worker", PoliticalLeaning.LEFT, Gender.MALE, 41),
    ("Financial Analyst", PoliticalLeaning.CENTER, Gender.MALE, 31)
]

class MisinformationGraph:
    """
    Manages the partial-observable social network graph.
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
        self.nx_graph = nx.Graph()

    # ─────────────────────────────────────────
    # GRAPH CONSTRUCTION
    # ─────────────────────────────────────────

    def build_from_config(
        self,
        num_nodes: int,
        avg_connections: int,
        infection_threshold: float,
        topology: str = "watts_strogatz"
    ):
        """
        Builds a random social network graph with exact topology per task.
        """
        self.rng = random.Random(self.seed)
        self.nodes = {}
        self.edges = []
        self.adjacency = {}
        self.infection_threshold = infection_threshold
        self.step = 0

        # 1. Topology Generation via NetworkX (Edge-case hardening)
        try:
            k = max(2, min(avg_connections, num_nodes - 1))
            m = max(1, min(int(avg_connections/2), num_nodes - 1))
            sub_n = max(3, int(num_nodes/2))
            
            if topology == "barabasi_albert":
                G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=self.seed)
            elif topology == "mixed":
                # Task 3 specific: bridge nodes
                G1 = nx.watts_strogatz_graph(n=sub_n, k=k, p=0.1, seed=self.seed)
                G2 = nx.watts_strogatz_graph(n=num_nodes - sub_n, k=k, p=0.1, seed=self.seed+1)
                G = nx.disjoint_union(G1, G2)
                # Add bridges explicitly
                for b in range(max(1, min(3, int(num_nodes/4)))):
                    n1 = self.rng.choice(list(G1.nodes()))
                    n2 = sub_n + self.rng.choice(list(G2.nodes()))
                    G.add_edge(n1, n2)
            else:
                G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=0.1, seed=self.seed)
                
            # Safety: Ensure basic connectivity for diffusion models
            if not nx.is_connected(G):
                comps = list(nx.connected_components(G))
                for i in range(len(comps)-1):
                    n1 = self.rng.choice(list(comps[i]))
                    n2 = self.rng.choice(list(comps[i+1]))
                    G.add_edge(n1, n2)
        except Exception:
            # Absolute fallback to guaranteed graph
            G = nx.erdos_renyi_graph(n=num_nodes, p=min(1.0, (avg_connections*2.0)/num_nodes), seed=self.seed)
            if not nx.is_connected(G):
                comps = list(nx.connected_components(G))
                for i in range(len(comps)-1):
                    n1 = self.rng.choice(list(comps[i]))
                    n2 = self.rng.choice(list(comps[i+1]))
                    G.add_edge(n1, n2)

        self.nx_graph = G

        # 2. Community Detection (Echo Chambers)
        communities = list(nx.community.louvain_communities(G, seed=self.seed))
        node_community_map = {}
        for idx, comm in enumerate(communities):
            for n in comm:
                node_community_map[n] = f"community_{idx}"

        # 3. Node Creation with Demographics
        for n in G.nodes():
            node_id = f"node_{n}"
            
            persona = self.rng.choice(PERSONAS)
            occ, lean, gender, age = persona
            
            # Skepticism depends on demographics
            base_skep = self.rng.uniform(0.2, 0.6)
            if occ == "Software Engineer" or occ == "Financial Analyst":
                base_skep += 0.3
            elif age > 60:
                base_skep -= 0.15
            
            skep = max(0.0, min(1.0, base_skep))

            # Assign content
            # Initially mostly clean, we will override origin later
            post = self._generate_payload("clean")
            
            self.nodes[node_id] = Node(
                node_id=node_id,
                status=NodeStatus.clean,
                influence_score=round(self.rng.uniform(0.1, 0.9), 2),
                skepticism_score=round(skep, 2),
                demographics=Demographics(
                    age=age,
                    gender=gender,
                    political_leaning=lean,
                    occupation=occ
                ),
                recent_post=post,
                user_persona=occ,
                community_id=node_community_map.get(n, "none"),
                is_bot=(self.rng.random() < 0.1), # 10% are bots
                neighbors=[]
            )
            self.adjacency[node_id] = []

        # 4. Edge Creation
        for u, v in G.edges():
            src_id = f"node_{u}"
            tgt_id = f"node_{v}"
            
            # Echo chamber multiplier
            weight = round(self.rng.uniform(0.1, 0.5), 2)
            if self.nodes[src_id].community_id == self.nodes[tgt_id].community_id:
                weight = min(1.0, weight * 2.5) # Explosive spread within community
                
            edge = Edge(source=src_id, target=tgt_id, weight=weight)
            self.edges.append(edge)
            self.adjacency[src_id].append(tgt_id)
            self.adjacency[tgt_id].append(src_id)
            self.nodes[src_id].neighbors.append(tgt_id)
            self.nodes[tgt_id].neighbors.append(src_id)

        # 5. Origin Assignment
        # Pick a node with high centrality
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
        sorted_nodes = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)
        self.origin_node_id = f"node_{sorted_nodes[0]}"

        # 6. Infect Origin
        self._infect_node_with_semantic(self.origin_node_id, step=0, tier=1)

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
        self.nx_graph = nx.Graph()

        for node_data in data["initial_snapshot"]["nodes"].values():
            node = Node(**node_data)
            self.nodes[node.node_id] = node
            self.adjacency[node.node_id] = node.neighbors.copy()
            
        for edge_data in data["initial_snapshot"]["edges"]:
            self.edges.append(Edge(**edge_data))
            self.nx_graph.add_edge(edge_data['source'].replace('node_',''), edge_data['target'].replace('node_',''))

    # ─────────────────────────────────────────
    # NODE OPERATIONS
    # ─────────────────────────────────────────

    def _generate_payload(self, tier_name: str) -> str:
        """Combines fragments to generate synthetically unique payload variations."""
        fragments = FRAGMENTS.get(tier_name, FRAGMENTS["clean"])
        prefix = self.rng.choice(fragments["prefix"])
        subject = self.rng.choice(fragments["subject"])
        suffix = self.rng.choice(fragments["suffix"])
        return f"{prefix} {subject} {suffix}"

    def _infect_node_with_semantic(self, node_id: str, step: int, tier: int = 1):
        """Mark node as infected and inject dynamic deceptive NLP payload."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.status == NodeStatus.clean:
            node.status = NodeStatus.infected
            node.infected_at_step = step
            
            # Change their post content to represent infection
            if node.is_bot:
                node.recent_post = self._generate_payload("cib")
            elif tier == 1:
                node.recent_post = self._generate_payload("infected")
            elif tier == 4:
                node.recent_post = self._generate_payload("cowardice")

    def quarantine_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        if node.status != NodeStatus.removed:
            node.status = NodeStatus.quarantined
            return True
        return False

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        node.status = NodeStatus.removed
        return True

    def inspect_node(self, node_id: str) -> dict:
        """
        Partial observability: Return semantic payload and demographics ONLY.
        Trigger adversarial bots if applicable.
        """
        if node_id not in self.nodes:
            return {}
        node = self.nodes[node_id]
        
        # Bot mechanic: Spooking adjacent bots
        for nb_id in node.neighbors:
            nb = self.nodes.get(nb_id)
            if nb and nb.is_bot and nb.status == NodeStatus.infected:
                nb.dormant_until = self.step + 3
                
        return {
            "node_id": node.node_id,
            "recent_post": node.recent_post,
            "user_persona": node.user_persona,
            "demographics": node.demographics.model_dump(),
            "skepticism_score": node.skepticism_score,
            "community_id": node.community_id
        }

    def trace_node(self, node_id: str) -> dict:
        """
        Return structural betweenness/bridges instead of direct answers.
        """
        if node_id not in self.nodes:
            return {}
            
        # Calculate centrality
        try:
            bc = nx.betweenness_centrality(self.nx_graph)
            n_idx = int(node_id.split("_")[1])
            score = round(bc.get(n_idx, 0.0), 4)
            bridges = list(nx.bridges(self.nx_graph))
            is_bridge = any(u == n_idx or v == n_idx for u, v in bridges)
        except:
            score = 0.0
            is_bridge = False
            
        return {
            "node_id": node_id,
            "betweenness_centrality": score,
            "is_bridge": is_bridge
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

    def infection_rate(self) -> float:
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
        return NetworkSnapshot(
            step=self.step,
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            total_infected=len(self.get_infected_nodes()),
            total_quarantined=len(self.get_quarantined_nodes()),
            infection_threshold=self.infection_threshold,
            origin_node_id=None if hide_origin else self.origin_node_id
        )

    def remap_edges(self, migration_rate: float = 0.1):
        """Dynamic Topology: Simulates users migrating away from quarantined nodes."""
        to_remove = []
        to_add = []
        for edge in self.edges:
            src = self.nodes.get(edge.source)
            tgt = self.nodes.get(edge.target)
            if src and tgt:
                # If either is quarantined or removed, small chance the edge breaks and remaps
                if src.status in [NodeStatus.quarantined, NodeStatus.removed] or tgt.status in [NodeStatus.quarantined, NodeStatus.removed]:
                    if self.rng.random() < migration_rate:
                        to_remove.append(edge)
                        # Find a random infected node to reconnect to (alt-hub)
                        infected_nodes = self.get_infected_nodes()
                        if infected_nodes:
                            new_tgt = self.rng.choice(infected_nodes)
                            # The node that is NOT quarantined remaps
                            clean_node_id = edge.source if tgt.status in [NodeStatus.quarantined, NodeStatus.removed] else edge.target
                            if clean_node_id != new_tgt:
                                to_add.append(Edge(source=clean_node_id, target=new_tgt, weight=self.rng.uniform(0.1, 0.4)))
                                
        for e in to_remove:
            if e in self.edges:
                self.edges.remove(e)
            if e.target in self.nodes[e.source].neighbors:
                self.nodes[e.source].neighbors.remove(e.target)
            if e.source in self.nodes[e.target].neighbors:
                self.nodes[e.target].neighbors.remove(e.source)
                
        for e in to_add:
            self.edges.append(e)
            if e.target not in self.nodes[e.source].neighbors:
                self.nodes[e.source].neighbors.append(e.target)
            if e.source not in self.nodes[e.target].neighbors:
                self.nodes[e.target].neighbors.append(e.source)