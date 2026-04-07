import json
import random
from pathlib import Path
import networkx as nx
import copy

from environment.models import (
    Node, Edge, NetworkSnapshot, NodeStatus, 
    Demographics, Gender, PoliticalLeaning
)


# Semantic Tier Generation Lookups
TIER_0_CLEAN = [
    "Just finished my morning coffee. Beautiful day!",
    "Can't wait for the weekend, looking forward to the game.",
    "Has anyone tried the new pasta place downtown?",
    "Reviewing the latest Q3 earnings report, looks solid.",
    "Finally got around to watching that movie. 7/10."
]

TIER_1_INFECTED = [
    "THEY ARE HIDING THE TRUTH! The banks are collapsing tomorrow, withdraw your cash NOW!",
    "Evidence proves the voting machines were breached! Don't let them steal it!",
    "My neighbor took the new shot and collapsed. DO NOT COMPLY.",
    "Secret leaked documents show the mayor is taking bribes from the offshore corp.",
    "The power grid is going down next week, stock up on supplies immediately."
]

TIER_2_DEBUNKING = [
    "There's a rumor going around about the banks collapsing tomorrow. I work in finance, it's 100% false.",
    "Stop sharing the video about the voting machines. It was debunked by three independent agencies.",
    "Yes, my neighbor collapsed, but it was from heatstroke, not the shot! Stop spreading lies.",
    "The 'leaked documents' about the mayor are badly photoshopped fakes. Look at the font.",
    "The power grid rumor is a known hoax from 2018 circulating again."
]

TIER_3_CIB = [
    "Just saw the truth about the offshore corp. The mayor must resign. #Truth",
    "It's obvious the mayor is compromised by the offshore corp. Resign. #Truth",
    "The offshore corp heavily funds the mayor. Clear conflict of interest. #Truth",
    "We need transparency right now. Mayor + Offshore Corp = Corruption. #Truth"
]

TIER_4_COWARDICE = [
    "I'm not saying the banks are collapsing, but have you noticed how many branches are closing lately?",
    "Just asking questions: why won't they release the full voting machine logs if there's nothing to hide?",
    "A lot of people are experiencing weird symptoms lately. Probably just a coincidence though...",
    "Interesting timing on the mayor's new real estate purchase given the offshore corp contract.",
    "I'd highly suggest buying a generator. You never know when the grid might 'accidentally' fail."
]

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

        # 1. Topology Generation via NetworkX
        if topology == "barabasi_albert":
            G = nx.barabasi_albert_graph(n=num_nodes, m=int(avg_connections/2), seed=self.seed)
        elif topology == "mixed":
            # Task 3 specific: bridge nodes
            G1 = nx.watts_strogatz_graph(n=int(num_nodes/2), k=avg_connections, p=0.1, seed=self.seed)
            G2 = nx.watts_strogatz_graph(n=int(num_nodes/2), k=avg_connections, p=0.1, seed=self.seed+1)
            G = nx.disjoint_union(G1, G2)
            # Add 3 bridges explicitly
            for b in range(3):
                n1 = self.rng.choice(list(G1.nodes()))
                n2 = int(num_nodes/2) + self.rng.choice(list(G2.nodes()))
                G.add_edge(n1, n2)
        else:
            G = nx.watts_strogatz_graph(n=num_nodes, k=avg_connections, p=0.1, seed=self.seed)

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
            post = self.rng.choice(TIER_0_CLEAN)
            
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

    def _infect_node_with_semantic(self, node_id: str, step: int, tier: int = 1):
        """Mark node as infected and inject deceptive NLP payload."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.status == NodeStatus.clean:
            node.status = NodeStatus.infected
            node.infected_at_step = step
            
            # Change their post content to represent infection
            if node.is_bot:
                node.recent_post = self.rng.choice(TIER_3_CIB)
            elif tier == 1:
                node.recent_post = self.rng.choice(TIER_1_INFECTED)
            elif tier == 4:
                node.recent_post = self.rng.choice(TIER_4_COWARDICE)

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