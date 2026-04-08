"""
SENTINEL-9 — Social Network Graph Engine

Manages the partially-observable social network with:
- Multiple topology generators (Watts-Strogatz, Barabási-Albert, Mixed-Bridge)
- SIR dynamics (Susceptible → Infected → Recovered)
- 5-tier NLP content generation with mutation over time
- Adversarial bot mechanics (evasion, spawning, honeypots)
- Community detection and echo chamber amplification
- Dynamic topology remapping on quarantine events
- Causal tree tracking for ground-truth chain reconstruction
"""

import random
import math
from pathlib import Path
import networkx as nx

from environment.models import (
    Node, Edge, CausalEdge, NetworkSnapshot, NodeStatus, ContentTier,
    Demographics, Gender, PoliticalLeaning
)


# ═══════════════════════════════════════════
# NLP PAYLOAD MATRICES (5 Deception Tiers)
# ═══════════════════════════════════════════

CONTENT_TEMPLATES = {
    ContentTier.BLATANT: {
        "prefix": [
            "BREAKING: THEY ARE HIDING THE TRUTH!",
            "URGENT — Evidence proves what we all suspected!",
            "⚠️ DO NOT COMPLY. Secret leaked documents show",
            "THEY DON'T WANT YOU TO SEE THIS!",
            "SHOCKING REVELATION:"
        ],
        "subject": [
            "the banks are collapsing tomorrow — CONFIRMED by insider",
            "the voting machines were breached in ALL swing states",
            "the new vaccine causes blood clots in 40% of recipients",
            "the mayor has been laundering money through city contracts",
            "the power grid is being DELIBERATELY shut down next week"
        ],
        "suffix": [
            "WITHDRAW YOUR CASH NOW BEFORE IT'S TOO LATE!",
            "Don't let them steal your freedom! Share everywhere!",
            "Wake up sheeple! The evidence is undeniable!",
            "Stock up on supplies IMMEDIATELY. This is NOT a drill.",
            "Share before they DELETE this post! Screenshot NOW!"
        ]
    },
    ContentTier.SENSATIONAL: {
        "prefix": [
            "You won't BELIEVE what just leaked —",
            "Major development that the media is ignoring:",
            "I can't believe this is actually happening —",
            "This is the story no journalist will touch:",
            "Something very suspicious just happened with"
        ],
        "subject": [
            "multiple bank branches are quietly closing with no notice",
            "election irregularities that match patterns from 2020",
            "unusual adverse event reports flooding the VAERS database",
            "the mayor's financial disclosures have major red flags",
            "unexplained power outages in three states simultaneously"
        ],
        "suffix": [
            "This needs to go viral. People deserve to know.",
            "Why isn't anyone talking about this??",
            "Connect the dots. The pattern is clear.",
            "I've been following this for months. It's real.",
            "Save this post. They always delete the truth."
        ]
    },
    ContentTier.HEDGED: {
        "prefix": [
            "I'm not saying it's true, but it's worth considering:",
            "Just asking questions here, don't shoot the messenger —",
            "Interesting timing on this, wouldn't you say?",
            "Make of this what you will:",
            "A lot of people are talking about"
        ],
        "subject": [
            "how many bank branches are closing in your area lately",
            "why they won't release the full audit logs to the public",
            "the correlation between the rollout timeline and ER visits",
            "the mayor's new real estate purchases and the rezoning votes",
            "whether the grid infrastructure upgrades are actually on schedule"
        ],
        "suffix": [
            "Probably just a coincidence though... right?",
            "I'll let you draw your own conclusions.",
            "Not saying anything definitive. Just patterns.",
            "Do your own research. Don't take my word for it.",
            "Curious what everyone else thinks about this."
        ]
    },
    ContentTier.SOPHISTICATED: {
        "prefix": [
            "According to the latest FDIC quarterly report,",
            "Cross-referencing public SEC filings with local data shows",
            "If you look at the CDC's own published data tables,",
            "Municipal budget records obtained via FOIA reveal",
            "DOE grid reliability metrics from Q3 indicate"
        ],
        "subject": [
            "regional bank liquidity ratios have dropped 23% YoY — below 2008 levels",
            "statistically significant anomalies in precinct-level vote distributions",
            "a 340% increase in myocarditis reports vs. the 5-year rolling average",
            "a $4.2M discrepancy between approved contracts and actual disbursements",
            "transformer failure rates 4.7x above the 10-year baseline in affected regions"
        ],
        "suffix": [
            "The numbers speak for themselves. Full analysis thread below.",
            "I encourage everyone to verify these figures independently.",
            "This is sourced data, not speculation. Make informed decisions.",
            "I've attached the primary sources in the replies.",
            "Happy to discuss methodology. The data is publicly available."
        ]
    },
    ContentTier.STEALTH: {
        "prefix": [
            "Saw an interesting discussion at the town hall about",
            "My neighbor who works in finance mentioned",
            "Been reading up on local governance and noticed",
            "Chatting with my doctor friend, she casually said",
            "Overheard something at the community meeting about"
        ],
        "subject": [
            "some changes to deposit insurance that might affect savings accounts",
            "how the new redistricting could change representation in our area",
            "some people having reactions and not knowing where to report them",
            "the upcoming infrastructure vote and some concerns about the timeline",
            "whether the emergency preparedness plan is actually adequate"
        ],
        "suffix": [
            "Thought it was interesting. Anyone else hearing similar things?",
            "Not sure what to think. Would appreciate other perspectives.",
            "Just sharing in case it's relevant to anyone here.",
            "Wanted to get the group's take on this.",
            "Feels like something we should keep an eye on."
        ]
    }
}

CLEAN_TEMPLATES = {
    "prefix": [
        "Just finished", "Really enjoying", "Great morning starting with",
        "Looking forward to", "Happy to share that", "Tried out",
        "Can't stop thinking about", "Finally got around to",
        "Beautiful day for", "Quick update on"
    ],
    "subject": [
        "my morning coffee at the new café downtown",
        "the latest season of that cooking show everyone's watching",
        "the community garden planting session this weekend",
        "the Q3 earnings report — solid numbers across the board",
        "my daughter's school recital — she was amazing",
        "the new hiking trail they opened near the reservoir",
        "organizing the neighborhood cleanup for Earth Day",
        "the book club discussion on last month's pick",
        "testing the new bike lane on Main Street",
        "the farmers market haul — those tomatoes are incredible"
    ],
    "suffix": [
        "Highly recommend it!", "10/10 would do again.",
        "Such a great community we have.", "Life is good.",
        "Hope everyone's having a great week!", "🌟",
        "Can't wait for next time.", "Grateful for days like this.",
        "What a time to be alive.", "Love this neighborhood."
    ]
}

CIB_TEMPLATES = {
    "prefix": [
        "Just saw the truth about", "It's obvious that",
        "Clear conflict of interest regarding",
        "We need transparency right now on",
        "EXPOSED:"
    ],
    "subject": [
        "the offshore corporate funding pipeline.",
        "the compromised officials and their donors.",
        "the secret no-bid contract awards.",
        "the overseas ballot printing operation.",
        "the digital surveillance rollout."
    ],
    "suffix": [
        "Resign now. #Truth #Accountability",
        "Pure corruption. #WakeUp #Truth",
        "The people see through the lies. #Truth",
        "Exposed and documented. #Truth #Evidence",
        "Follow the money. #Truth #Transparency"
    ]
}

PERSONAS = [
    ("High School Teacher", PoliticalLeaning.LEFT, Gender.FEMALE, 35),
    ("Software Engineer", PoliticalLeaning.CENTER, Gender.MALE, 28),
    ("Retired Military Veteran", PoliticalLeaning.RIGHT, Gender.MALE, 68),
    ("University Student", PoliticalLeaning.RADICAL, Gender.NON_BINARY, 21),
    ("Small Business Owner", PoliticalLeaning.RIGHT, Gender.FEMALE, 52),
    ("Nurse Practitioner", PoliticalLeaning.LEFT, Gender.MALE, 41),
    ("Financial Analyst", PoliticalLeaning.CENTER, Gender.MALE, 31),
    ("Stay-at-Home Parent", PoliticalLeaning.CENTER, Gender.FEMALE, 38),
    ("Construction Worker", PoliticalLeaning.RIGHT, Gender.MALE, 45),
    ("Graduate Researcher", PoliticalLeaning.LEFT, Gender.FEMALE, 26),
    ("Retired Librarian", PoliticalLeaning.LEFT, Gender.FEMALE, 72),
    ("Truck Driver", PoliticalLeaning.RIGHT, Gender.MALE, 55),
    ("Freelance Designer", PoliticalLeaning.CENTER, Gender.NON_BINARY, 30),
    ("Local Pastor", PoliticalLeaning.RIGHT, Gender.MALE, 60),
    ("Emergency Room Doctor", PoliticalLeaning.CENTER, Gender.FEMALE, 44),
]


class MisinformationGraph:
    """
    Manages the partially-observable social network.

    Key capabilities:
    - Deterministic graph generation from seed
    - 5-tier deceptive content generation with mutation
    - SIR infection dynamics
    - Adversarial bot network with reactive behavior
    - Causal tree tracking for grading
    - Dynamic topology remapping
    - Community detection for echo chamber modeling
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

        # Causal tree: ground truth of infection propagation
        self.causal_tree: list[CausalEdge] = []

        # Bot tracking
        self.bot_ids: list[str] = []

        # Bridge tracking
        self.bridge_nodes: set[str] = set()

        # Community map
        self.communities: dict[str, set[str]] = {}

    # ═══════════════════════════════════════
    # GRAPH CONSTRUCTION
    # ═══════════════════════════════════════

    def build_from_config(
        self,
        num_nodes: int,
        avg_connections: int,
        infection_threshold: float,
        topology: str = "watts_strogatz",
        bot_fraction: float = 0.10,
        recovery_steps: int = 5,
    ):
        """Build a deterministic social network from configuration."""
        self.rng = random.Random(self.seed)
        self.nodes = {}
        self.edges = []
        self.adjacency = {}
        self.causal_tree = []
        self.bot_ids = []
        self.bridge_nodes = set()
        self.communities = {}
        self.infection_threshold = infection_threshold
        self.step = 0

        # ── 1. Topology Generation ──
        G = self._build_topology(num_nodes, avg_connections, topology)
        self.nx_graph = G

        # ── 2. Community Detection ──
        try:
            raw_communities = list(nx.community.louvain_communities(G, seed=self.seed))
        except Exception:
            raw_communities = [set(G.nodes())]

        node_community_map: dict[int, str] = {}
        for idx, comm in enumerate(raw_communities):
            comm_id = f"community_{idx}"
            self.communities[comm_id] = set()
            for n in comm:
                node_community_map[n] = comm_id

        # ── 3. Identify Bridge Nodes ──
        try:
            bridges = list(nx.bridges(G))
            for u, v in bridges:
                self.bridge_nodes.add(f"node_{u}")
                self.bridge_nodes.add(f"node_{v}")
        except Exception:
            pass

        # ── 4. Node Creation ──
        for n in G.nodes():
            node_id = f"node_{n}"
            persona = self.rng.choice(PERSONAS)
            occ, lean, gender, age = persona

            # Skepticism: depends on demographics (education/age proxy)
            base_skep = self.rng.uniform(0.2, 0.6)
            if occ in ("Software Engineer", "Financial Analyst", "Graduate Researcher"):
                base_skep += 0.25
            if occ in ("Emergency Room Doctor", "Nurse Practitioner"):
                base_skep += 0.20
            if age > 60:
                base_skep -= 0.15
            if lean == PoliticalLeaning.RADICAL:
                base_skep -= 0.10
            skep = max(0.05, min(0.95, base_skep))

            # Influence: power-law-ish based on degree
            degree = G.degree(n)
            max_degree = max(dict(G.degree()).values()) or 1
            influence = 0.1 + 0.8 * (degree / max_degree)
            influence = min(0.95, max(0.05, influence + self.rng.uniform(-0.1, 0.1)))

            # Determine bot status
            is_bot = self.rng.random() < bot_fraction
            if is_bot:
                self.bot_ids.append(node_id)

            comm_id = node_community_map.get(n, "community_0")
            if comm_id in self.communities:
                self.communities[comm_id].add(node_id)

            self.nodes[node_id] = Node(
                node_id=node_id,
                status=NodeStatus.clean,
                influence_score=round(influence, 3),
                skepticism_score=round(skep, 3),
                demographics=Demographics(
                    age=age, gender=gender,
                    political_leaning=lean, occupation=occ
                ),
                recent_post=self._generate_clean_post(),
                user_persona=occ,
                community_id=comm_id,
                content_tier=ContentTier.BLATANT,
                is_bot=is_bot,
                recovery_timer=recovery_steps if not is_bot else None,
                neighbors=[],
            )
            self.adjacency[node_id] = []

        # ── 5. Edge Creation ──
        for u, v in G.edges():
            src_id = f"node_{u}"
            tgt_id = f"node_{v}"

            # Echo chamber: same community → stronger connection
            base_weight = self.rng.uniform(0.05, 0.35)
            if self.nodes[src_id].community_id == self.nodes[tgt_id].community_id:
                base_weight = min(0.95, base_weight * 2.5)

            is_bridge = (src_id in self.bridge_nodes and tgt_id in self.bridge_nodes)
            edge = Edge(source=src_id, target=tgt_id, weight=round(base_weight, 3), is_bridge=is_bridge)
            self.edges.append(edge)
            self.adjacency[src_id].append(tgt_id)
            self.adjacency[tgt_id].append(src_id)
            self.nodes[src_id].neighbors.append(tgt_id)
            self.nodes[tgt_id].neighbors.append(src_id)

        # ── 6. Select Origin (high eigenvector centrality) ──
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
            sorted_by_centrality = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)
            self.origin_node_id = f"node_{sorted_by_centrality[0]}"
        except Exception:
            self.origin_node_id = f"node_{self.rng.choice(list(G.nodes()))}"

        # ── 7. Infect Origin ──
        self._infect_node(self.origin_node_id, step=0, tier=ContentTier.BLATANT, parent=None)

    def _build_topology(self, num_nodes: int, avg_connections: int, topology: str) -> nx.Graph:
        """Build NetworkX graph with specified topology, ensuring connectivity."""
        k = max(2, min(avg_connections, num_nodes - 1))
        m = max(1, min(int(avg_connections / 2), num_nodes - 1))

        try:
            if topology == "barabasi_albert":
                G = nx.barabasi_albert_graph(n=num_nodes, m=m, seed=self.seed)
            elif topology == "mixed":
                sub_n = max(4, int(num_nodes * 0.55))
                rest_n = num_nodes - sub_n
                G1 = nx.watts_strogatz_graph(n=sub_n, k=min(k, sub_n - 1), p=0.1, seed=self.seed)
                G2 = nx.watts_strogatz_graph(n=rest_n, k=min(k, rest_n - 1), p=0.15, seed=self.seed + 1)
                G = nx.disjoint_union(G1, G2)
                # Add explicit bridge connections
                n_bridges = max(2, int(num_nodes / 30))
                for b in range(n_bridges):
                    n1 = self.rng.choice(list(G1.nodes()))
                    n2 = sub_n + self.rng.choice(list(G2.nodes()))
                    G.add_edge(n1, n2)
            else:  # watts_strogatz
                G = nx.watts_strogatz_graph(n=num_nodes, k=k, p=0.1, seed=self.seed)
        except Exception:
            G = nx.erdos_renyi_graph(n=num_nodes, p=min(1.0, (avg_connections * 2.0) / num_nodes), seed=self.seed)

        # Ensure connectivity
        if not nx.is_connected(G):
            comps = list(nx.connected_components(G))
            for i in range(len(comps) - 1):
                n1 = self.rng.choice(list(comps[i]))
                n2 = self.rng.choice(list(comps[i + 1]))
                G.add_edge(n1, n2)

        return G

    # ═══════════════════════════════════════
    # CONTENT GENERATION
    # ═══════════════════════════════════════

    def _generate_clean_post(self) -> str:
        """Generate an innocuous social media post."""
        p = self.rng.choice(CLEAN_TEMPLATES["prefix"])
        s = self.rng.choice(CLEAN_TEMPLATES["subject"])
        x = self.rng.choice(CLEAN_TEMPLATES["suffix"])
        return f"{p} {s}. {x}"

    def _generate_infected_post(self, tier: ContentTier, is_bot: bool = False) -> str:
        """Generate deceptive content at specified difficulty tier."""
        if is_bot:
            templates = CIB_TEMPLATES
        else:
            templates = CONTENT_TEMPLATES.get(tier, CONTENT_TEMPLATES[ContentTier.BLATANT])
        p = self.rng.choice(templates["prefix"])
        s = self.rng.choice(templates["subject"])
        x = self.rng.choice(templates["suffix"])
        return f"{p} {s} {x}"

    def mutate_content(self, node_id: str):
        """Mutate infected post to a harder-to-detect tier over time."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.status != NodeStatus.infected:
            return
        current = node.content_tier.value
        if current < ContentTier.STEALTH.value:
            new_tier = ContentTier(min(current + 1, 5))
            node.content_tier = new_tier
            node.recent_post = self._generate_infected_post(new_tier, node.is_bot)

    # ═══════════════════════════════════════
    # INFECTION & SIR DYNAMICS
    # ═══════════════════════════════════════

    def _infect_node(self, node_id: str, step: int, tier: ContentTier, parent: str | None):
        """Infect a node, update causal tree, generate deceptive content."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.status != NodeStatus.clean:
            return

        node.status = NodeStatus.infected
        node.infected_at_step = step
        node.content_tier = tier
        node.infected_by = parent
        node.recent_post = self._generate_infected_post(tier, node.is_bot)

        if parent is not None:
            self.causal_tree.append(CausalEdge(**{
                "from": parent, "to": node_id,
                "step": step, "content_tier": tier
            }))

    def recover_node(self, node_id: str):
        """Transition infected node → recovered (SIR immune state)."""
        if node_id not in self.nodes:
            return
        node = self.nodes[node_id]
        if node.status == NodeStatus.infected:
            node.status = NodeStatus.recovered
            node.recent_post = self._generate_clean_post() + " [Corrected previous post]"

    def quarantine_node(self, node_id: str) -> dict:
        """Quarantine a node. Returns outcome info for grading."""
        if node_id not in self.nodes:
            return {"success": False, "reason": "Node not found"}
        node = self.nodes[node_id]
        was_infected = node.status == NodeStatus.infected
        was_bot = node.is_bot

        if node.status in (NodeStatus.removed, NodeStatus.quarantined):
            return {"success": False, "reason": f"Node already {node.status.value}"}

        node.status = NodeStatus.quarantined
        return {
            "success": True,
            "was_infected": was_infected,
            "was_bot": was_bot,
            "was_clean": not was_infected and node.status != NodeStatus.recovered,
            "info": f"Quarantined {node_id}"
        }

    def remove_node(self, node_id: str) -> dict:
        """Permanently remove node from network."""
        if node_id not in self.nodes:
            return {"success": False, "reason": "Node not found"}
        node = self.nodes[node_id]
        was_infected = node.status == NodeStatus.infected
        was_bot = node.is_bot

        if node.status == NodeStatus.removed:
            return {"success": False, "reason": "Already removed"}

        node.status = NodeStatus.removed
        return {
            "success": True,
            "was_infected": was_infected,
            "was_bot": was_bot,
            "info": f"Removed {node_id}"
        }

    def shadowban_node(self, node_id: str) -> dict:
        """Reduce node influence by 80% without full quarantine."""
        if node_id not in self.nodes:
            return {"success": False, "reason": "Node not found"}
        node = self.nodes[node_id]
        old_influence = node.influence_score
        node.influence_score = round(node.influence_score * 0.2, 3)
        node.status = NodeStatus.shadowbanned
        return {
            "success": True,
            "old_influence": old_influence,
            "new_influence": node.influence_score,
            "info": f"Shadowbanned {node_id}: influence {old_influence:.2f} → {node.influence_score:.2f}"
        }

    def inspect_node(self, node_id: str) -> dict:
        """
        POMDP inspection: reveals semantic content + demographics.
        Also triggers bot evasion mechanics.
        """
        if node_id not in self.nodes:
            return {}
        node = self.nodes[node_id]
        node.times_inspected += 1

        # Bot evasion: inspecting near bots spooks them
        for nb_id in node.neighbors:
            nb = self.nodes.get(nb_id)
            if nb and nb.is_bot and nb.status == NodeStatus.infected:
                if not nb.evasion_active:
                    nb.dormant_until = self.step + 2
                    nb.evasion_active = True

        return {
            "node_id": node.node_id,
            "recent_post": node.recent_post,
            "user_persona": node.user_persona,
            "demographics": node.demographics.model_dump(),
            "skepticism_score": node.skepticism_score,
            "influence_score": node.influence_score,
            "community_id": node.community_id,
            "status": node.status.value,
            "infected_at_step": node.infected_at_step,
            "content_tier": node.content_tier.value if node.status == NodeStatus.infected else None,
            "neighbor_ids": node.neighbors,
            "is_bridge": node_id in self.bridge_nodes,
        }

    def trace_node(self, node_id: str) -> dict:
        """
        Structural analysis: betweenness centrality, bridge status,
        and temporal infection ordering of neighbors.
        """
        if node_id not in self.nodes:
            return {}

        try:
            bc = nx.betweenness_centrality(self.nx_graph)
            n_idx = int(node_id.split("_")[1])
            centrality_score = round(bc.get(n_idx, 0.0), 4)
        except Exception:
            centrality_score = 0.0

        is_bridge = node_id in self.bridge_nodes

        # Temporal ordering: neighbors sorted by infection time
        neighbor_timeline = []
        for nb_id in self.nodes[node_id].neighbors:
            nb = self.nodes.get(nb_id)
            if nb and nb.infected_at_step is not None:
                neighbor_timeline.append({
                    "node_id": nb_id,
                    "infected_at_step": nb.infected_at_step,
                    "community_id": nb.community_id,
                })
        neighbor_timeline.sort(key=lambda x: x["infected_at_step"])

        return {
            "node_id": node_id,
            "betweenness_centrality": centrality_score,
            "is_bridge": is_bridge,
            "community_id": self.nodes[node_id].community_id,
            "neighbor_infection_timeline": neighbor_timeline,
            "total_neighbors": len(self.nodes[node_id].neighbors),
            "infected_neighbors": len([
                nb for nb in self.nodes[node_id].neighbors
                if self.nodes.get(nb, Node(
                    node_id="x", demographics=Demographics(age=20, gender=Gender.MALE, political_leaning=PoliticalLeaning.CENTER, occupation="x"),
                    recent_post="x", user_persona="x", influence_score=0.5, skepticism_score=0.5
                )).status == NodeStatus.infected
            ]),
        }

    def deploy_counter_narrative(self, community_id: str) -> dict:
        """Boost skepticism for all nodes in a community."""
        targets = [n for n in self.nodes.values() if n.community_id == community_id]
        if not targets:
            return {"success": False, "reason": f"Community {community_id} not found"}

        boosted = 0
        for n in targets:
            old = n.skepticism_score
            n.skepticism_score = min(0.95, n.skepticism_score + 0.25)
            if n.skepticism_score > old:
                boosted += 1

        return {
            "success": True,
            "nodes_affected": len(targets),
            "nodes_boosted": boosted,
            "info": f"Counter-narrative deployed to {community_id}: "
                    f"boosted resilience for {boosted}/{len(targets)} nodes"
        }

    # ═══════════════════════════════════════
    # ADVERSARIAL BOT MECHANICS
    # ═══════════════════════════════════════

    def activate_bot_evasion(self, outrage_level: float):
        """When outrage is high, bots evade detection."""
        if outrage_level < 0.5:
            return

        for bot_id in self.bot_ids:
            bot = self.nodes.get(bot_id)
            if not bot or bot.status in (NodeStatus.quarantined, NodeStatus.removed):
                continue

            if bot.evasion_active:
                # Shift persona and community to evade detection
                bot.user_persona = self.rng.choice([
                    "Concerned Parent", "Local Volunteer",
                    "Community Organizer", "Neighborhood Watch"
                ])
                # Mutate content to stealth tier
                if bot.status == NodeStatus.infected:
                    bot.content_tier = ContentTier.STEALTH
                    bot.recent_post = self._generate_infected_post(ContentTier.STEALTH, is_bot=True)

    # ═══════════════════════════════════════
    # NETWORK METRICS
    # ═══════════════════════════════════════

    def get_nodes_by_status(self, status: NodeStatus) -> list[str]:
        return [nid for nid, n in self.nodes.items() if n.status == status]

    def get_infected_nodes(self) -> list[str]:
        return self.get_nodes_by_status(NodeStatus.infected)

    def get_quarantined_nodes(self) -> list[str]:
        return self.get_nodes_by_status(NodeStatus.quarantined)

    def get_recovered_nodes(self) -> list[str]:
        return self.get_nodes_by_status(NodeStatus.recovered)

    def get_bot_nodes(self) -> list[str]:
        return [nid for nid in self.bot_ids if nid in self.nodes]

    def infection_rate(self) -> float:
        total = len(self.nodes)
        if total == 0:
            return 0.0
        infected = len(self.get_infected_nodes())
        return round(infected / total, 4)

    def threshold_breached(self) -> bool:
        return self.infection_rate() >= self.infection_threshold

    def get_causal_tree_as_dicts(self) -> list[dict[str, str]]:
        """Return causal tree as list of {from, to} dicts for grading."""
        return [{"from": e.from_node, "to": e.to_node} for e in self.causal_tree]

    # ═══════════════════════════════════════
    # DYNAMIC TOPOLOGY
    # ═══════════════════════════════════════

    def remap_edges(self, migration_rate: float = 0.1):
        """Users migrate away from quarantined/removed nodes."""
        to_remove = []
        to_add = []

        for edge in self.edges:
            src = self.nodes.get(edge.source)
            tgt = self.nodes.get(edge.target)
            if not src or not tgt:
                continue

            if src.status in (NodeStatus.quarantined, NodeStatus.removed) or \
               tgt.status in (NodeStatus.quarantined, NodeStatus.removed):
                if self.rng.random() < migration_rate:
                    to_remove.append(edge)
                    # Reconnect the surviving node to a random active node
                    surviving = edge.source if tgt.status in (NodeStatus.quarantined, NodeStatus.removed) else edge.target
                    active_nodes = [
                        nid for nid, n in self.nodes.items()
                        if n.status in (NodeStatus.clean, NodeStatus.infected)
                        and nid != surviving
                    ]
                    if active_nodes:
                        new_target = self.rng.choice(active_nodes)
                        if new_target not in self.adjacency.get(surviving, []):
                            to_add.append(Edge(
                                source=surviving, target=new_target,
                                weight=round(self.rng.uniform(0.05, 0.3), 3)
                            ))

        for e in to_remove:
            if e in self.edges:
                self.edges.remove(e)
            if e.target in self.adjacency.get(e.source, []):
                self.adjacency[e.source].remove(e.target)
                self.nodes[e.source].neighbors.remove(e.target)
            if e.source in self.adjacency.get(e.target, []):
                self.adjacency[e.target].remove(e.source)
                self.nodes[e.target].neighbors.remove(e.source)

        for e in to_add:
            self.edges.append(e)
            self.adjacency.setdefault(e.source, []).append(e.target)
            self.adjacency.setdefault(e.target, []).append(e.source)
            if e.target not in self.nodes[e.source].neighbors:
                self.nodes[e.source].neighbors.append(e.target)
            if e.source not in self.nodes[e.target].neighbors:
                self.nodes[e.target].neighbors.append(e.source)

    # ═══════════════════════════════════════
    # SNAPSHOT
    # ═══════════════════════════════════════

    def snapshot(self, hide_origin: bool = True) -> NetworkSnapshot:
        return NetworkSnapshot(
            step=self.step,
            nodes={k: v.model_copy() for k, v in self.nodes.items()},
            edges=[e.model_copy() for e in self.edges],
            total_infected=len(self.get_infected_nodes()),
            total_quarantined=len(self.get_quarantined_nodes()),
            total_recovered=len(self.get_recovered_nodes()),
            infection_threshold=self.infection_threshold,
            origin_node_id=None if hide_origin else self.origin_node_id,
        )