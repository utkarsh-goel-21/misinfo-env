"""
Pre-generate deterministic network data files for all three tasks.

These JSON files ensure every judge and evaluator running the environment
sees the exact same network topology, infection spread, and origin node.

Usage:
    python generate_data.py

Output:
    environment/data/network_easy.json
    environment/data/network_medium.json
    environment/data/network_hard.json
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.tasks.task1_detection import Task1Detection
from environment.tasks.task2_tracing import Task2Tracing
from environment.tasks.task3_containment import Task3Containment

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "environment",
    "data",
)

SEED = 42


def snapshot_to_dict(snapshot) -> dict:
    """Convert a NetworkSnapshot to a JSON-serialisable dict."""
    return {
        "step": snapshot.step,
        "nodes": {
            node_id: {
                "node_id": node.node_id,
                "status": node.status.value,
                "influence_score": node.influence_score,
                "skepticism_score": node.skepticism_score,
                "demographics": node.demographics.model_dump(),
                "recent_post": node.recent_post,
                "user_persona": node.user_persona,
                "community_id": node.community_id,
                "is_bot": node.is_bot,
                "dormant_until": node.dormant_until,
                "infected_at_step": node.infected_at_step,
                "neighbors": node.neighbors,
            }
            for node_id, node in snapshot.nodes.items()
        },
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "weight": e.weight,
            }
            for e in snapshot.edges
        ],
        "total_infected": snapshot.total_infected,
        "total_quarantined": snapshot.total_quarantined,
        "infection_threshold": snapshot.infection_threshold,
        "origin_node_id": snapshot.origin_node_id,
    }


def generate_task_data(task_cls, task_name: str, filename: str):
    """Run one task, capture its initial snapshot, save to JSON."""
    print(f"  Generating {task_name}...")
    task = task_cls(seed=SEED)
    snapshot = task.reset()

    data = {
        "task": task_name,
        "seed": SEED,
        "origin_node_id": task.graph.origin_node_id,
        "initial_infected": task.graph.get_infected_nodes(),
        "network_size": len(task.graph.nodes),
        "infection_threshold": task.graph.infection_threshold,
        "spread_steps_before_episode": getattr(
            task, "PRE_SPREAD_STEPS", 0
        ),
        "initial_snapshot": snapshot_to_dict(snapshot),
    }

    out_path = os.path.join(DATA_DIR, filename)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    infected_count = len(data["initial_infected"])
    total = data["network_size"]
    print(
        f"    ✓ {filename} — "
        f"{total} nodes, "
        f"{infected_count} infected initially, "
        f"origin: {data['origin_node_id']}"
    )
    return data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating deterministic network data files...")
    print(f"  Seed: {SEED}")
    print(f"  Output: {DATA_DIR}\n")

    generate_task_data(
        Task1Detection,
        "task1_detection",
        "network_easy.json",
    )
    generate_task_data(
        Task2Tracing,
        "task2_tracing",
        "network_medium.json",
    )
    generate_task_data(
        Task3Containment,
        "task3_containment",
        "network_hard.json",
    )

    print("\n✓ All data files generated successfully.")
    print("  These files ensure grader reproducibility across machines.")


if __name__ == "__main__":
    main()
