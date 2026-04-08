"""
SENTINEL-9 — Comprehensive Test Suite

Tests every component: reset, step, POMDP, Brier scoring, graders,
reproducibility, budget, adversarial mechanics, edge cases.
"""

import pytest
from environment.env import MisinfoEnv
from environment.models import Action, ActionType, NodeStatus, TaskID


# ═══════════════════════════════════════
# RESET TESTS
# ═══════════════════════════════════════

class TestReset:
    def test_task1_reset(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        assert obs.task_id == TaskID.task1
        assert obs.network_size == 40
        assert obs.step_number == 0
        assert obs.max_steps == 10
        assert obs.infection_rate > 0
        assert len(obs.stream_reports) > 0
        assert obs.financial_budget == 10000.0
        assert obs.public_outrage_index == 0.0
        assert obs.brier_score_running == 0.0
        assert len(obs.revealed_nodes) == 0

    def test_task2_reset(self):
        env = MisinfoEnv(task_id="task2_tracing", seed=42)
        obs = env.reset()
        assert obs.task_id == TaskID.task2
        assert obs.network_size == 80
        assert obs.infection_rate > 0

    def test_task3_reset(self):
        env = MisinfoEnv(task_id="task3_containment", seed=42)
        obs = env.reset()
        assert obs.task_id == TaskID.task3
        assert obs.network_size == 150
        assert obs.actions_remaining is not None
        assert obs.actions_remaining == 5

    def test_invalid_task(self):
        with pytest.raises(ValueError):
            MisinfoEnv(task_id="task99_invalid", seed=42)


# ═══════════════════════════════════════
# STEP TESTS
# ═══════════════════════════════════════

class TestStep:
    def test_inspect_returns_data(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        target = obs.stream_reports[0]
        action = Action(action_type=ActionType.inspect, target_node_id=target, confidence=0.5)
        obs2, reward, done, info = env.step(action)
        assert info["action_valid"] is True
        assert obs2.inspection_results is not None
        assert target in obs2.inspection_results
        # Fog-of-war: target + neighbors should be revealed
        assert target in obs2.revealed_nodes
        assert len(obs2.revealed_nodes) > 1

    def test_quarantine_infected(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        # Find an actually infected node
        infected = env.task.graph.get_infected_nodes()
        target = infected[0]
        action = Action(action_type=ActionType.quarantine, target_node_id=target, confidence=0.9)
        obs2, reward, done, info = env.step(action)
        assert info["action_valid"] is True
        assert env.task.graph.nodes[target].status == NodeStatus.quarantined

    def test_quarantine_clean_node_penalty(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        # Find a clean node
        clean = [nid for nid, n in env.task.graph.nodes.items() if n.status == NodeStatus.clean]
        target = clean[0]
        action = Action(action_type=ActionType.quarantine, target_node_id=target, confidence=0.9)
        _, reward, _, info = env.step(action)
        # High confidence + wrong = high Brier
        assert info["brier_this_step"] > 0.5

    def test_invalid_action_type_for_task(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        # trace not allowed in task1
        action = Action(action_type=ActionType.trace, target_node_id="node_0", confidence=0.5)
        _, reward, _, info = env.step(action)
        assert "error" in info or reward.score < 0

    def test_invalid_node_id(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        action = Action(action_type=ActionType.inspect, target_node_id="node_9999", confidence=0.5)
        _, reward, _, info = env.step(action)
        assert reward.score < 0

    def test_step_after_done_raises(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        env.done = True
        action = Action(action_type=ActionType.inspect, target_node_id="node_0", confidence=0.5)
        with pytest.raises(RuntimeError):
            env.step(action)

    def test_trace_returns_timeline(self):
        env = MisinfoEnv(task_id="task2_tracing", seed=42)
        obs = env.reset()
        target = obs.stream_reports[0]
        action = Action(action_type=ActionType.trace, target_node_id=target, confidence=0.5)
        obs2, _, _, info = env.step(action)
        assert info["action_valid"] is True
        data = obs2.inspection_results[target]
        assert "betweenness_centrality" in data
        assert "neighbor_infection_timeline" in data
        assert "is_bridge" in data


# ═══════════════════════════════════════
# POMDP TESTS
# ═══════════════════════════════════════

class TestPOMDP:
    def test_fog_of_war_initially_empty(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        assert len(obs.revealed_nodes) == 0

    def test_inspect_reveals_neighbors(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        target = obs.stream_reports[0]
        action = Action(action_type=ActionType.inspect, target_node_id=target, confidence=0.5)
        obs2, _, _, _ = env.step(action)
        # Should reveal target + all its neighbors
        node = env.task.graph.nodes[target]
        expected = {target} | set(node.neighbors)
        assert set(obs2.revealed_nodes) == expected

    def test_stream_reports_contain_false_positives(self):
        """Over many resets, some stream report nodes should be clean."""
        false_positive_found = False
        for seed in range(10):
            env = MisinfoEnv(task_id="task1_detection", seed=seed)
            obs = env.reset()
            for node_id in obs.stream_reports:
                node = env.task.graph.nodes.get(node_id)
                if node and node.status == NodeStatus.clean:
                    false_positive_found = True
                    break
            if false_positive_found:
                break
        assert false_positive_found, "No false positives found across 10 seeds"


# ═══════════════════════════════════════
# BRIER SCORING TESTS
# ═══════════════════════════════════════

class TestBrier:
    def test_high_confidence_correct_low_brier(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        infected = env.task.graph.get_infected_nodes()
        # Quarantine infected node with high confidence
        action = Action(action_type=ActionType.quarantine, target_node_id=infected[0], confidence=0.95)
        _, _, _, info = env.step(action)
        # Brier = (0.95 - 1.0)² = 0.0025
        assert info["brier_this_step"] < 0.01

    def test_high_confidence_wrong_high_brier(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        clean = [nid for nid, n in env.task.graph.nodes.items() if n.status == NodeStatus.clean]
        # Quarantine clean node with high confidence
        action = Action(action_type=ActionType.quarantine, target_node_id=clean[0], confidence=0.95)
        _, _, _, info = env.step(action)
        # Brier = (0.95 - 0.0)² = 0.9025
        assert info["brier_this_step"] > 0.8

    def test_low_confidence_wrong_moderate_brier(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        clean = [nid for nid, n in env.task.graph.nodes.items() if n.status == NodeStatus.clean]
        action = Action(action_type=ActionType.quarantine, target_node_id=clean[0], confidence=0.3)
        _, _, _, info = env.step(action)
        # Brier = (0.3 - 0.0)² = 0.09
        assert info["brier_this_step"] < 0.15


# ═══════════════════════════════════════
# GRADER TESTS
# ═══════════════════════════════════════

class TestGraders:
    def test_grader1_perfect_score(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        # Quarantine ALL infected nodes with high confidence
        infected = env.task.graph.get_infected_nodes()
        for nid in infected:
            action = Action(action_type=ActionType.quarantine, target_node_id=nid, confidence=0.95)
            env.step(action)
        # Finish remaining steps
        while not env.done:
            action = Action(action_type=ActionType.inspect, target_node_id="node_0", confidence=0.5)
            env.step(action)
        # Should have high TPR
        state = env.state()
        assert state.cumulative_score > 0.2

    def test_grader2_correct_origin(self):
        env = MisinfoEnv(task_id="task2_tracing", seed=42)
        env.reset()
        origin = env.task.graph.origin_node_id
        tree = env.task.graph.get_causal_tree_as_dicts()
        chain = [{"from": origin, "to": tree[0]["to"]}] if tree else [{"from": origin, "to": "node_0"}]
        action = Action(action_type=ActionType.submit_causal_chain, confidence=0.8, causal_chain=chain)
        _, reward, done, _ = env.step(action)
        assert done is True
        assert reward.partial_credits["origin_correct"] is True
        assert reward.score > 0

    def test_grader3_scores_bounded(self):
        env = MisinfoEnv(task_id="task3_containment", seed=42)
        env.reset()
        # Take a few actions then submit
        for _ in range(3):
            action = Action(action_type=ActionType.inspect, target_node_id="node_0", confidence=0.5)
            env.step(action)
        chain = [{"from": env.task.graph.origin_node_id, "to": "node_1"}]
        action = Action(action_type=ActionType.submit_causal_chain, confidence=0.5, causal_chain=chain)
        _, reward, done, _ = env.step(action)
        assert -1.0 <= reward.score <= 1.0


# ═══════════════════════════════════════
# REPRODUCIBILITY TESTS
# ═══════════════════════════════════════

class TestReproducibility:
    def test_same_seed_same_episode(self):
        for task_id in ["task1_detection", "task2_tracing", "task3_containment"]:
            e1 = MisinfoEnv(task_id=task_id, seed=42)
            e2 = MisinfoEnv(task_id=task_id, seed=42)
            o1 = e1.reset()
            o2 = e2.reset()
            assert o1.infection_rate == o2.infection_rate
            assert o1.network_size == o2.network_size

    def test_different_seed_different_episode(self):
        e1 = MisinfoEnv(task_id="task1_detection", seed=42)
        e2 = MisinfoEnv(task_id="task1_detection", seed=99)
        o1 = e1.reset()
        o2 = e2.reset()
        # Highly unlikely to be identical
        assert o1.infection_rate != o2.infection_rate or o1.stream_reports != o2.stream_reports


# ═══════════════════════════════════════
# BUDGET TESTS
# ═══════════════════════════════════════

class TestBudget:
    def test_budget_depletes(self):
        env = MisinfoEnv(task_id="task3_containment", seed=42)
        obs = env.reset()
        initial = obs.financial_budget
        action = Action(action_type=ActionType.inspect, target_node_id="node_0", confidence=0.5)
        obs2, _, _, info = env.step(action)
        assert info["budget"] < initial
        assert info["budget"] == initial - 50.0  # inspect costs $50

    def test_expensive_action_costs(self):
        env = MisinfoEnv(task_id="task3_containment", seed=42)
        env.reset()
        # Remove costs $3000
        infected = env.task.graph.get_infected_nodes()
        action = Action(action_type=ActionType.remove, target_node_id=infected[0], confidence=0.8)
        _, _, _, info = env.step(action)
        assert info["budget"] == 10000.0 - 3000.0


# ═══════════════════════════════════════
# ADVERSARIAL TESTS
# ═══════════════════════════════════════

class TestAdversarial:
    def test_bots_exist_in_graph(self):
        env = MisinfoEnv(task_id="task3_containment", seed=42)
        env.reset()
        bots = env.task.graph.get_bot_nodes()
        assert len(bots) > 0

    def test_wrong_quarantine_increases_outrage(self):
        env = MisinfoEnv(task_id="task3_containment", seed=42)
        env.reset()
        assert env.public_outrage_index == 0.0
        clean = [nid for nid, n in env.task.graph.nodes.items() if n.status == NodeStatus.clean]
        action = Action(action_type=ActionType.quarantine, target_node_id=clean[0], confidence=0.8)
        env.step(action)
        assert env.public_outrage_index > 0

    def test_causal_tree_tracked(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        tree = env.task.graph.get_causal_tree_as_dicts()
        assert len(tree) > 0
        # Each edge should have 'from' and 'to' keys
        for edge in tree:
            assert "from" in edge
            assert "to" in edge


# ═══════════════════════════════════════
# STATE TESTS
# ═══════════════════════════════════════

class TestState:
    def test_state_returns_ground_truth(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        state = env.state()
        assert state.origin_node_id != ""
        assert len(state.infected_nodes) > 0
        assert len(state.causal_tree) > 0
        assert state.done is False

    def test_close_cleans_up(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        env.reset()
        env.close()
        assert env.task is None
        assert env.grader is None


# ═══════════════════════════════════════
# END-TO-END TESTS
# ═══════════════════════════════════════

class TestEndToEnd:
    def test_full_task1_episode(self):
        env = MisinfoEnv(task_id="task1_detection", seed=42)
        obs = env.reset()
        step_count = 0
        while not env.done:
            target = obs.stream_reports[0] if obs.stream_reports else "node_0"
            if step_count % 2 == 0:
                action = Action(action_type=ActionType.inspect, target_node_id=target, confidence=0.5)
            else:
                action = Action(action_type=ActionType.quarantine, target_node_id=target, confidence=0.7)
            obs, reward, done, info = env.step(action)
            step_count += 1
            assert -1.0 <= reward.score <= 1.0
        assert env.done

    def test_full_task2_episode_with_chain(self):
        env = MisinfoEnv(task_id="task2_tracing", seed=42)
        obs = env.reset()
        # Investigate then submit
        for _ in range(5):
            if obs.stream_reports:
                action = Action(action_type=ActionType.trace, target_node_id=obs.stream_reports[0], confidence=0.5)
                obs, _, _, _ = env.step(action)
            if env.done:
                break
        if not env.done:
            origin = env.task.graph.origin_node_id
            chain = [{"from": origin, "to": "node_0"}]
            action = Action(action_type=ActionType.submit_causal_chain, confidence=0.6, causal_chain=chain)
            obs, reward, done, _ = env.step(action)
            assert done is True
