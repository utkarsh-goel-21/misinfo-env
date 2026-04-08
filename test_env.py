from environment.env import MisinfoEnv
from environment.models import Action

env = MisinfoEnv("task3_containment", seed=42)
obs = env.reset()
print(env.state().financial_budget)
result = env.step(Action(action_type="shadowban", target_node_id="node_0", confidence=0.9))
print(env.state().financial_budget)
print("Done!")
