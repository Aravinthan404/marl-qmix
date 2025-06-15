from env import MultiAgentGridEnv

env = MultiAgentGridEnv(grid_size=5, num_agents=2)
obs = env.reset()
env.render()

done = False
while not done:
    actions = [4, 4]  # All agents stay
    obs, reward, done = env.step(actions)
    env.render()
    print(f"Reward: {reward}")
