import numpy as np
from env import MultiAgentGridEnv
import os
print("Loading agent.py from:", os.getcwd())
from agent import Agent as ImportedAgent
Agent = ImportedAgent 
import torch
import matplotlib.pyplot as plt
import os

# Debug check
print("✅ main.py started")

# Configuration
import sys
NUM_AGENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # Default to 2 agents if not specified
GRID_SIZE = 5
EPISODES = 200
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10

# Make sure results/ folder exists
if not os.path.exists("results"):
    os.makedirs("results")
    print("📁 Created 'results/' directory")

# Initialize environment and agents
env = MultiAgentGridEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS)
input_shape = (3, GRID_SIZE, GRID_SIZE)
agents = [Agent(input_shape=input_shape, num_actions=5) for _ in range(NUM_AGENTS)]

episode_rewards = []

for episode in range(EPISODES):
    print(f"\n▶️ Episode {episode + 1}/{EPISODES}")
    obs = env.reset()
    total_reward = 0
    done = False
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))

    while not done:
        actions = []
        for i, agent in enumerate(agents):
            obs_i = np.transpose(obs[i], (2, 0, 1))  # [HWC] → [CHW]
            action = agent.select_action(obs_i, epsilon)
            actions.append(action)

        next_obs, reward, done = env.step(actions)
        total_reward += reward

        for i, agent in enumerate(agents):
            transition = (
                np.transpose(obs[i], (2, 0, 1)),
                actions[i],
                reward,
                np.transpose(next_obs[i], (2, 0, 1)),
                float(done)
            )
            agent.store_transition(transition)
            agent.update()

        obs = next_obs

    if episode % TARGET_UPDATE_FREQ == 0:
        print(f"🔄 Updating target networks at episode {episode}")
        for agent in agents:
            agent.update_target_network()

    episode_rewards.append(total_reward)
    print(f"✅ Episode {episode} complete | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# Plot reward graph
plt.plot(episode_rewards)
plt.title("Total Reward per Episode (IQL Baseline)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plot_path = f"results/iql_rewards_{NUM_AGENTS}agents.png"
plt.savefig(plot_path)
plt.show()
print(f"\n📈 Training complete. Plot saved to: {plot_path}")
