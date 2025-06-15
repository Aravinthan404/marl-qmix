import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

from env import MultiAgentGridEnv
from qmix import QMIXTrainer

print("✅ QMIX main_qmix.py started")

# ===== Config =====
NUM_AGENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
GRID_SIZE = 5
EPISODES = 200
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
STATE_DIM = NUM_AGENTS * 3 * GRID_SIZE * GRID_SIZE  # Flattened joint observation

# ===== Ensure results folder =====
if not os.path.exists("results"):
    os.makedirs("results")
    print("📁 Created 'results/' directory")

# ===== Initialize environment and trainer =====
env = MultiAgentGridEnv(grid_size=GRID_SIZE, num_agents=NUM_AGENTS)
input_shape = (3, GRID_SIZE, GRID_SIZE)

trainer = QMIXTrainer(
    n_agents=NUM_AGENTS,
    input_shape=input_shape,
    n_actions=5,
    state_dim=STATE_DIM
)

episode_rewards = []

# ===== Training Loop =====
for episode in range(EPISODES):
    print(f"\n▶️ Episode {episode + 1}/{EPISODES}")
    obs = env.reset()
    total_reward = 0
    done = False
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** episode))

    while not done:
        actions = []
        for i, agent in enumerate(trainer.agents):
            obs_i = np.transpose(obs[i], (2, 0, 1))  # HWC → CHW
            action = agent.select_action(obs_i, epsilon)
            actions.append(action)

        next_obs, reward, done = env.step(actions)
        total_reward += reward

        # Store transition for joint update
        trainer.store_transition((
            [np.transpose(o, (2, 0, 1)) for o in obs],
            actions,
            reward,
            [np.transpose(o, (2, 0, 1)) for o in next_obs],
            float(done)
        ))

        trainer.update()
        obs = next_obs

    if episode % TARGET_UPDATE_FREQ == 0:
        print(f"🔄 Updating target networks at episode {episode}")
        trainer.update_target_networks()

    episode_rewards.append(total_reward)
    print(f"✅ Episode {episode} complete | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

# ===== Plot Results =====
plt.plot(episode_rewards)
plt.title(f"Total Reward per Episode (QMIX | {NUM_AGENTS} Agents)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()

plot_path = f"results/qmix_rewards_{NUM_AGENTS}agents.png"
plt.savefig(plot_path)
plt.show()
print(f"\n📈 Training complete. Plot saved to: {plot_path}")
