# 🚀 Multi-Agent Reinforcement Learning: IQL vs QMIX

A from-scratch implementation of Independent Q-Learning (IQL) and QMIX for cooperative multi-agent tasks in a custom grid world environment.

## 🌍 Environment

- Grid size: `5×5`
- Agents: Configurable (2 to 10)
- Objective: Any agent reaches the goal
- Reward: `+1` if goal is reached, else `0`

## 📦 Project Structure

marl-qmix/
- env.py       # Custom GridWorld environment
- agent.py     # DQN agent logic
- qmix.py      # QMIX trainer and mixing network
- main_iql.py  # Runs IQL training
- main_qmix.py # Runs QMIX training
- results/     # PNG reward curves


## 📈 Results

| Agents | IQL Avg Reward | QMIX Avg Reward |
|--------|----------------|-----------------|
| 2      | ~0.80          | ~0.75           |
| 4      | ~0.60          | ~0.70           |
| 6      | ~0.50          | ~0.65           |
| 8      | ~0.40          | ~0.60           |
| 10     | ~0.20          | ~0.55           |

## 🧠 Key Concepts

- Multi-Agent Reinforcement Learning (MARL)
- Value Decomposition (QMIX)
- DQN, replay buffers, epsilon-greedy
- Centralized training, decentralized execution (CTDE)

## 🚀 How to Run

```bash
# Train with IQL
python main_iql.py 4

# Train with QMIX
python main_qmix.py 4
- 
