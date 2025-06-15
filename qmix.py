import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random  # ✅ Use for proper sampling
from agent import Agent


class MixingNetwork(nn.Module):
    def __init__(self, n_agents, state_dim):
        super(MixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * 32)
        self.hyper_b1 = nn.Linear(state_dim, 32)
        self.hyper_w2 = nn.Linear(state_dim, 32)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, agent_qs, state):
        # agent_qs: [batch, n_agents]
        # state: [batch, state_dim]
        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(state)).view(batch_size, self.n_agents, 32)
        b1 = self.hyper_b1(state).view(batch_size, 1, 32)
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1  # [batch, 1, 32]
        hidden = torch.relu(hidden)

        w2 = torch.abs(self.hyper_w2(state)).view(batch_size, 32, 1)
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)
        q_total = torch.bmm(hidden, w2) + b2  # [batch, 1, 1]

        return q_total.view(-1, 1)


class QMIXTrainer:
    def __init__(self, n_agents, input_shape, n_actions, state_dim, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=10000):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size

        self.agents = [Agent(input_shape, n_actions) for _ in range(n_agents)]
        self.mixer = MixingNetwork(n_agents, state_dim)
        self.target_mixer = MixingNetwork(n_agents, state_dim)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.mixer_optimizer = optim.Adam(self.mixer.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self.replay_buffer = []

    def store_transition(self, transition):
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)  # [B, N, C, H, W]
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)  # [B, N]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)  # [B, 1]
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        B, N, C, H, W = states.shape
        state_flat = states.view(B, -1)
        next_state_flat = next_states.view(B, -1)

        # Get agent Q-values
        agent_qs = []
        target_qs = []
        for i in range(self.n_agents):
            q_vals = self.agents[i].policy_net(states[:, i]).gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_vals = self.agents[i].target_net(next_states[:, i]).max(1)[0]
            agent_qs.append(q_vals)
            target_qs.append(next_q_vals)

        agent_qs = torch.stack(agent_qs, dim=1)         # [B, N]
        target_qs = torch.stack(target_qs, dim=1)       # [B, N]

        # Mix with mixer networks
        q_total = self.mixer(agent_qs, state_flat)
        with torch.no_grad():
            target_q_total = self.target_mixer(target_qs, next_state_flat)
            targets = rewards + self.gamma * target_q_total * (1 - dones)

        # Loss and update
        loss = nn.MSELoss()(q_total, targets)
        self.mixer_optimizer.zero_grad()
        loss.backward()
        self.mixer_optimizer.step()

        # Update agent networks
        for agent in self.agents:
            agent.update()

    def update_target_networks(self):
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        for agent in self.agents:
            agent.update_target_network()
