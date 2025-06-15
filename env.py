import numpy as np

class MultiAgentGridEnv:
    def __init__(self, grid_size=5, num_agents=2):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.action_space = 5  # up, down, left, right, stay
        self.reset()

    def reset(self):
        self.agent_positions = []
        for _ in range(self.num_agents):
            pos = self._random_position(exclude=[])
            self.agent_positions.append(pos)
        self.goal_position = self._random_position(exclude=self.agent_positions)
        self.steps = 0
        return self._get_obs()

    def _random_position(self, exclude):
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in exclude:
                return pos

    def _get_obs(self):
        obs = np.zeros((self.num_agents, self.grid_size, self.grid_size, 3))
        for i, (x, y) in enumerate(self.agent_positions):
            obs[i, x, y, 0] = 1
            obs[i, self.goal_position[0], self.goal_position[1], 1] = 1
            for j, (ox, oy) in enumerate(self.agent_positions):
                if j != i:
                    obs[i, ox, oy, 2] = 1
        return obs

    def step(self, actions):
        rewards = 0
        done = False
        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            if action == 0 and x > 0: x -= 1
            elif action == 1 and x < self.grid_size - 1: x += 1
            elif action == 2 and y > 0: y -= 1
            elif action == 3 and y < self.grid_size - 1: y += 1
            self.agent_positions[i] = (x, y)
        for pos in self.agent_positions:
            if pos == self.goal_position:
                rewards = 1
                done = True
                break
        self.steps += 1
        if self.steps >= 30:
            done = True
        return self._get_obs(), rewards, done

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        gx, gy = self.goal_position
        grid[gx, gy] = 'G'
        for i, (x, y) in enumerate(self.agent_positions):
            grid[x, y] = str(i)
        print("\n".join(" ".join(row) for row in grid))
        print("-" * 10)
