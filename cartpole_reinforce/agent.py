"""
Implementation of neural network based reinforcement learning agent
"""
from copy import deepcopy

import torch
from torch.distributions import Categorical


class Agent(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size_1: int,
                 is_stochastic: bool = True):
        super(Agent, self).__init__()
        self.linear_1 = torch.nn.Linear(state_size, hidden_size_1)
        self.linear_2 = torch.nn.Linear(hidden_size_1, action_size)
        self.softmax = torch.nn.Softmax(dim=0)
        self.is_stochastic = is_stochastic

    def _forward(self, x: torch.Tensor):
        x = torch.relu(self.linear_1(x))
        x = self.linear_2(x)

        return self.softmax(x)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self._forward(state)
        action_prob_dist = Categorical(action_probs)
        # sample an action according to the softmax probabilities (stochastic policy)
        if self.is_stochastic:
            action = action_prob_dist.sample()
        else:
            action = torch.argmax(action_prob_dist.probs)

        return action, action_prob_dist.log_prob(action)

    def watch_me(self, env, steps: int=500):
        """Shows the agent acting in the environment"""
        state = env.reset()
        for i in range(steps):
            env.render()
            action, _ = self.get_action(state)
            next_state, reward, done, _ = env.step(action.item())
            state = next_state
            if done:
                state = env.reset()

        env.close()


def add_noise(params: dict, noise_std: float = 0.1, noise_mean: float = 0.):
    """Adds Gaussian noise to Agent parameters"""
    params_cand = deepcopy(params)
    for key, val in params_cand.items():
        if val is not None:
            shape = val.shape
            params_cand[key] = val + (torch.randn(shape) * noise_std + noise_mean)

    return params_cand
