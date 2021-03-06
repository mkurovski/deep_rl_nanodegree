"""
Implementation of neural network based reinforcement learning agent
"""
from copy import deepcopy

import numpy as np
import torch


class Agent(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size_1: int,
                 stochastic: bool = True):
        super(Agent, self).__init__()
        self.actions = list(range(action_size))
        self.linear_1 = torch.nn.Linear(state_size, hidden_size_1, bias=False)
        self.linear_2 = torch.nn.Linear(hidden_size_1, action_size, bias=False)
        self.softmax = torch.nn.Softmax(dim=0)
        self.stochastic = stochastic

    def _forward(self, state: np.ndarray):
        x = torch.from_numpy(state).float()
        x = self.linear_1(x)
        x = self.linear_2(x)

        return self.softmax(x)

    def get_action(self, state):
        action_probabilities = self._forward(state).detach().numpy()
        # sample an action according to the softmax probabilities (stochastic policy)
        if self.stochastic:
            action = np.random.choice(self.actions, size=1, p=action_probabilities)[0]
        else:
            action = np.argmax(action_probabilities)

        return action

    def get_params(self):
        params = dict.fromkeys(['weights_1', 'weights_2'])
        params['weights_1'] = self.linear_1.weight.detach()
        params['weights_2'] = self.linear_2.weight.detach()

        return params

    def set_params(self, new_params: dict):
        self.linear_1.weight = torch.nn.Parameter(new_params['weights_1'])
        self.linear_2.weight = torch.nn.Parameter(new_params['weights_2'])

    def watch_me(self, env, steps: int=500):
        """Shows the agent acting in the environment"""
        state = env.reset()
        for i in range(steps):
            env.render()
            action = self.get_action(state)
            next_state, reward, done, _ = env.step(action)
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
