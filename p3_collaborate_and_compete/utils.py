"""
Utilities used within Deep Reinforcement Learning Algorithms
"""
from collections import deque, namedtuple
import copy
import random
from typing import Dict, Tuple

import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
            buffer_size (int): maximum no. of samples in the replay buffer queue
            memory (deqeue): queue to keep examples up to a certain size
            batch_size (int): no. of samples that constitute a train batch
            num_agents (int): no. of agents acting in environment
            experience (namedtuple): container for one or several buffer samples
    """

    def __init__(self, params: dict):
        """Initialize a ReplayBuffer object.

        Args:
            params: parameters for replay buffer initialization
        """
        self.buffer_size = int(params['buffer_size'])
        self.memory = deque(maxlen=self.buffer_size)
        self.batch_size = params['batch_size']
        self.num_agents = params['num_agents']
        self.experience = namedtuple("Experience",
                                     field_names=["states",
                                                  "actions",
                                                  "rewards",
                                                  "next_states",
                                                  "dones"])
        random.seed(params['buffer_seed'])

    def add(self, states: torch.tensor, actions: torch.tensor, rewards: torch.tensor,
            next_states: torch.tensor, dones: torch.tensor):
        """Add a new experience to memory.

        Args:
            states:
            actions:
            rewards:
            next_states:
            dones:
        """
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self) -> Dict[int, Tuple[torch.tensor, torch.tensor, torch.tensor,
                                        torch.tensor, torch.tensor]]:
        """Randomly sample a batch of experiences from memory for multiple agents

        Returns:
            agents_experiences: dictionary with agent-specific tuples
                with five tensors each that comprise states, actions, rewards,
                next_states, and dones batch_wise following exactly that order,
                i.e. tensor objects of size
                (`batch_size`, dim) where dim is `state_size` for states
                and next_states, `action_size` for actions, and 1 for rewards and dones
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        agents_experiences = dict.fromkeys(range(self.num_agents))

        for no_agent in range(self.num_agents):
            states_batch = np.vstack(
                    [experience.states[no_agent] for experience in experiences])
            actions_batch = np.vstack(
                    [experience.actions[no_agent] for experience in experiences])
            rewards_batch = np.vstack(
                    [experience.rewards[no_agent] for experience in experiences])
            next_states_batch = np.vstack(
                    [experience.next_states[no_agent] for experience in experiences])
            dones_batch = np.vstack(
                    [experience.dones[no_agent] for experience in experiences]).astype(np.uint8)

            agents_experiences[no_agent] = (states_batch,
                                            actions_batch,
                                            rewards_batch,
                                            next_states_batch,
                                            dones_batch)

            agents_experiences[no_agent] = tuple(torch.from_numpy(batch).float().to(device)
                                                 for batch in agents_experiences[no_agent])

        return agents_experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        # self.state = None
        np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=x.size)
        self.state = x + dx
        return self.state
