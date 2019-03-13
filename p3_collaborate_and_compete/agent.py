"""
(MA)DDPG Agent implementation
"""
from typing import Dict, Tuple

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from utils import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment.

    Args:
        tau (float): interpolation parameter for soft model update

    """
    def __init__(self, state_size: int, action_size: int, agent_no: int, params: dict):
        """Initialize an Agent object.

        Args:
            state_size: dimension of each state
            action_size: dimension of each action
            agent_no: agent id
            params: architecture and hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = params['agent_seed']
        self.batch_size = params['batch_size']
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.critic_weight_decay = params['critic_weight_decay']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.update_step = params['update_step']
        self.num_agents = params['num_agents']

        random.seed(self.seed)
        self.t_step = 0
        self.agent_no = agent_no

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, params['first_hidden_units'],
                                 params['second_hidden_units'], self.seed).to(device)
        self.actor_target = Actor(state_size, action_size, params['first_hidden_units'],
                                  params['second_hidden_units'], self.seed).to(device)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * self.num_agents,
                                   action_size * self.num_agents,
                                   params['first_hidden_units'],
                                   params['second_hidden_units'],
                                   self.seed).to(device)
        self.critic_target = Critic(state_size * self.num_agents,
                                    action_size * self.num_agents,
                                    params['first_hidden_units'],
                                    params['second_hidden_units'],
                                    self.seed).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.lr_critic,
                                           weight_decay=self.critic_weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, self.seed, sigma=params['noise_sigma'])

    def step(self, memory: object, agents: Dict[int, object]):
        """Save experience in replay memory, and use random sample
        from buffer to learn every `update_step` if there are enough samples
        in the buffer to form a batch

        Args:
            memory: fixed-size buffer to store experience tuples
            agents: object references to Agent instances within the environment
        """
        self.t_step += 1

        if (len(memory) >= self.batch_size) & (self.t_step % self.update_step == 0):
            agents_experiences = memory.sample()
            self.learn(agents_experiences, agents)

    def act(self, state: np.array, add_noise: bool = True, scale: float = 1.0) -> np.array:
        """Returns actions for given state as per current policy.

        Args:
            state:
            add_noise: whether to add noise to actions for exploration during training
                or not (for evaluation)
            scale: noise scaling parameter

        Returrns:
            action:
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * scale
        np.clip(action, -1, 1)
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences: Dict[int, Tuple[torch.tensor, torch.tensor, torch.tensor,
                                                 torch.tensor, torch.tensor]],
              agents: Dict[int, object]):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Args:
            experiences: dictionary with agent-specific tuples
                with five tensors each that comprise states, actions, rewards,
                next_states, and dones batch_wise following exactly that order,
                i.e. tensor objects of size
                (`batch_size`, dim) where dim is `state_size` for states
                and next_states, `action_size` for actions, and 1 for rewards and dones
            agents: object references to Agent instances within the environment
        """
        self_rewards = experiences[self.agent_no][2]
        self_dones = experiences[self.agent_no][4]

        joint_next_states = torch.cat([experiences[no][3]
                                       for no in range(self.num_agents)], dim=1)

        # compute actions_next applying ea. agents target policy
        # on its next_states observations
        joint_actions_next = torch.cat([agents[no].actor_target(experiences[no][3])
                                        for no in range(self.num_agents)], dim=1)

        # --------------------------- update critic ---------------------------- #
        # compute the Q_targets (y) using the agent's target critic network with
        # on the next_states observations of all agents and joint_actions_next
        Q_targets_next = self.critic_target(joint_next_states, joint_actions_next)
        Q_targets = self_rewards + (self.gamma * Q_targets_next * (1 - self_dones))

        joint_states = torch.cat([experiences[no][0]
                                  for no in range(self.num_agents)], dim=1)
        joint_actions = torch.cat([experiences[no][1]
                                   for no in range(self.num_agents)], dim=1)

        # compute Q_expected applying the local critic to joint state observations
        # and all agents' actions
        Q_expected = self.critic_local(joint_states, joint_actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        joint_actions_pred = torch.cat([agents[no].actor_local(experiences[no][0])
                                        for no in range(self.num_agents)], dim=1)

        # Compute actor loss
        actor_loss = -self.critic_local(joint_states, joint_actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
