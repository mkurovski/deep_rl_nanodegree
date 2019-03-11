"""
Module adapted from
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum/ddpg_agent.py
for training the OpenAI Gym's Pendulum environment
"""
import copy
from collections import namedtuple, deque

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agent_no, params):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
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
        self.actor_local = Actor(state_size, action_size, self.seed).to(device)
        self.actor_target = Actor(state_size, action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * self.num_agents,
                                   action_size * self.num_agents,
                                   self.seed).to(device)
        self.critic_target = Critic(state_size * self.num_agents,
                                    action_size * self.num_agents,
                                    self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.lr_critic,
                                           weight_decay=self.critic_weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, self.seed, sigma=params['noise_sigma'])

    def step(self, memory, agents):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.t_step += 1

        # if we need to control the learning of both agents we can also set again a random seed here
        if (len(memory) > self.batch_size) & (self.t_step % self.update_step == 0):
            experiences = memory.sample()
            self.learn(experiences, agents)

    def act(self, state, add_noise=True, scale=1.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * scale
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agents):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            states, actions, rewards, next states, dones
        """
        # self_states = experiences[self.agent_no][0]
        # self_actions = experiences[self.agent_no][1]

        self_rewards = experiences[self.agent_no][2]
        self_dones = experiences[self.agent_no][4]

        # compute actions_next based on states (0) or states_next (3)?
        joint_actions_next = torch.cat([agents[no].actor_target(experiences[no][3])
                                        for no in range(self.num_agents)], dim=1)

        joint_next_states = torch.cat([experiences[no][3]
                                       for no in range(self.num_agents)], dim=1)

        Q_targets_next = self.critic_target(joint_next_states, joint_actions_next)

        Q_targets = self_rewards + (self.gamma * Q_targets_next * (1 - self_dones))

        joint_states = torch.cat([experiences[no][0]
                                  for no in range(self.num_agents)], dim=1)
        joint_actions = torch.cat([experiences[no][1]
                                   for no in range(self.num_agents)], dim=1)

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

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        # self.seed = random.seed(seed)
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


# TODO: MK - edited
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, params):
        """Initialize a ReplayBuffer object.

        Params:
            params (dict):
        """
        self.memory = deque(maxlen=int(params['buffer_size']))  # internal memory (deque)
        self.batch_size = params['batch_size']
        self.experience = namedtuple("Experience",
                                     field_names=["states", "actions", "rewards",
                                                  "next_states", "dones"])
        self.num_agents = params['num_agents']
        random.seed(params['buffer_seed'])

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory for multiple agents"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # return a tuple with ea. element refering to an agent experience batch
        agent_experiences = dict.fromkeys(range(self.num_agents))
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
                    [experience.dones[no_agent] for experience in experiences]).astype(
                np.uint8)

            agent_experiences[no_agent] = (states_batch,
                                           actions_batch,
                                           rewards_batch,
                                           next_states_batch,
                                           dones_batch)

            agent_experiences[no_agent] = tuple(torch.from_numpy(batch).float().to(device)
                                                for batch in agent_experiences[no_agent])

        return agent_experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
