from collections import deque, namedtuple
import random

import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(torch.nn.Module):
    """
    Neural Network Model to approximate action value function Q
    """
    def __init__(self, input_size: int = 37, output_size: int = 4,
                 hidden_layer_sizes: list = [64],
                 hidden_layer_activation_fn=torch.relu,
                 seed=42):
        """
        Initializes model parameters

        Args:
            input_size: state space size of the environment
            output_size:  action space size of the agent
            hidden_layer_sizes: units per hidden layer
            hidden_layer_activation_fn: activation function for hidden layer logits
            seed: random seed
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.hidden_layer_activation_fn = hidden_layer_activation_fn

        self.fc_input = torch.nn.Linear(input_size, hidden_layer_sizes[0])
        self.fc_output = torch.nn.Linear(hidden_layer_sizes[0], output_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass trough the model

        Args:
            state: instance(s) representing state to pass through the network

        Returns:
            output_logit: action values
        """
        input_logit = self.fc_input(state.float())
        input_activ = self.hidden_layer_activation_fn(input_logit)
        output_logits = self.fc_output(input_activ)

        return output_logits


class Agent:
    """Interacts with and learns from the environment"""
    def __init__(self, state_size: int, action_size: int, seed: int, hyperparams: dict):
        """
        Initializes agent instance

        Args:
            state_size: number of values to express a state observation
            action_size: number of unique actions the agent can perform
            seed: random seed
            hyperparams: instantiation and training hyperparameters
        """
        self.epsilon = hyperparams['eps_start']
        self.min_epsilon = hyperparams['eps_min']
        self.action_space = np.arange(action_size)
        self.seed = random.seed(seed)

        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(),
                                          lr=hyperparams['learn_rate'])

        self.buffer_size = int(1e5)
        self.batch_size = hyperparams['batch_size']
        self.gamma = hyperparams['gamma']

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        self.t_step = 0
        self.update_interval = hyperparams['update_interval']
        self.tau = hyperparams['tau']

    def get_action(self, state: np.array) -> int:
        if np.random.uniform() > self.epsilon:
            return self._get_greedy_action(torch.from_numpy(state))
        else:
            return np.random.choice(self.action_space)

    def _get_greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action_values = self.q_network(state)
        return torch.argmax(action_values, dim=0).item()

    def _needs_update(self):
        return self.t_step % self.update_interval == 0

    def _has_enough_samples(self):
        return len(self.memory) >= self.batch_size

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self._needs_update() & self._has_enough_samples():
            experiences = self.memory.sample()
            self.learn(experiences)

    def update_epsilon(self, eps_decay: float):
        self.epsilon = max(self.epsilon*eps_decay, self.min_epsilon)

    def learn(self, experiences: tuple):
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) Tensors
        """
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)

        # Compute and minimize the loss
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.q_network, self.target_q_network)

    def soft_update(self, local_model, target_model):
        """
        Performs soft update of target Q-network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


# Copied from https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/dqn_agent.py
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
