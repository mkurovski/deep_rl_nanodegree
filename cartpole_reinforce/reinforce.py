from collections import deque
from typing import List, Tuple

import numpy as np
import torch


def sample_trajectory(agent, env, gamma: float = 1., max_steps: int = 500) ->\
        (List[Tuple[np.array, int]], List[float], float):
    """
    Samples a single trajectories using `agent` in `env`
    and collects the state-action-sequence and optionally the
    log probabilities for the chose actions
    """
    state_action_sequence = []
    log_probabilities = []
    rewards = []

    state = env.reset()
    for step in range(max_steps):
        action, log_prob = agent.get_action(state)

        next_state, reward, done, _ = env.step(action.item())

        state_action_sequence.append((state, action.item()))
        log_probabilities.append(log_prob)
        rewards.append(reward * gamma ** step)

        state = next_state

        if done:
            break

    cum_reward = sum(rewards)

    return state_action_sequence, log_probabilities, cum_reward


def reinforce(agent, env, learning_rate: float = 0.01, num_episodes: int = 1000):
    """
    Samples `num_episodes` trajectories
    and performs gradient ascent step for each of them
    """
    return_hist = []
    return_deque = deque(maxlen=100)

    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    for episode in range(num_episodes):
        state_action_sequence, log_probabilities, cum_reward = \
            sample_trajectory(agent, env)

        policy_loss = torch.cat([-log_prob.unsqueeze(0) * cum_reward
                                 for log_prob in log_probabilities]).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        return_hist.append(cum_reward)
        return_deque.append(cum_reward)

        if episode % 100 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(episode,
                                                             np.mean(return_deque)))
        if np.mean(return_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                    episode - 100, np.mean(return_deque)))
            break

    return return_hist
