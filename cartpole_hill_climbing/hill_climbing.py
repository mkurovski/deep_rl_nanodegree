"""
Module that implements different flavors and helpers of Hill Climbing Algorithms
for Policy-based strategies towards Reinforcement Learning
"""
from typing import List

import numpy as np

from agent import Agent, add_noise


def run_single_episode(agent, env) -> int:
    """
    Runs agent within an environment for a single episode
    and returns the cumulative reward
    """
    G = 0
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        G += reward
        state = next_state
        if done:
            break

    return G


def perform_hill_climbing(agent, env, n_episodes: int = 500,
                          population_size: int = 10,
                          use_adaptive_noise: bool = False,
                          initial_noise_std: float = 0.1,
                          noise_scaling_factor: float = 2.0,
                          min_noise_std: float = 0.01,
                          max_noise_std: float = 2.0,
                          initial_noise_mean: float = 0.0) -> (Agent, List[int]):
    """
    Performs Basic Hill Climbing Algorithm

    For `population_size` > 1 this method turns into steepest ascent hill climbing
        by picking selecting the parameters as update candidate that yield the highest return

    For `use_adaptive_noise` == True this method adapts the noise variance by
        decreasing `noise_std` for increasing returns and vice versa
        within the predefined boundaries `min_noise_std` and `max_noise_std`

    Args:
        agent:
        env:
        n_episodes:
        population_size:
        use_adaptive_noise:
        initial_noise_std:
        noise_scaling_factor:
        min_noise_std:
        max_noise_std:
        initial_noise_mean:

    Returns:
        agent:
    """
    noise_std = initial_noise_std
    noise_mean = initial_noise_mean

    best_params = agent.get_params()
    # estimated return of current policy
    G_best = run_single_episode(agent, env)
    return_history = [G_best]

    print("Start with Return Estimate G = %s" % G_best)

    for episode in range(n_episodes):
        params_cand_list = []
        G_list = []
        for _ in range(population_size):
            cand_params = add_noise(best_params)
            agent.set_params(cand_params)
            G = run_single_episode(agent, env)
            params_cand_list.append(cand_params)
            G_list.append(G)

        best_cand_idx = np.argmax(G_list)
        if G_list[best_cand_idx] > G_best:
            G_best = G_list[best_cand_idx]
            best_params = params_cand_list[best_cand_idx]
            print("Episode %  s - Improved G to G_best = %s" % (episode, G_best))
            if use_adaptive_noise:
                noise_std = max(min_noise_std, noise_std / noise_scaling_factor)
        else:
            if use_adaptive_noise:
                noise_std = min(max_noise_std, noise_std * noise_scaling_factor)

        agent.set_params(best_params)
        return_history.append(G)

    return agent, return_history
