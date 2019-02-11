"""
Module that implements different flavors and helpers of Hill Climbing Algorithms
for Policy-based strategies towards Reinforcement Learning
"""
from typing import Dict, List

import numpy as np
import torch

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
                          cem_frac: float = 0.2,
                          use_adaptive_noise: bool = False,
                          use_cem: bool = False,
                          use_evolution: bool = False,
                          initial_noise_std: float = 0.01,
                          noise_scaling_factor: float = 2.0,
                          min_noise_std: float = 0.001,
                          max_noise_std: float = 2.0,
                          initial_noise_mean: float = 0.0) -> (Agent, List[int]):
    """
    Performs Basic Hill Climbing Algorithm

    For `population_size` > 1 this method turns into steepest ascent hill climbing
        by picking selecting the parameters as update candidate that yield the highest return

    For `use_adaptive_noise` == True this method adapts the noise variance by
        decreasing `noise_std` for increasing returns and vice versa
        within the predefined boundaries `min_noise_std` and `max_noise_std`

    For `use_cem` == True this method uses Cross Entropy to determine the current return
        and parameters taking the average of the `cem_frac`*`population_size`
        candidates from the population

    For `use_evolution` == True this method uses Evolution Strategies to determine
        the candidate parameters by taking a return-weighted sum of the population policies

    Args:
        agent:
        env:
        n_episodes:
        population_size:
        cem_frac:
        use_adaptive_noise:
        use_cem:
        use_evolution:
        initial_noise_std:
        noise_scaling_factor:
        min_noise_std:
        max_noise_std:
        initial_noise_mean:

    Returns:
        agent:
        return_history:
    """
    n_best = max(1, int(cem_frac*population_size))

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

        if use_cem:
            G_cand, params_cand = get_cross_entropy_candidate(G_list, params_cand_list, n_best=n_best)
        elif use_evolution:
            params_cand = get_evolution_strategy_candidate(G_list, params_cand_list)
            agent.set_params(params_cand)
            G_cand = run_single_episode(agent, env)
        else:
            cand_idx = np.argmax(G_list)
            G_cand, params_cand = G_list[cand_idx], params_cand_list[cand_idx]

        if G_cand > G_best:
            G_best = G_cand
            best_params = params_cand
            print("Episode %  s - Improved G to G_best = %s" % (episode, G_best))
            if use_adaptive_noise:
                noise_std = max(min_noise_std, noise_std / noise_scaling_factor)
        else:
            if use_adaptive_noise:
                noise_std = min(max_noise_std, noise_std * noise_scaling_factor)

        agent.set_params(best_params)
        return_history.append(G)

    return agent, return_history


def get_cross_entropy_candidate(G_list: List[int], params_cand_list: List[dict],
                                n_best: int) -> (int, dict):
    """Applies CEM to candidates"""
    assert(len(G_list) >= n_best)

    best_idxs = np.argsort(G_list)[::-1][:n_best]
    cem_G = sum([G_list[idx] for idx in best_idxs])/n_best
    cem_cand_params = dict.fromkeys(params_cand_list[0].keys())

    for key in cem_cand_params.keys():
        tensor_list = [params_cand_list[idx][key] for idx in best_idxs]
        cum_tensor = torch.stack(tensor_list)
        cem_cand_params[key] = torch.mean(cum_tensor, dim=0)

    return cem_G, cem_cand_params


def get_evolution_strategy_candidate(G_list: List[int], params_cand_list: List[dict]) -> dict:
    """Applies Evolution Strategy to candidates"""
    cand_params = dict.fromkeys(params_cand_list[0].keys())

    for key in cand_params.keys():
        tensor_list = [params_cand_list[idx][key]*G for idx, G in enumerate(G_list)]
        cum_tensor = torch.stack(tensor_list)
        cand_params[key] = torch.sum(cum_tensor, dim=0) / sum(G_list)

    return cand_params
