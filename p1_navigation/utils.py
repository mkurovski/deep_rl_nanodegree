import torch


def sample_experience_tuples(agent, env, num_samples: int) -> list:
    """Uses neural network as q-value function approximator
    and applies epsilon-greedy policy to determine action
    and obtain reward as well as the next state returned by the environment

    Returns a list of experience tuples [(S, A, R, S')]
    """
    brain_name = 'BananaBrain'
    assert(num_samples > 0)
    experience_tuples = []

    state = torch.Tensor(env.reset(train_mode=False)[brain_name].vector_observations[0])
    for _ in range(num_samples):
        action = agent.get_action(state)
        env_info = env.step(action)[brain_name]
        reward = env_info.rewards[0]
        next_state = torch.Tensor(env_info.vector_observations[0])
        experience_tuples.append((state, action, reward, next_state))
        done = env_info.local_done[0]
        if not done:
            state = next_state
        else:
            state = env.reset(train_mode=False)[brain_name].vector_observations[0]

    return experience_tuples
