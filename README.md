# Deep Reinforcement Learning Nanodegree
Project Solutions for my [Deep Reinforcement Learning Nanodegree at Udacity](https://eu.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)


## Installation

1. Clone this repository with `git clone git@github.com:squall-1002/deep_rl_nanodegree.git`
2. Set up the conda environment `drlnd.yml` with [Anaconda](https://www.anaconda.com/): `conda env create -f drl.yml`
3. Activate the conda environment with `conda activate drl`


## Project 1: Navigation - training a robot to collect yellow(!) bananas

1. Download the Unity Environment from one of the applicable links below, place the file in `p1_navigation` and unzip the file:
	* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
	* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
	* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
	* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. In order to train your yellow banana picking agent, navigate into `p1_navigation` and start your JuPyter Notebook server with `jupyter notebook` within the activated environment.
3. Open `navigation_solution.ipynb` which is the notebook that guides you through the steps involved to
	* set everything up,
	* train the agent
	* and evaluate its performance.

You will see the a small window popping up that shows the agent quickly navigating through the environment and - hopefully - collecting the right colored bananas. You will also observe it getting better over time. Therefore, observe the average score counter in the notebook that tells you the average scores the agent achieved across the recent 100 episodes.

The environment is considered solved when the average score of the most recent 100 episodes surpasses `+13`.

<img src="img/p1_bananas.png" width="30%">



### Project Details

The state for the quadratic environment, which contains purple and yellow bananas, is represented with a real-valued vector of size 37. The agent can act upon this environment by moving forward, backward as well as turning left and right. These four actions constitute the action space as follows:

* `0`: move forward
* `1`: move backward
* `2`: turn left
* `3`: turn right

Rewards for collectings bananas are as follows:

* yellow banana: `+1`
* purple banana: `-1`

You may change the following hyperparameters in the dictionary `hyperparams` that is used to create an Agent instance:

* `eps_start`: Start probability to choose a random action when following an epsilon-greedy strategy
* `eps_min`: Minimum probability for epsilon-greedy strategy
* `eps_decay`: Decay Factor for `epsilon` applied every episode
* `learn_rate`: portion of the gradient to use for updating the parameters of the neural network that is used to approximate the action values during training
* `batch_size`: Number of single step experiences (state, action, reward, next state) to constitute a minibatch that is used for the gradient descent update
* `gamma`: discount factor used within the TD-target as part of the parameter update within (Deep) Q-Learning
* `update_interval`: number of steps to perform before updating the target network
* `tau`: interpolation parameter for target network update

Besides these hyperparameters you may also change the number of episodes `n_episodes` and the maximum number of steps `max_steps` before we force an otherwise unfinished episode to end. Both parameters are used for the `perform_dqn_training` method that trains our agent.

## Project 2: Continuous Control - train a double-joined arm to reach bananas

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. In order to train version 2 with 20 agents, navigate into `p2_continuous_control` and start your JuPyter Notebook server with `jupyter notebook` within the activated environment.
3. Open `continuous_control_solution.ipynb` which is the notebook that guides you through the steps involved to
	* set everything up,
	* train the agent
	* and evaluate its performance.

The environment is considered solved when the average score of the episode-wise 20-agent averages surpasses `+30` for the most recent 100 episodes.

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

![Trained Agent][image1]

### Project Details

The state for each agent, which is a double-joint arm, is represented with a real-valued vector of size 33 that correspond to position, rotation, velocity, and angular velocities of the two arm Rigidbodies. The agent can act upon this environment by moving forward, backward as well as turning left and right. The continuous action space is of size four and corresponds to torque applicable to two joints with valid values in the interval `[-1, 1]`

The agent's goal is to reach the sphere that move around and keep its hand with the sphere. For each timestep the agent adheres to this target it receives a reward of `0.1`.

You may change the following hyperparameters in the dictionary `hyperparams` that is used to create an Agent instance:

* `buffer_size`: Maximum number of samples that can be stored in the replay buffer queue
* `batch_size`: Number of single step experiences (state, action, reward, next state) to constitute a minibatch that is used for an agent update
* `update_step`: How many steps to sample before conducting an agent update
* `agent_seed`: Random seed used to initialize the neural network parameters and sampling generators
* `env_seed`: Random seed to initialize the environment
* `gamma`: discount factor used within the TD-target as part of the parameter update within (Deep) Q-Learning
* `tau`: interpolation parameter for soft target network update
* `lr_actor`: Learning rate for the Adam Optimizer used for updating the network parameters of the actor
* `lr_critic`: Learning rate for the Adam Optimizer used for updating the network parameters of the critic

Besides these hyperparameters you may also change the number of episodes `n_episodes` and the maximum number of steps `max_steps` before we force an otherwise unfinished episode to end. Both parameters are used for the `perform_ddpg_training` method that trains our agent.

## Project 3: Collaboration and Competition - train multiple agents to play tennis

tbd