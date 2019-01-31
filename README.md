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

ment is considered solved when the average score of the most recent 100 episodes surpasses `+13`.

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
* `learn_rate`: portion of the gradient to use for updating the parameters of the neural network that is used to approximate the action values during
* `batch_size`: Number of single step experiences (state, action, reward, next state) to constitute a minibatch that is used for the gradient descent update
* `gamma`: discount factor used within the TD-target as part of the parameter update within (Deep) Q-Learning
* `update_interval`: number of steps to perform before updating the target network
* `tau`: interpolation parameter for target network update

Besides these hyperparameters you may also change the number of episodes `n_episodes` and the maximum number of steps `max_steps` before we force an otherwise unfinished episode to end. Both parameters are used for the `perform_dqn_training` method that trains our agent.

**TODO** Add `train_mode = False` environment to show trained agent in test mode and potentially before (before training, training, after training)

## Project 2: Continuous Control - train a double-joined arm to reach bananas

tbd

## Project 3: Collaboration and Competition - train multiple agents to play tennis

tbd