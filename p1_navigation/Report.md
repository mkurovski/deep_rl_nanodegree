# Project 1: Navigation - Report

This report describes the solution for the navigation evironment along with the algorithm, chosen network architecture and hyperparameters. I evaluate the performance across training in each episode and illustrate it along with target and average performance in the figure below. In addition, I conclude with a few ideas on further improvements.

The model parameters for the best performing agent, i.e. the Q-network parameters can be found in `best_banana_picker_agent.pth`. Please notice that they will be overwritten once you trigger training using the provided JuPyter notebook.

## Learning Algorithm

* learning algorithm: Q-Learning with a neural network for q-value function approximation (see and reference Mnih paper)
* hyperparameters (see README.md, choice of SGD flavor is also a hyperparameter, here it is Adaptive momentum, but it could also be vanilla SGD)

To train the agent maximizing its cumulated discounted reward per episode, I used deep Q-learning with experience replay presented in the [DeepMind paper from 2015 by Mnih et al.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) with the following hyperparameter choice:

* `eps_start`: 1.0
* `eps_min`: 0.01
* `eps_decay`: 0.995
* `learn_rate`: 0.0005
* `batch_size`: 64
* `gamma`: 0.99
* `update_interval`: 4
* `tau`: 0.001

The deep Q-network serves to approximate the best action-values (Q) and is a standard deep feedforward neural network. It consists of three layers (input, hidden, output) with 37 input units (37 continues values that represent a state), 64 hidden units that are activated using the rectified linear unit (ReLU) function, and 4 output units that represented the action values. All layers are fully connected without employing dropout regularization or batch normalization. Adaptive Momentum (Adam) is the stochatic gradient descent flavor I use for gradient computation and parameter updates.

## Performance Curve

We can see three lines in the figure below that depicts the performance of our agent across training. The blue line shows the score (sum of rewards per episode) obtained during each episode. The green line is the average score for the latest 100 episodes. The constant red line illustrates the threshold to surpass in order to consider our environment solved which is `+13` for this task.

In this run the agent was able to solve the environment in Episode 538, i.e. in this episode the average score across the most recent 100 episodes crossed was equal or higher than 13 for the first time.

We can also see that the agent keeps improving beyond that point. For example in this run where it finishes with an average score of `14.61` after 1000 episodes. Further training could further improve the performance, but will potentially find its threshold below 20 indicated by the decreasing marginal improvement.

![](performance_plot.png)

## Ideas for Future Work

* Improve implemented Replay Buffer moving towards **Prioritized Replay**, i.e. picking training samples not uniformly, but weighted by their expected contribution to loss minimization by prioritizing samples with high TD error higher as samples with lower TD error.
* Increase the complexity and thus the capacity of the underlying **neural network** that approximates the q-value function in order to obtain better q-values that yield better decisions and thus higher rewards.
* Increase **input resolution** by shifting from a 37-dimensional state representation to a more accurate state resolution, for example by using the frame sequences with raw pixels and a set of convolutional layers in the beginning of the neural network that are followed by a set of fully-connected layers
* Configure **small negative reward for each non-banana-collecting movement** to encourage the agent to pick up bananas in the most efficient way and thus potentially increase the number of bananas collected per episode
* Perform **extensive hyperparameters search** across:
	* learning rate
	* update interval 
	* discount factor
	* epsilon (minimum, decay factor)
	* train batch size
	* interpolation parameter for target network update


