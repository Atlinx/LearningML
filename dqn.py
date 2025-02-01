import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
import math
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import itertools
from collections import deque

env = gym.make("CartPole-v1")

class Network(nn.Module):
	"""Neural network that takes in CartPole-v1 states and outputs q-values for every possible action.

	Args:
			nn (_type_): _description_
	"""
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(4, 64), # input 4 (size of state) -> inner layer 64
			nn.Tanh(), 				# activation func
			nn.Linear(64, 2)  # inner layer 64 -> output 2 (size of actions)
			)
	
	# forward propagation of NN
	def forward(self, t):
		t = self.net(t)
		return t
	
	def act(self, state):
		# preprocess state before feeding to nn
		# convert state into a tensor so we can feed it into the NN
		state_t = torch.as_tensor(state, dtype=torch.float32)
		# self() -> passes the inputs through the neural network
		#						essentially calls forward()
		# nn expects a batch dimension (usually trained on multiple samples at once)
		# we only want to train on one sample, but we still need to add batch dimension
		q_values = self(state_t.unsqueeze(0))
		# q_values contains all possible q_values
		# we want to choose the action (index) that has the highest value
		action = torch.argmax(q_values, dim=1).item()
		return action

# annealing
def decay(eps_start, eps_end, eps_decayrate, current_step):
	#                          1            ^ (eps_decayrate * current_step)
	# eps_start +  -----------------------
	#               (eps_start - eps_end)

	return eps_end + (eps_start - eps_end) * np.exp(-1 * eps_decayrate * current_step)

eps_count = 0   # number of training cycles
batch_size = 32 # batches per cycle in training stage
gamma = 0.99    # Bellman equation (horizon)

online_net = Network() # calculate initial guess of Q-value
target_net = Network() # calculate optimal Q-value
target_net.load_state_dict(online_net.state_dict()) # target_net + online_net should be identical

epsilon_start = 1
epsilon_end = 0.001
epsilon_decayrate = 0.003

episode_durations = [] # episode -> how long a "round" is
optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4) # optimize loss function

replay_memory = deque(maxlen=50000) # memory has size of 50k, after more, then it starts forgetting
avg_window_steps = deque([])

for t in range(1000):
	state, _ = env.reset()

	# while loop, to iterate as long as we need until the game ends (win or lose)
	for step in itertools.count():
		eps_count += 1
		epsilon = decay(epsilon_start, epsilon_end, epsilon_decayrate, eps_count)
		
		# take action (explore or exploit)
		if random.random() < epsilon:
			# explore -> get random action
			action = env.action_space.sample()
		else:
			# exploit
			action = online_net.act(state)
		
		s1, reward, term, trunc, info = env.step(action)
		done = term or trunc
		# state -> s1 from taking action
		experience = (state, action, reward, done, s1)
		replay_memory.append(experience)
		state = s1 # update state

		# train the network using a random batch of experiences from replay memory
		# we can only start training when len(replay_memory) >= batch_size
		if len(replay_memory) >= batch_size:
			experiences = random.sample(replay_memory, batch_size)
			# split state, action, and rewards into their own arrays
			states = np.asarray([e[0] for e in experiences])
			actions =  np.array([e[1] for e in experiences])
			rewards =  np.array([e[2] for e in experiences])
			dones =  np.array([e[3] for e in experiences])
			new_states =  np.array([e[4] for e in experiences])

			states_t = torch.as_tensor(states, dtype=torch.float32)
			actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
			rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
			dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
			new_states_t = torch.as_tensor(new_states, dtype=torch.float32)

			# guess what q-values are for each state in batch
			q_values: torch.Tensor = online_net(states_t)
			# find q-value for action we took in each batch
			# actions_t = [[0], [1], [1], [0], ...]
			# q_values = [[0.23, 0.43], [0.23, 0.43], ...]
			#								^     ^
			#					action_0   action_1
			# for each experience use each action to index into q_values array
			action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

			# predict optimal q-value
			target_q_output: torch.Tensor = target_net(new_states_t)
			# .max() returns          [[max values], [indices of max_values]]
			# we only care about max values --^
			target_q_values = target_q_output.max(dim=1, keepdim=True)[0]
			# bellman equation
			# q(s_t, a_t) = r_t + gamma * max Q(s_t+1, a_t+1)
			# we multiply by (1 - dones_t), because if we finished the game, then there are no future states
			optimal_q_values = rewards_t + gamma * (1 - dones_t) * target_q_values

			# calculate loss (difference) between our action's q-values and the optimal q-values
			# this difference exists if we didn't pick the optimal action at each state 
			loss = nn.functional.smooth_l1_loss(action_q_values, optimal_q_values)
			
			# reset the optimizer
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if eps_count % 1000 == 0:
			# every 1000 steps, reset target_net
			target_net.load_state_dict(online_net.state_dict())
		
		if done:
			avg_window_steps.append(step)
			if len(avg_window_steps) > 5:
				avg_window_steps.popleft()
			avg_steps = np.average(avg_window_steps)
			print(f"s: {step}   10-avg: {avg_steps}")
			if avg_steps >= 499:
				# game is done, run a simulation
				env = gym.make("CartPole-v1", render_mode="human")
				state, _ = env.reset()
				while True:
					action = online_net.act(state)
					state, _, term, trunc, _ = env.step(action)
					done = term or trunc
					env.render()
					if done:
						state, _ = env.reset()
			break