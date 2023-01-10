import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		#self.l1 = nn.Linear(state_dim, 256)
		#self.l2 = nn.Linear(256, 256)
		#self.l3 = nn.Linear(state_dim, action_dim, bias=False)
		
		self.max_action = max_action
		#with torch.no_grad():
			#for param in self.parameters():
				#param.clamp_(0,50)
		self.weight = nn.Parameter(torch.randn((state_dim, action_dim)))

	def forward(self, state):
		#a = F.relu(self.l1(state))
		#a = F.relu(self.l2(a))
		#return self.max_action * torch.tanh(self.l3(a))
		#return self.l3(state)
		return torch.matmul(state,torch.abs(self.weight))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()
		size = 64
		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, size)
		self.l2 = nn.Linear(size, size)
		self.l3 = nn.Linear(size, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, size)
		self.l5 = nn.Linear(size, size)
		self.l6 = nn.Linear(size, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class Critic2(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic2, self).__init__()
		size = 32
		# Q1 architecture
		self.l1s = nn.Linear(state_dim, size)
		self.l1a = nn.Linear(action_dim, size)
		self.l2 = nn.Linear(2 * size, size)
		self.l3 = nn.Linear(size, 1)

		# Q2 architecture
		self.l4s = nn.Linear(state_dim, size)
		self.l4a = nn.Linear(action_dim, size)
		self.l5 = nn.Linear(2 * size, size)
		self.l6 = nn.Linear(size, 1)


	def forward(self, state, action):
		q1a = self.l1a(action)
		q1s = self.l1s(state)

		q1 = F.relu(torch.cat([q1a,q1s], 1))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2a = self.l4a(action)
		q2s = self.l4s(state)

		q2 = F.relu(torch.cat([q2a,q2s], 1))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		q1a = self.l1a(action)
		q1s = self.l1s(state)

		q1 = F.relu(torch.cat([q1a,q1s], 1))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=8e-4) #TODO higher learning rate

		self.critic = Critic(state_dim, action_dim).to(device) #TODO different critic
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-4) #TODO higher learning rate

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise * 20
			).clamp(-self.noise_clip, self.noise_clip)

			noise = 0 #TODO remove noise
			
			next_action = (
				self.actor_target(next_state) + noise ##TODO actor instead actor_target
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		#critic einfacher machhen
		

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		nn.utils.clip_grad_value_(self.critic.parameters(), clip_value=1.0) #TODO clipped grads
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			nn.utils.clip_grad_value_(self.actor.parameters(), clip_value=1.0) #TODO clipped grads
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			#with torch.no_grad():
				#for param in self.actor.parameters():
					#param.clamp_(1e-12, 50)
			

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
