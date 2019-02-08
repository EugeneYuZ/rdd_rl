from collections import namedtuple
import random
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.utils import *
from dqn_agent import DQNAgent


Transition = namedtuple('Transition', ('pre_state', 'state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.pre_state = None

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = (self.pre_state, args[0], args[1], args[2], args[3])
        self.memory[self.position] = Transition(*transition)
        if args[2] is None:
            self.pre_state = None
        else:
            self.pre_state = args[0]
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OneStepDRQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=1, target_update_frequency=10, saving_dir=None):
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)
        self.memory = ReplayMemory(memory_size)
        self.hidden_size = 0
        if self.policy_net:
            self.hidden_size = self.policy_net.hidden_size
        self.hidden = None

    def getInitialHidden(self):
        """
        get initial hidden state of all 0
        :return:
        """
        return (torch.zeros(1, self.hidden_size, device=self.device, requires_grad=False),
                torch.zeros(1, self.hidden_size, device=self.device, requires_grad=False))

    @staticmethod
    def getNonInitialPreStateBatch(mini_batch):
        non_final_pre_states = torch.cat([s for s in mini_batch.pre_state
                                          if s is not None])
        return non_final_pre_states

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            q_values, self.hidden = self.policy_net(state, self.hidden)
            return q_values

    def optimizeModel(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        mini_batch = Transition(*zip(*transitions))

        non_initial_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                  mini_batch.pre_state)), device=self.device, dtype=torch.uint8)
        hidden_state = self.getInitialHidden()
        non_initial_pre_states = self.getNonInitialPreStateBatch(mini_batch)
        hidden_batch = (hidden_state[0].repeat(self.batch_size, 1), hidden_state[1].repeat(self.batch_size, 1))
        _, non_initial_hidden = self.policy_net(non_initial_pre_states,
                                                (hidden_batch[0][non_initial_mask],
                                                 hidden_batch[1][non_initial_mask]))
        hidden_batch[0][non_initial_mask] = non_initial_hidden[0]
        hidden_batch[1][non_initial_mask] = non_initial_hidden[1]

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                mini_batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = self.getNonFinalNextStateBatch(mini_batch)
        state_batch = self.getStateBatch(mini_batch)
        action_batch = torch.cat(mini_batch.action)
        reward_batch = torch.cat(mini_batch.reward)

        state_action_values, hidden_batch = self.policy_net(state_batch, hidden_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        hidden_batch = (hidden_batch[0][non_final_mask], hidden_batch[1][non_final_mask])
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, hidden_batch)[0].max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def trainOneEpisode(self, *args, **kwargs):
        self.hidden = None
        DQNAgent.trainOneEpisode(self, *args, **kwargs)
