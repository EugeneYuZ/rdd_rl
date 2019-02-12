from collections import namedtuple, deque
import random
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.utils import *
from dqn_agent import DQNAgent
from drqn_agent import DRQNAgent

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))


class SliceReplayMemory:
    def __init__(self, capacity, sequence_len):
        self.memory = deque()
        self.local_memory = []
        self.capacity = capacity
        self.sequence_len = sequence_len

    def push(self, *args):
        state, action, next_state, reward = args
        if type(state) is tuple:
            state = map(lambda x: x.to('cpu'), state)
        else:
            state = state.to('cpu')
        action = action.to('cpu')
        if next_state is not None:
            if type(next_state) is tuple:
                next_state = map(lambda x: x.to('cpu'), next_state)
            else:
                next_state = next_state.to('cpu')
        reward = reward.to('cpu')
        if next_state is not None:
            self.local_memory.append(Transition(state, action, next_state, reward, 0, 0))

        else:
            self.local_memory.append(Transition(state, action, next_state, reward, 1, 0))
            while len(self.local_memory) < self.sequence_len:
                self.local_memory.append(Transition(
                    None,
                    torch.tensor([[0]], dtype=torch.long),
                    None,
                    torch.tensor([0.]),
                    0,
                    1
                ))
            self.memory.append(self.local_memory)
            while self.__len__() > self.capacity:
                self.memory.popleft()
            self.local_memory = []

    def sample(self, batch_size):
        sample = []
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - self.sequence_len)
            transitions = episode[start:start + self.sequence_len]
            sample.append(transitions)
        return sample

    def __len__(self):
        return sum(map(len, self.memory))


class DRQNSliceAgent(DRQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=1, target_update_frequency=1000, saving_dir=None,
                 min_mem=10000, sequence_len=32):
        DRQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                           target_update_frequency, saving_dir, min_mem)
        self.memory = SliceReplayMemory(memory_size, sequence_len)
        self.state_padding = torch.zeros(self.env.observation_space.shape).unsqueeze(0)

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values, self.hidden = self.policy_net(state, self.hidden)
            q_values = q_values.squeeze(0)
            return q_values

    def unzipMemory(self, memory):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        final_mask_batch = []
        pad_mask_batch = []

        padding = self.state_padding

        for episode in memory:
            episode_transition = Transition(*zip(*episode))
            state_batch.append(torch.cat([s if s is not None else padding
                                          for s in episode_transition.state]))
            action_batch.append(torch.cat(episode_transition.action))
            next_state_batch.append(torch.cat([s if s is not None else padding
                                               for s in episode_transition.next_state]))
            reward_batch.append(torch.cat(episode_transition.reward))
            final_mask_batch.append(torch.tensor(list(episode_transition.final_mask), dtype=torch.uint8))
            pad_mask_batch.append(torch.tensor(list(episode_transition.pad_mask), dtype=torch.uint8))

        state = torch.stack(state_batch).to(self.device)
        action = torch.stack(action_batch).to(self.device)
        next_state = torch.stack(next_state_batch).to(self.device)
        reward = torch.stack(reward_batch).to(self.device)
        final_mask = torch.stack(final_mask_batch).to(self.device)
        pad_mask = torch.stack(pad_mask_batch)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask
