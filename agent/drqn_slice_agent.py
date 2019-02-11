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
        self.current_len = 0

    def push(self, *args):
        state, action, next_state, reward = args
        state = state.to('cpu')
        action = action.to('cpu')
        if next_state is not None:
            next_state = next_state.to('cpu')
        reward = reward.to('cpu')
        if next_state is not None:
            self.local_memory.append(Transition(state, action, next_state, reward, 0, 0))

        else:
            next_state = torch.zeros_like(state)
            self.local_memory.append(Transition(state, action, next_state, reward, 1, 0))
            while len(self.local_memory) < self.sequence_len:
                self.local_memory.append(Transition(
                    torch.zeros_like(state),
                    torch.tensor([[0]], dtype=torch.long),
                    torch.zeros_like(state),
                    0,
                    0,
                    1
                ))
            self.memory.append(self.local_memory)
            self.current_len += len(self.local_memory)
            if self.current_len > self.capacity:
                self.memory.popleft()
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_final_mask, batch_pad_mask = [], [], [], [], [], []
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - self.sequence_len)
            transitions = episode[start:start + self.sequence_len]
            batch = Transition(*zip(*transitions))

            batch_state.append(torch.cat(list(batch.state)))
            batch_next_state.append(torch.cat(list(batch.next_state)))
            batch_action.append(torch.cat(list(batch.action)))
            batch_reward.append(torch.tensor(list(batch.reward)))
            batch_final_mask.append(torch.tensor(list(batch.final_mask), dtype=torch.uint8))
            batch_pad_mask.append(torch.tensor(list(batch.pad_mask), dtype=torch.uint8))

        return Transition(batch_state, batch_action, batch_next_state, batch_reward, batch_final_mask, batch_pad_mask)

    def __len__(self):
        return len(self.memory)


class DRQNSliceAgent(DRQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=1, target_update_frequency=1000, saving_dir=None,
                 min_mem=None, sequence_len=32):
        DRQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                           target_update_frequency, saving_dir, min_mem)
        self.memory = SliceReplayMemory(memory_size, sequence_len)

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values, self.hidden = self.policy_net(state, self.hidden)
            q_values = q_values.squeeze(0)
            return q_values

    def unzipMemory(self, memory):
        state = torch.stack(memory.state).to(self.device)
        action = torch.stack(memory.action).to(self.device)
        next_state = torch.stack(memory.next_state).to(self.device)
        reward = torch.stack(memory.reward).to(self.device)
        final_mask = torch.stack(memory.final_mask).to(self.device)
        non_final_mask = 1 - final_mask
        pad_mask = torch.stack(memory.pad_mask)
        non_pad_mask = 1 - pad_mask

        return state, action, next_state, reward, final_mask, non_pad_mask
