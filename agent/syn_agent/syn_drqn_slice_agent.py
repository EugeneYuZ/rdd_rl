from collections import namedtuple, deque
import random
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.utils import *

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))


class SliceReplayMemory:
    def __init__(self, capacity, sequence_len):
        self.memory = deque()
        self.capacity = capacity
        self.sequence_len = sequence_len

    def push(self, episode):
        for transition in episode:
            if type(transition.state) is tuple:
                transition.state = map(lambda x: x.to('cpu'), transition.state)
            else:
                transition.state = transition.state.to('cpu')

            if transition.next_state is not None:
                if type(transition.next_state) is tuple:
                    transition.next_state = map(lambda x: x.to('cpu'), transition.next_state)
                else:
                    transition.next_state = transition.next_state.to('cpu')

            transition.action = transition.action.to('cpu')
            transition.reward = transition.reward.to('cpu')

        self.memory.append(episode)
        while self.__len__() > self.capacity:
            self.memory.popleft()

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