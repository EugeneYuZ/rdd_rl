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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))


class SliceReplayMemory:
    def __init__(self, capacity, sequence_len):
        self.memory = deque(maxlen=capacity)
        self.local_memory = []
        self.capacity = capacity
        self.sequence_len = sequence_len

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
            self.local_memory = []

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward, batch_final_mask, batch_pad_mask = [], [], [], [], [], []
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size)
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - self.sequence_len)
            transitions = episode[start:start + self.sequence_len]
            batch = Transition(*zip(*transitions))

            batch_state.append(torch.cat(list(batch.state)))
            batch_next_state.append(torch.cat(list(batch.next_state)))
            batch_action.append(torch.cat(list(batch.action)))
            batch_reward.append(torch.tensor(list(batch.reward)))
            batch_final_mask.append(torch.tensor(list(batch.final_mask), dtype=torch.float32))
            batch_pad_mask.append(torch.tensor(list(batch.pad_mask), dtype=torch.uint8))

        return Transition(batch_state, batch_action, batch_next_state, batch_reward, batch_final_mask, batch_pad_mask)

    def __len__(self):
        return len(self.memory)


class DRQNSliceAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=1, target_update_frequency=1000, saving_dir=None,
                 min_mem=None, sequence_len=32):
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)
        self.memory = SliceReplayMemory(memory_size, sequence_len)
        self.hidden = None
        if min_mem is None or min_mem < batch_size+1:
            min_mem = batch_size + 1
        self.min_mem = min_mem

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values, self.hidden = self.policy_net(state, self.hidden)
            q_values = q_values.squeeze(0)
            return q_values

    def optimizeModel(self):
        if len(self.memory) < self.min_mem:
            return

        mini_batch = self.memory.sample(self.batch_size)

        state = torch.stack(mini_batch.state).to(self.device)
        action = torch.stack(mini_batch.action).to(self.device)
        next_state = torch.stack(mini_batch.next_state).to(self.device)
        reward = torch.stack(mini_batch.reward).to(self.device)
        final_mask = torch.stack(mini_batch.final_mask).to(self.device)
        non_final_mask = 1 - final_mask
        pad_mask = torch.stack(mini_batch.pad_mask)
        non_pad_mask = 1 - pad_mask

        output, _ = self.policy_net(state)
        state_action_values = output.gather(2, action).squeeze(2)
        target_output, _ = self.target_net(next_state)
        target_values = target_output.max(2)[0].detach()
        expected_values = reward + non_final_mask * self.gamma * target_values

        loss = F.mse_loss(state_action_values[non_pad_mask], expected_values[non_pad_mask])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def trainOneEpisode(self, num_episodes, max_episode_steps=100, save_freq=100, render=False, print_step=True):
        self.hidden = None
        DQNAgent.trainOneEpisode(self, num_episodes, max_episode_steps, save_freq, render, print_step)


