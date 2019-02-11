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


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class EpisodicReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [[]]
        self.position = 0

    def push(self, *args):
        state, action, next_state, reward = args
        state = state.to('cpu')
        action = action.to('cpu')
        if next_state is not None:
            next_state = next_state.to('cpu')
        reward = reward.to('cpu')

        if self.memory[-1] and self.memory[-1][-1].next_state is None:
            self.memory.append([])
            if len(self.memory) > self.capacity:
                self.memory = self.memory[1:]

        self.memory[-1].append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        if self.memory[-1] and self.memory[-1][-1].next_state is None:
            return random.sample(self.memory, batch_size)
        else:
            return random.sample(self.memory[:-1], batch_size)

    def __len__(self):
        return len(self.memory)


class DRQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=1, target_update_frequency=1000, saving_dir=None, min_mem=None):
        """
        base class for lstm dqn agent
        :param model_class: sub class of torch.nn.Module. class reference of the model
        :param model: initial model of the policy net. could be None if loading from checkpoint
        :param env: environment
        :param exploration: exploration object. Must have function value(step) which returns e
        :param gamma: gamma
        :param memory_size: size of the memory
        :param batch_size: size of the mini batch for one step update
        :param target_update_frequency: the frequency for updating target net (in steps)
        :param saving_dir: the directory for saving checkpoint
        """
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)
        self.memory = EpisodicReplayMemory(memory_size)
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

    def unzipMemory(self, memory):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        padding = torch.zeros_like(memory[0][0].state)

        memory.sort(key=lambda x: len(x), reverse=True)

        for episode in memory:
            episode_transition = Transition(*zip(*episode))
            state_batch.append(torch.cat(episode_transition.state))
            action_batch.append(torch.cat(episode_transition.action))
            non_final_next_states = torch.cat([s if s is not None else padding
                                               for s in episode_transition.next_state])
            next_state_batch.append(non_final_next_states)
            reward_batch.append(torch.cat(episode_transition.reward))

        episode_size = map(lambda x: x.shape[0], state_batch)

        padded_state = nn.utils.rnn.pad_sequence(state_batch, True).to(self.device)
        padded_next_state = nn.utils.rnn.pad_sequence(next_state_batch, True).to(self.device)
        padded_action = nn.utils.rnn.pad_sequence(action_batch, True).to(self.device)
        padded_reward = nn.utils.rnn.pad_sequence(reward_batch, True).to(self.device)

        final_mask = torch.zeros_like(padded_reward, dtype=torch.uint8)
        for i in range(len(episode_size)):
            final_mask[i, episode_size[i] - 1] = 1

        non_pad_mask = torch.ones_like(padded_reward, dtype=torch.uint8)
        for i in range(len(episode_size)):
            non_pad_mask[i, episode_size[i]:] = 0

        return padded_state, padded_action, padded_next_state, padded_reward, final_mask, non_pad_mask

    def optimizeModel(self):
        if len(self.memory) < self.min_mem:
            return
        mini_memory = self.memory.sample(self.batch_size)
        # mini_memory.sort(key=lambda x: len(x), reverse=True)

        state_batch, action_batch, next_state_batch, reward_batch, final_mask, non_pad_mask = self.unzipMemory(mini_memory)

        state_action_values, _ = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(2, action_batch).squeeze(2)
        target_state_action_values, _ = self.target_net(next_state_batch)
        target_state_action_values = target_state_action_values.max(2)[0].detach()

        expected_state_action_values = reward_batch

        target_state_action_values[final_mask] = 0
        expected_state_action_values += self.gamma * target_state_action_values

        loss = F.mse_loss(state_action_values[non_pad_mask], expected_state_action_values[non_pad_mask])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def trainOneEpisode(self, num_episodes, max_episode_steps=100, save_freq=100, render=False, print_step=True):
        self.hidden = None
        DQNAgent.trainOneEpisode(self, num_episodes, max_episode_steps, save_freq, render, print_step)

