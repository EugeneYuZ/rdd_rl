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
        if self.memory[-1] and self.memory[-1][-1].next_state is None:
            self.memory.append([])
            if len(self.memory) > self.capacity:
                self.memory = self.memory[1:]

        self.memory[-1].append(Transition(*args))

    def sample(self, batch_size):
        if self.memory[-1] and self.memory[-1][-1].next_state is None:
            return random.sample(self.memory, batch_size)
        else:
            return random.sample(self.memory[:-1], batch_size)

    def __len__(self):
        return len(self.memory)


class DRQNAgent(DQNAgent):
    def __init__(self, model_class, model=None, env=None, exploration=None,
                 gamma=0.99, memory_size=10000, batch_size=1, target_update_frequency=10, saving_dir=None):
        """
        base class for lstm dqn agent
        :param model_class: sub class of torch.nn.Module. class reference of the model
        :param model: initial model of the policy net. could be None if loading from checkpoint
        :param env: environment
        :param exploration: exploration object. Must have function value(step) which returns e
        :param gamma: gamma
        :param memory_size: size of the memory
        :param batch_size: size of the mini batch for one step update
        :param target_update_frequency: the frequency for updating target net (in episode)
        :param saving_dir: the directory for saving checkpoint
        """
        DQNAgent.__init__(self, model_class, model, env, exploration, gamma, memory_size, batch_size,
                          target_update_frequency, saving_dir)
        self.memory = EpisodicReplayMemory(memory_size)
        self.hidden = None

    # def getInitialHidden(self, size=1):
    #     """
    #     get initial hidden state of all 0
    #     :return:
    #     """
    #     return (torch.zeros(1, size, self.hidden_size, device=self.device, requires_grad=False),
    #             torch.zeros(1, size, self.hidden_size, device=self.device, requires_grad=False))

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values, self.hidden = self.policy_net(state, self.hidden)
            q_values = q_values.squeeze(0)
            return q_values

    @staticmethod
    def unzipTransition(transition):
        state = transition.state
        next_state = transition.next_state
        reward = transition.reward
        action = transition.action

        return state, action, reward, next_state

    # def optimizeModel(self):
    #     if len(self.memory) < self.batch_size + 1:
    #         return
    #     memory = self.memory.sample(self.batch_size)
    #
    #     loss = 0
    #
    #     for transitions in memory:
    #         hidden = self.getInitialHidden()
    #         for transition in transitions:
    #             state, action, reward, next_state = self.unzipTransition(transition)
    #             q_values, hidden = self.policy_net((state, hidden))
    #             q_value = q_values.gather(1, action)
    #             if next_state is not None:
    #                 hidden_clone = (hidden[0].clone(), hidden[1].clone())
    #                 next_q_values, _ = self.target_net((next_state, hidden_clone))
    #                 next_q_value = next_q_values.max(1)[0].detach()
    #             else:
    #                 next_q_value = torch.zeros(1, device=self.device)
    #             expected_q_value = (next_q_value * self.gamma) + reward
    #
    #             advantage = expected_q_value - q_value
    #
    #             loss = loss + 0.5 * advantage.pow(2)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     for param in self.policy_net.parameters():
    #         param.grad.data.clamp_(-1, 1)
    #     self.optimizer.step()

    @staticmethod
    def unzipMemory(memory):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        padding = torch.zeros_like(memory[0][0].state)

        for episode in memory:
            episode_transition = Transition(*zip(*episode))
            state_batch.append(torch.cat(episode_transition.state))
            action_batch.append(torch.cat(episode_transition.action))
            # non_final_next_states = torch.cat([s for s in episode_transition.next_state
            #                                    if s is not None])
            non_final_next_states = torch.cat([s if s is not None else padding
                                               for s in episode_transition.next_state])
            next_state_batch.append(non_final_next_states)
            reward_batch.append(torch.cat(episode_transition.reward))

        return state_batch, action_batch, next_state_batch, reward_batch

    def optimizeModel(self):
        if len(self.memory) < self.batch_size + 1:
            return
        mini_memory = self.memory.sample(self.batch_size)
        mini_memory.sort(key=lambda x: len(x), reverse=True)

        state_batch, action_batch, next_state_batch, reward_batch = self.unzipMemory(mini_memory)
        episode_size = map(lambda x: x.shape[0], state_batch)
        padded_state = nn.utils.rnn.pad_sequence(state_batch, True)
        padded_next_state = nn.utils.rnn.pad_sequence(next_state_batch, True)
        padded_action = nn.utils.rnn.pad_sequence(action_batch, True)
        padded_reward = nn.utils.rnn.pad_sequence(reward_batch, True)

        state_action_values, _ = self.policy_net(padded_state, episode_size=episode_size)
        state_action_values = state_action_values.gather(2, padded_action).squeeze(2)
        target_state_action_values, _ = self.target_net(padded_next_state, episode_size=episode_size)
        target_state_action_values = target_state_action_values.max(2)[0].detach()

        expected_state_action_values = padded_reward

        final_mask = torch.zeros_like(target_state_action_values, dtype=torch.uint8)
        for i in range(len(episode_size)):
            final_mask[i, episode_size[i] - 1] = 1
        target_state_action_values[final_mask] = 0
        expected_state_action_values += self.gamma * target_state_action_values

        non_pad_mask = torch.ones_like(target_state_action_values, dtype=torch.uint8)
        for i in range(len(episode_size)):
            non_pad_mask[i, episode_size[i]:] = 0

        loss = F.mse_loss(state_action_values[non_pad_mask], expected_state_action_values[non_pad_mask])

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # def train(self, num_episodes, max_episode_steps=100, save_freq=100, render=False):
    #     while self.episodes_done < num_episodes:
    #         self.hidden = self.getInitialHidden()
    #         self.trainOneEpisode(num_episodes, max_episode_steps, save_freq, render)
    #     self.save_checkpoint()

    def trainOneEpisode(self, num_episodes, max_episode_steps=100, save_freq=100, render=False):
        self.hidden = None
        DQNAgent.trainOneEpisode(self, num_episodes, max_episode_steps, save_freq, render)

