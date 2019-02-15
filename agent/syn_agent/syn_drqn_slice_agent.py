from collections import namedtuple, deque
import random
import time
import os

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from syn_dqn_agent import SynDQNAgent

from util.utils import *
from gym_test.wrapper import wrap_drqn


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'final_mask', 'pad_mask'))


class SliceReplayMemory:
    def __init__(self, capacity, sequence_len):
        self.memory = deque()
        self.capacity = capacity
        self.sequence_len = sequence_len

    def push(self, episode):
        self.memory.append(episode)
        while self.__len__() > self.capacity:
            self.memory.popleft()

    def sample(self, batch_size):
        sample = []
        batch_size = min(batch_size, len(self.memory))
        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, replace=False)
        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]
            start = random.randint(0, len(episode) - self.sequence_len)
            transitions = episode[start:start + self.sequence_len]
            sample.append(transitions)
        return sample

    def __len__(self):
        return sum(map(len, self.memory))


class SynDRQNAgent(SynDQNAgent):
    def __init__(self, model, envs, exploration,
                 gamma=0.99, memory_size=100000, batch_size=64, target_update_frequency=1000, saving_dir=None,
                 min_mem=1000, sequence_len=10):
        SynDQNAgent.__init__(self, model, envs, exploration, gamma, memory_size, batch_size, target_update_frequency,
                             saving_dir, min_mem)
        self.memory = SliceReplayMemory(memory_size, sequence_len)
        self.hidden = None
        self.local_memory = [[] for _ in range(self.n_env)]
        self.sequence_len = sequence_len

    def forwardPolicyNet(self, x):
        with torch.no_grad():
            state = x.unsqueeze(1)
            q_values, self.hidden = self.policy_net(state, self.hidden)
            q_values = q_values.squeeze(1)
            return q_values

    def unzipMemory(self, memory):
        state_batch = []
        action_batch = []
        next_state_batch = []
        reward_batch = []
        final_mask_batch = []
        pad_mask_batch = []

        padding = self.state_padding.to('cpu')

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

    def optimizeModel(self):
        if len(self.memory) < self.min_mem:
            return
        mini_memory = self.memory.sample(self.batch_size)

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

    def pushMemory(self, states, actions, next_states, rewards, dones):
        for i, idx in enumerate(self.alive_idx):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            reward = rewards[i]
            done = dones[i]

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

            if not done:
                self.local_memory[idx].append(Transition(state, action, next_state, reward, 0, 0))
            else:
                self.local_memory[idx].append(Transition(state, action, next_state, reward, 1, 0))
                while len(self.local_memory[idx]) < self.sequence_len:
                    self.local_memory[idx].append(Transition(
                        None,
                        torch.tensor([[0]], dtype=torch.long),
                        None,
                        torch.tensor([0.]),
                        0,
                        1
                    ))
                self.memory.push(self.local_memory[idx])
                self.local_memory[idx] = []

    def trainOneEpisode(self, num_episodes, max_episode_steps=100, save_freq=100):
        self.hidden = None
        SynDQNAgent.trainOneEpisode(self, num_episodes, max_episode_steps, save_freq)



