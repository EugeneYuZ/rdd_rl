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

    def forwardPolicyNet(self, state):
        with torch.no_grad():
            q_values, self.hidden = self.policy_net((state, self.hidden))
            return q_values

    @staticmethod
    def unzipTransition(transition):
        state = transition.state
        next_state = transition.next_state
        reward = transition.reward
        action = transition.action

        return state, action, reward, next_state

    def optimizeModel(self):
        if len(self.memory) < self.batch_size + 1:
            return
        memory = self.memory.sample(self.batch_size)

        loss = 0

        for transitions in memory:
            hidden = self.getInitialHidden()
            for transition in transitions:
                state, action, reward, next_state = self.unzipTransition(transition)
                q_values, hidden = self.policy_net((state, hidden))
                q_value = q_values.gather(1, action)
                if next_state is not None:
                    hidden_clone = (hidden[0].clone(), hidden[1].clone())
                    next_q_values, _ = self.target_net((next_state, hidden_clone))
                    next_q_value = next_q_values.max(1)[0].detach()
                else:
                    next_q_value = torch.zeros(1, device=self.device)
                expected_q_value = (next_q_value * self.gamma) + reward

                advantage = expected_q_value - q_value

                loss = loss + 0.5 * advantage.pow(2)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes, max_episode_steps=100, save_freq=100):
        while self.episodes_done < num_episodes:
            self.hidden = self.getInitialHidden()
            self.trainOneEpisode(num_episodes, max_episode_steps, save_freq)
        self.save_checkpoint()

    def save_checkpoint(self):
        time_stamp = time.strftime('%Y%m%d%H%M%S', time.gmtime())
        filename = os.path.join(self.saving_dir, 'checkpoint' + time_stamp + '.pth.tar')
        state = {
            'episode': self.episodes_done,
            'steps': self.steps_done,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'hidden': self.hidden,
            'hidden_size': self.hidden_size,
            'memory': self.memory,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        torch.save(state, filename)

    def load_checkpoint(self, time_stamp):
        filename = os.path.join(self.saving_dir, 'checkpoint' + time_stamp + '.pth.tar')
        print 'loading checkpoint: ', filename
        checkpoint = torch.load(filename)
        self.episodes_done = checkpoint['episode']
        self.steps_done = checkpoint['steps']
        self.hidden = checkpoint['hidden']
        self.hidden_size = checkpoint['hidden_size']
        self.memory = checkpoint['memory']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']

        self.policy_net = self.model_class()
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_net = self.policy_net.to(self.device)
        self.policy_net.train()

        self.target_net = self.model_class()
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.target_net = self.target_net.to(self.device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
